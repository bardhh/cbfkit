"""simulator.

This module contains the functions responsible for simulating the trajectories
of (controlled) dynamical systems over

Functions
---------
-function(a): description

Notes
-----
Various notes here

Examples
--------
>>> import title
>>> run code
"""

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import time

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
from jax import Array, random

from cbfkit.utils.user_types import (
    ControllerCallable,
    ControllerData,
    Covariance,
    DynamicsCallable,
    EstimatorCallable,
    IntegratorCallable,
    NominalControllerCallable,
    PerturbationCallable,
    PlannerCallable,
    PlannerData,
    SensorCallable,
    SimulationResults,
    State,
    StlTrajectoryCostCallable,
    Time,
)

from .backend import stepper
from .callbacks import LoggingCallback, ProgressCallback, SimulationCallback
from .simulator_jit import INTEGRATION_NAN_ERROR, simulator_jit
from .ui import create_progress, print_error, print_jit_status, print_warning
from .utils import SimulationStepData


SOLVER_STATUS_MAP = {
    -99: "NO_STATUS_AVAILABLE",
    -10: "INTEGRATION_NAN_ERROR",
    -2: "NAN_INPUT_DETECTED",
    -1: "NAN_DETECTED",
    0: "UNSOLVED (Likely Infeasible)",
    1: "SOLVED",
    2: "MAX_ITER_REACHED",
    3: "PRIMAL_INFEASIBLE",
    4: "DUAL_INFEASIBLE",
    5: "MAX_ITER_REACHED (UNSOLVED)",
}


def _format_error_status(status_code: Any) -> str:
    """Formats a solver status code into a human-readable string."""
    if isinstance(status_code, (int, float, jnp.ndarray, np.ndarray)):
        # Handle scalar arrays or ints
        try:
            code = int(status_code)
            if code in SOLVER_STATUS_MAP:
                return f"{SOLVER_STATUS_MAP[code]} (Status: {code})"
            else:
                return f"Status: {code}"
        except (ValueError, TypeError):
            pass
    return str(status_code)


def _check_simulation_status(
    controller_data_keys: List[str],
    controller_data_values: List[Array],
    planner_data_keys: List[str],
    planner_data_values: List[Array],
    nan_detected: bool = False,
) -> None:
    """Checks for simulation errors and prints warnings if found."""
    # Sentinel: Explicit check for NaNs
    if nan_detected:
        print_error("Simulation failed due to NaNs in state trajectory.")

    # Check controller errors
    if "error" in controller_data_keys:
        idx = controller_data_keys.index("error")
        errors = controller_data_values[idx]
        if jnp.any(errors):
            # Find first error index
            first_error_idx = int(jnp.argmax(errors).item())

            # Try to get error data/status
            status_msg = ""
            if "error_data" in controller_data_keys:
                idx_data = controller_data_keys.index("error_data")
                error_data = controller_data_values[idx_data]
                # Get status at the point of failure
                status = error_data[first_error_idx].item()
                status_msg = f" ({_format_error_status(status)})"

            print_warning(
                f"Simulation stopped early due to controller error at step {first_error_idx}{status_msg}."
            )

    # Sentinel: Warn if solver hit MAX_ITER_REACHED (status 2) but was accepted
    # We check 'sub_data_solver_status' (explicit) or 'error_data' (legacy/implicit)
    status_key = None
    if "sub_data_solver_status" in controller_data_keys:
        status_key = "sub_data_solver_status"
    elif "error_data" in controller_data_keys:
        status_key = "error_data"

    if status_key:
        idx_data = controller_data_keys.index(status_key)
        status_codes = controller_data_values[idx_data]
        # Check for status 2 (MAX_ITER_REACHED)
        max_iter_mask = status_codes == 2

        # Sentinel: Filter out cases where error occurred (avoid confusing "accepted" message)
        if "error" in controller_data_keys:
            idx_err = controller_data_keys.index("error")
            errors = controller_data_values[idx_err]
            # Ensure errors array matches shape (it should, as both are stacked per-step)
            # errors is boolean mask of errors
            max_iter_mask = max_iter_mask & (~errors)

        if jnp.any(max_iter_mask):
            count = int(jnp.sum(max_iter_mask).item())
            print_warning(
                f"Solver reached max iterations in {count} steps. "
                "Solutions were accepted but may be suboptimal."
            )

    # Check planner errors
    if "error" in planner_data_keys:
        idx = planner_data_keys.index("error")
        errors = planner_data_values[idx]
        if jnp.any(errors):
            first_error_idx = int(jnp.argmax(errors).item())
            print_warning(
                f"Simulation stopped early due to planner error at step {first_error_idx}."
            )


def _default_sensor(
    t: Time,
    x: Array,
    *,
    sigma: Optional[Array] = None,
    key: Optional[Array] = None,
    **kwargs: Any,
) -> Array:
    return x


def _default_estimator(t, y, z, u, c):
    return y, c if c is not None else jnp.zeros((len(y), len(y)))


def _default_perturbation(x, u, f, g):
    def p(key):
        return jnp.zeros_like(x)

    return p


def simulator(
    dt: float,
    num_steps: int,
    dynamics: DynamicsCallable,
    integrator: IntegratorCallable,
    planner: Optional[PlannerCallable],
    nominal_controller: Optional[NominalControllerCallable],
    controller: Optional[ControllerCallable],
    sensor: Optional[SensorCallable],
    estimator: Optional[EstimatorCallable],
    perturbation: Optional[PerturbationCallable],
    sigma: Optional[Array],
    key: Array,
    callbacks: List[SimulationCallback] = [],
    stl_trajectory_cost: Optional[StlTrajectoryCostCallable] = None,
) -> Callable[
    [Array, Optional[ControllerData], Optional[PlannerData]],
    Iterator[SimulationStepData],
]:
    """Generates an iterator for simulating the dynamical system over a fixed horizon.

    ...
    """
    # Handle defaults for Optional callables
    sensor_func: SensorCallable = sensor if sensor is not None else _default_sensor
    estimator_func: EstimatorCallable = estimator if estimator is not None else _default_estimator
    perturbation_func: PerturbationCallable = (
        perturbation if perturbation is not None else _default_perturbation
    )

    sigma_val: Array
    if sigma is None:
        sigma_val = jnp.zeros(0)  # Dummy sigma
    else:
        sigma_val = sigma

    assert sensor_func is not None
    assert estimator_func is not None
    assert perturbation_func is not None
    assert sigma_val is not None

    # Define step function
    step = stepper(
        dynamics=dynamics,
        sensor=sensor_func,
        planner=planner,
        nominal_controller=nominal_controller,
        controller=controller,
        estimator=estimator_func,
        perturbation=perturbation_func,
        integrator=integrator,
        sigma=sigma_val,
        dt=dt,
        key=key,
        stl_trajectory_cost=stl_trajectory_cost,
    )

    def simulate_iter(
        x: Array,
        controller_data: Optional[ControllerData] = None,
        planner_data: Optional[PlannerData] = None,
    ) -> Iterator[SimulationStepData]:
        if controller_data is None:
            controller_data = ControllerData()
        if planner_data is None:
            planner_data = PlannerData()
        # No info on control/estimation
        u = None
        z = None
        c = None  # Initialize covariance

        # Initialize callbacks
        for cb in callbacks:
            cb.on_start(total_steps=num_steps, dt=dt)

        # Simulate remaining timesteps
        xs_list = [x.reshape(-1, 1)]
        # If planner_data already has history, we should respect it, but here we assume
        # the simulation controls the history accumulation for the current run.
        # If planner_data.xs was passed in, we might need to prepend it to xs_list.
        if planner_data.xs is not None:
            # This converts existing history to a list of (state_dim, 1) arrays
            # This might be slow for long history, but it's a one-time cost at start.
            # However, efficient simulation usually starts from fresh or continues.
            # Let's just use the list for new steps.
            pass

        for s in range(num_steps):
            x_ret, u_ret, z_ret, c_ret, controller_data, planner_data = step(
                dt * s, x, u, z, c, controller_data, planner_data
            )

            # Sentinel: Check for NaNs
            nan_detected = jnp.any(jnp.isnan(x_ret))
            if nan_detected:
                controller_data = controller_data._replace(
                    error=jnp.array(True), error_data=jnp.array(INTEGRATION_NAN_ERROR)
                )

            u = u_ret
            z = z_ret
            c = c_ret
            # Clamp state if NaN (use previous x)
            x = x if nan_detected else x_ret

            # Efficient accumulation
            xs_list.append(x.reshape(-1, 1))

            # Update planner_data.xs ONLY if needed (e.g. for STL or if planner requires it)
            # We check if stl_trajectory_cost is provided or if planner is active.
            # To avoid O(N^2) at every step, we only stack if strictly necessary.
            # But if the planner interface *requires* .xs to be the full trajectory array every step,
            # we are bound by that interface.
            # Assuming we can optimize:
            if stl_trajectory_cost is not None:
                # We must pay the cost if STL cost depends on full trajectory
                xs = jnp.concatenate(xs_list, axis=1)
                planner_data = planner_data._replace(xs=xs)
            elif planner_data.xs is not None:
                # If planner_data.xs is being maintained, we might need to update it.
                # But if stl_trajectory_cost is None, maybe we can defer?
                # Current behavior was to ALWAYS update. Let's optimize to only update
                # if we suspect the planner needs it (which is hard to know without inspection).
                # For safety/compatibility, we can maintain the behavior but use concatenate which is slightly better?
                # No, concatenate on list is O(N). Doing it N times is O(N^2).
                # We will simply NOT update planner_data.xs every step unless forced.
                # BUT, the `step` function might have used `planner_data.xs`.
                # The `step` function takes `planner_data`. If `planner` inside `step` reads `xs`, it needs it.
                # If `planner` is None, `step` doesn't use `xs`.
                if planner is not None:
                    xs = jnp.concatenate(xs_list, axis=1)
                    planner_data = planner_data._replace(xs=xs)

            # Use the list for step data to avoid JAX array overhead for logging
            step_data = SimulationStepData(
                state=x,
                control=u,
                estimate=z,
                covariance=c,
                controller_keys=list(controller_data._asdict().keys()),
                controller_values=list(controller_data._asdict().values()),
                planner_keys=list(planner_data._asdict().keys()),
                planner_values=list(planner_data._asdict().values()),
            )

            for cb in callbacks:
                cb.on_step(step_idx=s, time=dt * s, data=step_data)

            yield step_data

            if controller_data.complete:
                msg = "GOAL REACHED!"
                for cb in callbacks:
                    cb.on_end(success=True, message=msg)
                break

            if controller_data.error:
                err_msg = (
                    _format_error_status(controller_data.error_data)
                    if controller_data.error_data is not None
                    else "Unknown error"
                )
                msg = f"CONTROLLER ERROR: {err_msg}"
                for cb in callbacks:
                    cb.on_end(success=False, message=msg)
                break

            if planner_data.error:
                msg = "PLANNER ERROR"
                for cb in callbacks:
                    cb.on_end(success=False, message=msg)
                break
        else:
            # Loop finished naturally
            for cb in callbacks:
                cb.on_end(success=True, message="")

    return simulate_iter


def execute(
    x0: State,
    dt: float,
    num_steps: int,
    dynamics: DynamicsCallable,
    integrator: IntegratorCallable,
    planner: Optional[PlannerCallable] = None,
    nominal_controller: Optional[NominalControllerCallable] = None,
    controller: Optional[ControllerCallable] = None,
    sensor: Optional[Union[SensorCallable, None]] = None,
    estimator: Optional[Union[EstimatorCallable, None]] = None,
    perturbation: Optional[Union[PerturbationCallable, None]] = None,
    sigma: Optional[Union[Array, None]] = None,
    key: Optional[Union[Array, None]] = None,
    filepath: Optional[str] = None,
    verbose: Optional[bool] = True,
    controller_data: Optional[Union[ControllerData, Dict[str, Any]]] = None,
    planner_data: Optional[Union[PlannerData, Dict[str, Any]]] = None,
    initial_covariance: Optional[Covariance] = None,
    stl_trajectory_cost: Optional[StlTrajectoryCostCallable] = None,
    use_jit: bool = False,
    jit_progress: bool = False,
    jit_progress_interval: int = 50,
) -> SimulationResults:
    """Executes a complete simulation of the dynamical system.

    This function runs the simulation for `num_steps` starting from `x0`.
    It can execute either in a standard Python loop or using JAX JIT compilation
    for performance.

    Args:
        x0 (State): Initial state vector.
        dt (float): Simulation timestep (seconds).
        num_steps (int): Number of steps to simulate.
        dynamics (DynamicsCallable): True system dynamics function.
        integrator (IntegratorCallable): Numerical integrator.
        planner (Optional[ControllerCallable], optional): Trajectory planner. Defaults to None.
        nominal_controller (Optional[ControllerCallable], optional): Nominal controller.
        Defaults to None.
        controller (Optional[ControllerCallable], optional): Safety controller (e.g., CBF).
        Defaults to None.
        sensor (Optional[SensorCallable], optional): Sensor model. Defaults to None.
        estimator (Optional[EstimatorCallable], optional): State estimator. Defaults to None.
        perturbation (Optional[PerturbationCallable], optional): Disturbance model.
        Defaults to None.
        sigma (Optional[Array], optional): Noise covariance/parameters. Defaults to None.
        key (Optional[Array], optional): Random key for noise. Defaults to None.
        filepath (Optional[str], optional): Path to save log file. Defaults to None.
        verbose (Optional[bool], optional): Print progress/status. Defaults to True.
        controller_data (ControllerData, optional): Initial controller data. Defaults to None.
        planner_data (PlannerData, optional): Initial planner data. Defaults to None.
        initial_covariance (Optional[Covariance], optional): Initial estimator covariance.
        Defaults to None.
        stl_trajectory_cost (Optional[Any], optional): STL cost object. Defaults to None.
        use_jit (bool, optional): If True, uses `jax.jit` for faster execution.
            JIT compilation requires that all callables are JIT-compatible. Defaults to False.
        jit_progress (bool, optional): Show a host-side progress bar during JIT execution.
            Disabled by default to avoid overhead; requires `verbose=True` to display.
        jit_progress_interval (int, optional): Number of steps between progress updates when
            `jit_progress` is enabled. Defaults to 50.

    Returns
    -------
        SimulationResults object containing:
            - states (Array): Trajectory of states (num_steps x state_dim).
            - controls (Array): Trajectory of control inputs.
            - estimates (Array): Trajectory of state estimates.
            - covariances (Array): Trajectory of covariances.
            - controller_keys (List[str]): Names of logged controller data fields.
            - controller_values (List[Array]): Logged controller data values.
            - planner_keys (List[str]): Names of logged planner data fields.
            - planner_values (List[Array]): Logged planner data values.
    """
    # Setup callbacks
    callbacks: List[SimulationCallback] = []
    if verbose:
        callbacks.append(ProgressCallback())

    logging_callback = None
    if filepath is not None:
        logging_callback = LoggingCallback(filepath)
        callbacks.append(logging_callback)

    # Generate key for randomization
    if key is None:
        key = random.PRNGKey(0)  # type: ignore

    # Ensure data structures are NamedTuples
    if controller_data is None:
        controller_data = ControllerData()
    elif isinstance(controller_data, dict):
        controller_data = ControllerData(**controller_data)

    if planner_data is None:
        planner_data = PlannerData()
    elif isinstance(planner_data, dict):
        planner_data = PlannerData(**planner_data)

    if jit_progress and jit_progress_interval <= 0:
        raise ValueError("jit_progress_interval must be positive when jit_progress is enabled.")

    if use_jit:
        # JIT Execution Path

        # Initialize data structures (already ensured to be NamedTuples or None)
        c_data = controller_data
        p_data = planner_data

        # Handle defaults (locally, as simulator_jit expects non-None)
        # This logic matches simulator() but must be explicit for JIT call prep
        _sensor = sensor if sensor is not None else _default_sensor
        _estimator = estimator if estimator is not None else _default_estimator
        _perturbation = perturbation if perturbation is not None else _default_perturbation
        sigma_val = sigma if sigma is not None else jnp.zeros(0)

        progress_bar = None
        progress_task_id = None
        progress_hook = None
        if verbose and jit_progress:
            progress_bar = create_progress(total=num_steps, description="JIT Simulation")
            progress_bar.start()
            progress_task_id = progress_bar.add_task("JIT Simulation", total=num_steps)
            last_step_reported = -1

            def progress_hook(step_idx: int) -> None:
                nonlocal last_step_reported
                if progress_bar is not None and progress_task_id is not None:
                    step_delta = step_idx - last_step_reported
                    if step_delta > 0:
                        progress_bar.update(progress_task_id, advance=step_delta)
                        last_step_reported = step_idx

        if verbose:
            print_jit_status("Warming up JIT...")

        prime_key1, prime_key2, prime_key3 = random.split(key, 3)  # type: ignore

        if planner is not None:
            _, p_data = planner(0.0, x0, None, prime_key1, p_data)  # type: ignore
            # Bolt: Strip sampled_x_traj from p_data to avoid carrying it in JIT loop
            p_data = p_data._replace(sampled_x_traj=None)

        if controller is not None:
            f_dummy, g_dummy = dynamics(x0)
            u_nom_dummy = jnp.zeros((g_dummy.shape[1],))
            _, c_data = controller(0.0, x0, u_nom_dummy, prime_key3, c_data)  # type: ignore

        if verbose:
            if progress_bar is not None:
                print_jit_status(
                    f"JIT compilation/execution started. Progress updates every "
                    f"{jit_progress_interval} steps."
                )
            else:
                print_jit_status(
                    "JIT compilation/execution started. No progress bar will be shown."
                )

        start_time = time.time()
        xs, us, zs, cs, c_datas, p_datas = simulator_jit(
            dt=dt,
            num_steps=num_steps,
            dynamics=dynamics,
            integrator=integrator,
            planner=planner,
            nominal_controller=nominal_controller,
            controller=controller,
            sensor=_sensor,
            estimator=_estimator,
            perturbation=_perturbation,
            sigma=sigma_val,
            key=key,  # type: ignore
            initial_state=x0,
            initial_controller_data=c_data,
            initial_planner_data=p_data,
            initial_covariance=initial_covariance,
            progress_callback=progress_hook,
            progress_interval=jit_progress_interval,
        )
        # Ensure progress callbacks flush before printing completion.
        xs.block_until_ready()
        elapsed = time.time() - start_time

        if verbose:
            print_jit_status(f"JIT execution completed in {elapsed:.4f}s.")

        if progress_bar is not None:
            progress_bar.stop()

        # If logging was requested, we must simulate the callbacks behavior
        if logging_callback:
            logging_callback.on_start(num_steps, dt)

            # Optimization (Bolt): Use bulk logging instead of per-step loop
            c_keys = list(c_datas._fields)
            p_keys = list(p_datas._fields)

            log_dict = {
                "state": list(np.array(xs)),
                "control": list(np.array(us)),
                "estimate": list(np.array(zs)),
                "covariance": list(np.array(cs)),
            }

            def process_bulk_data(keys, data_obj, prefix):
                for k in keys:
                    val = getattr(data_obj, k)
                    # val could be Array(T, ...), Dict[str, Array(T, ...)], or None
                    if val is None:
                        log_dict[f"{prefix}_{k}"] = [None] * num_steps
                    elif isinstance(val, dict):
                        # Unstack dict of arrays -> list of dicts
                        # First convert to numpy to speed up iteration
                        val_np = {}
                        for sk, sv in val.items():
                            try:
                                if isinstance(sv, tuple):
                                    val_np[sk] = list(zip(*sv))
                                else:
                                    val_np[sk] = list(np.array(sv))
                            except Exception:
                                val_np[sk] = [None] * num_steps
                        # zip now works on lists of values
                        vals = [dict(zip(val_np.keys(), t)) for t in zip(*val_np.values())]
                        log_dict[f"{prefix}_{k}"] = vals
                    else:
                        log_dict[f"{prefix}_{k}"] = list(np.array(val))

            process_bulk_data(c_keys, c_datas, "controller")
            process_bulk_data(p_keys, p_datas, "planner")

            logging_callback.log_bulk(log_dict)
            logging_callback.on_end(success=True)

        # Fast return path for JIT: Extract data directly from stacked arrays
        # This bypasses the expensive object creation loop + format_return_data

        controller_data_keys: List[str] = []
        controller_data_values: List[Array] = []
        for key in c_datas._fields:
            key_str = str(key)
            val = getattr(c_datas, key_str)
            if val is None:
                continue
            if isinstance(val, dict):
                # Flatten dictionary fields (e.g., sub_data)
                for sub_k, sub_v in val.items():
                    if isinstance(sub_v, (dict, list, tuple)):
                        continue
                    controller_data_keys.append(f"{key_str}_{sub_k}")
                    controller_data_values.append(sub_v)
                continue
            if isinstance(val, (list, tuple)):
                continue

            # We assume JIT results are arrays.
            # If a field was None in input and stayed None, it's None.
            controller_data_keys.append(key_str)
            controller_data_values.append(val)

        planner_data_keys: List[str] = []
        planner_data_values: List[Array] = []
        for key in p_datas._fields:
            key_str = str(key)
            val = getattr(p_datas, key_str)
            if val is None or isinstance(val, (dict, list, tuple)):
                continue
            planner_data_keys.append(key_str)
            planner_data_values.append(val)

        nan_detected = jnp.any(jnp.isnan(xs))
        _check_simulation_status(
            controller_data_keys,
            controller_data_values,
            planner_data_keys,
            planner_data_values,
            nan_detected=bool(nan_detected),
        )

        return SimulationResults(
            xs,
            us,
            zs,
            cs,
            controller_data_keys,
            controller_data_values,
            planner_data_keys,
            planner_data_values,
        )

    # Python Execution Path
    simulate_iter = simulator(
        dt,
        num_steps,
        dynamics,
        integrator,
        planner,
        nominal_controller,
        controller,
        sensor=sensor,
        estimator=estimator,
        perturbation=perturbation,
        sigma=sigma,
        key=key,  # type: ignore
        callbacks=callbacks,
        stl_trajectory_cost=stl_trajectory_cost,
    )

    # Run simulation from initial state
    simulation_data: Tuple[SimulationStepData, ...] = tuple(  # type: ignore
        simulate_iter(
            x0,
            controller_data,
            planner_data,
        )
    )

    formatted_data = format_return_data(simulation_data)

    (
        _,
        _,
        _,
        _,
        controller_data_keys,
        controller_data_values,
        planner_data_keys,
        planner_data_values,
    ) = formatted_data

    # Check states for NaNs
    nan_detected = False
    if len(formatted_data.states) > 0:
        nan_detected = jnp.any(jnp.isnan(formatted_data.states))

    _check_simulation_status(
        controller_data_keys,
        controller_data_values,
        planner_data_keys,
        planner_data_values,
        nan_detected=bool(nan_detected),
    )

    return formatted_data


def format_return_data(
    data: Tuple[SimulationStepData, ...],
) -> SimulationResults:
    """Extracts simulation data into JAX arrays."""
    if not data:
        return SimulationResults(
            jnp.array([]),
            jnp.array([]),
            jnp.array([]),
            jnp.array([]),
            [],
            [],
            [],
            [],
        )

    # Optimization: Transpose tuple of NamedTuples to NamedTuple of tuples
    # This avoids iterating over `data` multiple times and speeds up array stacking
    transposed = SimulationStepData(*zip(*data))

    states = jnp.stack(transposed.state)
    controls = jnp.stack(transposed.control)
    estimates = jnp.stack(transposed.estimate)
    covariances = jnp.stack(transposed.covariance)

    controller_data_keys = []
    controller_data_values = []
    planner_data_keys = []
    planner_data_values = []

    def process_keys_values(keys, values_tuple_of_lists):
        processed_keys = []
        processed_values = []

        if not values_tuple_of_lists:
            return processed_keys, processed_values

        # zip(*tuple_of_lists) -> list of tuples (per key)
        vals_by_key = list(zip(*values_tuple_of_lists))

        if not vals_by_key:
            return processed_keys, processed_values

        for i, key in enumerate(keys):
            vals = vals_by_key[i]

            # Check consistency and stackability
            first_valid = next((v for v in vals if v is not None), None)
            if first_valid is None or isinstance(first_valid, (dict, str, list, tuple)):
                continue

            # Handle mixed None/Numeric (e.g., error_data appearing mid-simulation)
            if any(v is None for v in vals):
                if isinstance(first_valid, (int, float, jnp.ndarray, np.ndarray)):
                    # Replace None with default
                    default_val = -99
                    # Check if float to use NaN instead
                    is_float = False
                    if hasattr(first_valid, "dtype"):
                        is_float = jnp.issubdtype(first_valid.dtype, jnp.floating)
                    elif isinstance(first_valid, float):
                        is_float = True

                    if is_float:
                        default_val = jnp.nan

                    vals = [v if v is not None else default_val for v in vals]
                else:
                    # Non-numeric mixed with None (unsupported)
                    continue

            try:
                # jnp.stack is more efficient than jnp.array for stacking existing arrays
                arr = jnp.stack(vals)
                processed_keys.append(key)
                processed_values.append(arr)
            except ValueError:
                # Skip fields that cannot be stacked
                pass
        return processed_keys, processed_values

    if len(data) > 0:
        controller_data_keys, controller_data_values = process_keys_values(
            data[0].controller_keys, transposed.controller_values
        )
        planner_data_keys, planner_data_values = process_keys_values(
            data[0].planner_keys, transposed.planner_values
        )

    return SimulationResults(
        states,
        controls,
        estimates,
        covariances,
        controller_data_keys,
        controller_data_values,
        planner_data_keys,
        planner_data_values,
    )
