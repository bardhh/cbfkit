"""Simulation engine for controlled dynamical systems.

Provides ``execute()`` to run a full simulation pipeline:
Planner -> Nominal Controller -> Safety Controller (CBF-CLF-QP) -> Plant Dynamics -> Integrator -> Sensor -> Estimator.
"""

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import os
import time

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
from jax import Array, random

from cbfkit.controllers.utils import setup_controller
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
)

from .backend import stepper
from .callbacks import LoggingCallback, ProgressCallback, SimulationCallback
from .formatting import format_return_data
from .simulator_jit import INTEGRATION_NAN_ERROR, simulator_jit
from .status import (
    SOLVER_STATUS_MAP,
    _check_simulation_status,
    _default_estimator,
    _default_perturbation,
    _default_sensor,
    _format_error_status,
)
from .ui import create_progress, print_jit_status
from .utils import SimulationStepData


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
    callbacks: Optional[List[SimulationCallback]] = None,
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
    controller_func: Optional[ControllerCallable] = (
        setup_controller(controller) if controller is not None else None
    )

    sigma_val: Array
    if sigma is None:
        sigma_val = jnp.zeros(0)  # Dummy sigma
    else:
        sigma_val = sigma

    _callbacks = callbacks if callbacks is not None else []

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
        controller=controller_func,
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
        for cb in _callbacks:
            cb.on_start(total_steps=num_steps, dt=dt)

        # Pre-allocate trajectory buffer (avoids O(N^2) concatenation in loop)
        needs_trajectory = stl_trajectory_cost is not None or planner is not None
        if needs_trajectory:
            xs_buf = jnp.zeros((x.shape[0], num_steps + 1))
            xs_buf = xs_buf.at[:, 0].set(x)

        for s in range(num_steps):
            x_ret, u_ret, z_ret, c_ret, controller_data, planner_data = step(
                dt * s, x, u, z, c, controller_data, planner_data
            )

            # Check for NaNs
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

            # O(1) trajectory update via pre-allocated buffer
            if needs_trajectory:
                xs_buf = xs_buf.at[:, s + 1].set(x)
                planner_data = planner_data._replace(xs=xs_buf[:, : s + 2])

            # Strip sampled_x_traj before logging to avoid accumulating
            # massive MPPI sample arrays (num_samples * state_dim * horizon per step)
            planner_data_for_log = planner_data._replace(sampled_x_traj=None)

            # Use the list for step data to avoid JAX array overhead for logging
            step_data = SimulationStepData(
                state=x,
                control=u,
                estimate=z,
                covariance=c,
                controller_keys=list(ControllerData._fields),
                controller_values=list(controller_data),
                planner_keys=list(PlannerData._fields),
                planner_values=list(planner_data_for_log),
            )

            for cb in _callbacks:
                cb.on_step(step_idx=s, time=dt * s, data=step_data)

            yield step_data

            if controller_data.complete:
                msg = "GOAL REACHED!"
                for cb in _callbacks:
                    cb.on_end(success=True, message=msg)
                break

            if controller_data.error:
                err_msg = (
                    _format_error_status(controller_data.error_data)
                    if controller_data.error_data is not None
                    else "Unknown error"
                )
                msg = f"CONTROLLER ERROR: {err_msg}"
                for cb in _callbacks:
                    cb.on_end(success=False, message=msg)
                break

            if planner_data.error:
                msg = "PLANNER ERROR"
                for cb in _callbacks:
                    cb.on_end(success=False, message=msg)
                break
        else:
            # Loop finished naturally
            for cb in _callbacks:
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
    goal: Optional[State] = None,
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
        goal (State, optional): Constant reference state (e.g., target pose).
            If provided, creates a constant trajectory in planner_data.
            Mutually exclusive with `planner_data.x_traj`. Defaults to None.
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
    # Validate dynamics output — single call, reused for all checks
    x0 = jnp.atleast_1d(jnp.asarray(x0))
    try:
        f_check, g_check = dynamics(x0)
    except Exception as e:
        raise ValueError(
            f"Dynamics evaluation failed for initial state 'x0' with shape {x0.shape}.\n"
            f"Ensure 'x0' has the correct dimensions for the system.\n"
            f"Original error: {e}"
        ) from e

    if f_check.shape != x0.shape:
        msg = (
            f"Shape mismatch: Initial state 'x0' has shape {x0.shape}, "
            f"but dynamics drift 'f' has shape {f_check.shape}.\n"
            "The state vector must match the dynamics output shape."
        )
        if x0.ndim == 2 and x0.shape[1] == 1 and f_check.ndim == 1:
            msg += "\nTip: Pass a 1D array for 'x0' (e.g., use x0.ravel() or x0.flatten())."
        elif x0.shape[0] < f_check.shape[0]:
            msg += f"\nTip: System expects {f_check.shape[0]} states, but got {x0.shape[0]}."
        raise ValueError(msg)

    if f_check.ndim != 1:
        msg = (
            f"Dynamics function returned `f` with shape {f_check.shape}. "
            "Expected 1D array (shape (n,)).\n"
        )
        if f_check.ndim == 2 and f_check.shape[1] == 1:
            msg += (
                "It appears `f` is a column vector (n, 1). "
                "Please squeeze it to (n,) (e.g., using jnp.squeeze or .flatten())."
            )
        raise ValueError(msg)

    if g_check.ndim != 2:
        raise ValueError(
            f"Dynamics function returned `g` with shape {g_check.shape}. "
            "Expected 2D array (shape (n, m))."
        )

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
        seed = 0
        env_seed = os.environ.get("CBFKIT_SEED")
        if env_seed is not None:
            try:
                seed = int(env_seed)
            except ValueError:
                pass

        key = random.PRNGKey(seed)  # type: ignore

    if controller is not None:
        controller = setup_controller(controller)

    # Ensure data structures are NamedTuples
    if controller_data is None:
        controller_data = ControllerData()
    elif isinstance(controller_data, dict):
        controller_data = ControllerData(**controller_data)

    # Process goal
    if goal is not None:
        goal_arr = jnp.atleast_1d(jnp.array(goal))
        if goal_arr.ndim == 1:
            goal_arr = goal_arr.reshape(-1, 1)

        if planner_data is None:
            planner_data = PlannerData(x_traj=goal_arr)
        elif isinstance(planner_data, dict):
            if planner_data.get("x_traj") is not None:
                raise ValueError("Cannot specify both 'goal' and 'planner_data[\"x_traj\"]'.")
            planner_data["x_traj"] = goal_arr
            # Convert to NamedTuple later
        elif isinstance(planner_data, PlannerData):
            if planner_data.x_traj is not None:
                raise ValueError("Cannot specify both 'goal' and 'planner_data.x_traj'.")
            planner_data = planner_data._replace(x_traj=goal_arr)

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
            # Strip sampled_x_traj from p_data to avoid carrying it in JIT loop
            p_data = p_data._replace(sampled_x_traj=None)

        if controller is not None:
            u_nom_dummy = jnp.zeros((g_check.shape[1],))
            _, c_data = controller(0.0, x0, u_nom_dummy, prime_key3, c_data)  # type: ignore

        # Ensure error_data is initialized to enable NaN reporting in JIT loop.
        # If controller is None, the loop propagates this structure, allowing us to report errors.
        if controller is None and c_data.error_data is None:
            c_data = c_data._replace(error_data=jnp.array(-99, dtype=jnp.int32))

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
