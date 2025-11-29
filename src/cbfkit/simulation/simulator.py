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

from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.tree_util
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
    State,
    Time,
)

from .backend import stepper
from .callbacks import LoggingCallback, ProgressCallback, SimulationCallback
from .simulator_jit import simulator_jit
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
    callbacks: List[SimulationCallback] = [],
    stl_trajectory_cost=None,
) -> Callable[
    [Array, Optional[ControllerData], Optional[PlannerData]],
    Iterator[SimulationStepData],
]:
    """Generates an iterator for simulating the dynamical system over a fixed horizon.

    ...
    """
    # Handle defaults for Optional callables
    sensor_func: SensorCallable
    if sensor is None:

        def _default_sensor(
            t: Time,
            x: Array,
            *,
            sigma: Optional[Array] = None,
            key: Optional[Array] = None,
            **kwargs: Any,
        ) -> Array:
            return x

        sensor_func = _default_sensor
    else:
        sensor_func = sensor

    estimator_func: EstimatorCallable
    if estimator is None:

        def _default_estimator(t, y, z, u, c):
            return y, c

        estimator_func = _default_estimator
    else:
        estimator_func = estimator

    perturbation_func: PerturbationCallable
    if perturbation is None:

        def _default_perturbation(x, u, f, g):
            def p(key):
                return jnp.zeros(x.shape)

            return p

        perturbation_func = _default_perturbation
    else:
        perturbation_func = perturbation

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
        if planner_data.xs is None:
            xs = jnp.copy(x.reshape(-1, 1))
        else:
            xs = planner_data.xs

        for s in range(num_steps):
            x_ret, u_ret, z_ret, c_ret, controller_data, planner_data = step(
                dt * s, x, u, z, c, controller_data, planner_data
            )
            u = u_ret
            z = z_ret
            c = c_ret
            x = x_ret

            xs = jnp.append(xs, x.reshape(-1, 1), axis=1)
            planner_data = planner_data._replace(xs=xs)

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
                    controller_data.error_data
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
    controller_data: Optional[ControllerData] = None,
    planner_data: Optional[PlannerData] = None,
    initial_covariance: Optional[Covariance] = None,
    stl_trajectory_cost=None,
    use_jit: bool = False,
) -> Tuple[Array, Array, Array, Array, List[str], List[Array], List[str], List[Array]]:
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

    Returns
    -------
        Tuple containing:
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

    if use_jit:
        # JIT Execution Path

        # Initialize data structures (already ensured to be NamedTuples or None)
        c_data = controller_data
        p_data = planner_data

        # Handle defaults (locally, as simulator_jit expects non-None)
        # This logic matches simulator() but must be explicit for JIT call prep
        _sensor = sensor if sensor else (lambda t, x, **k: x)
        _estimator = (
            estimator
            if estimator
            else (lambda t, y, z, u, c: (y, c if c is not None else jnp.zeros((len(y), len(y)))))
        )
        _perturbation = (
            perturbation if perturbation else (lambda x, u, f, g: (lambda k: jnp.zeros(x.shape)))
        )
        sigma_val = sigma if sigma is not None else jnp.zeros(0)

        if verbose:
            print("Warming up JIT...")

        prime_key1, prime_key2, prime_key3 = random.split(key, 3)  # type: ignore

        if planner is not None:
            _, p_data = planner(0.0, x0, None, prime_key1, p_data)  # type: ignore

        if controller is not None:
            f_dummy, g_dummy = dynamics(x0)
            u_nom_dummy = jnp.zeros((g_dummy.shape[1],))
            _, c_data = controller(0.0, x0, u_nom_dummy, prime_key3, c_data)  # type: ignore

        if verbose:
            print("JIT compilation/execution started. No progress bar will be shown.")

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
        )

        if verbose:
            print("JIT execution completed.")

        # Unpack JIT results into SimulationStepData list
        c_keys = list(c_datas._fields)
        p_keys = list(p_datas._fields)

        # We reconstruct the list of step data to support the return format and logging
        simulation_data_list: List[SimulationStepData] = []

        for t in range(num_steps):
            # Use tree_map to extract the t-th slice of the PyTree
            # This correctly handles nested structures (like dicts in sub_data)
            c_data_t = jax.tree_util.tree_map(lambda x: x[t], c_datas)
            p_data_t = jax.tree_util.tree_map(lambda x: x[t], p_datas)

            c_val_t = [getattr(c_data_t, k) for k in c_keys]
            p_val_t = [getattr(p_data_t, k) for k in p_keys]

            simulation_data_list.append(
                SimulationStepData(
                    state=xs[t],
                    control=us[t],
                    estimate=zs[t],
                    covariance=cs[t],
                    controller_keys=c_keys,
                    controller_values=c_val_t,
                    planner_keys=p_keys,
                    planner_values=p_val_t,
                )
            )

        simulation_data = tuple(simulation_data_list)

        # If logging was requested, we must simulate the callbacks behavior or use the data directly
        if logging_callback:
            # Populate the logging callback with the JIT data
            # This effectively "logs" the JIT run after the fact
            logging_callback.on_start(num_steps, dt)
            for idx, step_data in enumerate(simulation_data):
                logging_callback.on_step(idx, idx * dt, step_data)
            logging_callback.on_end(success=True)  # JIT runs don't fail mid-stream in the same way

        return format_return_data(simulation_data)

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

    return format_return_data(simulation_data)


def format_return_data(
    data: Tuple[SimulationStepData, ...],
) -> Tuple[Array, Array, Array, Array, List[str], List[Array], List[str], List[Array]]:
    """Extracts simulation data into JAX arrays."""

    states = jnp.array([step.state for step in data])
    controls = jnp.array([step.control for step in data])
    estimates = jnp.array([step.estimate for step in data])
    covariances = jnp.array([step.covariance for step in data])

    controller_data_keys = []
    controller_data_values = []
    planner_data_keys = []
    planner_data_values = []

    if len(data) > 0:
        # Controller Keys
        raw_c_keys = data[0].controller_keys
        for i, key in enumerate(raw_c_keys):
            vals = [step.controller_values[i] for step in data]

            # Check consistency and stackability
            first_valid = next((v for v in vals if v is not None), None)
            if first_valid is None or isinstance(first_valid, (dict, str, list, tuple)):
                continue
            if any(v is None for v in vals):
                continue

            try:
                arr = jnp.array(vals)
                controller_data_keys.append(key)
                controller_data_values.append(arr)
            except ValueError:
                # Skip fields that cannot be stacked (e.g. variable shapes)
                pass

        # Planner Keys
        raw_p_keys = data[0].planner_keys
        for i, key in enumerate(raw_p_keys):
            vals = [step.planner_values[i] for step in data]

            first_valid = next((v for v in vals if v is not None), None)
            if first_valid is None or isinstance(first_valid, (dict, str, list, tuple)):
                continue
            if any(v is None for v in vals):
                continue

            try:
                arr = jnp.array(vals)
                planner_data_keys.append(key)
                planner_data_values.append(arr)
            except ValueError:
                # Skip fields that cannot be stacked
                pass

    return (
        states,
        controls,
        estimates,
        covariances,
        controller_data_keys,
        controller_data_values,
        planner_data_keys,
        planner_data_values,
    )
