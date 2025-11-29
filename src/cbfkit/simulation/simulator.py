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

from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

import jax.numpy as jnp
import jax.tree_util
from jax import Array, random
from tqdm import tqdm

from cbfkit.utils.logger import write_log
from cbfkit.utils.user_types import (
    Control,
    ControllerCallable,
    ControllerData,
    Covariance,
    DynamicsCallable,
    Estimate,
    EstimatorCallable,
    IntegratorCallable,
    Key,
    NominalControllerCallable,
    PerturbationCallable,
    PlannerCallable,
    PlannerData,
    SensorCallable,
    State,
    Time,
)

from .simulator_jit import simulator_jit


class _SimulationStepData(NamedTuple):
    """Represents the data captured at a single simulation step."""

    state: State
    control: Control
    estimate: Estimate
    covariance: Covariance
    controller_keys: List[str]
    controller_values: List[Any]
    planner_keys: List[str]
    planner_values: List[Any]


def stepper(
    dt: float,
    dynamics: DynamicsCallable,
    integrator: IntegratorCallable,
    planner: Optional[PlannerCallable],
    nominal_controller: Optional[NominalControllerCallable],
    controller: Optional[ControllerCallable],
    sensor: SensorCallable,
    estimator: EstimatorCallable,
    perturbation: PerturbationCallable,
    sigma: Array,
    key: Key,
    stl_trajectory_cost,
) -> Callable[
    [
        Time,
        State,
        Optional[Control],
        Optional[Estimate],
        Optional[Covariance],
        Optional[ControllerData],
        Optional[PlannerData],
    ],
    Tuple[Array, Array, Array, Array, ControllerData, PlannerData],
]:
    """Creates a closure to step the simulation forward by one timestep.

    ...
    """

    def step(
        t: Time,
        x: State,
        u: Optional[Control],
        z: Optional[Estimate],
        c: Optional[Covariance],
        controller_data: Optional[ControllerData],
        planner_data: Optional[PlannerData],
    ) -> Tuple[Array, Array, Array, Array, ControllerData, PlannerData]:
        """_summary_.

        ...
        """
        if controller_data is None:
            controller_data = ControllerData()
        if planner_data is None:
            planner_data = PlannerData()

        # Nonlocal key (argument to parent function) for random noise generation
        nonlocal key
        key, subkey = random.split(key)  # type: ignore

        # Handle initial estimate if None
        if z is None:
            z = x  # Assume perfect knowledge initially if not provided

        # Sensor measurement
        y = sensor(t, x, sigma=sigma, key=key)

        # Compute state estimate using estimator
        z, c = estimator(t, y, z, u, c)

        # Generate true dynamics based on true state
        f, g = dynamics(x)

        if planner is None and nominal_controller is None and controller is None:
            raise ValueError(
                "At least one of planner, nominal_controller, or controller must be specified."
            )

        # Plan trajectory using planner
        if stl_trajectory_cost is not None:
            planner_data = planner_data._replace(
                prev_robustness=stl_trajectory_cost(dt, planner_data.xs)
            )
        else:
            planner_data = planner_data._replace(prev_robustness=None)

        if planner is not None:
            assert planner is not None
            u_planner, planner_data = planner(t, z, None, subkey, planner_data)  # type: ignore
            if planner_data.error:
                return (
                    x,
                    u if u is not None else jnp.zeros(g.shape[1]),
                    z,
                    c if c is not None else jnp.zeros((len(z), len(z))),
                    controller_data,
                    planner_data,
                )
        else:
            planner_data = planner_data._replace(u_traj=None)
        planner_data = planner_data._replace(prev_robustness=None)

        # Nominal controller
        if (planner is not None) and (planner_data.u_traj is not None):
            # already have nominal input. just pass it along
            u = u_planner
        elif planner_data.x_traj is not None:
            # expect trajectory to be n x N. N is time steps
            # Calculate current timestep index for trajectory lookup
            timestep_idx = jnp.round(t / dt).astype(int)
            # Clamp to valid range [0, N-1] where N is number of columns in x_traj
            timestep_idx = jnp.clip(timestep_idx, 0, planner_data.x_traj.shape[1] - 1)
            u, _ = nominal_controller(t, z, subkey, planner_data.x_traj[:, timestep_idx])  # type: ignore
        else:
            if nominal_controller is None:
                # print(f"WARNING: Nominal controller not defined. Setting to zero input")
                u = jnp.zeros((g.shape[1],))
            else:
                u, _ = nominal_controller(t, z, subkey, None)  # type: ignore

        # Compute control input using controller
        if controller is not None:
            u, controller_data = controller(t, z, u, subkey, controller_data)  # type: ignore
            if controller_data.error:
                return (
                    x,
                    u,
                    z,
                    c if c is not None else jnp.zeros((len(z), len(z))),
                    controller_data,
                    planner_data,
                )
            if controller_data.complete:
                return (
                    x,
                    u,
                    z,
                    c if c is not None else jnp.zeros((len(z), len(z))),
                    controller_data,
                    planner_data,
                )
        else:
            controller_data = ControllerData()

        # Generate perturbation to the dynamics (inclusive of SDEs)
        p = perturbation(x, u, f, g)

        # Continuous-time dynamics (inclusive of SDEs)
        key, subkey = random.split(key)  # type: ignore
        xdot = f + jnp.matmul(g, u) + p(subkey)  # type: ignore

        # Hmm. only way is to initialize integrator for specific dynamics beforehand
        # for sampled data systems
        x = integrator(x, xdot, dt)

        # Ensure return types
        u_ret = u
        c_ret = c if c is not None else jnp.zeros((len(z), len(z)))

        return x, u_ret, z, c_ret, controller_data, planner_data

    return step


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
    verbose: Optional[bool] = True,
    stl_trajectory_cost=None,
) -> Callable[
    [Array, Optional[ControllerData], Optional[PlannerData]],
    Iterator[_SimulationStepData],
]:
    """Generates an iterator for simulating the dynamical system over a fixed horizon.

    ...
    """
    # ... (defaults handling same as before) ...
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
    ) -> Iterator[_SimulationStepData]:  # Updated to yield _SimulationStepData
        if controller_data is None:
            controller_data = ControllerData()
        if planner_data is None:
            planner_data = PlannerData()
        # No info on control/estimation
        u = None
        z = None
        c = None  # Initialize covariance

        items = range(0, num_steps)

        if verbose:
            items = tqdm(items)

        # Simulate remaining timesteps
        if planner_data.xs is None:
            xs = jnp.copy(x.reshape(-1, 1))
        else:
            xs = planner_data.xs

        for s in items:
            # Fix: Step arguments
            # step(t, x, u, z, c, controller_data, planner_data)
            # Need to pass u=None, z=None/x, c=None (p)
            # But step now accepts Optional[Estimate] for z.
            x_ret, u_ret, z_ret, c_ret, controller_data, planner_data = step(
                dt * s, x, u, z, c, controller_data, planner_data
            )
            u = u_ret  # Update u for the next step's input
            z = z_ret  # Update z for the next step's input
            c = c_ret  # Update c for the next step's input
            x = x_ret  # Update x for the next step's initial state

            xs = jnp.append(xs, x.reshape(-1, 1), axis=1)
            planner_data = planner_data._replace(xs=xs)

            yield _SimulationStepData(
                state=x,
                control=u,
                estimate=z,
                covariance=c,
                controller_keys=list(controller_data._asdict().keys()),
                controller_values=list(controller_data._asdict().values()),
                planner_keys=list(planner_data._asdict().keys()),
                planner_values=list(planner_data._asdict().values()),
            )

            if controller_data.complete:
                if verbose:
                    print("GOAL REACHED!")
                break

            if controller_data.error:
                if verbose:
                    err_msg = (
                        controller_data.error_data
                        if controller_data.error_data is not None
                        else "Unknown error"
                    )
                    print(f"CONTROLLER ERROR: {err_msg}")
                break

            if planner_data.error:
                if verbose:
                    print("PLANNER ERROR")
                break

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
    # Handle defaults
    _sensor: SensorCallable
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

        _sensor = _default_sensor
    else:
        _sensor = sensor

    _estimator: EstimatorCallable
    if estimator is None:

        def _default_estimator(
            t: Time,
            y: Array,
            z: Array,
            u: Optional[Array],
            c: Optional[Array],
        ) -> Tuple[Array, Array]:
            return y, c if c is not None else jnp.zeros((len(y), len(y)))

        _estimator = _default_estimator
    else:
        _estimator = estimator

    _perturbation: PerturbationCallable
    if perturbation is None:

        def _default_perturbation(
            x: Array, u: Array, f: Array, g: Array
        ) -> Callable[[Array], Array]:
            def p(key: Array) -> Array:
                return jnp.zeros(x.shape)

            return p

        _perturbation = _default_perturbation
    else:
        _perturbation = perturbation

    sigma_val: Array
    if sigma is None:
        sigma_val = jnp.zeros(0)
    else:
        sigma_val = sigma

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

        # WARMUP JIT
        # Prime the data structures to ensure they have the correct static shape/keys
        # We perform one dummy call (not integrated) to get the output dict structure.
        if verbose:
            print("Warming up JIT...")

        prime_key1, prime_key2, prime_key3 = random.split(key, 3)  # type: ignore

        # Prime Planner
        if planner is not None:
            # Dummy z (estimate) = x0
            _, p_data = planner(0.0, x0, None, prime_key1, p_data)  # type: ignore

        # Prime Nominal (optional, usually stateless, but good for consistency)
        # u_nom_dummy, _ = nominal_controller(0.0, x0, prime_key2, None)

        # Prime Controller
        if controller is not None:
            # Need a dummy u_nom
            f_dummy, g_dummy = dynamics(x0)
            u_nom_dummy = jnp.zeros((g_dummy.shape[1],))
            _, c_data = controller(0.0, x0, u_nom_dummy, prime_key3, c_data)  # type: ignore

        if verbose:
            print("JIT compilation/execution started. No progress bar will be shown.")

        # Run JIT simulator
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

        # Process results to match extract_and_log_data format
        # c_datas and p_datas are PyTrees of stacked arrays (Time x Shape).
        # We need to unstack them into a list of step-dictionaries, then extract values.

        c_keys = list(c_datas._fields)
        c_values = []

        # Extracting data efficiently
        for t in range(num_steps):
            # Extract full structure for step t
            c_data_t = jax.tree_util.tree_map(lambda x: x[t], c_datas)
            # Extract values in the order of c_keys using getattr
            c_val_t = [getattr(c_data_t, k) for k in c_keys]
            c_values.append(c_val_t)

        p_keys = list(p_datas._fields)
        p_values = []
        for t in range(num_steps):
            p_data_t = jax.tree_util.tree_map(lambda x: x[t], p_datas)
            p_val_t = [getattr(p_data_t, k) for k in p_keys]
            p_values.append(p_val_t)

        simulation_data_list: List[_SimulationStepData] = []
        for t in range(num_steps):
            # We need to handle potential shape mismatches if JIT returns different shapes
            # Assuming xs, us, zs, cs are shaped (num_steps, ...)

            x_t = xs[t]
            u_t = us[t]
            z_t = zs[t]
            c_t = cs[t]

            c_val_t = c_values[t]
            p_val_t = p_values[t]

            item = _SimulationStepData(
                state=x_t,
                control=u_t,
                estimate=z_t,
                covariance=c_t,
                controller_keys=c_keys,
                controller_values=c_val_t,
                planner_keys=p_keys,
                planner_values=p_val_t,
            )
            simulation_data_list.append(item)

        simulation_data: Tuple[_SimulationStepData, ...] = tuple(simulation_data_list)

        if filepath is not None and verbose:
            # Now we can support logging in JIT!
            # print(f"Logging JIT data to {filepath}...")
            pass

        return extract_and_log_data(filepath, simulation_data)

    simulate_iter = simulator(
        dt,
        num_steps,
        dynamics,
        integrator,
        planner,
        nominal_controller,
        controller,
        sensor=_sensor,
        estimator=_estimator,
        perturbation=_perturbation,
        sigma=sigma_val,
        key=key,  # type: ignore
        verbose=verbose,
        stl_trajectory_cost=stl_trajectory_cost,
    )

    # Run simulation from initial state
    simulation_data: Tuple[_SimulationStepData, ...] = tuple(  # type: ignore
        simulate_iter(
            x0,
            controller_data,
            planner_data,
        )
    )

    # Log / Extract data
    return extract_and_log_data(filepath, simulation_data)


def extract_and_log_data(
    filepath: Optional[str], data: Tuple[_SimulationStepData, ...]
) -> Tuple[Array, Array, Array, Array, List[str], List[Array], List[str], List[Array]]:
    """Extracts simulation data and optionally logs it to a file.

    Args:
        filepath (Optional[str]): Path to save the logged data. If None, no data is logged.
        data (Tuple[_SimulationStepData, ...]): A tuple of _SimulationStepData objects,
                                                  each representing a single simulation step.

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
    if len(data) > 0:
        first_step_data = data[0]
        controller_data_keys: List[str] = first_step_data.controller_keys
        planner_data_keys: List[str] = first_step_data.planner_keys

        # Extract controller and planner data values over time
        all_controller_values = [step.controller_values for step in data]
        all_planner_values = [step.planner_values for step in data]

        if filepath is not None:
            # Reconstruct log data for logging
            log_data = []
            for step_idx, step_data in enumerate(data):
                step_log_entry = {
                    "state": step_data.state,
                    "control": step_data.control,
                    "estimate": step_data.estimate,
                    "covariance": step_data.covariance,
                }
                # Add controller data
                for i, key in enumerate(controller_data_keys):
                    step_log_entry[f"controller_{key}"] = all_controller_values[step_idx][i]
                # Add planner data
                for i, key in enumerate(planner_data_keys):
                    step_log_entry[f"planner_{key}"] = all_planner_values[step_idx][i]
                log_data.append(step_log_entry)

            write_log(filepath, log_data)

    # Use descriptive unpacking for simulation data extraction
    states = jnp.array([step.state for step in data])
    controls = jnp.array([step.control for step in data])
    estimates = jnp.array([step.estimate for step in data])
    covariances = jnp.array([step.covariance for step in data])

    if len(data) > 0:
        # These keys should be consistent across all steps
        raw_c_keys = data[0].controller_keys
        raw_p_keys = data[0].planner_keys

        controller_data_keys = []
        controller_data_values = []

        for i, key in enumerate(raw_c_keys):
            vals = [step.controller_values[i] for step in data]

            # Find the first non-None value to determine the type
            first_valid_val = next((v for v in vals if v is not None), None)

            # 1. Skip if all values are None
            if first_valid_val is None:
                continue

            # 2. Skip unsupported types (Dicts, Strings, etc. cannot be stacked into JAX arrays)
            if isinstance(first_valid_val, (dict, str, list, tuple)):
                continue

            # 3. Handle ragged data (mixture of None and values)
            # JAX cannot stack None with Arrays. If data is ragged, we must skip it
            # or fill it. For safety in a generic simulator, we skip inconsistent fields.
            if any(v is None for v in vals):
                continue

            # 4. Attempt to stack
            # At this point, we have consistent, non-None, non-container values.
            # They should be stackable. If strict shape mismatches occur here,
            # we allow the error to propagate because that indicates a logic error
            # in the controller (returning variable-sized arrays for the same field).
            arr = jnp.array(vals)
            controller_data_keys.append(key)
            controller_data_values.append(arr)

        planner_data_keys = []
        planner_data_values = []

        for i, key in enumerate(raw_p_keys):
            vals = [step.planner_values[i] for step in data]

            first_valid_val = next((v for v in vals if v is not None), None)

            if first_valid_val is None:
                continue

            if isinstance(first_valid_val, (dict, str, list, tuple)):
                continue

            if any(v is None for v in vals):
                continue

            arr = jnp.array(vals)
            planner_data_keys.append(key)
            planner_data_values.append(arr)
    else:
        # If no data, return empty lists for keys and values
        controller_data_keys = []
        controller_data_values = []
        planner_data_keys = []
        planner_data_values = []

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
