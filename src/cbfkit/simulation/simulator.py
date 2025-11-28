"""
simulator
================

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

import jax.numpy as jnp
import jax.tree_util
from jax import Array, jit, random
from tqdm import tqdm

from cbfkit.utils.jax_stl import *
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
    PerturbationCallable,
    PlannerCallable,
    PlannerData,
    SensorCallable,
    State,
)

from .simulator_jit import simulator_jit


def stepper(
    dt: float,
    dynamics: DynamicsCallable,
    integrator: IntegratorCallable,
    planner: PlannerCallable,
    nominal_controller: ControllerCallable,
    controller: ControllerCallable,
    sensor: SensorCallable,
    estimator: EstimatorCallable,
    perturbation: PerturbationCallable,
    sigma: Array,
    key: random.PRNGKey,
    stl_trajectory_cost,
) -> Tuple[State, Dict[str, Any]]:
    """Step function to take the simulation forward one timestep. Designed
    to work generically with broad classes of dynamics, controllers, and
    estimators.

    Args:
        dynamics (Callable): function handle to compute true system dynamics
        sensor (Callable): function handle to generate new state sensor
        controller (Callable): function handle to compute control input
        estimator (Callable): function handle to compute new state estimate
        integrator (Callable): function handle to integrate over timestep for new state
        dt (float) timestep (sec)

    Returns:
        step (Callable): function handle for stepping forward in simulation time


    """

    def step(
        t: float,
        x: State,
        u: Control,
        z: Estimate,
        c: Covariance,
        controller_data: ControllerData,
        planner_data: PlannerData,
    ) -> Tuple[Array, Array, Array, Array, ControllerData, PlannerData]:
        """_summary_

        Args:
            t (float): time (sec)
            x (Array): state vector
            u (Array): control input vector
            z (Array): state estimate vector
            c (Array): covariance matrix of state estimate

        Returns:
            x (Array): state vector
            u (Array): control input vector
            z (Array): state estimate vector
            c (Array): covariance matrix of state estimate
            data (dict): contains other relevant simulation data
        """
        # Nonlocal key (argument to parent function) for random noise generation
        nonlocal key
        key, subkey = random.split(key)

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
            u_planner, planner_data = planner(t, z, None, subkey, planner_data)
            if planner_data.error:
                return x, u, z, c, controller_data, planner_data
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
            u, _ = nominal_controller(t, z, subkey, planner_data.x_traj[:, timestep_idx])
        else:
            if nominal_controller == None:
                # print(f"WARNING: Nominal controller not defined. Setting to zero input")
                u = jnp.zeros((g.shape[1],))
            else:
                u, _ = nominal_controller(t, z, subkey, None)

        # Compute control input using controller
        if controller is not None:
            u, controller_data = controller(t, z, u, subkey, controller_data)
            if controller_data.error:
                return x, u, z, c, controller_data, planner_data
            if controller_data.complete:
                return x, u, z, c, controller_data, planner_data
        else:
            controller_data = ControllerData()

        # Generate perturbation to the dynamics (inclusive of SDEs)
        p = perturbation(x, u, f, g)

        # Continuous-time dynamics (inclusive of SDEs)
        key, subkey = random.split(key)
        xdot = f + jnp.matmul(g, u) + p(subkey)

        # Hmm. only way is to initialize integrator for specific dynamics beforehand
        # for sampled data systems
        x = integrator(x, xdot, dt)

        return x, u, z, c, controller_data, planner_data

    return step


def simulator(
    dt: float,
    num_steps: int,
    dynamics: DynamicsCallable,
    integrator: IntegratorCallable,
    planner: PlannerCallable,
    nominal_controller: ControllerCallable,
    controller: ControllerCallable,
    sensor: SensorCallable,
    estimator: EstimatorCallable,
    perturbation: PerturbationCallable,
    sigma: Array,
    key: random.PRNGKey,
    verbose: Optional[bool] = True,
    stl_trajectory_cost=None,
) -> Callable[[Array], Iterator[Tuple[Array, Array, Array, Array, List[str], List[Array]]]]:
    """Generates function handle for the iterator that carries out the simulation of the
    dynamical system.

    Args:
        dynamics (Callable): function handle to compute true system dynamics
        sensor (Callable): function handle to generate new state sensor
        controller (Callable): function handle to compute control input
        estimator (Callable): function handle to compute new state estimate
        dt (float) timestep (sec)
        num_steps (int): number of timesteps to simulate
        key (random.PRNGKey): key for generating random stochastic noise

    Returns:
        iterator eventually returning the following:
            x (Array): state vector
            u (Array): control input vector
            z (Array): state estimate vector
            p (Array): state estimate covariance matrix
            keys (list): list of keys in data dict
            vals (list): lists of values contained in data

    """
    # Define step function
    step = stepper(
        dynamics=dynamics,
        sensor=sensor,
        planner=planner,
        nominal_controller=nominal_controller,
        controller=controller,
        estimator=estimator,
        perturbation=perturbation,
        integrator=integrator,
        sigma=sigma,
        dt=dt,
        key=key,
        stl_trajectory_cost=stl_trajectory_cost,
    )

    def simulate_iter(
        x: Array,
        controller_data: ControllerData = None,
        planner_data: PlannerData = None,
    ) -> Iterator[Tuple[Array, Array, Array, Array, List[str], List[Array]]]:
        if controller_data is None:
            controller_data = ControllerData()
        if planner_data is None:
            planner_data = PlannerData()
        # No info on control/estimation
        u = None
        z = None
        p = None

        items = range(0, num_steps)

        if verbose:
            items = tqdm(items)

        xs = jnp.copy(x.reshape(-1, 1))

        # Simulate remaining timesteps
        planner_data = planner_data._replace(xs=xs)
        for s in items:
            x, u, z, p, controller_data, planner_data = step(
                dt * s, x, u, z, p, controller_data, planner_data
            )
            # Removed log() call here
            xs = jnp.append(xs, x.reshape(-1, 1), axis=1)
            planner_data = planner_data._replace(
                xs=jnp.append(planner_data.xs, x.reshape(-1, 1), axis=1)
            )
            # None  # xs
            yield x, u, z, p, list(controller_data._asdict().keys()), list(
                controller_data._asdict().values()
            ), list(planner_data._asdict().keys()), list(planner_data._asdict().values())

            if controller_data.complete:
                if verbose:
                    print("GOAL REACHED!")
                break

            if controller_data.error:
                if verbose:
                    print(
                        f"CONTROLLER ERROR: {controller_data.error_data if controller_data.error_data is not None else 'Unknown error'}"
                    )
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
    planner: Optional[Union[ControllerCallable, None]] = None,
    nominal_controller: Optional[Union[ControllerCallable, None]] = None,
    controller: Optional[Union[ControllerCallable, None]] = None,
    sensor: Optional[Union[SensorCallable, None]] = None,
    estimator: Optional[Union[EstimatorCallable, None]] = None,
    perturbation: Optional[Union[PerturbationCallable, None]] = None,
    sigma: Optional[Union[Array, None]] = None,
    key: Optional[Union[random.PRNGKey, None]] = None,
    filepath: Optional[str] = None,
    verbose: Optional[bool] = True,
    controller_data: ControllerData = None,
    planner_data: PlannerData = None,
    initial_covariance: Optional[Covariance] = None,
    stl_trajectory_cost=None,
    use_jit: bool = False,
) -> Tuple[Array, List[str], List[Array]]:
    """This function simulates the dynamical system for a given number of steps..."""
    if nominal_controller is None:
        pass

    if controller is None:
        pass

    if planner is None:
        pass

    if sensor is None:
        pass

    if estimator is None:
        pass

    if perturbation is None:

        def perturbation(x, _u, _f, _g):
            def p(_subkey):
                return jnp.zeros(x.shape)

            return p

    if sigma is None:
        pass

    # Generate key for randomization
    if key is None:
        key = random.PRNGKey(0)

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

        prime_key1, prime_key2, prime_key3 = random.split(key, 3)

        # Prime Planner
        if planner is not None:
            # Dummy z (estimate) = x0
            _, p_data = planner(0.0, x0, None, prime_key1, p_data)

        # Prime Nominal (optional, usually stateless, but good for consistency)
        # u_nom_dummy, _ = nominal_controller(0.0, x0, prime_key2, None)

        # Prime Controller
        if controller is not None:
            # Need a dummy u_nom
            f_dummy, g_dummy = dynamics(x0)
            u_nom_dummy = jnp.zeros((g_dummy.shape[1],))
            _, c_data = controller(0.0, x0, u_nom_dummy, prime_key3, c_data)

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
            sensor=sensor,
            estimator=estimator,
            perturbation=perturbation,
            sigma=sigma,
            key=key,
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

        simulation_data = []
        for t in range(num_steps):
            # We need to handle potential shape mismatches if JIT returns different shapes
            # Assuming xs, us, zs, cs are shaped (num_steps, ...)

            x_t = xs[t]
            u_t = us[t]
            z_t = zs[t]
            c_t = cs[t]

            c_val_t = c_values[t]
            p_val_t = p_values[t]

            item = (x_t, u_t, z_t, c_t, c_keys, c_val_t, p_keys, p_val_t)
            simulation_data.append(item)

        simulation_data = tuple(simulation_data)

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
        sensor,
        estimator,
        perturbation,
        sigma,
        key,
        verbose,
        stl_trajectory_cost,
    )

    # Run simulation from initial state
    simulation_data = tuple(
        simulate_iter(
            x0,
            controller_data=controller_data,
            planner_data=planner_data,
        )
    )

    # Log / Extract data
    return extract_and_log_data(filepath, simulation_data)


#! Finish this function
def extract_and_log_data(filepath: str, data):
    """_summary_"""

    if len(data) > 0:
        controller_data_keys = data[0][4]
        controller_data_values = [sim_data[5] for sim_data in data]

        if filepath is not None:
            # Reconstruct log data
            log_data = []
            for i, values in enumerate(controller_data_values):
                # values is a list of values corresponding to keys
                entry = dict(zip(controller_data_keys, values))
                log_data.append(entry)

            write_log(filepath, log_data)

    #! Somehow make these more modular?
    states = jnp.array([sim_data[0] for sim_data in data])
    controls = jnp.array([sim_data[1] for sim_data in data])
    estimates = jnp.array([sim_data[2] for sim_data in data])
    covariances = jnp.array([sim_data[3] for sim_data in data])

    if len(data) > 0:
        controller_data_keys = data[0][4]
        controller_data_values = [sim_data[5] for sim_data in data]
        planner_data_keys = data[0][6]
        planner_data_values = [sim_data[7] for sim_data in data]
    else:
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
    )  # type: ignore[return-value]
