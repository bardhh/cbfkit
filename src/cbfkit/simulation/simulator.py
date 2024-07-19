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

from cbfkit.utils.jax_stl import *
from typing import Any, Dict, Iterator, Optional, Tuple, List, Callable, Union
from tqdm import tqdm
import jax.numpy as jnp
from jax import random, Array, jit
from cbfkit.utils.logger import log, write_log
from cbfkit.utils.user_types import (
    PlannerCallable,
    ControllerCallable,
    DynamicsCallable,
    EstimatorCallable,
    IntegratorCallable,
    PerturbationCallable,
    SensorCallable,
    State,
    Control,
    Estimate,
    Covariance,
)


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
        controller_data: list,
        planner_data: list,
    ) -> Tuple[Array, Array, Array, Array, Dict[str, Any]]:
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

        # TODO: error when none of the planner, nominal controller and controller are specified

        # Plan trajectory using planner
        if stl_trajectory_cost is not None:
            planner_data["prev_robustness"] = stl_trajectory_cost(dt, planner_data["xs"])
        else:
            planner_data["prev_robustness"] = None
        if planner is not None:
            u_planner, planner_data = planner(t, z, None, subkey, planner_data)
            if "error" in planner_data.keys():
                if planner_data["error"]:
                    return x, u, z, c, controller_data, planner_data
        else:
            planner_data["u_traj"] = None
            # planner_data = {"u_traj": None, "prev_robustness": None, "x_traj": None}
        planner_data["prev_robustness"] = None

        # Nominal controller
        if (planner is not None) and (planner_data["u_traj"] is not None):
            # already have nominal input. just pass it along
            u = u_planner
        else:
            if planner_data["x_traj"] is None:
                print(f"ERROR: Planner output wrong. no states passed")
                exit()
            # Now pass the first waypoint for trajectory tracking
            if nominal_controller == None:
                print(f"WARNING: Nominal controller not defined. Setting to zero input")
                u = jnp.zeros((g.shape[1],))
            else:
                # expect trajectory to be n x N. N is time steps
                u, _ = nominal_controller(t, z, subkey, planner_data["x_traj"][:, 0])

        # Compute control input using controller
        if controller is not None:
            # print(f"hello")
            u, controller_data = controller(t, z, u, subkey, controller_data)
            if "error" in controller_data.keys():
                if controller_data["error"]:
                    return x, u, z, c, controller_data, planner_data
            if "complete" in controller_data.keys():
                if controller_data["complete"]:
                    return x, u, z, c, controller_data, planner_data
        else:
            controller_data = {}

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
        x: Array, controller_data: list = None, planner_data: list = None
    ) -> Iterator[Tuple[Array, Array, Array, Array, List[str], List[Array]]]:
        # No info on control/estimation
        u = None
        z = None
        p = None

        items = range(0, num_steps)

        if verbose:
            items = tqdm(items)

        xs = jnp.copy(x.reshape(-1, 1))

        # Simulate remaining timesteps
        planner_data["xs"] = xs
        for s in items:
            x, u, z, p, controller_data, planner_data = step(
                dt * s, x, u, z, p, controller_data, planner_data
            )
            log(controller_data)
            xs = jnp.append(xs, x.reshape(-1, 1), axis=1)
            planner_data["xs"] = jnp.append(planner_data["xs"], x.reshape(-1, 1), axis=1)
            # None  # xs
            yield x, u, z, p, list(controller_data.keys()), list(controller_data.values()), list(
                planner_data.keys()
            ), list(planner_data.values())

            if "complete" in controller_data.keys():
                if controller_data["complete"]:
                    print("GOAL REACHED!")
                    break

            if "error" in controller_data.keys():
                if controller_data["error"]:
                    # print(f"CONTROLLER ERROR: \n {controller_data[""]}")
                    break

            if "error" in planner_data.keys():
                if planner_data["error"]:
                    print("CONTROLLER ERROR")
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
    controller_data=None,
    planner_data=None,
    stl_trajectory_cost=None,
) -> Tuple[Array, List[str], List[Array]]:
    """This function simulates the dynamical system for a given number of steps,
    and returns a tuple consisting of 1) an array containing the state, control,
    estimate, and covariance trajectories, 2) a list containing keys to a dict
    object containing additional data, and 3) a list of objects corresponding to
    the data accessed by those keys.

    Args:
        x0 (State): initial (ground truth) state of the system
        dynamics (DynamicsCallable): specifies system dynamics
        sensor (SensorCallable): function for sensing the (complete or partial) state
        planner (PlannerCallable): function for computing the control input and state trajectories
        nominal_controller (ControllerCallable): function for computing the control input u
        controller (ControllerCallable): function for computing the control input u
        estimator (EstimatorCallable): function for estimating the state x
        integrator (IntegratorCallable): function for numerically integrating the state forward in time
        dt (Time): length of simulation timestep (sec)
        num_steps (int): total number of timesteps in simulation. final time = num_steps * dt
        filepath (Optional[str], optional): location to save file. Defaults to None (no save).

    Returns:
        Tuple[Array, List[str], List[Array]]: _description_
    """
    if nominal_controller is None:
        #! Load some kind of default controller
        pass

    if controller is None:
        #! Load some kind of default controller
        pass

    if planner is None:
        pass

    if sensor is None:
        #! Load some kind of default sensor
        pass

    if estimator is None:
        #! Load some kind of default estimator
        pass

    if perturbation is None:

        def perturbation(x, _u, _f, _g):
            def p(_subkey):
                return jnp.zeros(x.shape)

            return p

    if sigma is None:
        #! Load some kind of default sigma
        pass

    # Generate key for randomization
    if key is None:
        key = random.PRNGKey(0)

    # Define simulator
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
    if filepath is not None:
        write_log(filepath)

    #! Somehow make these more modular?
    states = jnp.array([sim_data[0] for sim_data in data])
    controls = jnp.array([sim_data[1] for sim_data in data])
    estimates = jnp.array([sim_data[2] for sim_data in data])
    covariances = jnp.array([sim_data[3] for sim_data in data])
    controller_data_keys = data[0][4]
    controller_data_values = [sim_data[5] for sim_data in data]
    planner_data_keys = data[0][6]
    planner_data_values = [sim_data[7] for sim_data in data]

    return states, controls, estimates, covariances, controller_data_keys, controller_data_values, planner_data_keys, planner_data_values  # type: ignore[return-value]
