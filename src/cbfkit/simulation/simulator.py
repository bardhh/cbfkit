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

from typing import Any, Dict, Iterator, Optional, Tuple, List, Callable, Union
from tqdm import tqdm
import jax.numpy as jnp
from jax import random, Array
from cbfkit.utils.logger import log, write_log
from cbfkit.utils.user_types import (
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
    controller: ControllerCallable,
    sensor: SensorCallable,
    estimator: EstimatorCallable,
    perturbation: PerturbationCallable,
    sigma: Array,
    key: random.PRNGKey,
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

        # Print Progress

        # Sensor measurement
        y = sensor(t, x, sigma=sigma, key=key)

        # Compute state estimate using estimator
        z, c = estimator(t, y, z, u, c)

        # Generate true dynamics based on true state
        f, g = dynamics(x)

        # Compute control input using controller
        if controller is not None:
            u, data = controller(t, z)
            if "error" in data.keys():
                if data["error"]:
                    return x, u, z, c, data
            if "complete" in data.keys():
                if data["complete"]:
                    return x, u, z, c, data
        else:
            u = jnp.zeros((g.shape[1],))
            data = {}

        # Generate perturbation to the dynamics (inclusive of SDEs)
        p = perturbation(x, u, f, g)

        # Continuous-time dynamics (inclusive of SDEs)
        key, subkey = random.split(key)
        xdot = f + jnp.matmul(g, u) + p(subkey)

        # Integrate to generate next step
        x = integrator(x, xdot, dt)

        return x, u, z, c, data

    return step


def simulator(
    dt: float,
    num_steps: int,
    dynamics: DynamicsCallable,
    integrator: IntegratorCallable,
    controller: ControllerCallable,
    sensor: SensorCallable,
    estimator: EstimatorCallable,
    perturbation: PerturbationCallable,
    sigma: Array,
    key: random.PRNGKey,
    verbose: Optional[bool] = True,
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
        controller=controller,
        estimator=estimator,
        perturbation=perturbation,
        integrator=integrator,
        sigma=sigma,
        dt=dt,
        key=key,
    )

    def simulate_iter(
        x: Array,
    ) -> Iterator[Tuple[Array, Array, Array, Array, List[str], List[Array]]]:
        # No info on control/estimation
        u = None
        z = None
        p = None

        items = range(0, num_steps)

        if verbose:
            items = tqdm(items)

        # Simulate remaining timesteps
        for s in items:
            x, u, z, p, data = step(dt * s, x, u, z, p)
            log(data)

            yield x, u, z, p, list(data.keys()), list(data.values())

            if "complete" in data.keys():
                if data["complete"]:
                    print("GOAL REACHED!")
                    break

            if "error" in data.keys():
                if data["error"]:
                    print("CONTROLLER ERROR")
                    break

    return simulate_iter


def execute(
    x0: State,
    dt: float,
    num_steps: int,
    dynamics: DynamicsCallable,
    integrator: IntegratorCallable,
    controller: Optional[Union[ControllerCallable, None]] = None,
    sensor: Optional[Union[SensorCallable, None]] = None,
    estimator: Optional[Union[EstimatorCallable, None]] = None,
    perturbation: Optional[Union[PerturbationCallable, None]] = None,
    sigma: Optional[Union[Array, None]] = None,
    key: Optional[Union[random.PRNGKey, None]] = None,
    filepath: Optional[str] = None,
    verbose: Optional[bool] = True,
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
        controller (ControllerCallable): function for computing the control input u
        estimator (EstimatorCallable): function for estimating the state x
        integrator (IntegratorCallable): function for numerically integrating the state forward in time
        dt (Time): length of simulation timestep (sec)
        num_steps (int): total number of timesteps in simulation. final time = num_steps * dt
        filepath (Optional[str], optional): location to save file. Defaults to None (no save).

    Returns:
        Tuple[Array, List[str], List[Array]]: _description_
    """
    if controller is None:
        #! Load some kind of default controller
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
        controller,
        sensor,
        estimator,
        perturbation,
        sigma,
        key,
        verbose,
    )

    # Run simulation from initial state
    simulation_data = tuple(simulate_iter(x0))

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
    data_keys = data[0][4]
    data_values = [sim_data[5] for sim_data in data]

    return states, controls, estimates, covariances, data_keys, data_values  # type: ignore[return-value]
