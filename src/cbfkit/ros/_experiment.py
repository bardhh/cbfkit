import jax.numpy as jnp
from jax import random, Array
from typing import Any, Dict, Iterator, Optional, Tuple, List, Callable
from cbfkit.utils.logger import log, write_log
from cbfkit.utils.user_types import (
    ControllerCallable,
    EstimatorCallable,
    SensorCallable,
    Time,
    State,
    Control,
    Estimate,
    Covariance,
)


def stepper(
    sensor: SensorCallable,
    controller: ControllerCallable,
    estimator: EstimatorCallable,
) -> Tuple[State, Dict[str, Any]]:
    """Step function to take the simulation forward one timestep. Designed
    to work generically with broad classes of dynamics, controllers, and
    estimators.

    Args:
        dynamics (Callable): function handle to compute true system dynamics
        sensor (Callable): function handle to generate new state sensor
        controller (Callable): function handle to compute control input
        estimator (Callable): function handle to compute new state estimate
        dt (float) timestep (sec)

    Returns:
        step (Callable): function handle for stepping forward in simulation time


    """

    def step(
        t: Time,
        u: Control,
        z: Estimate,
        p: Covariance,
    ) -> Tuple[Array, Array, Array, Array, Dict[str, Any]]:
        """_summary_

        Args:
            t (float): time (sec)
            u (Array): control input vector
            z (Array): state estimate vector
            p (Array): covariance matrix of state estimate

        Returns:
            u (Array): control input vector
            z (Array): state estimate vector
            p (Array): covariance matrix of state estimate
            data (dict): contains other relevant simulation data
        """
        #! Need to implement some kind of sensor
        y = sensor()

        # Compute state estimate using estimator
        z, p = estimator(t, y, z, u, p)

        # Compute control input using controller
        # this is wrapped in a function to publish
        # the input to the correct ROS node
        u, data = controller(t, z)

        return u, z, p, data

    return step


def experimenter(
    sensor: SensorCallable,
    controller: ControllerCallable,
    estimator: EstimatorCallable,
    dt: Time,
    num_steps: int,
) -> Callable[[Array], Iterator[tuple[Array, Array, Array, Array, List[str], List[Array]]]]:
    """Generates function handle for the iterator that carries out the simulation of the
    dynamical system.

    Args:
        sensor (Callable): function handle to generate new state sensor
        controller (Callable): function handle to compute control input
        estimator (Callable): function handle to compute new state estimate
        dt (float) timestep (sec)
        num_steps (int): number of timesteps to simulate

    Returns:
        iterator eventually returning the following:
            u (Array): control input vector
            z (Array): state estimate vector
            p (Array): state estimate covariance matrix
            keys (list): list of keys in data dict
            vals (list): lists of values contained in data

    """
    # Define step function
    step = stepper(
        sensor=sensor,
        controller=controller,
        estimator=estimator,
    )

    def simulate_iter() -> Iterator[tuple[Array, Array, Array, Array, List[str], List[Array]]]:
        # No info on control/estimation
        u = None
        z = None
        p = None

        # Simulate remaining timesteps
        for s in range(0, num_steps):
            u, z, p, data = step(dt * s, u, z, p)
            log(data)

            yield u, z, p, list(data.keys()), list(data.values())

    return simulate_iter


# This function simulates the dynamical system for a given number of steps,
# returning a tuple of all states. Optionally, it can also write the logged data to a file.
def experiment(
    z0: Array,
    sensor: SensorCallable,
    controller: ControllerCallable,
    estimator: EstimatorCallable,
    dt: Time,
    num_steps: int,
    filepath: Optional[str] = None,
) -> Tuple[Array, List[str], List[Array]]:
    """This function simulates the dynamical system for a given number of steps,
    and returns a tuple consisting of 1) an array containing the state, control,
    estimate, and covariance trajectories, 2) a list containing keys to a dict
    object containing additional data, and 3) a list of objects corresponding to
    the data accessed by those keys.

    Args:
        z0 (State): initial (estimated) state of the system
        sensor (SensorCallable): _description_
        controller (ControllerCallable): _description_
        estimator (EstimatorCallable): _description_
        dt (Time): _description_
        num_steps (int): _description_
        filepath (Optional[str], optional): _description_. Defaults to None.

    Returns:
        Tuple[Array, List[str], List[Array]]: _description_
    """
    # Define simulator
    experiment_iter = experimenter(sensor, controller, estimator, dt, num_steps)

    # Run simulation from initial state
    experiment_data = tuple(experiment_iter(z0))

    # Log / Extract data
    return extract_and_log_data(filepath, experiment_data)


#! Finish this function
def extract_and_log_data(filepath: str, data):
    """_summary_"""
    if filepath is not None:
        write_log(filepath)

    #! Somehow make these more modular?
    controls = jnp.array([sim_data[0] for sim_data in data])
    estimates = jnp.array([sim_data[1] for sim_data in data])
    covariances = jnp.array([sim_data[2] for sim_data in data])
    data_keys = data[0][1]
    data_values = [sim_data[2] for sim_data in data]

    return controls, estimates, covariances, data_keys, data_values  # type: ignore[return-value]
