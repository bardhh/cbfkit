from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array

from cbfkit.utils.logger import write_log
from cbfkit.utils.user_types import (
    Control,
    ControllerCallable,
    Covariance,
    Estimate,
    EstimatorCallable,
    SensorCallable,
    State,
    Time,
)


def stepper(
    sensor: Callable[[], Array],
    controller: Callable[[float, Array], Tuple[Array, Dict[str, Any]]],
    estimator: EstimatorCallable,
) -> Callable[
    [float, Optional[Array], Array, Optional[Array]], Tuple[Array, Array, Array, Dict[str, Any]]
]:
    """Build a single-step estimator/controller update callable for ROS2 experiments."""

    def step(
        t: Time,
        u: Optional[Control],
        z: Estimate,
        p: Optional[Covariance],
    ) -> Tuple[Array, Array, Array, Dict[str, Any]]:
        """Run one estimator/controller update step."""
        y = sensor()

        # Compute state estimate using estimator
        z, p = estimator(t, y, z, u, p)

        # Compute control input using controller
        # this is wrapped in a function to publish
        # the input to the correct ROS node
        u, data = controller(float(t), z)

        return u, z, p, data

    return step


def experimenter(
    sensor: Callable[[], Array],
    controller: Callable[[float, Array], Tuple[Array, Dict[str, Any]]],
    estimator: EstimatorCallable,
    dt: float,
    num_steps: int,
) -> Callable[[Array], Iterator[tuple[Array, Array, Array, List[str], List[Array]]]]:
    """Build an iterator that runs the ROS2 experiment for a fixed number of steps."""
    # Define step function
    step = stepper(
        sensor=sensor,
        controller=controller,
        estimator=estimator,
    )

    def simulate_iter(z0: Array) -> Iterator[tuple[Array, Array, Array, List[str], List[Array]]]:
        # No info on control/estimation
        u = None
        z = z0
        p = None

        # Simulate remaining timesteps
        for s in range(0, num_steps):
            u, z, p, data = step(dt * s, u, z, p)
            # log(data)

            yield u, z, p, list(data.keys()), list(data.values())

    return simulate_iter


# This function simulates the dynamical system for a given number of steps,
# returning a tuple of all states. Optionally, it can also write the logged data to a file.
def experiment(
    z0: Array,
    sensor: Callable[[], Array],
    controller: Callable[[float, Array], Tuple[Array, Dict[str, Any]]],
    estimator: EstimatorCallable,
    dt: float,
    num_steps: int,
    filepath: Optional[str] = None,
) -> Tuple[Array, List[str], List[Array]]:
    """Run the full ROS2 experiment loop and collect trajectories plus auxiliary logs."""
    # Define simulator
    experiment_iter = experimenter(sensor, controller, estimator, dt, num_steps)

    # Run simulation from initial state
    experiment_data = tuple(experiment_iter(z0))

    # Log / Extract data
    return extract_and_log_data(filepath, experiment_data)


def extract_and_log_data(filepath: Optional[str], data):
    """Extract logged arrays and optional metadata from experiment output."""
    if filepath is not None and len(data) > 0:
        keys = data[0][3]
        values_list = [step[4] for step in data]
        log_data = [dict(zip(keys, vals)) for vals in values_list]
        write_log(filepath, log_data)

    controls = jnp.array([sim_data[0] for sim_data in data])
    estimates = jnp.array([sim_data[1] for sim_data in data])
    covariances = jnp.array([sim_data[2] for sim_data in data])
    if len(data) > 0:
        data_keys = data[0][3]
        data_values = [sim_data[4] for sim_data in data]
    else:
        data_keys = []
        data_values = []

    return controls, estimates, covariances, data_keys, data_values  # type: ignore[return-value]
