from typing import Callable, List
from jax import jit, Array


def dynamics(
    subscriptions: List[str], generate_dynamics: Callable[[List[str]], Callable[[], Array]]
):
    """
    Provides an interface to the ROS node that publishes the dynamics data
    as deemed relevant by subscriptions.

    Args:
        subscriptions

    Returns:
        dyn (callable): generates a measurement of the state

    """
    dyn = generate_dynamics(subscriptions)

    return dyn
