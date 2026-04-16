from typing import Callable, List, Tuple

import jax.numpy as jnp
from jax import Array, jit

from cbfkit.utils.user_types import DynamicsCallable


def generate_augmented_dynamics(
    list_of_dynamics: List[DynamicsCallable],
) -> Callable[[Array], Tuple[Array, Array]]:
    """Given a list of dynamics functions, generates a function to compute the augmented system
    dynamics (f and g).

    Args:
        list_of_dynamics (List[DynamicsCallable]): each entry is a dynamics function returning f(x), g(x) given x

    Returns
    -------
        Callable[[Array], Array]: new dynamics function
    """

    @jit
    def dynamics(z: Array) -> Tuple[Array, Array]:
        """

        Args:
            z (Array): augmented state vector

        Returns
        -------
        """
        # first set of dynamics
        f, g = list_of_dynamics[0](z)

        # append with remaining dynamics
        for dynamics in list_of_dynamics[1:]:
            fd, gd = dynamics(z)
            f = jnp.hstack([f, fd])
            g = jnp.vstack([g, gd])

        return f, g

    return dynamics


def generate_t_dynamics(n_controls: int) -> Callable[[Array], Tuple[Array, Array]]:
    """Generates dynamics function for time evolution."""

    def t_dynamics(_z: Array) -> Tuple[Array, Array]:
        """Computes the dynamics for the evolution of time.

        Args:
            _z (Array): augmented state vector (unused)

        Returns
        -------
            Tuple[Array, Array]: f and g for time dynamics (i.e., [1.0] and n_controls * [0])
        """
        return jnp.array([1.0]), jnp.zeros((1, n_controls))

    return t_dynamics


def generate_w_dynamics(
    n_states: int, n_weights: int, n_controls: int
) -> Callable[[Array], Tuple[Array, Array]]:
    """Generates adaptation weight dynamics for 2nd order system (i.e., this is just the wdot not
    the w2dot terms).

    Args:
        n_states (int): number of states
        n_weights (int): number of weights
        n_controls (int): number of controls

    Returns
    -------
        Callable[[Array], Tuple[Array, Array]]: function to compute f(z) and g(z) control-affine dynamics
    """

    def w_dynamics(z: Array) -> Tuple[Array, Array]:
        """Computes adaptation weight dynamics (wdot, not w2dot).

        Args:
            z (Array): augmented state vector

        Returns
        -------
            Tuple[Array, Array]: f(z) and g(z) control-affine dynamics
        """
        return (
            z[n_states + n_weights : n_states + 2 * n_weights],
            jnp.zeros((n_weights, n_controls)),
        )

    return w_dynamics
