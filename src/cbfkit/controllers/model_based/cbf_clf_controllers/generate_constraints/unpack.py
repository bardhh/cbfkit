"""
unpack.py


"""

from typing import List, Dict, Any
from jax import Array
import jax.numpy as jnp


def unpack_for_cbf(
    control_limits: Array,
    barriers: List,
    lyapunovs: List,
    **kwargs: Dict[str, Any],
):
    """Unpacks information required to generate CLF constraints of arbitrary type.

    Args:
        control_limits (Array): _description_
        functions (List): _description_
        lyapunovs (List): _description_

    Returns:
        _type_: _description_
    """
    tunable = False
    n_con = len(control_limits)
    n_bfs = len(barriers[0])

    # Initialize constraint matrix/vector a/b for a*u <= b
    b_cbf = jnp.zeros((n_bfs,))
    a_cbf = jnp.zeros((n_bfs, n_con))

    # Check whether Lyapunov functions are relaxable
    if "relaxable_clf" not in kwargs:
        n_lfs = 0
    elif kwargs["relaxable_clf"]:
        n_lfs = len(lyapunovs[0])
        n_con -= n_lfs

    # Check whether Barrier functions are tunable
    if "tunable_class_k" not in kwargs:
        pass
    elif kwargs["tunable_class_k"]:
        n_con -= n_bfs
        tunable = True

    return n_con, n_bfs, n_lfs, a_cbf, b_cbf, tunable


def unpack_for_clf(
    control_limits: Array,
    lyapunovs: List,
    barriers: List,
    **kwargs: Dict[str, Any],
):
    """Unpacks information required to generate CLF constraints of arbitrary type.

    Args:
        control_limits (Array): _description_
        functions (List): _description_
        barriers (List): _description_

    Returns:
        _type_: _description_
    """
    relaxable = False
    n_con = len(control_limits)
    n_lfs = len(lyapunovs[0])

    # Initialize constraint matrix/vector a/b for a*u <= b
    b_clf = jnp.zeros((n_lfs,))
    a_clf = jnp.zeros((n_lfs, n_con))

    # Check for tunable barrier functions
    if "tunable_class_k" not in kwargs:
        n_bfs = 0
    elif kwargs["tunable_class_k"]:
        n_bfs = len(barriers[0])
        n_con -= n_bfs

    # Check whether Lyapunov functions are relaxable
    if "relaxable_clf" not in kwargs:
        pass
    elif kwargs["relaxable_clf"]:
        n_con -= n_lfs
        relaxable = True

    return n_con, n_bfs, n_lfs, a_clf, b_clf, relaxable
