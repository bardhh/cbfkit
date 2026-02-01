"""unpack.py."""

from typing import Any, Dict

import jax.numpy as jnp
from jax import Array

from cbfkit.utils.user_types import CertificateCollection


def unpack_for_cbf(
    control_limits: Array,
    barriers: CertificateCollection,
    lyapunovs: CertificateCollection,
    **kwargs: Dict[str, Any],
):
    """Unpacks information required to generate CLF constraints of arbitrary type.

    Args:
        control_limits (Array): _description_
        functions (CertificateCollection): _description_
        lyapunovs (CertificateCollection): _description_

    Returns
    -------
        _type_: _description_
    """
    tunable = False
    relaxable = False
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
    else:
        n_lfs = 0

    # Check whether Barrier functions are tunable or relaxable
    if "tunable_class_k" in kwargs and kwargs["tunable_class_k"]:
        n_con -= n_bfs
        tunable = True
    elif "relaxable_cbf" in kwargs and kwargs["relaxable_cbf"]:
        n_con -= n_bfs
        relaxable = True

    return n_con, n_bfs, n_lfs, a_cbf, b_cbf, tunable, relaxable


def unpack_for_clf(
    control_limits: Array,
    lyapunovs: CertificateCollection,
    barriers: CertificateCollection,
    **kwargs: Dict[str, Any],
):
    """Unpacks information required to generate CLF constraints of arbitrary type.

    Args:
        control_limits (Array): _description_
        functions (CertificateCollection): _description_
        barriers (CertificateCollection): _description_

    Returns
    -------
        _type_: _description_
    """
    relaxable = False
    n_con = len(control_limits)
    n_lfs = len(lyapunovs[0])

    # Initialize constraint matrix/vector a/b for a*u <= b
    b_clf = jnp.zeros((n_lfs,))
    a_clf = jnp.zeros((n_lfs, n_con))

    # Check for tunable barrier functions
    if "tunable_class_k" in kwargs and kwargs["tunable_class_k"]:
        n_bfs = len(barriers[0])
        n_con -= n_bfs
    elif "relaxable_cbf" in kwargs and kwargs["relaxable_cbf"]:
        n_bfs = len(barriers[0])
        n_con -= n_bfs
    else:
        n_bfs = 0

    # Check whether Lyapunov functions are relaxable
    if "relaxable_clf" not in kwargs:
        pass
    elif kwargs["relaxable_clf"]:
        n_con -= n_lfs
        relaxable = True

    return n_con, n_bfs, n_lfs, a_clf, b_clf, relaxable
