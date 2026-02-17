
import jax.numpy as jnp
from cbfkit.controllers.cbf_clf.generate_constraints.unpack import unpack_for_clf
from cbfkit.utils.user_types import CertificateCollection

def test_unpack_for_clf_relaxable_cbf():
    """
    Test that unpack_for_clf correctly identifies relaxable CBFs and adjusts
    n_con (number of control inputs) accordingly.

    If n_con is not reduced by n_bfs, subsequent constraint generation will
    broadcast the control derivative matrix (LgV) into the slack variable columns,
    incorrectly coupling the CBF slack to the CLF constraint.
    """
    n_u = 1
    n_bfs = 1
    n_lfs = 1

    # control_limits has length n_u + n_bfs (because of relaxable_cbf)
    # Note: If relaxable_clf were also True, it would be n_u + n_bfs + n_lfs.
    # Here we isolate relaxable_cbf.

    control_limits = jnp.zeros(n_u + n_bfs)

    barriers = CertificateCollection([lambda x: x]*n_bfs, [], [], [], [])
    lyapunovs = CertificateCollection([lambda x: x]*n_lfs, [], [], [], [])

    kwargs = {"relaxable_cbf": True}

    n_con, n_bfs_out, n_lfs_out, a_clf, b_clf, relaxable = unpack_for_clf(
        control_limits, lyapunovs, barriers, **kwargs
    )

    # If correct, n_con should be n_u (1).
    # If bug exists, n_con will be n_u + n_bfs (2).
    assert n_con == n_u, f"n_con should be {n_u}, but got {n_con}"
    assert n_bfs_out == n_bfs
    assert a_clf.shape[1] == len(control_limits)
    # Check that a_clf was initialized with correct dimensions
    # unpack_for_clf returns a_clf of shape (n_lfs, original_control_limits_len)
    # wait, unpack_for_clf initializes a_clf:
    # a_clf = jnp.zeros((n_lfs, n_con_original))
    # where n_con_original = len(control_limits) passed to it.

    # Wait, the broadcasting bug happens in vanilla_clfs.py using the returned a_clf and n_con.
    # unpack_for_clf itself just returns n_con and initialized arrays.
    pass

def test_unpack_for_clf_tunable_cbf():
    """Test that tunable class k is also handled correctly."""
    n_u = 1
    n_bfs = 1
    n_lfs = 1

    control_limits = jnp.zeros(n_u + n_bfs)
    barriers = CertificateCollection([lambda x: x]*n_bfs, [], [], [], [])
    lyapunovs = CertificateCollection([lambda x: x]*n_lfs, [], [], [], [])

    kwargs = {"tunable_class_k": True}

    n_con, n_bfs_out, n_lfs_out, a_clf, b_clf, relaxable = unpack_for_clf(
        control_limits, lyapunovs, barriers, **kwargs
    )

    assert n_con == n_u
