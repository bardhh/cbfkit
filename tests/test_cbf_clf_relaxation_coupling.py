
import jax.numpy as jnp
import pytest
from cbfkit.controllers.cbf_clf.generate_constraints.zeroing_cbfs import generate_compute_zeroing_cbf_constraints
from cbfkit.controllers.cbf_clf.generate_constraints.vanilla_clfs import generate_compute_vanilla_clf_constraints

def test_cbf_relaxation_coupling():
    n_controls = 2
    n_cbfs = 2
    control_limits = jnp.ones((4, 2)) # 2 controls + 2 slack

    def dyn_func(x):
        return jnp.zeros((2,)), jnp.zeros((2, 2))

    barriers = (
        [lambda t, x: 1.0]*n_cbfs,
        [lambda t, x: jnp.zeros((2,))]*n_cbfs,
        [lambda t, x: jnp.zeros((2,2))]*n_cbfs,
        [lambda t, x: 0.0]*n_cbfs,
        [lambda h: h]*n_cbfs
    )

    compute_cbf = generate_compute_zeroing_cbf_constraints(
        control_limits=control_limits,
        dyn_func=dyn_func,
        barriers=barriers,
        relaxable_cbf=True,
        scale_cbf=10.0
    )

    a_cbf, _, _ = compute_cbf(0.0, jnp.zeros(2))
    # n_con (controls) = 2. Slack vars start at index 2.
    # a_cbf shape is (n_cbfs, n_total) = (2, 4)
    relaxation_block = a_cbf[:, 2:]

    # Check if diagonal
    diag_elements = jnp.diag(relaxation_block)
    expected_diag = -10.0 * jnp.ones(n_cbfs)

    # Assert diagonal elements are correct
    assert jnp.allclose(diag_elements, expected_diag), f"Diagonal elements mismatch. Expected -10, got {diag_elements}"

    # Assert off-diagonal elements are zero
    off_diag = relaxation_block - jnp.diag(diag_elements)
    assert jnp.allclose(off_diag, 0.0), f"Off-diagonal elements are not zero: \n{off_diag}"

def test_cbf_tunable_coupling():
    n_controls = 2
    n_cbfs = 2
    control_limits = jnp.ones((4, 2)) # 2 controls + 2 tunable

    def dyn_func(x):
        return jnp.zeros((2,)), jnp.zeros((2, 2))

    # bc_x will be alpha(h(x))
    # h(x) = [1.0, 2.0]
    barriers = (
        [lambda t, x: 1.0, lambda t, x: 2.0],
        [lambda t, x: jnp.zeros((2,))]*n_cbfs,
        [lambda t, x: jnp.zeros((2,2))]*n_cbfs,
        [lambda t, x: 0.0]*n_cbfs,
        [lambda h: h]*n_cbfs
    )

    compute_cbf = generate_compute_zeroing_cbf_constraints(
        control_limits=control_limits,
        dyn_func=dyn_func,
        barriers=barriers,
        tunable_class_k=True,
        scale_cbf=1.0 # scalar multiplier
    )

    a_cbf, _, _ = compute_cbf(0.0, jnp.zeros(2))
    tunable_block = a_cbf[:, 2:]

    expected = jnp.diag(jnp.array([-1.0, -2.0]))
    assert jnp.allclose(tunable_block, expected), f"Tunable block mismatch. Expected:\n{expected}\nGot:\n{tunable_block}"

def test_clf_relaxation_coupling():
    n_controls = 2
    n_clfs = 2
    control_limits = jnp.ones((4, 2))

    def dyn_func(x):
        return jnp.zeros((2,)), jnp.zeros((2, 2))

    lyapunovs = (
        [lambda t, x: 1.0]*n_clfs,
        [lambda t, x: jnp.zeros((2,))]*n_clfs,
        [lambda t, x: jnp.zeros((2,2))]*n_clfs,
        [lambda t, x: 0.0]*n_clfs,
        [lambda h: h]*n_clfs
    )

    compute_clf = generate_compute_vanilla_clf_constraints(
        control_limits=control_limits,
        dyn_func=dyn_func,
        lyapunovs=lyapunovs,
        relaxable_clf=True,
        scale_clf=5.0
    )

    a_clf, _, _ = compute_clf(0.0, jnp.zeros(2))
    # a_clf shape (n_clfs, n_total) = (2, 4)
    relaxation_block = a_clf[:, 2:]

    expected = -5.0 * jnp.eye(n_clfs)
    assert jnp.allclose(relaxation_block, expected), f"CLF Relaxation block mismatch. Expected:\n{expected}\nGot:\n{relaxation_block}"
