"""
Tests for detecting NaNs in QP inputs (q, g, h) before solving.
Ensures that the controller returns error code -2 (NAN_INPUT_DETECTED) and does not silently fail or return undefined behavior.
"""

import jax.numpy as jnp
from jax import random
import pytest

from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.utils.user_types import ControllerData

# Simple 1D dynamics: x_dot = u
def simple_dynamics(x):
    return jnp.array([0.0]), jnp.array([[1.0]])

# Valid barrier: h(x) = x
def valid_barrier_func(t, x): return x[0]
def valid_barrier_grad(t, x): return jnp.array([1.0])
def valid_barrier_hess(t, x): return jnp.array([[0.0]])
def valid_partial_t(t, x): return 0.0
def valid_condition(val): return val

# Barrier that returns NaN gradient
def nan_barrier_grad(t, x): return jnp.array([jnp.nan])

@pytest.mark.parametrize("use_jit", [False, True])
def test_nan_nominal_input_detection(use_jit):
    """
    Case 1: Nominal control input (u_nom) contains NaN.
    Expected: error=True, error_data=-2 (NAN_INPUT_DETECTED), u=NaN.
    """
    controller = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([10.0]),
        dynamics_func=simple_dynamics,
        barriers=None,  # No barriers needed for this test case
        relaxable_cbf=False
    )

    # If testing JIT, wrap the controller
    if use_jit:
        from jax import jit
        controller = jit(controller)

    x = jnp.array([0.0])
    t = 0.0
    u_nom_nan = jnp.array([jnp.nan])  # Introduce NaN
    key = random.PRNGKey(0)
    data = ControllerData()

    u, data = controller(t, x, u_nom_nan, key, data)

    assert data.error, "Controller failed to report error for NaN nominal input"
    # Verify error code -2 (NAN_INPUT_DETECTED)
    # Note: data.error_data might be scalar or array depending on JIT/device, checking value.
    assert data.error_data == -2, f"Expected error code -2, got {data.error_data}"

    # Safety invariant: output should be NaN if input was invalid
    assert jnp.all(jnp.isnan(u)), f"Expected NaN output, got {u}"

@pytest.mark.parametrize("use_jit", [False, True])
def test_nan_barrier_gradient_detection(use_jit):
    """
    Case 2: Barrier function gradient contains NaN.
    Expected: error=True, error_data=-2 (NAN_INPUT_DETECTED).
    """
    # Create barrier collection with NaN gradient
    barriers = ([valid_barrier_func], [nan_barrier_grad], [valid_barrier_hess], [valid_partial_t], [valid_condition])

    controller = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([10.0]),
        dynamics_func=simple_dynamics,
        barriers=barriers,
        relaxable_cbf=False
    )

    if use_jit:
        from jax import jit
        controller = jit(controller)

    x = jnp.array([0.0])
    t = 0.0
    u_nom = jnp.array([0.0])  # Valid nominal input
    key = random.PRNGKey(0)
    data = ControllerData()

    u, data = controller(t, x, u_nom, key, data)

    assert data.error, "Controller failed to report error for NaN barrier gradient"
    assert data.error_data == -2, f"Expected error code -2, got {data.error_data}"
    assert jnp.all(jnp.isnan(u)), f"Expected NaN output, got {u}"
