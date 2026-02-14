
import pytest
import jax.numpy as jnp
from jax import random

from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.utils.user_types import ControllerData

def test_relaxable_cbf_constraint():
    """Test that relaxable CBF constraints allow the QP to succeed even when strictly infeasible.

    Scenario:
        x_dot = u
        Constraint: |u| <= 0.5 (Hard input limit)
        Barrier: h(x) = x >= 0
        State: x = -2.0 (Deeply unsafe)
        Alpha = 1.0
        Required u >= -alpha * h(x) - Lfh(x) = -1.0 * (-2.0) - 0 = 2.0
        Conflict: u >= 2.0 AND u <= 0.5 -> STRICTLY INFEASIBLE.

    Relaxation:
        With relaxable_cbf=True, the QP should introduce a slack variable delta.
        u + delta >= 2.0
        Minimize u^2 + penalty * delta^2
        Result should be u = 0.5 (saturated at hard limit), delta = 1.5.
    """

    def dynamics(x):
        return jnp.array([0.0]), jnp.array([[1.0]])

    def h(t, x): return x[0]
    def dhdx(t, x): return jnp.array([1.0])
    def d2hdx2(t, x): return jnp.array([[0.0]])
    def partial_t(t, x): return 0.0
    def condition(val): return 1.0 * val

    barriers = ([h], [dhdx], [d2hdx2], [partial_t], [condition])

    # 1. Strict Case (Should Fail)
    controller_strict = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([0.5]),
        dynamics_func=dynamics,
        barriers=barriers,
        nominal_input=None,
        relaxable_cbf=False,
    )

    x = jnp.array([-2.0])
    t = 0.0
    key = random.PRNGKey(0)
    data = ControllerData()
    u_nom = jnp.array([0.0])

    u_strict, data_strict = controller_strict(t, x, u_nom, key, data)

    # Expect failure (NaN return due to Aegis policy)
    assert data_strict.error, "Strict QP should fail for infeasible constraints"
    assert jnp.all(jnp.isnan(u_strict)), f"Strict QP should return NaN on failure, got {u_strict}"

    # 2. Relaxable Case (Should Succeed)
    controller_relaxable = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([0.5]),
        dynamics_func=dynamics,
        barriers=barriers,
        nominal_input=None,
        relaxable_cbf=True,
        slack_penalty_cbf=100.0,
    )

    u_relaxable, data_relaxable = controller_relaxable(t, x, u_nom, key, data)

    # Expect success
    assert not data_relaxable.error, "Relaxable QP should succeed even when strictly infeasible"
    assert not jnp.any(jnp.isnan(u_relaxable)), "Relaxable QP should return valid control"

    # Check that control is saturated at the hard limit (0.5) because slack handles the rest
    assert jnp.allclose(u_relaxable, 0.5, atol=1e-3), f"Relaxable QP should saturate control at 0.5, got {u_relaxable}"


def test_relaxable_clf_constraint():
    """Test that relaxable CLF constraints allow the QP to succeed when conflicting with input limits.

    Scenario:
        x_dot = u
        Constraint: |u| <= 0.5 (Hard input limit)
        Lyapunov: V(x) = x^2
        State: x = 2.0
        Alpha = 1.0
        Required dot(V) <= -alpha * V
        2 * x * u <= -1.0 * x^2
        2 * 2.0 * u <= -4.0
        4 * u <= -4.0 => u <= -1.0
        Conflict: u <= -1.0 AND u >= -0.5 -> STRICTLY INFEASIBLE.

    Relaxation:
        With relaxable_clf=True, slack delta is used.
        4 * u - delta <= -4.0 (assuming additive relaxation, check implementation)
        Minimize u^2 + penalty * delta^2
        Result should be u = -0.5 (saturated), delta handles the rest.
    """

    def dynamics(x):
        return jnp.array([0.0]), jnp.array([[1.0]])

    def V(t, x): return x[0]**2
    def dVdx(t, x): return jnp.array([2*x[0]])
    def d2Vdx2(t, x): return jnp.array([[2.0]])
    def partial_t(t, x): return 0.0
    def condition(val): return -1.0 * val # Exponential stability

    lyapunovs = ([V], [dVdx], [d2Vdx2], [partial_t], [condition])

    # 1. Strict Case (Should Fail)
    controller_strict = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([0.5]),
        dynamics_func=dynamics,
        lyapunovs=lyapunovs,
        nominal_input=None,
        relaxable_clf=False,
    )

    x = jnp.array([2.0])
    t = 0.0
    key = random.PRNGKey(0)
    data = ControllerData()
    u_nom = jnp.array([0.0])

    u_strict, data_strict = controller_strict(t, x, u_nom, key, data)

    assert data_strict.error, "Strict CLF QP should fail for infeasible constraints"
    assert jnp.all(jnp.isnan(u_strict)), f"Strict CLF QP should return NaN, got {u_strict}"

    # 2. Relaxable Case (Should Succeed)
    controller_relaxable = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([0.5]),
        dynamics_func=dynamics,
        lyapunovs=lyapunovs,
        nominal_input=None,
        relaxable_clf=True,
        slack_penalty_clf=100.0,
    )

    u_relaxable, data_relaxable = controller_relaxable(t, x, u_nom, key, data)

    assert not data_relaxable.error, "Relaxable CLF QP should succeed"
    assert not jnp.any(jnp.isnan(u_relaxable)), "Relaxable CLF QP should return valid control"

    # Check that control is saturated at -0.5
    assert jnp.allclose(u_relaxable, -0.5, atol=1e-3), f"Relaxable CLF QP should saturate at -0.5, got {u_relaxable}"

if __name__ == "__main__":
    test_relaxable_cbf_constraint()
    test_relaxable_clf_constraint()
