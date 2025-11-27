import jax.numpy as jnp
from jax import random

from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.certificates import certificate_package

def test_infeasible_qp_behavior():
    """
    Test that the QP solver handles infeasibility correctly.
    Scenario:
        x_dot = u
        Constraint: u <= 0.5
        Barrier: h(x) = x >= 0
        State: x = -2.0
        Alpha = 1.0
        Required u >= -alpha * x = 2.0
        Conflict: u >= 2.0 AND u <= 0.5 -> Infeasible.
    """
    
    def dynamics(x):
        return jnp.array([0.0]), jnp.array([[1.0]])

    # Barrier definitions
    def h(t, x): return x[0]
    def dhdx(t, x): return jnp.array([1.0])
    def d2hdx2(t, x): return jnp.array([[0.0]])
    
    # Certificate package
    # We use a dummy condition that just returns alpha * h
    # Condition: dot(h) >= -alpha * h
    # But in cbfkit, condition function returns the required value for dot(h) + ...
    # Actually, zeroing_cbfs uses: b = dbf_t + Lfh + bc_x
    # bc_x is the class K function value (alpha * h)
    # So we need bc_x = 1.0 * x[0]
    
    def condition(val): return 1.0 * val # alpha = 1.0
    
    barrier_pkg = certificate_package(h, dhdx, d2hdx2, 1)
    # We need to construct the "barriers" tuple expected by the controller
    # (functions, jacobians, hessians, partials, conditions)
    # Wait, the controller expects a list of these.
    
    # Manually constructing the tuple structure as expected by the generator
    # barriers = ([h], [dhdx], [d2hdx2], [partial_t], [condition])
    
    def partial_t(t, x): return 0.0
    
    barriers = (
        [h],
        [dhdx],
        [d2hdx2],
        [partial_t],
        [condition]
    )
    
    # 1. Vanilla Case (Should fail and return 0.0)
    controller_vanilla = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([0.5]),
        dynamics_func=dynamics,
        barriers=barriers,
        nominal_input=None # Zero nominal
    )
    
    x = jnp.array([-2.0])
    t = 0.0
    key = random.PRNGKey(0)
    data = {}
    u_nom = jnp.array([0.0])
    
    # Execute
    u_vanilla, data_vanilla = controller_vanilla(t, x, u_nom, key, data)
    
    # Expect u=0 because QP failed and fallback is 0
    assert jnp.abs(u_vanilla[0]) < 1e-6, f"Vanilla QP should fail and return 0, got {u_vanilla}"
    assert data_vanilla["error"] == True
    
    
    # 2. Relaxable Case (Should succeed and return saturated limit 0.5)
    controller_relaxable = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([0.5]),
        dynamics_func=dynamics,
        barriers=barriers,
        nominal_input=None,
        relaxable_cbf=True,
        slack_penalty_cbf=100.0
    )
    
    u_relaxable, data_relaxable = controller_relaxable(t, x, u_nom, key, data)
    
    # Expect u=0.5 (saturated) because slack allows violation
    # The QP will try to get as close to u_nom=0 as possible while minimizing slack
    # Constraint: 1*u + delta >= 2.0
    # Cost: u^2 + 100*delta^2
    # u will be pushed to limit 0.5. delta will take up the rest (1.5).
    
    print(f"Relaxable U: {u_relaxable}")
    assert jnp.abs(u_relaxable[0] - 0.5) < 1e-3, f"Relaxable QP should return max control 0.5, got {u_relaxable}"
    assert data_relaxable["error"] == False

if __name__ == "__main__":
    test_infeasible_qp_behavior()
