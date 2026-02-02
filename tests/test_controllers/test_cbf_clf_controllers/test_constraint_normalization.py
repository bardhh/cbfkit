import jax.numpy as jnp
from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.controllers.cbf_clf.generate_constraints import (
    generate_compute_zeroing_cbf_constraints,
    generate_compute_vanilla_clf_constraints,
)
from cbfkit.utils.user_types import CertificateCollection, ControllerData

def test_constraint_normalization_prevents_unsafe_solution():
    """
    Verifies that the controller normalizes constraints with small gradients,
    preventing the QP solver (with loose tolerance) from ignoring them.

    Scenario:
    - 1 state, 1 control.
    - Constraint: Lg*u <= -Lf - alpha
    - Lg = 1e-9 (small gradient)
    - RHS = -1e-9 (small violation/requirement)
    - Implied constraint: 1e-9 * u <= -1e-9  =>  u <= -1

    Without normalization:
    - Solver sees 1e-9 * u <= -1e-9.
    - If solver tolerance is 1e-3, u=0 yields residual 1e-9 < 1e-3.
    - Solver accepts u=0.
    - u=0 is UNSAFE (violation).

    With normalization (Active when p_mat=None):
    - Row norm ~ 1e-9.
    - Scale factor = min(1/1e-9, 1e8) = 1e8.
    - Scaled constraint: 0.1 * u <= -0.1.
    - If u=0, residual is 0.1 > 1e-3.
    - Solver rejects u=0.
    - Solver finds u = -1 (residual 0).
    """

    # 1. Setup Dynamics
    # f(x) = 0, g(x) = 1e-9
    epsilon = 1e-9

    def dynamics(x):
        return jnp.zeros((1,)), jnp.array([[epsilon]])

    # 2. Setup Barrier
    # We want Lf*h + Lg*h * u >= -alpha(h)
    # => -Lg*h * u <= Lf*h + alpha(h)
    # We want final form: 1e-9 * u <= -1e-9
    # So we need -Lg*h = 1e-9 => Lg*h = -1e-9.
    # But g = 1e-9. So dh/dx must be -1.
    # Lf*h = 0 (since f=0).
    # alpha(h) = -1e-9.

    # Let's mock the certificate collection directly to control values precisely
    # h(x) is not used if we provide fixed derivatives/conditions?
    # Actually, generate_compute_zeroing_cbf_constraints calls functions.

    # We can use a custom CertificateCollection where we inject fixed values
    # or just define functions that return what we want.

    def h_func(t, x): return 0.0 # Dummy, returns scalar
    def grad_h(t, x): return jnp.array([-1.0]) # dh/dx = -1, returns vector (1,)
    def hess_h(t, x): return jnp.zeros((1, 1)) # returns matrix (1, 1)
    def partial_h(t, x): return 0.0 # returns scalar

    # alpha(h(x)) needs to return -1e-9.
    # usually condition is `lambda x: gamma * h(x)`
    # We can just return fixed value.
    # Note: condition functions in CertificateCollection usually take 1 arg (value of h),
    # but generate_compute_zeroing_cbf_constraints passes (t, x) to the original condition callable?
    # No, wait.
    # generate_compute_certificate_values_list_comprehension:
    # bc_values = [lc(lf) for lc, lf in zip(conditions, bf_x)]
    # So lc takes the VALUE of h(x).

    def condition(h_val): return -epsilon

    # Note: certificate structure is (funcs, jacs, hess, parts, conditions)
    # But conditions expects a certain signature?
    # checking generate_compute_zeroing_cbf_constraints...
    # It calls `term = conditions[i](t, x)`

    barriers = CertificateCollection(
        [h_func], [grad_h], [hess_h], [partial_h], [condition]
    )

    # 3. Create Controller
    # Must use p_mat=None to trigger normalization
    setup_controller = cbf_clf_qp_generator(
        generate_compute_zeroing_cbf_constraints,
        generate_compute_vanilla_clf_constraints
    )

    controller = setup_controller(
        control_limits=jnp.array([10.0]), # Limit |u| <= 10
        dynamics_func=dynamics,
        barriers=barriers,
        p_mat=None,
        relaxable_cbf=False, # Hard constraint
    )

    # 4. Run Controller
    t = 0.0
    x = jnp.array([0.0])
    u_nom = jnp.array([0.0])

    # JIT compilation happens on first call
    # We must pass a valid ControllerData structure (even if empty) to satisfy type expectations in JAX
    u, data = controller(t, x, u_nom, jnp.array([0]), ControllerData())

    # 5. Verify
    # We expect u <= -1.
    # Ideally u = -1 (minimizing deviation from u_nom=0)
    print(f"Computed u: {u}")

    assert not jnp.isnan(u).any(), "Controller returned NaN (Solver failure?)"
    assert u[0] <= -0.9, f"Safety violation! Expected u <= -0.9, got {u[0]}. Normalization might be broken."

if __name__ == "__main__":
    test_constraint_normalization_prevents_unsafe_solution()
