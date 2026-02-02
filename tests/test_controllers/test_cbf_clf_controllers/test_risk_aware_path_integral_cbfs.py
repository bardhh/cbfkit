
import jax.numpy as jnp
import jax.scipy.special as jsp
from cbfkit.certificates.conditions.barrier_conditions.path_integral_barrier import right_hand_side
from cbfkit.controllers.cbf_clf.generate_constraints.risk_aware_path_integral_cbfs import generate_compute_ra_pi_cbf_constraints
from cbfkit.controllers.cbf_clf.utils.risk_aware_params import RiskAwareParams

def test_ra_pi_constraints_output():
    # Setup
    rho = 0.05
    gamma = 0.5
    eta = 2.0
    time_period = 1.0

    # Dynamics (mock)
    def dyn_func(x):
        # f(x) = 0, g(x) = 0
        return jnp.zeros_like(x), jnp.zeros((x.shape[0], 2))

    # Barrier (mock)
    # The framework seems to expect (t, x) or handle it.
    # generate_compute_certificate_values calls lf(t, x).
    def h(t, x): return x[0]
    def grad_h(t, x): return jnp.array([1.0, 0.0])
    def hess_h(t, x): return jnp.zeros((2, 2))
    def partial_h(t, x): return 0.0 # Time derivative

    condition = right_hand_side(rho, gamma, eta, time_period)

    # CertificateCollection tuple structure: (cbf, grad, hess, partials, conditions)
    barriers = ([h], [grad_h], [hess_h], [partial_h], [condition])

    control_limits = jnp.array([10.0, 10.0])

    ra_params = RiskAwareParams(
        t_max=time_period,
        p_bound=rho,
        gamma=jnp.array([gamma]),
        eta=eta,
        sigma=lambda x: jnp.zeros((x.shape[0], 1))
    )

    # Generate constraints
    compute_constraints = generate_compute_ra_pi_cbf_constraints(
        control_limits=control_limits,
        dyn_func=dyn_func,
        barriers=barriers,
        ra_cbf_params=ra_params
    )

    t = 0.0
    x = jnp.array([0.0, 0.0])

    # Run
    a, b, data = compute_constraints(t, x)

    # Check b
    # b should match 1 - gamma - term (since integrator_states=0)
    # This verifies that generate_constraints logic does not incorrectly add offsets.

    term = jnp.sqrt(2 * time_period) * eta * jsp.erfinv(1 - 2 * rho)
    expected_bc_x = 1 - gamma - term

    assert jnp.abs(b[0] - expected_bc_x) < 1e-5, f"Expected {expected_bc_x}, got {b[0]}"
