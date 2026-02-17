"""
Regression test for CLF relaxation scheme.
Ensures that we use additive relaxation (preserving authority near V=0)
rather than multiplicative relaxation (which vanishes near V=0).
"""

import jax
import jax.numpy as jnp
import pytest
from cbfkit.controllers.cbf_clf.generate_constraints.vanilla_clfs import (
    generate_compute_vanilla_clf_constraints,
)
from cbfkit.utils.user_types import CertificateCollection, EMPTY_CERTIFICATE_COLLECTION


def test_clf_additive_relaxation():
    # 1. Define minimal system
    # 1D system: dx/dt = 0 + 1*u
    def dynamics(x):
        return jnp.zeros((1,)), jnp.ones((1, 1))

    # 2. Define CLF: V(x) = x^2
    # Gradient: 2x
    # Hessian: 2
    # Partial t: 0
    # Condition: dV/dt <= -0.5 V

    # Note: V, partial_t, and condition must return scalars so that stacking
    # results in (n_lfs,) arrays, not (n_lfs, 1) which causes broadcast errors.
    def V(t, x):
        return jnp.sum(x**2)

    def grad_V(t, x):
        return 2 * x

    def hess_V(t, x):
        return jnp.array([[2.0]])

    def partial_t_V(t, x):
        return 0.0

    def condition(v):
        return -0.5 * v

    lyapunovs = CertificateCollection([V], [grad_V], [hess_V], [partial_t_V], [condition])

    # 3. Setup parameters
    # Normal control limits [-10, 10]
    # Scale factor for slack variable
    scale_clf = 100.0

    # We need to augment control limits for the slack variable.
    # [u_min, u_max, slack_min, slack_max]
    # The generator expects control_limits array to match total vars (controls + slacks)
    u_lim = 10.0
    slack_lim = 1e9

    # 1 control + 1 slack
    control_limits = jnp.array([u_lim, slack_lim])

    # 4. Generate constraints function
    compute_constraints = generate_compute_vanilla_clf_constraints(
        control_limits=control_limits,
        dyn_func=dynamics,
        barriers=EMPTY_CERTIFICATE_COLLECTION,
        lyapunovs=lyapunovs,
        relaxable_clf=True,
        scale_clf=scale_clf,
    )

    # 5. Evaluate at a state very close to equilibrium
    t = 0.0
    x = jnp.array([1e-6])  # V(x) = 1e-12

    # Run JIT compiled function (implicit or explicit)
    # Note: generate_compute_vanilla_clf_constraints returns a jitted function usually
    a_clf, b_clf, _ = compute_constraints(t, x)

    # 6. Analyze the constraint matrix A (LHS)
    # Form: [LgV, -scale * delta] <= ...
    # a_clf shape should be (1, 2) -> 1 constraint, 2 vars (u, delta)

    assert a_clf.shape == (1, 2), f"Expected shape (1, 2), got {a_clf.shape}"

    # The coefficient for delta (last column)
    delta_coeff = a_clf[0, 1]

    # 7. Check if it is additive
    # Additive: coeff = -scale_clf
    # Multiplicative: coeff approx -scale_clf * V(x) or similar

    expected_coeff = -scale_clf

    # Check exact match (it's set by assignment)
    assert jnp.isclose(delta_coeff, expected_coeff), (
        f"Relaxation coefficient {delta_coeff} does not match additive scale {expected_coeff}. "
        f"Is it using multiplicative relaxation?"
    )

    # Explicitly check it's not tiny (which would happen if it depended on V=1e-12)
    # If it were multiplicative: coeff ~ 100 * 1e-12 = 1e-10
    # Additive: coeff = -100
    assert (
        abs(delta_coeff) > 1.0
    ), "Relaxation coefficient is too small, likely vanishing near equilibrium!"


if __name__ == "__main__":
    test_clf_additive_relaxation()
