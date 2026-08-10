"""UKF must produce finite estimates from an all-zero initial covariance.

The simulator seeds the estimator covariance with ``jnp.zeros((n, n))`` when the
caller passes no ``initial_covariance``. LAPACK's Cholesky of a zero matrix is
NaN (zero pivot), which used to NaN every sigma point on the first update and
poison the whole pipeline (the reach_goal/ukf.py "NaN/Inf in g_mat at rows: [6]"
failure). A zero prior is legitimate: its factor is zero, sigma points collapse
to the mean, and the predict step restores spread via Q.
"""

import jax.numpy as jnp

from cbfkit.estimators.kalman_filters.ukf import ct_ukf_dtmeas


def _dynamics(x):
    return jnp.zeros_like(x), jnp.eye(x.shape[0])


def test_ukf_step_finite_from_zero_covariance():
    n = 2
    estimator = ct_ukf_dtmeas(
        Q=0.05 * jnp.eye(n),
        R=0.1 * jnp.eye(n),
        dynamics=_dynamics,
        h=lambda x: x,
        dt=0.05,
    )
    z0 = jnp.array([0.3, -0.4])
    p0 = jnp.zeros((n, n))  # the simulator's default seed
    y = jnp.array([0.35, -0.38])
    u = jnp.zeros(n)

    z1, p1 = estimator(0.0, y, z0, u, p0)[:2]
    assert bool(jnp.all(jnp.isfinite(z1))), f"NaN estimate from zero-covariance seed: {z1}"
    assert bool(jnp.all(jnp.isfinite(p1))), f"NaN covariance from zero-covariance seed: {p1}"

    # Covariance must regain spread from Q (not stay pinned at zero),
    # and a second step must remain finite.
    assert float(jnp.trace(p1)) > 0.0
    z2, p2 = estimator(0.05, y, z1, u, p1)[:2]
    assert bool(jnp.all(jnp.isfinite(z2))) and bool(jnp.all(jnp.isfinite(p2)))


def test_ukf_nonzero_covariance_path_unchanged():
    """The zero-seed guard must not perturb the regular PD-covariance path."""
    n = 2
    estimator = ct_ukf_dtmeas(
        Q=0.05 * jnp.eye(n),
        R=0.1 * jnp.eye(n),
        dynamics=_dynamics,
        h=lambda x: x,
        dt=0.05,
    )
    z0 = jnp.array([0.3, -0.4])
    p0 = 0.2 * jnp.eye(n)
    y = jnp.array([0.35, -0.38])
    z1, p1 = estimator(0.0, y, z0, jnp.zeros(n), p0)[:2]
    assert bool(jnp.all(jnp.isfinite(z1))) and bool(jnp.all(jnp.isfinite(p1)))
    # PD prior + PD noise: posterior trace stays strictly positive
    assert float(jnp.trace(p1)) > 0.0
