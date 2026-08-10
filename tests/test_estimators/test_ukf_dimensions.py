"""Tests for UKF behavior when the measurement dimension differs from the state dimension.

The update step's unscented transform draws sigma points over the *state*, so it must be
parameterized by the state dimension. Parameterizing it by the measurement dimension
builds only ``2 * dim(y) + 1`` points and spreads them along the first ``dim(y)`` columns
of the Cholesky factor of P, so the reconstructed covariance
``sum_i wc_i (s_i - z)(s_i - z)^T`` has rank ``dim(y)`` instead of rank ``dim(x)``.

That defect is invisible in two situations, which is why it stayed latent:

* ``dim(y) == dim(x)`` -- every sensor shipped with the library is full-state, so the two
  dimensions always agreed and the sigma-point set was correct by accident.
* the measurement reads exactly the *leading* ``dim(y)`` state coordinates. The Cholesky
  factor is lower-triangular, so its first ``m`` columns still reproduce the exact
  top-left ``m x m`` block of P, and both the innovation covariance and the cross
  covariance come out right anyway.

It is very much visible as soon as the measurement touches a state coordinate outside
that leading block (velocity-only sensing of ``[position, velocity, acceleration]``, say),
where the filter silently disagrees with the exact Kalman filter it should reproduce.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from cbfkit.estimators import ct_ukf_dtmeas
from cbfkit.estimators.kalman_filters.ukf import generate_sigma_points

DT = 0.05
N_STEPS = 50

# 1-D constant-acceleration model: state is [position, velocity, acceleration].
A_CONST_ACCEL = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])


def _linear_dynamics(A):
    """Builds a dynamics callable xdot = A x + 0 u for the given state matrix."""
    n = A.shape[0]

    def dynamics(x):
        return jnp.matmul(jnp.asarray(A), x), jnp.zeros((n, 1))

    return dynamics


def _linear_measurement(H):
    """Builds y = H x, applied either to one state or to a stack of sigma points."""

    def h(x):
        return jnp.matmul(x, jnp.asarray(H).T)

    return h


def _simulate_measurements(A, H, x0, noise_std, seed=0):
    """Rolls the forward-Euler truth trajectory and its noisy measurements."""
    rng = np.random.default_rng(seed)
    F = np.eye(A.shape[0]) + DT * A
    x = np.asarray(x0, dtype=float)
    states, measurements = [], []
    for _ in range(N_STEPS):
        x = F @ x
        states.append(x.copy())
        measurements.append(H @ x + rng.normal(0.0, noise_std, size=(H.shape[0],)))
    return np.array(states), measurements


def _run_estimator(estimator, measurements, z0, P0):
    """Steps the estimator over a measurement sequence; returns estimate/covariance history."""
    z, P = jnp.asarray(z0), jnp.asarray(P0)
    u = jnp.zeros((1,))
    estimates, covariances = [], []
    for k, y in enumerate(measurements):
        z, P = estimator(k * DT, jnp.asarray(y), z, u, P)
        estimates.append(np.asarray(z))
        covariances.append(np.asarray(P))
    return np.array(estimates), np.array(covariances)


def _run_reference_kalman_filter(A, H, Q, R, measurements, z0, P0):
    """Exact discrete-time KF matching the UKF's forward-Euler predict and its update."""
    F = np.eye(A.shape[0]) + DT * A
    x, P = np.asarray(z0, dtype=float), np.asarray(P0, dtype=float)
    estimates, covariances = [], []
    for y in measurements:
        x = F @ x
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R
        C = P @ H.T
        K = C @ np.linalg.inv(S)
        x = x + K @ (y - H @ x)
        P = P - K @ S @ K.T
        estimates.append(x.copy())
        covariances.append(P.copy())
    return np.array(estimates), np.array(covariances)


def _assert_finite_and_psd(covariances):
    for k, P in enumerate(covariances):
        assert np.all(np.isfinite(P)), f"covariance is not finite at step {k}: {P}"
        symmetric = 0.5 * (P + P.T)
        assert np.allclose(P, symmetric, atol=1e-9), f"covariance is not symmetric at step {k}"
        eigenvalues = np.linalg.eigvalsh(symmetric)
        assert eigenvalues.min() > -1e-9, f"covariance is not PSD at step {k}: {eigenvalues}"


@pytest.mark.parametrize("n_states, n_measurements", [(3, 1), (3, 2), (4, 2), (2, 2), (2, 3)])
def test_update_draws_sigma_points_over_the_state(n_states, n_measurements):
    """The update's sigma-point set is 2 * dim(x) + 1 points wide dim(x), whatever dim(y) is."""
    observed_shapes = []
    H = np.eye(n_measurements, n_states)

    def recording_measurement(x):
        observed_shapes.append(x.shape)
        return _linear_measurement(H)(x)

    estimator = ct_ukf_dtmeas(
        Q=1e-4 * jnp.eye(n_states),
        R=1e-2 * jnp.eye(n_measurements),
        dynamics=_linear_dynamics(np.zeros((n_states, n_states))),
        h=recording_measurement,
        dt=DT,
    )
    z, P = estimator(
        0.0,
        jnp.zeros((n_measurements,)),
        jnp.zeros((n_states,)),
        jnp.zeros((1,)),
        jnp.eye(n_states),
    )

    assert observed_shapes, "measurement model was never applied to the sigma points"
    assert observed_shapes[0] == (2 * n_states + 1, n_states)
    assert z.shape == (n_states,)
    assert P.shape == (n_states, n_states)
    assert np.all(np.isfinite(np.asarray(P)))


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_sigma_points_reproduce_the_mean_and_covariance(n):
    """The defining property of the unscented transform, and the reason L must be dim(x).

    A set built with L < n reconstructs a rank-L covariance instead of P.
    """
    rng = np.random.default_rng(0)
    M = rng.normal(size=(n, n))
    P = M @ M.T + n * np.eye(n)
    z = rng.normal(size=(n,))

    s, wa, wc = generate_sigma_points(n, scheme=1)(jnp.asarray(z), jnp.asarray(P))
    s, wa, wc = np.asarray(s), np.asarray(wa), np.asarray(wc)

    assert s.shape == (2 * n + 1, n)
    np.testing.assert_allclose(wa.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(wa @ s, z, atol=1e-10)

    deviations = s - z
    reconstructed = np.einsum("i,ij,ik->jk", wc, deviations, deviations)
    np.testing.assert_allclose(reconstructed, P, atol=1e-10)


@pytest.mark.parametrize(
    "label, H",
    [
        ("position-only", np.array([[1.0, 0.0, 0.0]])),
        ("velocity-only", np.array([[0.0, 1.0, 0.0]])),
        ("acceleration-only", np.array([[0.0, 0.0, 1.0]])),
        ("position-plus-velocity", np.array([[1.0, 1.0, 0.0]])),
        ("velocity-and-acceleration", np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])),
        ("full-state", np.eye(3)),
    ],
)
def test_matches_exact_kalman_filter_for_linear_systems(label, H):
    """For a linear model the UKF must reproduce the exact KF, for any measurement geometry.

    The sigma-point set carries the mean and covariance exactly through a linear map, so
    any disagreement here means the unscented transform itself is malformed.
    """
    n_measurements = H.shape[0]
    Q = 1e-4 * np.eye(3)
    R = 1e-2 * np.eye(n_measurements)
    z0 = np.zeros(3)
    P0 = np.diag([0.1, 1.0, 1.0])

    _, measurements = _simulate_measurements(
        A_CONST_ACCEL, H, [0.0, 1.0, -0.2], noise_std=0.05, seed=1
    )

    estimator = ct_ukf_dtmeas(
        Q=jnp.asarray(Q),
        R=jnp.asarray(R),
        dynamics=_linear_dynamics(A_CONST_ACCEL),
        h=_linear_measurement(H),
        dt=DT,
    )
    estimates, covariances = _run_estimator(estimator, measurements, z0, P0)
    expected_estimates, expected_covariances = _run_reference_kalman_filter(
        A_CONST_ACCEL, H, Q, R, measurements, z0, P0
    )

    np.testing.assert_allclose(
        estimates, expected_estimates, atol=1e-9, rtol=1e-7, err_msg=f"{label}: estimate drift"
    )
    np.testing.assert_allclose(
        covariances,
        expected_covariances,
        atol=1e-9,
        rtol=1e-7,
        err_msg=f"{label}: covariance drift",
    )


def test_partial_measurement_runs_and_converges():
    """dim(x) = 3, dim(y) = 1: runs 50 steps, stays finite and PSD, beats the initial guess."""
    H = np.array([[1.0, 0.0, 0.0]])
    x0_true = np.array([0.0, 1.0, -0.2])
    z0 = np.zeros(3)
    P0 = np.diag([0.1, 1.0, 1.0])

    states, measurements = _simulate_measurements(A_CONST_ACCEL, H, x0_true, noise_std=0.1, seed=3)

    estimator = ct_ukf_dtmeas(
        Q=1e-4 * jnp.eye(3),
        R=1e-2 * jnp.eye(1),
        dynamics=_linear_dynamics(A_CONST_ACCEL),
        h=_linear_measurement(H),
        dt=DT,
    )
    estimates, covariances = _run_estimator(estimator, measurements, z0, P0)

    assert estimates.shape == (N_STEPS, 3)
    assert np.all(np.isfinite(estimates))
    _assert_finite_and_psd(covariances)

    initial_error = np.linalg.norm(z0 - x0_true)
    final_error = np.linalg.norm(estimates[-1] - states[-1])
    assert (
        final_error < initial_error
    ), f"estimate did not improve: initial {initial_error:.4f}, final {final_error:.4f}"


def test_unmeasured_state_is_corrected_through_the_cross_covariance():
    """Velocity is never measured directly, so only the cross-covariance can correct it."""
    H = np.array([[1.0, 0.0, 0.0]])
    x0_true = np.array([0.0, 1.0, 0.0])
    z0 = np.zeros(3)
    P0 = np.diag([0.1, 1.0, 1.0])

    states, measurements = _simulate_measurements(A_CONST_ACCEL, H, x0_true, noise_std=0.05, seed=5)

    estimator = ct_ukf_dtmeas(
        Q=1e-4 * jnp.eye(3),
        R=1e-2 * jnp.eye(1),
        dynamics=_linear_dynamics(A_CONST_ACCEL),
        h=_linear_measurement(H),
        dt=DT,
    )
    estimates, _ = _run_estimator(estimator, measurements, z0, P0)

    initial_velocity_error = abs(z0[1] - x0_true[1])
    final_velocity_error = abs(estimates[-1][1] - states[-1][1])
    assert final_velocity_error < 0.5 * initial_velocity_error, (
        "velocity was not corrected through the cross-covariance: "
        f"initial error {initial_velocity_error:.4f}, final error {final_velocity_error:.4f}"
    )


def test_full_state_measurement_is_unaffected():
    """dim(y) == dim(x) is the pre-existing path and must keep behaving identically."""
    H = np.eye(3)
    x0_true = np.array([0.0, 1.0, -0.2])
    z0 = np.zeros(3)
    P0 = np.diag([0.1, 1.0, 1.0])

    states, measurements = _simulate_measurements(A_CONST_ACCEL, H, x0_true, noise_std=0.1, seed=7)

    estimator = ct_ukf_dtmeas(
        Q=1e-4 * jnp.eye(3),
        R=1e-2 * jnp.eye(3),
        dynamics=_linear_dynamics(A_CONST_ACCEL),
        h=_linear_measurement(H),
        dt=DT,
    )
    estimates, covariances = _run_estimator(estimator, measurements, z0, P0)

    _assert_finite_and_psd(covariances)
    assert np.linalg.norm(estimates[-1] - states[-1]) < np.linalg.norm(z0 - x0_true)
