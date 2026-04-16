from typing import Callable, Optional, Tuple, Union, cast

import jax
import jax.numpy as jnp
from jax import Array

from cbfkit.integration import forward_euler as integrate
from cbfkit.utils.user_types import DynamicsCallable, EstimatorCallable, Time


def ct_ukf_dtmeas(
    Q: Array, R: Array, dynamics: DynamicsCallable, h: Callable, dt: float
) -> EstimatorCallable:
    """Function defining the continuous-time UKF with discrete-time measurements.

    Arguments:
        dynamics (DynamicsCallable): function handle to computing nonlinear dynamics

    Returns
    -------
        step_ukf (Callable): function handle to compute the next UKF observer state
    """
    predict = predict_ct_dtmeas(Q, dynamics, dt)
    update = update_dtmeas(R, h)

    def step_ukf(
        t: Time,
        y: Array,
        z: Optional[Array] = None,
        u: Optional[Array] = None,
        P: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Continuous-time implementation of Unscented Kalman Filter (UKF) with discrete-time
        measurements.

        Arguments:
            t (float): time (in sec)
            x (Array): observer state
            u (Array): control input
            y (Array): measurement
            P (Array): Kalman covariance matrix

        Returns
        -------
            x_new (Array): updated observer state
            P_new (Array): updated Kalman covariance matrix
        """
        if z is None or u is None or P is None:
            return initialize(y, R)

        x_predicted, P_predicted, s_predicted = predict(t, z, u, P)
        x_new, P_new = update(x_predicted, y, P_predicted)

        return x_new, P_new

    return step_ukf


def initialize(y: Array, R: Array) -> Tuple[Array, Array]:
    """Initialization for the continuous-time EKF with discrete-time measurements.

    Arguments:
        y (Array): measurement of state
        R (Array): measurement noise covariance matrix (proxy for initial covariance)


    Returns
    -------
        z0 (Array): initial estimate of state
        P0 (Array): initial covariance of state estimate
    """
    return y, R


def predict_ct_dtmeas(
    Q: Array, dynamics: DynamicsCallable, dt: float
) -> Callable[[Time, Array, Array, Array], Tuple[Array, Array, Array]]:
    """Function defining the prediction step for the continuous-time EKF with discrete-time
    measurements.

    Arguments:
        Q (Array): positive definite process noise covariance
        dynamics (DynamicsCallable): function handle to computing the nonlinear system dynamics
        dt (float): timestep (sec)

    Returns
    -------
        predict (Callable): function handle to compute EKF state and covariance matrix based on system model
    """
    sigma_points = generate_sigma_points(Q.shape[0], scheme=1)

    def predict(_t: Time, z: Array, u: Array, P: Array) -> Tuple[Array, Array, Array]:
        """Implementation of prediction step for the continuous-time EKF with discrete-time
        measurements.

        Arguments:
            t (Time): time (sec)
            z (Array): observer state
            u (Array): control input
            P (Array): Kalman covariance matrix

        Returns
        -------
            zk: predicted observer state
            Pk: predicted covariance matrix
            sk: predicted sigma points (through dynamics)
        """
        s, wa, wc = sigma_points(z, P)

        # Propagate sigma points through dynamics (vectorized)
        def _propagate_one(ss):
            f, g = dynamics(ss)
            val = f + jnp.matmul(g, u)
            return integrate(ss, lambda _: val, dt)

        sk = jax.vmap(_propagate_one)(s)

        # Compute predicted state estimate
        zk = jnp.dot(wa, sk)

        # Compute predicted covariance
        Pk = jnp.dot(wc, jnp.einsum("ij,ik->jik", sk - zk, sk - zk)) + Q

        return zk, Pk, sk

    return predict


def update_dtmeas(
    R: Array, h: Callable[[Array], Array]
) -> Callable[[Array, Array, Array], Tuple[Array, Array]]:
    """Function defining the update step for (any) EKF with discrete-time measurements.

    Arguments:
        R (Array): measurement noise covariance matrix
        h (Callable): measurement model
        dhdx (Callable): linearized measurement model


    Returns
    -------
        update (Callable): function handle to compute the updated EKF state and covariance matrix
    """
    sigma_points = generate_sigma_points(R.shape[0], scheme=1)

    def update(z: Array, y: Array, P: Array) -> Tuple[Array, Array]:
        """Update step for (any)) EKF with discrete-time measurements.

        Arguments:
            z (Array): predicted observer state
            y (Array): measurement
            P (Array): predicted Kalman covariance matrix

        Returns
        -------
            z_new (Array): updated observer state
            P_new (Array): updated Kalman covariance matrix
        """
        s, wa, wc = sigma_points(z, P)

        # Transform sigma points through measurement model
        s_predicted = h(s)

        # Generate measurement estimate, covariance, and cross covariance
        y_pred = jnp.dot(wa, s_predicted)
        S_pred = (
            jnp.dot(wc, jnp.einsum("ij,ik->jik", s_predicted - y_pred, s_predicted - y_pred)) + R
        )
        C_pred = jnp.dot(wc, jnp.einsum("ij,ik->jik", s - z, s_predicted - y_pred))

        # Compute UKF updated state and covariance estimates
        K = jnp.matmul(C_pred, jnp.linalg.inv(S_pred))
        z_new = z + jnp.matmul(K, y - y_pred)
        P_new = P - jnp.matmul(jnp.matmul(K, S_pred), K.T)

        return z_new, P_new

    return update


def generate_sigma_points(
    L: int, alpha: float = 0.05, beta: float = 2.0, kappa: float = 1.0, scheme: int = 0
) -> Callable[[Array, Array], Tuple[Array, Array, Array]]:
    """Generates the sigma points for the Unscented Transform.

    Arguments:
        z (Array): state estimate vector
        alpha (float): sigma point parameter
        beta (float): sigma point parameter
        kappa (float): sigma point parameter

    Returns
    -------
        sigma_points (Array): sigma points for unscented transform
        Wa (Array): weights for state estimation
        Wc (Array): weights for covariance estimation
    """
    # Initialize sigma points and weights
    Wa = jnp.zeros((2 * L + 1,))
    Wc = jnp.zeros((2 * L + 1,))

    if scheme == 0:
        # Set sigma points and weights
        Wa = Wa.at[0].set((alpha**2 * kappa - L) / (alpha**2 * kappa))
        Wc = Wc.at[0].set(Wa[0] + 1 - alpha**2 + beta)

        # Compute weights
        Wa = Wa.at[1:].set(1 / (2 * alpha**2 * kappa))
        Wc = Wc.at[1:].set(1 / (2 * alpha**2 * kappa))

    elif scheme == 1:
        # Set sigma points and weights
        Wa = Wa.at[0].set(0.0)
        Wc = Wc.at[0].set(0.0)

        # Compute weights
        Wa = Wa.at[1:].set(1 - Wa[0] / (2 * L))
        Wc = Wc.at[1:].set(1 - Wa[0] / (2 * L))

    elif scheme == 2:
        # Set sigma points and weights
        Wa = Wa.at[0].set((3 - L) / 3)
        Wc = Wc.at[0].set((3 - L) / 3)

        # Compute weights
        Wa = Wa.at[1:].set(1 / 6)
        Wc = Wc.at[1:].set(1 / 6)

    Wa = Wa / jnp.sum(Wa)
    Wc = Wc / jnp.sum(Wc)

    def sigma_points(z: Array, P: Array) -> Tuple[Array, Array, Array]:
        """

        Args:
            z (Array): current state estimate
            P (Array): current UKF covariance matrix

        Returns
        -------
        """
        # Cholesky decomposition of covariance matrix
        A = jnp.linalg.cholesky(P)

        # Initialize sigma points and weights
        sigma_points = jnp.zeros((2 * L + 1, len(z)))

        # Set sigma points and weights
        sigma_points = sigma_points.at[0, :].set(z)

        # Generate indices for the two ranges
        indices_1 = jnp.arange(1, L + 1)
        indices_2 = jnp.arange(L + 1, 2 * L + 1)

        if scheme == 0:
            # Calculate the first set of sigma points in one line using broadcasting
            sigma_points = sigma_points.at[indices_1, :].set(
                z + alpha * jnp.sqrt(kappa) * A[:, indices_1 - 1].T
            )

            # Calculate the second set of sigma points in one line using broadcasting
            sigma_points = sigma_points.at[indices_2, :].set(
                z - alpha * jnp.sqrt(kappa) * A[:, indices_2 - L - 1].T
            )

        elif scheme == 1:
            # Calculate the first set of sigma points in one line using broadcasting
            sigma_points = sigma_points.at[indices_1, :].set(
                z + jnp.sqrt(L / (1 - Wa[0])) * A[:, indices_1 - 1].T
            )

            # Calculate the second set of sigma points in one line using broadcasting
            sigma_points = sigma_points.at[indices_2, :].set(
                z - jnp.sqrt(L / (1 - Wa[0])) * A[:, indices_2 - L - 1].T
            )

        elif scheme == 2:
            # Calculate the first set of sigma points in one line using broadcasting
            sigma_points = sigma_points.at[indices_1, :].set(
                z + jnp.sqrt(3) * A[:, indices_1 - 1].T
            )

            # Calculate the second set of sigma points in one line using broadcasting
            sigma_points = sigma_points.at[indices_2, :].set(
                z - jnp.sqrt(3) * A[:, indices_2 - L - 1].T
            )

        return sigma_points, Wa, Wc

    return sigma_points
