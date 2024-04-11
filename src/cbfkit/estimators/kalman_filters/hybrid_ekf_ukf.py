import jax.numpy as jnp
from jax import Array
from typing import Tuple, Callable, Union, Optional

from cbfkit.utils.user_types import DynamicsCallable
from cbfkit.integration import forward_euler as integrate

from .ekf import ct_ekf_dtmeas
from .ukf import ct_ukf_dtmeas


def ct_hybrid_ekf_ukf_dtmeas(
    Q: Array,
    R: Array,
    dynamics: DynamicsCallable,
    dfdx: Callable,
    h: Callable,
    dhdx: Callable,
    dt: float,
) -> Callable[[float, Array, Array, Array], Tuple[Array, Array]]:
    """Function defining the continuous-time hybrid EKF/UKF estimator with
    discrete-time measurements.

    Args:
        Q (Array): process noise covariance matrix
        R (Array): measurement noise covariance matrix
        dynamics (DynamicsCallable): function handle to computing nonlinear dynamics
        dfdx (Callable): gradient of nonlinear dynamics model
        h (Callable): measurement model
        dhdx (Callable): gradient of measurement model
        dt (float): timestep length (sec)

    Returns:
        step_hybrid_ekf_ukf (Callable): function handle to compute the next observer state

    """
    step_ekf = ct_ekf_dtmeas(Q, R, dynamics, dfdx, h, dhdx, dt)
    step_ukf = ct_ukf_dtmeas(Q, R, dynamics, h, dt)

    def step_hybrid_ekf_ukf(
        t: float,
        y: Array,
        z: Optional[Union[Array, None]] = None,
        u: Optional[Union[Array, None]] = None,
        P: Optional[Union[Array, None]] = None,
    ):
        """"""

        if z is None and u is None and P is None:
            return initialize(y, R)

        ze, Pe = step_ekf(t, y, z, u, P[: len(z), :])
        zu, Pu = step_ukf(t, y, z, u, P[len(z) :, :])

        covariance_norm_ratio = jnp.linalg.norm(Pu, ord="fro") / jnp.linalg.norm(Pe, ord="fro")
        delta = 1 - jnp.exp(jnp.log(0.5) * covariance_norm_ratio)

        z_new = delta * ze + (1 - delta) * zu
        P_new = jnp.vstack([Pe, Pu])

        return z_new, P_new

    return step_hybrid_ekf_ukf


#! Possibly implement this in a better fashion in the future
def initialize(y: Array, R: Array) -> Array:
    """Initialization for the continuous-time EKF with discrete-time measurements.

    Arguments:
        y (Array): measurement of state
        R (Array): measurement noise covariance matrix (proxy for initial covariance)


    Returns:
        z0 (Array): initial estimate of state
        P0 (Array): initial covariance of state estimate

    """
    return y, jnp.vstack([R, R])


def predict_ct_dtmeas(
    Q: Array, dynamics: DynamicsCallable, dfdx: Callable, dt: float
) -> Callable[[float, Array, Array, Array], Tuple[Array, Array]]:
    """Function defining the prediction step for the continuous-time EKF with discrete-time measurements.

    Arguments:
        Q (Array): positive definite process noise covariance
        dynamics (DynamicsCallable): function handle to computing the nonlinear system dynamics
        dfdx (Callable): linearized dynamics model
        dt (float): timestep (sec)

    Returns:
        predict (Callable): function handle to compute EKF state and covariance matrix based on system model

    """

    def predict(t: float, z: Array, u: Array, P: Array) -> Tuple[Array, Array]:
        """Implementation of prediction step for the continuous-time EKF with discrete-time measurements.

        Arguments:
            t (float): time (sec)
            z (Array): observer state
            u (Array): control input
            P (Array): Kalman covariance matrix

        Returns:
            xk: predicted observer state
            Pk: predicted covariance matrix

        """
        # Compute xdot from system dynamics
        f, g, _ = dynamics(z)
        zdot = f + jnp.matmul(g, u)
        zk = integrate(z, zdot, dt)

        # Compute Pdot from covariance dynamics
        Ff, Fg, _ = dfdx(z)
        F = Ff + jnp.einsum("ijk,j->ik", Fg, u)
        # F = Ff + jnp.einsum('ijk,j->ki', Fg, u)
        Pdot = jnp.matmul(F, P) + jnp.matmul(P, F.T) + Q
        Pk = integrate(P, Pdot, dt)

        return zk, Pk

    return predict


def update_dtmeas(
    R: Array, h: Callable[[Array], Array], dhdx: Callable[[Array], Array]
) -> Callable[[float, Array, Array], Tuple[Array, Array]]:
    """Function defining the update step for (any) EKF with discrete-time measurements.

    Arguments:
        R (Array): measurement noise covariance matrix
        h (Callable): measurement model
        dhdx (Callable): linearized measurement model


    Returns:
        update (Callable): function handle to compute the updated EKF state and covariance matrix

    """

    def update(z: Array, y: Array, P: Array) -> Tuple[Array, Array]:
        """Update step for (any)) EKF with discrete-time measurements.

        Arguments:
            z (Array): predicted observer state
            y (Array): measurement
            P (Array): predicted Kalman covariance matrix

        Returns:
            x_new (Array): updated observer state
            P_new (Array): updated Kalman covariance matrix

        """
        H = dhdx(z)
        K = jnp.matmul(jnp.matmul(P, H.T), jnp.linalg.inv(jnp.matmul(jnp.matmul(H, P), H.T) + R))
        z_new = z + jnp.matmul(K, y - h(z))
        P_new = jnp.matmul(jnp.eye(P.shape[0]) - jnp.matmul(K, H), P)

        return z_new, P_new

    return update
