import jax.numpy as jnp
from jax import Array
from cbfkit.utils.user_types import CertificateTuple
from cbfkit.systems.single_integrator.certificates.lyapunov_function_catalog import position


def fxts_lyapunov_conditions(c1: float, c2: float, e1: float, e2: float):
    """Generates function for computing RHS of Lyapunov conditions for FxTS.

    Args:
        c1 (float): convergence constant 1
        c2 (float): convergence constant 2
        e1 (float): exponential constant 1
        e2 (float): exponential constant 2

    Returns:
        callable[float]: FxTS Lyapunov conditions
    """
    return ([lambda V: jnp.where(V > 0, -c1 * V**e1 - c2 * V**e2, 0.0)],)


def fxts_lyapunov(
    goal: Array, r: float, c1: float, c2: float, e1: float, e2: float
) -> CertificateTuple:
    """Generates Lyapunov functions, jacobians, hessians for use in CLF control law.

    Args:
        goal (Array): goal location
        rr (float): goal set radius
        c1 (float): convergence constant 1
        c2 (float): convergence constant 2
        e1 (float): exponential constant 1
        e2 (float): exponential constant 2

    Returns:
        CertificateTuple: all inforrmation needed for CLF constraint in QP
    """

    return position(goal, r) + fxts_lyapunov_conditions(c1, c2, e1, e2)
