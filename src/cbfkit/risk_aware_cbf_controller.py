import jax.numpy as jnp
from typing import Callable
from jax import grad, jit
from jax.interpreters.xla import DeviceArray
from kvxopt import matrix, solvers
from numpy import array as arr
from scipy.linalg import block_diag
from scipy.special import erfinv

global INTEGRATOR_STATE


def risk_aware_cbf_controller(
    nominal_input: Callable,
    dynamics_func: Callable,
    barrier_funcs: Callable,
    barrier_jacobians: Callable,
    barrier_hessians: Callable,
    control_limits: DeviceArray = jnp.array([100.0, 100.0]),
    alpha: float = 1,
    R: DeviceArray = None,
) -> Callable[DeviceArray, DeviceArray]:
    """
    Compute the solution to a control barrier function based quadratic program seeking to minimize the deviation
    from some nominal input.

    Args:
    nominal_input: the function in charge of computing the nominal input
    dynamics_func: the dynamics function describing the system to be controlled.
    barrier_func: the barrier function which should be less than or equal to zero within a particular set.
    barrier_jacobian: the Jacobian of the barrier function.
    control_limits: denotes the maximum control input in magnitude (e.g., for a car it's acceleration and steering rate).
    Q: state-weight parameter in the QP.
    R: input-weight parameter in the QP.
    P: positive definite matrix to compute final cost

    Returns:
    u: Optimal control input
    """
    global INTEGRATOR_STATE
    if R is None:
        R = jnp.eye(len(control_limits), dtype=float)

    M = len(control_limits)
    L = len(barrier_funcs)
    INTEGRATOR_STATE = jnp.zeros((L,))

    #! HARD CODED -- NEEDS TO CHANGE
    T = 10.0
    rhod = 0.1
    eta = 0.0
    gamma = 

    def integrate(state, derivative, dt=0.05):
        return state + derivative * dt

    def controller(x):
        dynamics = dynamics_func(x)
        N = len(dynamics[0])
        if len(dynamics) == 2:
            # Deterministic Model
            dynamics_f, dynamics_g = dynamics
            dynamics_s = jnp.zeros((N, N))
        elif len(dynamics) == 3:
            # Stochastic Model
            dynamics_f, dynamics_g, dynamics_s = dynamics

        H = matrix(arr(R, dtype=float))
        f = matrix(arr(-2 * R @ nominal_input(x), dtype=float))

        # Formulate the input constraints
        Au = block_diag_matrix(M)
        bu = interleave_arrays(control_limits, control_limits)

        # Formulate CBF constraint(s)
        Acbf = jnp.zeros((L, M))
        bcbf = jnp.zeros((L,))
        trace_term = jnp.zeros((L,))
        for ib, (bf, bj, bh) in enumerate(zip(barrier_funcs, barrier_jacobians, barrier_hessians)):
            func = risk_aware_barrier_transform(bf(x))
            jaco = risk_aware_jacobian_transform(bj(x))
            hess = risk_aware_hessian_transform(bh(x))
            trace_term = trace_term.at[ib].set(0.5 * jnp.trace(dynamics_s.T @ bh(x) @ dynamics_s))
            h = 1 - INTEGRATOR_STATE[ib] - gamma - (jnp.sqrt(2) * T * eta) * erfinv(1 - rhod)
            Acbf = Acbf.at[ib, :].set(jaco @ dynamics_g)
            bcbf = bcbf.at[ib].set(alpha * h - jaco @ dynamics_f - trace_term)

        # Formulate complete set of inequality constraints
        A = matrix(arr(jnp.vstack([Au, Acbf]), dtype=float))
        b = matrix(arr(jnp.hstack([bu, bcbf]), dtype=float))

        # Solve the QP
        sol = qp_solver(H, f, A, b)

        # Saturate the solution if necessary
        u = jnp.squeeze(jnp.array(sol[:M]))
        u = jnp.clip(u, -control_limits, control_limits)

        # Update integrator state
        for ib, (bf, bj, bh) in enumerate(zip(barrier_funcs, barrier_jacobians, barrier_hessians)):
            derivative = bj @ (dynamics_f + dynamics_g @ u) + trace_term[ib]
            integrator_state = integrator_state.at[ib].set(
                integrate(integrator_state[ib], derivative)
            )

        return u

    return controller


def adaptive_control_barrier_function_controller(
    nominal_input: Callable,
    dynamics_func: Callable,
    barrier_funcs: Callable,
    barrier_jacobians: Callable,
    control_limits: DeviceArray = jnp.array([100.0, 100.0]),
    alpha: float = 1,
    R: DeviceArray = None,
) -> Callable[DeviceArray, DeviceArray]:
    """
    Compute the solution to a control barrier function based quadratic program seeking to minimize the deviation
    from some nominal input.

    Args:
    nominal_input: the function in charge of computing the nominal input
    dynamics_func: the dynamics function describing the system to be controlled.
    barrier_func: the barrier function which should be less than or equal to zero within a particular set.
    barrier_jacobian: the Jacobian of the barrier function.
    control_limits: denotes the maximum control input in magnitude (e.g., for a car it's acceleration and steering rate).
    Q: state-weight parameter in the QP.
    R: input-weight parameter in the QP.
    P: positive definite matrix to compute final cost

    Returns:
    u: Optimal control input
    """
    M = len(control_limits)
    L = len(barrier_funcs)

    if R is None:
        u_weights = jnp.array(M * [1])
        a_weights = jnp.array(L * [100])
        R = jnp.diag(jnp.hstack([u_weights, a_weights]))

    def controller(x):
        dynamics_f, dynamics_g = dynamics_func(x)

        unom = nominal_input(x)
        anom = jnp.array(L * [alpha])
        H = matrix(arr(R, dtype=float))
        f = matrix(arr(-2 * R @ jnp.hstack([unom, anom]), dtype=float))

        # Formulate the input constraints
        alpha_limit = 100
        upper_limits = jnp.hstack([control_limits, jnp.array(L * [alpha_limit])])
        lower_limits = jnp.hstack([control_limits, jnp.array(L * [0])])
        Au = block_diag_matrix(M + L)
        bu = interleave_arrays(upper_limits, lower_limits)

        # Formulate CBF constraint(s)
        Acbf = jnp.zeros((L, M + L))
        bcbf = jnp.zeros((L,))
        for ib, (bf, bj) in enumerate(zip(barrier_funcs, barrier_jacobians)):
            Acbf = Acbf.at[ib, :M].set(-bj(x) @ dynamics_g)
            Acbf = Acbf.at[ib, M + ib].set(-bf(x))
            bcbf = bcbf.at[ib].set(bj(x) @ dynamics_f)

        # Formulate complete set of inequality constraints
        A = matrix(arr(jnp.vstack([Au, Acbf]), dtype=float))
        b = matrix(arr(jnp.hstack([bu, bcbf]), dtype=float))

        # Solve the QP
        sol = qp_solver(H, f, A, b)

        # Saturate the solution if necessary
        u = jnp.squeeze(jnp.array(sol[:M]))
        u = jnp.clip(u, -control_limits, control_limits)

        return u

    return controller


def qp_solver(H, f, A, b, G=None, h=None):
    """
    Solve a quadratic program using the cvxopt solver.

    Args:
    H: quadratic cost matrix.
    f: linear cost vector.
    A: linear constraint matrix.
    b: linear constraint vector.
    G: quadratic constraint matrix.
    h: quadratic constraint vector.

    Returns:
    sol['x']: Solution to the QP
    """
    # Use the cvxopt library to solve the quadratic program
    P = matrix(H)
    q = matrix(f)
    A = matrix(A)
    b = matrix(b)

    if G is None and h is None:
        sol = solvers.qp(P, q, A, b)
    else:
        G = matrix(G)
        h = matrix(h)
        sol = solvers.qp(P, q, G=G, h=h, A=A, b=b)

    return sol["x"]


def block_diag_matrix(n_blocks):
    block = jnp.array([1, -1])
    return jnp.array([block_diag(*([block] * n_blocks)).T])[0, :, :]


def interleave_arrays(a, b):
    return jnp.ravel(jnp.column_stack((a, b)))

def risk_aware_barrier_transform(h):
    return jnp.exp(-h)

def risk_aware_jacobian_transform(h, dhdx):
    return -jnp.exp(-h) * dhdx

def risk_aware_hessian_transform(h, dhdx, d2hdx2):
    return jnp.exp(-h) * (dhdx.T @ dhdx - d2hdx2)