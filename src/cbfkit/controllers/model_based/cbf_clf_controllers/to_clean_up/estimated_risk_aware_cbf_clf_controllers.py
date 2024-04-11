import jax.numpy as jnp
from jax import Array, lax
from numpy import array as arr
from typing import List, Union, Callable, Optional
from cbfkit.utils.user_types import (
    BarrierCollectionCallable,
    ControllerCallable,
    ControllerCallableReturns,
    DynamicsCallable,
    LyapunovCollectionCallable,
    State,
)
from cbfkit.utils.numerical_integration import forward_euler as integrator
from cbfkit.utils.solvers import qp_solver
from .utils import (
    block_diag_matrix_from_vec,
    interleave_arrays,
    generate_compute_input_constraints,
    generate_compute_estimated_risk_aware_cbf_clf_constraints,
    generate_compute_estimated_risk_aware_path_integral_cbf_clf_constraints,
    stochastic_barrier_transform,
    stochastic_hessian_transform,
    stochastic_jacobian_transform,
)
from cbfkit.estimators.kalman_filters.ekf import get_global_k_ekf


class Params:
    """Object to hold values of globally-tracked parameters."""

    integrator_states: Union[Array, None] = None

    def __init__(
        self,
        t_max: Optional[Union[float, None]] = None,
        p_bound: Optional[Union[float, None]] = None,
        gamma: Optional[Union[float, None]] = None,
        eta: Optional[Union[float, None]] = None,
        varsigma: Optional[Union[Array, None]] = None,
    ):
        self.t_max = t_max
        self.p_bound = p_bound
        self.gamma = gamma
        self.eta = eta
        self.varsigma = varsigma
        self.epsilon = 0.5
        self.lambda_h = 1.0


class RiskAwareParams:
    """Object to hold values of globally-tracked parameters."""

    integrator_states: Union[Array, None] = None

    def __init__(
        self,
        t_max: Optional[Union[float, None]] = None,
        p_bound_b: Optional[Union[float, None]] = None,
        gamma_b: Optional[Union[float, None]] = None,
        eta_b: Optional[Union[float, None]] = None,
        p_bound_v: Optional[Union[float, None]] = None,
        gamma_v: Optional[Union[float, None]] = None,
        eta_v: Optional[Union[float, None]] = None,
        varsigma: Optional[Union[Array, None]] = None,
    ):
        self.ra_cbf = Params(t_max, p_bound_b, gamma_b, eta_b, varsigma)
        self.ra_clf = Params(t_max, p_bound_v, gamma_v, eta_v, varsigma)


def fxt_cbf_clf_controller(
    nominal_input: ControllerCallable,
    dynamics_func: DynamicsCallable,
    barriers: BarrierCollectionCallable = lambda: ([], [], [], []),
    lyapunovs: LyapunovCollectionCallable = lambda: ([], [], [], [], []),
    control_limits: Array = jnp.array([100.0, 100.0]),
    alpha: Array = jnp.array([1.0]),
    R: Union[Array, None] = None,
    t_max: Optional[Union[float, None]] = None,
    p_bound_b: Optional[Union[float, None]] = None,
    p_bound_v: Optional[Union[float, None]] = None,
    gamma_b: Optional[Union[float, None]] = None,
    gamma_v: Optional[Union[float, None]] = None,
    eta_b: Optional[Union[float, None]] = None,
    eta_v: Optional[Union[float, None]] = None,
    varsigma: Optional[Union[Array, None]] = None,
) -> ControllerCallable:
    """
    Compute the solution to a control barrier function/control Lyapunov function-based
    quadratic program seeking to minimize the deviation from some nominal input.

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

    if R is None:
        Rmat = jnp.eye(M, dtype=float)
    else:
        Rmat = R

    ra_params = RiskAwareParams(
        t_max, p_bound_b, gamma_b, eta_b, p_bound_v, gamma_v, eta_v, varsigma
    )
    compute_input_constraints = generate_compute_input_constraints(control_limits)
    compute_cbf_clf_constraints = generate_compute_estimated_risk_aware_cbf_clf_constraints(
        ra_params, dynamics_func, barriers, lyapunovs, control_limits, alpha
    )

    # @jit
    def controller(t: float, x: State) -> ControllerCallableReturns:
        u_nom, _ = nominal_input(t, x)
        H = Rmat
        f = jnp.matmul(-2 * Rmat, u_nom)
        Gu, hu = compute_input_constraints(t, x)
        Gc, hc, complete = compute_cbf_clf_constraints(t, x)
        if complete:
            return u_nom, {"complete": True}

        # Concatenate
        G = jnp.vstack([Gu, Gc])
        h = jnp.hstack([hu, hc])

        # Solve the QP
        sol, status = qp_solver(
            H, f, G=G, h=h, A=jnp.zeros((1, H.shape[1])), b=jnp.zeros((1,)), lib="cvxopt"
        )
        u = lax.cond(
            status,
            lambda _fake: jnp.array(sol[:M]).reshape((M,)),
            lambda _fake: jnp.zeros((M,)),
            0,
        )
        # print(u)
        terminate = lax.cond(
            not status,
            lambda _fake: True,
            lambda _fake: False,
            0,
        )

        # Saturate the solution if necessary
        u = jnp.squeeze(jnp.array(sol[:M]))
        u = jnp.clip(u, -control_limits, control_limits)

        # logging data
        data = {
            # "cbfs": bf_x,
            # "clfs": lf_x,
            "complete": terminate,
            "feasible_area": lax.cond(status, lambda _fake: 1, lambda _fake: 0, 0),
            "sol": jnp.array(sol),
            "u": u,
            "u_nom": u_nom,
        }

        return u, data

    return controller


def pi_cbf_clf_controller(
    nominal_input: ControllerCallable,
    dynamics_func: DynamicsCallable,
    barriers: BarrierCollectionCallable = lambda: ([], [], [], []),
    lyapunovs: LyapunovCollectionCallable = lambda: ([], [], [], [], []),
    control_limits: Array = jnp.array([100.0, 100.0]),
    alpha: Array = jnp.array([1.0]),
    R: Union[Array, None] = None,
    t_max: Optional[Union[float, None]] = None,
    p_bound_b: Optional[Union[float, None]] = None,
    p_bound_v: Optional[Union[float, None]] = None,
    gamma_b: Optional[Union[float, None]] = None,
    gamma_v: Optional[Union[float, None]] = None,
    eta_b: Optional[Union[float, None]] = None,
    eta_v: Optional[Union[float, None]] = None,
    varsigma: Optional[Union[Array, None]] = None,
    dt: float = 1e-2,
) -> ControllerCallable:
    """
    Compute the solution to a control barrier function/control Lyapunov function-based
    quadratic program seeking to minimize the deviation from some nominal input.

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

    if R is None:
        Rmat = jnp.eye(M, dtype=float)
    else:
        Rmat = R

    ra_params = RiskAwareParams(
        t_max, p_bound_b, gamma_b, eta_b, p_bound_v, gamma_v, eta_v, varsigma
    )
    compute_input_constraints = generate_compute_input_constraints(control_limits)
    compute_cbf_clf_constraints = (
        generate_compute_estimated_risk_aware_path_integral_cbf_clf_constraints(
            ra_params, dynamics_func, barriers, lyapunovs, control_limits, alpha
        )
    )
    lyapunov_functions, lyapunov_jacobians, lyapunov_hessians, _, _ = lyapunovs()

    # @jit
    def controller(t: float, x: State) -> ControllerCallableReturns:
        nonlocal ra_params
        dynamics_f, dynamics_g, dynamics_s = dynamics_func(x)
        u_nom, _ = nominal_input(t, x)
        H = Rmat
        f = jnp.matmul(-2 * Rmat, u_nom)
        Gu, hu = compute_input_constraints(t, x)
        Gc, hc, complete = compute_cbf_clf_constraints(t, x)
        if complete:
            return u_nom, {"complete": True}

        # Concatenate
        G = jnp.vstack([Gu, Gc])
        h = jnp.hstack([hu, hc])

        # Solve the QP
        sol, status = qp_solver(
            H, f, G=G, h=h, A=jnp.zeros((1, H.shape[1])), b=jnp.zeros((1,)), lib="cvxopt"
        )
        u = lax.cond(
            status,
            lambda _fake: jnp.array(sol[:M]).reshape((M,)),
            lambda _fake: jnp.zeros((M,)),
            0,
        )
        # print(u)
        terminate = lax.cond(
            not status,
            lambda _fake: True,
            lambda _fake: False,
            0,
        )

        # Saturate the solution if necessary
        u = jnp.squeeze(jnp.array(sol[:M]))
        u = jnp.clip(u, -control_limits, control_limits)

        # logging data
        data = {
            # "cbfs": bf_x,
            # "clfs": lf_x,
            "complete": terminate,
            "feasible_area": lax.cond(status, lambda _fake: 1, lambda _fake: 0, 0),
            "sol": jnp.array(sol),
            "u": u,
            "u_nom": u_nom,
        }

        lj_x = jnp.stack([lj(t, x) for lj in lyapunov_jacobians])
        lh_x = jnp.stack([lh(t, x) for lh in lyapunov_hessians])

        Lf1 = jnp.matmul(lj_x, dynamics_f)
        Lg1 = jnp.matmul(lj_x, dynamics_g)

        k_mat = get_global_k_ekf()
        lipschitz_term = (
            ra_params.ra_clf.lambda_h
            * ra_params.ra_clf.epsilon
            * jnp.linalg.norm(jnp.matmul(lj_x, k_mat))
        )
        varsigma_by_k = jnp.matmul(ra_params.ra_clf.varsigma, k_mat)
        Ls = jnp.array(
            [
                0.5 * jnp.trace(jnp.matmul(jnp.matmul(varsigma_by_k.T, lh_ii), varsigma_by_k))
                for lh_ii in lh_x
            ]
        )
        generator1 = Lf1 + jnp.matmul(Lg1, u) + lipschitz_term + Ls

        new_integrator_state = ra_params.ra_clf.integrator_states + generator1 * dt
        ra_params.ra_clf.integrator_states = ra_params.ra_clf.integrator_states.at[:].set(
            new_integrator_state
        )

        return u, data

    return controller


def risk_aware_cbf_clf_controller(
    nominal_input: ControllerCallable,
    dynamics_func: DynamicsCallable,
    barriers: BarrierCollectionCallable = lambda: ([], [], [], []),
    lyapunovs: LyapunovCollectionCallable = lambda: ([], [], [], [], []),
    control_limits: Array = jnp.array([1e6, 1e6]),
    alpha: Array = jnp.array([1.0]),
    R: Union[Array, None] = None,
) -> ControllerCallable:
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

    Returns:
    u: Optimal control input
    """
    barrier_functions, barrier_jacobians, barrier_hessians, barrier_partials = barriers()
    (
        lyapunov_functions,
        lyapunov_jacobians,
        lyapunov_hessians,
        lyapunov_partials,
        lyapunov_conditions,
    ) = lyapunovs()

    if R is None:
        Rmat = jnp.eye(len(control_limits), dtype=float)
    else:
        Rmat = R

    M = len(control_limits)
    L = len(barrier_functions)
    K = len(lyapunov_functions)
    RA_GLOBALS.integrator_states = jnp.zeros((L + K,))

    if len(alpha) != L:
        alpha = jnp.array(L * alpha.min())

    #! TO DO -- Implement Programmatically
    eps = 0.01
    T = 2.0
    rhob = 0.1
    # pv = 0.5
    pv = 0.9

    def controller(t: float, x: State) -> ControllerCallableReturns:
        dynamics_f, dynamics_g, dynamics_s = dynamics_func(x)
        N = len(dynamics_f)
        if t == 0:
            RA_GLOBALS.integrator_states = jnp.zeros((L + K,))
            RA_GLOBALS.rb = stochastic_barrier_transform(
                jnp.array([bf(t, x) for bf in barrier_functions])
            )
            RA_GLOBALS.rv = jnp.array([lf(t, x) for lf in lyapunov_functions])
            RA_GLOBALS.eta_b = jnp.linalg.norm(
                jnp.array([1 / jnp.sqrt(2), 1 / jnp.sqrt(2), 0.0]) @ dynamics_s
            )
            RA_GLOBALS.eta_v = jnp.linalg.norm(jnp.array([10.0, 10.0, 0.0]) @ dynamics_s)

        u_nom, _ = nominal_input(t, x)
        H = arr(Rmat, dtype=float)
        f = arr(-2 * Rmat @ u_nom, dtype=float)

        # Formulate the input constraints
        Au = block_diag_matrix_from_vec(M)
        bu = interleave_arrays(control_limits, control_limits)

        # Formulate CBF constraint(s)
        Acbf = jnp.zeros((L, M))
        bcbf = jnp.zeros((L,))
        bf_x = jnp.zeros((L,))
        bj_x = jnp.zeros((L, N))
        bh_x = jnp.zeros((L, N, N))
        if L > 0:
            bf_x = jnp.stack([bf(t, x) for bf in barrier_functions])
            bj_x = jnp.stack(
                [
                    stochastic_jacobian_transform(bf_x[ii], bj(t, x))
                    for ii, bj in enumerate(barrier_jacobians)
                ]
            )
            bh_x = jnp.stack(
                [
                    stochastic_hessian_transform(bf_x[ii], bj(t, x), bh(t, x))
                    for ii, (bj, bh) in enumerate(zip(barrier_jacobians, barrier_hessians))
                ]
            )
            dbf_t = jnp.stack(
                [
                    stochastic_jacobian_transform(bf_x[ii], bt(t, x))
                    for ii, bt in enumerate(barrier_partials)
                ]
            )
            h_vals = (
                1
                - RA_GLOBALS.integrator_states[:L]
                - RA_GLOBALS.rb
                - (RA_GLOBALS.eta_b * jnp.sqrt(2 * T)) * erfinv(1 - rhob)
            )
            traces = jnp.array(
                [0.5 * jnp.trace(dynamics_s.T @ bh_ii @ dynamics_s) for bh_ii in bh_x]
            )

            Acbf = Acbf.at[:, :].set(jnp.matmul(bj_x, dynamics_g))
            bcbf = bcbf.at[:].set(
                jnp.multiply(jnp.array(alpha), h_vals)
                - dbf_t
                - jnp.matmul(bj_x, dynamics_f)
                - traces
            )

        # Formulate CLF constraint(s)
        Aclf = jnp.zeros((K, M))
        bclf = jnp.zeros((K,))
        lf_x = jnp.zeros((K,))
        lj_x = jnp.zeros((K, N))
        lh_x = jnp.zeros((K, N, N))
        w_vals = 0
        if K > 0:
            lf_x = jnp.stack([lf(t, x) for lf in lyapunov_functions])
            lj_x = jnp.stack([lj(t, x) for lj in lyapunov_jacobians])
            lh_x = jnp.stack([lh(t, x) for lh in lyapunov_hessians])
            dlf_t = jnp.stack([lj(t, x) for lj in lyapunov_partials])
            w_vals = (
                RA_GLOBALS.integrator_states[-K:]
                + RA_GLOBALS.rv
                + (RA_GLOBALS.eta_v * jnp.sqrt(2 * T)) * erfinv(pv)
            )
            lc_x = jnp.stack([lc(w_vals[ii]) for ii, lc in enumerate(lyapunov_conditions)])
            traces = jnp.array(
                [
                    0.5 * jnp.trace(jnp.matmul(jnp.matmul(dynamics_s.T, lh_ii), dynamics_s))
                    for lh_ii in lh_x
                ]
            )

            idxs = jnp.where(w_vals > eps and lf_x > 0)  # only impose when W positive
            Aclf = Aclf.at[idxs, :].set(jnp.matmul(lj_x, dynamics_g))
            bclf = bclf.at[idxs].set(lc_x - dlf_t - jnp.matmul(lj_x, dynamics_f) - traces)

        # Formulate complete set of inequality constraints
        A = arr(jnp.vstack([Au, Acbf, Aclf]), dtype=float)
        b = arr(jnp.hstack([bu, bcbf, bclf]), dtype=float)

        # Solve the QP
        sol, status = qp_solver(H, f, A, b)

        if not status:
            # sol = jnp.array([0.0, 0.0])
            raise ValueError("INFEASIBLE QP")

        # Saturate the solution if necessary
        u = jnp.squeeze(jnp.array(sol[:M]))
        u = jnp.clip(u, -control_limits, control_limits)

        # logging data
        data = {
            "cbfs": bf_x,
            "clfs": lf_x,
            "sol": jnp.array(sol),
            "u": u,
            "u_nom": u_nom,
            "w": w_vals,
        }

        # Update integrator state (discrete-implementation variant)
        jacobians = jnp.vstack([bj_x, lj_x])
        hessians = jnp.vstack([bh_x, lh_x])
        generators = jnp.matmul(
            jacobians, dynamics_f + jnp.matmul(dynamics_g, u)
        ) + 0.5 * jnp.trace(
            jnp.matmul(jnp.matmul(dynamics_s.T, hessians), dynamics_s), axis1=1, axis2=2
        )
        xnext = integrator(x, (dynamics_f + jnp.matmul(dynamics_g, u)), dt=DT)
        INTEGRATOR_STATES = update_integrator_states_discrete(
            t,
            xnext,
            u,
            dynamics_func,
            generators,
            barrier_functions,
            barrier_jacobians,
            barrier_hessians,
            lyapunov_functions,
            lyapunov_jacobians,
            lyapunov_hessians,
        )
        # INTEGRATOR_STATES += generators2  # - generators1
        RA_GLOBALS.integrator_states = integrator(
            RA_GLOBALS.integrator_states, (generators1 + generators2) / 2, dt=0.01
        )

        # dx = dynamics_f + jnp.matmul(dynamics_g, u)
        # if L > 0:
        #     INTEGRATOR_STATES = INTEGRATOR_STATES.at[:L].set(
        #         jnp.stack([bf(t, integrator(x, dx, dt=DT)) - bf(t, x) for bf in barrier_functions])
        #     )
        # if K > 0:
        #     INTEGRATOR_STATES = INTEGRATOR_STATES.at[-K:].set(
        #         jnp.stack([lf(t, integrator(x, dx, dt=DT)) - lf(t, x) for lf in lyapunov_functions])
        #     )

        return u, data

    return controller


def adaptive_risk_aware_cbf_clf_controller(
    nominal_input: ControllerCallable,
    dynamics_func: DynamicsCallable,
    barriers: BarrierCollectionCallable = lambda: ([], [], [], []),
    lyapunovs: LyapunovCollectionCallable = lambda: ([], [], [], [], []),
    control_limits: Array = jnp.array([100.0, 100.0]),
    alpha: Array = jnp.array([1.0]),
    R: Union[Array, None] = None,
) -> ControllerCallable:
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
    barrier_functions, barrier_jacobians, barrier_hessians, barrier_partials = barriers()
    (
        lyapunov_functions,
        lyapunov_jacobians,
        lyapunov_hessians,
        lyapunov_partials,
        lyapunov_conditions,
    ) = lyapunovs()

    M = len(control_limits)
    L = len(barrier_functions)
    K = len(lyapunov_functions)

    if R is None:
        u_weights = jnp.array(M * [1])
        a_weights = jnp.array(L * [1000])
        d_weights = jnp.array(K * [1000])
        Rmat = jnp.diag(jnp.hstack([u_weights, a_weights, d_weights], dtype=float))
    else:
        Rmat = R

    RA_GLOBALS.INTEGRATOR_STATES = jnp.zeros((L + K,))

    if len(alpha) != L:
        a_nom = jnp.array(L * [alpha.min()])
    else:
        a_nom = jnp.array(alpha)

    #! TO DO -- Implement Programmatically
    eps = 0.1
    T = 2.0
    rhob = 0.5
    pv = 0.01

    def controller(t: float, x: State) -> ControllerCallableReturns:
        dynamics_f, dynamics_g, dynamics_s = dynamics_func(x)
        N = len(dynamics_f)

        if t == 0:
            RA_GLOBALS.RB = stochastic_barrier_transform(
                jnp.array([bf(t, x) for bf in barrier_functions])
            )
            RA_GLOBALS.RV = jnp.array([lf(t, x) for lf in lyapunov_functions])
            RA_GLOBALS.ETAB = jnp.linalg.norm(
                jnp.array([1 / jnp.sqrt(2), 1 / jnp.sqrt(2), 0.0]) @ dynamics_s
            )
            RA_GLOBALS.ETAV = jnp.linalg.norm(jnp.array([10.0, 10.0, 0.0]) @ dynamics_s)

        u_nom, _ = nominal_input(t, x)
        H = arr(Rmat, dtype=float)
        f = arr(-2 * Rmat @ jnp.hstack([u_nom, a_nom, jnp.zeros((K,))]), dtype=float)

        # Formulate the input constraints
        alpha_limit = 10
        delta_limit = 1e6
        upper_limits = jnp.hstack(
            [control_limits, jnp.array(L * [alpha_limit]), jnp.array(K * [delta_limit])]
        )
        lower_limits = jnp.hstack([control_limits, jnp.array(L * [0]), jnp.array(K * [0])])
        Au = block_diag_matrix_from_vec(M + L + K)
        bu = interleave_arrays(upper_limits, lower_limits)

        # # Formulate CBF constraint(s)
        # Acbf = jnp.zeros((L, M + L))
        # bcbf = jnp.zeros((L,))
        # trace_term = jnp.zeros((L,))
        # for ib, (bf, bj, bh) in enumerate(zip(barrier_funcs, barrier_jacobians, barrier_hessians)):
        #     bf_val, bj_val, bh_val = bf(x), bj(x), bh(x)
        #     jaco = stochastic_jacobian_transform(bf_val, bj_val)
        #     hess = stochastic_hessian_transform(bf_val, bj_val, bh_val)
        #     trace_term = trace_term.at[ib].set(0.5 * jnp.trace(dynamics_s.T @ hess @ dynamics_s))
        #     h = 1 - INTEGRATOR_STATE[ib] - gamma - (jnp.sqrt(2) * T * eta) * erfinv(1 - rhob)
        #     Acbf = Acbf.at[ib, :M].set(jaco @ dynamics_g)
        #     Acbf = Acbf.at[ib, M + ib].set(-h)
        #     bcbf = bcbf.at[ib].set(-jaco @ dynamics_f - trace_term[ib])

        # Formulate CBF constraint(s)
        Acbf = jnp.zeros((L, M + L + K))
        bcbf = jnp.zeros((L,))
        bf_x = jnp.zeros((L,))
        bj_x = jnp.zeros((L, N))
        bh_x = jnp.zeros((L, N, N))
        if L > 0:
            bf_x = jnp.stack([bf(t, x) for bf in barrier_functions])
            bj_x = jnp.stack(
                [
                    stochastic_jacobian_transform(bf_x[ii], bj(t, x))
                    for ii, bj in enumerate(barrier_jacobians)
                ]
            )
            bh_x = jnp.stack(
                [
                    stochastic_hessian_transform(bf_x[ii], bj(t, x), bh(t, x))
                    for ii, (bj, bh) in enumerate(zip(barrier_jacobians, barrier_hessians))
                ]
            )
            dbf_t = jnp.stack(
                [
                    stochastic_jacobian_transform(bf_x[ii], bt(t, x))
                    for ii, bt in enumerate(barrier_partials)
                ]
            )
            h_vals = (
                1
                - RA_GLOBALS.INTEGRATOR_STATES[:L]
                - RA_GLOBALS.RB
                - (RA_GLOBALS.ETAB * jnp.sqrt(2 * T)) * erfinv(1 - rhob)
            )
            traces = jnp.array(
                [0.5 * jnp.trace(dynamics_s.T @ bh_ii @ dynamics_s) for bh_ii in bh_x]
            )

            Acbf = Acbf.at[:, :M].set(jnp.matmul(bj_x, dynamics_g))
            Acbf = Acbf.at[:, M : M + L].set(-jnp.diag(h_vals))
            bcbf = bcbf.at[:].set(-dbf_t - jnp.matmul(bj_x, dynamics_f) - traces)

        # Formulate CLF constraint(s)
        Aclf = jnp.zeros((K, M + L + K))
        bclf = jnp.zeros((K,))
        lf_x = jnp.zeros((K,))
        lj_x = jnp.zeros((K, N))
        lh_x = jnp.zeros((K, N, N))
        if K > 0:
            lf_x = jnp.stack([lf(t, x) for lf in lyapunov_functions])
            lj_x = jnp.stack([lj(t, x) for lj in lyapunov_jacobians])
            lh_x = jnp.stack([lh(t, x) for lh in lyapunov_hessians])
            dlf_t = jnp.stack([lj(t, x) for lj in lyapunov_partials])
            w_vals = (
                RA_GLOBALS.INTEGRATOR_STATES[-K:]
                + RA_GLOBALS.RV
                + (RA_GLOBALS.ETAV * jnp.sqrt(2 * T)) * erfinv(pv)
            )
            lc_x = jnp.stack([lc(w_vals[ii]) for ii, lc in enumerate(lyapunov_conditions)])
            traces = jnp.array(
                [
                    0.5 * jnp.trace(jnp.matmul(jnp.matmul(dynamics_s.T, lh_ii), dynamics_s))
                    for lh_ii in lh_x
                ]
            )

            idxs = jnp.where(w_vals > eps)  # only impose when W positive
            Aclf = Aclf.at[idxs, :M].set(jnp.matmul(lj_x, dynamics_g))
            Aclf = Aclf.at[idxs, -K:].set(jnp.diag(-jnp.ones((K,))))
            bclf = bclf.at[idxs].set(lc_x - dlf_t - jnp.matmul(lj_x, dynamics_f) - traces)

        # Formulate complete set of inequality constraints
        A = arr(jnp.vstack([Au, Acbf, Aclf]), dtype=float)
        b = arr(jnp.hstack([bu, bcbf, bclf]), dtype=float)

        # Solve the QP
        sol, status = qp_solver(H, f, A, b)

        if not status:
            raise ValueError("INFEASIBLE QP")

        # Saturate the solution if necessary
        u = jnp.squeeze(jnp.array(sol[:M]))
        u = jnp.clip(u, -control_limits, control_limits)

        # logging data
        data = {
            "cbfs": bf_x,
            "clfs": lf_x,
            "sol": jnp.array(sol),
            "u": u,
            "u_nom": u_nom,
        }

        # Update integrator state
        # dx = dynamics_f + jnp.matmul(dynamics_g, u)
        # if L > 0:
        #     INTEGRATOR_STATES = INTEGRATOR_STATES.at[:L].set(
        #         jnp.stack(
        #             [bf(t, integrator(x, dx, dt=DT)) - bf(t, x) for bf in barrier_functions]
        #         )
        #     )
        # if K > 0:
        #     INTEGRATOR_STATES = INTEGRATOR_STATES.at[-K:].set(
        #         jnp.stack(
        #             [lf(t, integrator(x, dx, dt=DT)) - lf(t, x) for lf in lyapunov_functions]
        #         )
        #     )
        jacobians = jnp.vstack([bj_x, lj_x])
        hessians = jnp.vstack([bh_x, lh_x])
        generators = jnp.matmul(
            jacobians, dynamics_f + jnp.matmul(dynamics_g, u)
        ) + 0.5 * jnp.trace(
            jnp.matmul(jnp.matmul(dynamics_s.T, hessians), dynamics_s), axis1=1, axis2=2
        )
        RA_GLOBALS.INTEGRATOR_STATES = integrator(INTEGRATOR_STATES, generators, dt=DT)

        return u, data

    return controller


def update_integrator_states_discrete(
    t: float,
    xnext: Array,
    u: Array,
    dynamics_func: DynamicsCallable,
    generators: Array,
    barrier_functions: List[Callable[[float, Array], Array]],
    barrier_jacobians: List[Callable[[float, Array], Array]],
    barrier_hessians: List[Callable[[float, Array], Array]],
    lyapunov_functions: List[Callable[[float, Array], Array]],
    lyapunov_jacobians: List[Callable[[float, Array], Array]],
    lyapunov_hessians: List[Callable[[float, Array], Array]],
) -> Array:
    """Updates integrator states based on discrete-time (simulation time) integrator implementation."""
    global INTEGRATOR_STATES
    N = xnext.shape[0]
    L = len(barrier_functions)
    K = len(lyapunov_functions)
    dynamics_fnext, dynamics_gnext, dynamics_snext = dynamics_func(xnext)

    bf_xnext = jnp.zeros((L,))
    bj_xnext = jnp.zeros((L, N))
    bh_xnext = jnp.zeros((L, N, N))
    if L > 0:
        bf_xnext = jnp.stack([bf(t, xnext) for bf in barrier_functions])
        bj_xnext = jnp.stack(
            [
                stochastic_jacobian_transform(bf_xnext[ii], bj(t, xnext))
                for ii, bj in enumerate(barrier_jacobians)
            ]
        )
        bh_xnext = jnp.stack(
            [
                stochastic_hessian_transform(bf_xnext[ii], bj(t, xnext), bh(t, xnext))
                for ii, (bj, bh) in enumerate(zip(barrier_jacobians, barrier_hessians))
            ]
        )

    lj_xnext = jnp.zeros((K, N))
    lh_xnext = jnp.zeros((K, N, N))
    if K > 0:
        lj_xnext = jnp.stack([lj(t, xnext) for lj in lyapunov_jacobians])
        lh_xnext = jnp.stack([lh(t, xnext) for lh in lyapunov_hessians])

    jacobians_next = jnp.vstack([bj_xnext, lj_xnext])
    hessians_next = jnp.vstack([bh_xnext, lh_xnext])
    generators_next = jnp.matmul(
        jacobians_next, dynamics_fnext + jnp.matmul(dynamics_gnext, u)
    ) + 0.5 * jnp.trace(
        jnp.matmul(jnp.matmul(dynamics_snext.T, hessians_next), dynamics_snext), axis1=1, axis2=2
    )

    w1 = 1.0
    # return integrator(INTEGRATOR_STATES, generators, dt=DT)
    return integrator(INTEGRATOR_STATES, w1 * generators + (1 - w1) * generators_next, dt=DT)

    # dx = dynamics_f + jnp.matmul(dynamics_g, u)
    # if L > 0:
    #     INTEGRATOR_STATES = INTEGRATOR_STATES.at[:L].set(
    #         jnp.stack(
    #             [bf(t, integrator(x, dx, dt=DT)) - bf(t, x) for bf in barrier_functions]
    #         )
    #     )
    # if K > 0:
    #     INTEGRATOR_STATES = INTEGRATOR_STATES.at[-K:].set(
    #         jnp.stack(
    #             [lf(t, integrator(x, dx, dt=DT)) - lf(t, x) for lf in lyapunov_functions]
    #         )
    #     )
