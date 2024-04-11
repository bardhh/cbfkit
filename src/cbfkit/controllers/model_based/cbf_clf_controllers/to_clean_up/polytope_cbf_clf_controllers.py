import jax.numpy as jnp
from jax import Array, value_and_grad
from typing import Union, Dict, Any
from numpy import array as arr


from cbfkit.utils.user_types import (
    BarrierCollectionCallable,
    ControllerCallable,
    ControllerCallableReturns,
    DynamicsCallable,
    LyapunovCollectionCallable,
    State,
)
from cbfkit.optimization.solvers import qp_solver
from .utils import (
    block_diag_matrix_from_vec,
    interleave_arrays,
    compute_enclosed_area,
    generate_compute_cbf_clf_constraints,
    generate_compute_adaptive_cbf_clf_constraints_fcn,
    approximate_inscribing_ellipse_volume,
    polytope_volume_ellipse_layer,
)


def polytope_cbf_clf_controller(
    nominal_input: ControllerCallable,
    dynamics_func: DynamicsCallable,
    barriers: BarrierCollectionCallable = lambda: ([], [], [], []),
    lyapunovs: LyapunovCollectionCallable = lambda: ([], [], [], [], []),
    control_limits: Array = jnp.array([100.0, 100.0]),
    alpha: Array = jnp.array([1.0]),
    adaptive: bool = False,
    R: Union[Array, None] = None,
) -> ControllerCallable:
    """
    Compute the solution to a control barrier function/control Lyapunov function based quadratic program seeking to minimize the deviation
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
    L = len(barriers()[0])
    K = len(lyapunovs()[0])
    if len(alpha) != L:
        alpha = jnp.array(L * alpha.min())

    if adaptive:
        n_decision_vars = M + L + K
        compute_cbf_clf_constraints = generate_compute_adaptive_cbf_clf_constraints_fcn(
            dynamics_func, barriers, lyapunovs, control_limits, alpha
        )
    else:
        n_decision_vars = M
        compute_cbf_clf_constraints = generate_compute_cbf_clf_constraints_fcn(
            dynamics_func, barriers, lyapunovs, control_limits, alpha
        )

    n_constraints = 2 * n_decision_vars + L + K
    ellipse_cvxpylayer = polytope_volume_ellipse_layer(n_decision_vars, n_constraints)

    def compute_ellipse_volume(t: float, x: State):
        A, b = compute_cbf_clf_constraints(t, x)
        sol = ellipse_cvxpylayer(A, b.reshape(-1, 1))
        B = sol[0]
        return approximate_inscribing_ellipse_volume(B)

    ellipse_val_and_grad = value_and_grad(compute_ellipse_volume, argnums=(0, 1))

    if R is None:
        Rmat = jnp.eye(n_decision_vars, dtype=float)
        if adaptive:
            Rmat = Rmat.at[M:, M:].set(100 * jnp.ones((L + K)))
    else:
        Rmat = R

    def controller(t: float, x: State) -> ControllerCallableReturns:
        dynamics_f, dynamics_g, _ = dynamics_func(x)
        u_nom, _ = nominal_input(t, x)
        if adaptive:
            u_nom = jnp.hstack([u_nom, alpha, jnp.zeros((K,))])
        H = arr(Rmat, dtype=float)
        f = arr(-2 * Rmat @ u_nom, dtype=float)
        A0, b0 = compute_cbf_clf_constraints(t, x)

        # Compute value and gradient of incribing ellipse of feasible region
        try:
            hvol, hvol_grad = ellipse_val_and_grad(t, x)
        except:
            hvol, hvol_grad = -1000, [0, jnp.zeros((dynamics_g.shape[0]))]
        dhvoldt, dhvoldx = hvol_grad[0], hvol_grad[1]
        Avol = jnp.zeros((1, n_decision_vars))
        Avol = Avol.at[0, :M].set(-jnp.matmul(dhvoldx, dynamics_g))
        bvol = dhvoldt + jnp.matmul(dhvoldx, dynamics_f) + 0.1 * alpha.min() * hvol
        A = arr(jnp.vstack([A0, Avol]), dtype=float)
        b = arr(jnp.hstack([b0, bvol]), dtype=float)

        # feasible_area = compute_enclosed_area(A, b, control_limits)

        # Solve the QP
        sol, status = qp_solver(H, f, A, b)

        if not status:
            # Backup Control: use Lyapunov control to brake and steer to zero
            u = jnp.array([(-2 * x[2] * x[3] - x[3] ** 2 - 1 / 2 * x[2] ** 2) / x[2], -5 * x[6]])
            print("infeasible")
        else:
            u = jnp.squeeze(jnp.array(sol[:M]))

        # Saturate the solution if necessary
        u = jnp.squeeze(jnp.array(sol[:M]))
        u = jnp.clip(u, -control_limits, control_limits)

        # logging data
        data = {
            # "cbfs": bf_x,
            # "clfs": lf_x,
            # "feasible_area": feasible_area,
            "sol": jnp.array(sol),
            "u": u,
            "u_nom": u_nom,
        }

        return u, data

    return controller


def adaptive_cbf_clf_controller(
    nominal_input: ControllerCallable,
    dynamics_func: DynamicsCallable,
    barriers: BarrierCollectionCallable,
    lyapunovs: LyapunovCollectionCallable,
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
    barrier_functions, barrier_jacobians, _, barrier_partials = barriers()
    lyapunov_functions, lyapunov_jacobians, _, lyapunov_partials, lyapunov_conditions = lyapunovs()

    M = len(control_limits)
    L = len(barrier_functions)
    K = len(lyapunov_functions)

    if R is None:
        u_weights = jnp.array(M * [1])
        a_weights = jnp.array(L * [1e3])
        d_weights = jnp.array(K * [1e2])
        Rmat = jnp.diag(jnp.hstack([u_weights, a_weights, d_weights]))
    else:
        Rmat = R

    if len(alpha) != L:
        a_nom = jnp.array(L * alpha.min())
    else:
        a_nom = jnp.array(alpha)

    def controller(t: float, x: State) -> ControllerCallableReturns:
        dynamics_f, dynamics_g, _ = dynamics_func(x)

        u_nom, _ = nominal_input(t, x)
        H = arr(Rmat, dtype=float)
        f = arr(-2 * Rmat @ jnp.hstack([u_nom, a_nom, jnp.zeros((K,))]), dtype=float)

        # Formulate the input constraints
        alpha_limit = 100
        delta_limit = 1000
        upper_limits = jnp.hstack(
            [control_limits, jnp.array(L * [alpha_limit]), jnp.array(K * [delta_limit])]
        )
        lower_limits = jnp.hstack([control_limits, jnp.array(L * [0]), jnp.array(K * [0])])
        Au = block_diag_matrix_from_vec(M + L + K)
        bu = interleave_arrays(upper_limits, lower_limits)

        # Formulate CBF constraint(s)
        Acbf = jnp.zeros((L, M + L + K))
        bcbf = jnp.zeros((L,))
        bf_x = jnp.zeros((L,))
        if L > 0:
            bf_x = jnp.stack([bf(t, x) for bf in barrier_functions])
            bj_x = jnp.stack([bj(t, x) for bj in barrier_jacobians])
            dbf_t = jnp.stack([bt(t, x) for bt in barrier_partials])
            Acbf = Acbf.at[:, :M].set(-jnp.matmul(bj_x, dynamics_g))
            Acbf = Acbf.at[:, M : M + L].set(-jnp.diag(bf_x))
            bcbf = bcbf.at[:].set(dbf_t + jnp.matmul(bj_x, dynamics_f))

            # # Suppress angular control
            # Acbf = Acbf.at[:, 1].set(0)

        # Formulate CLF constraint(s)
        Aclf = jnp.zeros((K, M + L + K))
        bclf = jnp.zeros((K,))
        lf_x = jnp.zeros((K,))
        if K > 0:
            lf_x = jnp.stack([lf(t, x) for lf in lyapunov_functions])
            lj_x = jnp.stack([lj(t, x) for lj in lyapunov_jacobians])
            dlf_t = jnp.stack([lt(t, x) for lt in lyapunov_partials])
            lc_x = jnp.stack(
                [lc(lf(t, x)) for lc, lf in zip(lyapunov_conditions, lyapunov_functions)]
            )

            Aclf = Aclf.at[:, :M].set(jnp.matmul(lj_x, dynamics_g))
            Aclf = Aclf.at[:, M + L :].set(-jnp.ones((K,)))
            bclf = bclf.at[:].set(-dlf_t - jnp.matmul(lj_x, dynamics_f) + lc_x)

        # Formulate complete set of inequality constraints
        A = arr(jnp.vstack([Au, Acbf, Aclf]), dtype=float)
        b = arr(jnp.hstack([bu, bcbf, bclf]), dtype=float)

        # Solve the QP
        sol, status = qp_solver(H, f, A, b)
        u = jnp.squeeze(jnp.array(sol[:M]))

        # Saturate the solution if necessary
        u = jnp.clip(u, -control_limits, control_limits)

        # logging data
        data = {
            "cbfs": bf_x,
            "clfs": lf_x,
            "sol": jnp.array(sol),
            "u": u,
            "u_nom": u_nom,
        }

        return u, data

    return controller
