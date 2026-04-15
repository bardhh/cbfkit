"""
cbf_clf_qp_generator.py
================

Generates the function for generating CBF-CLF-QP control laws of various forms.

Functions
---------
-cbf_clf_qp_generator: produces the generating function based on

Notes
-----
Used in the generation of vanilla, robust, stochastic, risk-aware, and risk-aware path integral
CBF-CLF-QP control laws.

Examples
--------
>>> from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
>>> from cbfkit.controllers.cbf_clf.generate_constraints import (
>>>     generate_compute_zeroing_cbf_constraints,
>>>     generate_compute_vanilla_clf_constraints,
>>> )
>>> vanilla_cbf_clf_qp_controller = cbf_clf_qp_generator(
>>>     generate_compute_zeroing_cbf_constraints,
>>>     generate_compute_vanilla_clf_constraints,
>>> )
"""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import jax.numpy as jnp
import jax.debug as jdebug
from jax import Array, jit, lax, tree_util

from cbfkit.certificates import concatenate_certificates
from cbfkit.optimization.quadratic_program.solver_registry import (
    QpSolution,
    get_solver,
)
from cbfkit.utils.user_types import (
    EMPTY_CERTIFICATE_COLLECTION,
    CbfClfQpConfig,
    CbfClfQpData,
    CbfClfQpGenerator,
    CertificateCollection,
    CertificateInput,
    Control,
    ControllerCallable,
    ControllerCallableReturns,
    ControllerData,
    DynamicsCallable,
    GenerateComputeCertificateConstraintCallable,
    Key,
    QpSolverCallable,
    State,
)

from .generate_constraints import (
    generate_compute_cbf_clf_constraints,
    generate_compute_input_constraints,
)


def _normalize_certificate_collection(
    cert_collection: Optional[CertificateInput], name: str
) -> CertificateCollection:
    """Validates and normalizes certificate collection structure.

    Accepts:
    - None (returns empty collection)
    - CertificateCollection (returns as is)
    - List[CertificateCollection] (concatenates and returns)
    - Tuple/List of length 5 (wraps in CertificateCollection)
    """
    if cert_collection is None:
        return EMPTY_CERTIFICATE_COLLECTION

    collection = None

    # Case 1: Already a CertificateCollection
    if isinstance(cert_collection, CertificateCollection):
        collection = cert_collection

    # Case 2: List/Tuple of CertificateCollections (user passed [c1, c2] or [])
    elif isinstance(cert_collection, (list, tuple)):
        if len(cert_collection) == 0:
            return EMPTY_CERTIFICATE_COLLECTION
        if isinstance(cert_collection[0], CertificateCollection):
            collection = concatenate_certificates(*cert_collection)

    if collection is None:
        # Case 3: Raw tuple of length 5 (legacy structure: (funcs, jacs, hess, parts, conds))
        # Check if it's iterable
        try:
            iter(cert_collection)
        except TypeError:
            raise TypeError(
                f"'{name}' must be a CertificateCollection (tuple/list of length 5) or a list of them, but got {type(cert_collection)}."
            )

        # Check length
        if len(cert_collection) != 5:
            raise ValueError(
                f"Invalid structure for '{name}'. Expected a CertificateCollection with 5 elements "
                "(functions, jacobians, hessians, partials, conditions), "
                f"but got a collection of length {len(cert_collection)}. "
                "Did you pass a list of barrier functions directly? You must provide the derivatives and conditions as well, "
                "or use a helper like 'cbfkit.certificates.CertificateCollection'."
            )

        collection = CertificateCollection(*cert_collection)

    # Validate component consistency
    # Ensure all component lists have the same length to prevent obscure JAX errors downstream.
    lengths = [len(x) for x in collection]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"Inconsistent component lengths in '{name}'. "
            f"All components (functions, jacobians, hessians, partials, conditions) must have the same length. "
            f"Got lengths: functions={lengths[0]}, jacobians={lengths[1]}, hessians={lengths[2]}, "
            f"partials={lengths[3]}, conditions={lengths[4]}."
        )

    return collection


def cbf_clf_qp_generator(
    generate_compute_cbf_constraints: GenerateComputeCertificateConstraintCallable,
    generate_compute_clf_constraints: GenerateComputeCertificateConstraintCallable,
) -> CbfClfQpGenerator:
    """Function for producing a generating function for CBF-CLF-QP control laws of various forms.

    Args:
        generate_compute_cbf_constraints (GenerateComputeCertificateConstraintCallable)
        generate_compute_clf_constraints (GenerateComputeCertificateConstraintCallable)

    Returns
    -------
        (CbfClfQpGenerator): function for generating CBF-CLF-QP control law
    """

    def generate_cbf_clf_controller(
        control_limits: Array,
        dynamics_func: DynamicsCallable,
        barriers: Optional[CertificateInput] = EMPTY_CERTIFICATE_COLLECTION,
        lyapunovs: Optional[CertificateInput] = EMPTY_CERTIFICATE_COLLECTION,
        p_mat: Optional[Union[Array, None]] = None,
        *,
        solver: Optional[QpSolverCallable] = None,
        relaxable_clf: bool = True,
        relaxable_cbf: bool = False,
        tunable_class_k: bool = False,
        slack_bound_cbf: Optional[float] = None,
        slack_bound_clf: float = 1e9,
        slack_penalty_cbf: float = 2e3,
        slack_penalty_clf: float = 2e3,
        **kwargs: Any,
    ) -> ControllerCallable:
        """Produces the function to deploy a CBF-CLF-QP control law.

        Args:
            control_limits (Array): symmetric actuation constraints [u1_bar, u2_bar, etc.].
                Can be a scalar for 1D systems.
            dynamics_func (DynamicsCallable): function to compute dynamics based on current state
            barriers (CertificateInput): collection of barrier functions,
                gradients, hessians, dh/dt, conditions. Can be a single collection, a list of them, or a legacy tuple.
            lyapunovs (CertificateInput): collection of lyapunov functions,
                gradients, hessians, dV/dt,  conditions. Can be a single collection, a list of them, or a legacy tuple.
            p_mat (Optional[Union[Array, None]] = None): objective function matrix (quadratic term)
            solver (Optional[QpSolverCallable]): QP solver callable from
                ``get_solver()``.  Defaults to ``get_solver("jaxopt")`` when
                ``None``.  Must be JIT-compatible (currently only jaxopt).
            relaxable_clf (bool): whether to treat CLF as a soft constraint (default: True).
            relaxable_cbf (bool): whether to treat CBF as a soft constraint (default: False).
            tunable_class_k (bool): whether to tune the Class K function parameter (default: False).
            slack_bound_cbf (float): Maximum slack for CBF constraints (default: 100.0 or 1e4).
            slack_bound_clf (float): Maximum slack for CLF constraints (default: 1e9).
            slack_penalty_cbf (float): Penalty weight for CBF slack variables (default: 2e3).
            slack_penalty_clf (float): Penalty weight for CLF slack variables (default: 2e3).
            **kwargs (CbfClfQpConfig): keyword arguments.

        Returns
        -------
            ControllerCallable: function for computing control input based on CBF-CLF-QP
        """
        # Resolve solver — default to jaxopt
        solve_qp: QpSolverCallable
        if solver is not None:
            if not getattr(solver, "jit_compatible", False):
                solver_name = getattr(solver, "solver_name", "unknown")
                raise ValueError(
                    f"Solver {solver_name!r} is not JIT-compatible. "
                    f"The CBF-CLF-QP controller is JIT-compiled and requires a "
                    f"JIT-compatible solver. Use get_solver('jaxopt', ...) instead."
                )
            solve_qp = solver
        else:
            solve_qp = get_solver("jaxopt")

        # Update kwargs to ensure downstream functions see the configuration
        kwargs.update(
            {
                "relaxable_clf": relaxable_clf,
                "relaxable_cbf": relaxable_cbf,
                "tunable_class_k": tunable_class_k,
                "slack_bound_cbf": slack_bound_cbf,
                "slack_bound_clf": slack_bound_clf,
                "slack_penalty_cbf": slack_penalty_cbf,
                "slack_penalty_clf": slack_penalty_clf,
            }
        )

        # Atlas: Support scalar inputs for 1D systems
        control_limits = jnp.atleast_1d(jnp.asarray(control_limits))

        # Validate configuration to prevent silent failures (e.g., NaNs from negative penalties)
        if slack_penalty_cbf < 0:
            raise ValueError(
                f"Invalid configuration: 'slack_penalty_cbf' must be non-negative, but got {slack_penalty_cbf}."
            )
        if slack_penalty_clf < 0:
            raise ValueError(
                f"Invalid configuration: 'slack_penalty_clf' must be non-negative, but got {slack_penalty_clf}."
            )
        if slack_bound_cbf is not None and slack_bound_cbf <= 0:
            raise ValueError(
                f"Invalid configuration: 'slack_bound_cbf' must be positive, but got {slack_bound_cbf}."
            )
        if slack_bound_clf <= 0:
            raise ValueError(
                f"Invalid configuration: 'slack_bound_clf' must be positive, but got {slack_bound_clf}."
            )
        if jnp.any(jnp.asarray(control_limits) < 0):
            raise ValueError(
                f"Invalid configuration: 'control_limits' elements must be non-negative (defining symmetric bounds |u|<=limit), "
                f"but got {control_limits}."
            )

        complete = False
        n_con = len(control_limits)

        def verified_dynamics_func(x: State) -> Tuple[Array, Array]:
            f, g = dynamics_func(x)
            if f.ndim != 1:
                raise ValueError(
                    f"Dynamics drift term 'f' must be a 1D array of shape (n_states,), "
                    f"but got shape {f.shape}. Ensure your dynamics function returns a flat array, "
                    f"not a column vector."
                )
            if g.ndim != 2:
                raise ValueError(
                    f"Dynamics control term 'g' must be a 2D array of shape (n_states, n_controls), "
                    f"but got shape {g.shape}. If you have 1 control input, ensure 'g' is (n, 1)."
                )
            if f.shape[0] != g.shape[0]:
                raise ValueError(
                    f"State dimension mismatch: drift 'f' has {f.shape[0]} states, "
                    f"but control matrix 'g' has {g.shape[0]} states."
                )
            if g.shape[1] != n_con:
                raise ValueError(
                    f"Control dimension mismatch: 'control_limits' implies {n_con} inputs, "
                    f"but 'g' has {g.shape[1]} columns."
                )
            return f, g

        # Normalize barriers and lyapunovs
        barriers = _normalize_certificate_collection(barriers, "barriers")
        lyapunovs = _normalize_certificate_collection(lyapunovs, "lyapunovs")

        # Get number of functions for semantic mapping in error reporting
        n_cbf_total = len(barriers.functions)
        n_clf_total = len(lyapunovs.functions)

        if tunable_class_k:
            b_funcs, _, _, _, _ = barriers
            n_bfs = len(b_funcs)
            slack_cbf_val = slack_bound_cbf if slack_bound_cbf is not None else 100.0
            control_limits = jnp.hstack([control_limits, slack_cbf_val * jnp.ones((n_bfs,))])
        elif relaxable_cbf:
            b_funcs, _, _, _, _ = barriers
            n_bfs = len(b_funcs)
            slack_cbf_val = slack_bound_cbf if slack_bound_cbf is not None else 1e4
            control_limits = jnp.hstack([control_limits, slack_cbf_val * jnp.ones((n_bfs,))])
        else:
            n_bfs = 0

        # Default to relaxable CLF for robustness unless explicitly disabled
        if relaxable_clf:
            l_funcs, _, _, _, _ = lyapunovs
            n_lfs = len(l_funcs)
            control_limits = jnp.hstack([control_limits, slack_bound_clf * jnp.ones((n_lfs,))])
        else:
            n_lfs = 0

        # Use direct penalty weights in P matrix (no variable scaling).
        # Variable scaling (delta_raw = delta_phys / sqrt(penalty)) causes OSQP
        # ill-conditioning: tiny constraint coefficients + huge bounds = large
        # dynamic range that OSQP cannot handle reliably.
        # Direct approach: P = diag([1, penalty_cbf, penalty_clf]), no transformation.
        scale_cbf = 1.0
        scale_clf = 1.0
        auto_p_mat = p_mat is None

        # Precompute scaling vector for solution recovery
        sol_scaling_vector = jnp.ones((n_con + n_bfs + n_lfs,))

        if auto_p_mat:
            penalty_cbf = slack_penalty_cbf
            penalty_clf = slack_penalty_clf

            # Clamp penalty to avoid catastrophic P matrix condition numbers.
            # OSQP with float64 can handle weights up to ~1e8 reliably.
            MAX_WEIGHT = 1.0e8
            weight_cbf = float(min(penalty_cbf, MAX_WEIGHT))
            weight_clf = float(min(penalty_clf, MAX_WEIGHT))

            p_mat = jnp.diag(
                jnp.hstack(
                    [
                        jnp.ones((n_con,)),
                        weight_cbf * jnp.ones((n_bfs,)),
                        weight_clf * jnp.ones((n_lfs,)),
                    ]
                )
            )
            # No variable transformation: scale stays 1.0, control_limits unchanged.

        # Pass scales to constraint generators (scale=1.0 means no transformation)
        if auto_p_mat:
            kwargs["scale_cbf"] = scale_cbf
            kwargs["scale_clf"] = scale_clf

        # Pre-compute solution scaling vector to avoid repetitive slice updates in JIT loop
        sol_scaling_vector = jnp.ones((n_con + n_bfs + n_lfs,))
        if auto_p_mat:
            if n_bfs > 0:
                sol_scaling_vector = sol_scaling_vector.at[n_con : n_con + n_bfs].set(scale_cbf)
            if n_lfs > 0:
                sol_scaling_vector = sol_scaling_vector.at[
                    n_con + n_bfs : n_con + n_bfs + n_lfs
                ].set(scale_clf)

        # Ensure p_mat is defined for the controller closure
        assert p_mat is not None

        compute_input_constraints = generate_compute_input_constraints(control_limits)

        # Pre-compute static input constraints (box limits + tunable non-negativity)
        # to avoid repeated creation and stacking inside the JIT loop.
        dummy_t = 0.0
        dummy_x = jnp.zeros((1,))  # Shape irrelevant for static constraints
        g_mat_static, h_vec_static = compute_input_constraints(dummy_t, dummy_x)

        if "tunable_class_k" in kwargs and kwargs["tunable_class_k"] and n_bfs > 0:
            # Constraints: -delta <= 0  (delta >= 0)
            # tunable parameters are located at indices [n_con : n_con + n_bfs]
            n_total_vars = n_con + n_bfs + n_lfs
            g_pos = jnp.zeros((n_bfs, n_total_vars))

            # Set -1.0 for the tunable columns using vector indexing
            row_indices = jnp.arange(n_bfs)
            col_indices = n_con + jnp.arange(n_bfs)
            g_pos = g_pos.at[row_indices, col_indices].set(-1.0)
            h_pos = jnp.zeros((n_bfs,))

            g_mat_static = jnp.vstack([g_mat_static, g_pos])
            h_vec_static = jnp.hstack([h_vec_static, h_pos])

        compute_cbf_clf_constraints = generate_compute_cbf_clf_constraints(
            generate_compute_cbf_constraints,
            generate_compute_clf_constraints,
            control_limits,
            verified_dynamics_func,
            barriers,
            lyapunovs,
            **kwargs,
        )

        def _log_nan_indices(q_mask, g_mask, h_mask):
            import numpy as np

            q_idx = np.where(q_mask)[0]
            g_rows = np.where(np.any(g_mask, axis=1))[0]
            h_idx = np.where(h_mask)[0]

            if len(q_idx) > 0:
                print(f"      -> NaN/Inf in q_vec at indices: {q_idx}")
            if len(g_rows) > 0:
                print(f"      -> NaN/Inf in g_mat at rows: {g_rows}")
            if len(h_idx) > 0:
                print(f"      -> NaN/Inf in h_vec at indices: {h_idx}")

            # Semantic mapping
            offset = 0

            # Input limits
            n_inputs_constraints = 2 * n_con
            bad_inputs = h_idx[(h_idx >= offset) & (h_idx < offset + n_inputs_constraints)]
            if len(bad_inputs) > 0:
                print(f"      - Input Limits: {bad_inputs - offset}")
            offset += n_inputs_constraints

            # Tunable constraints
            is_tunable = kwargs.get("tunable_class_k", False)
            if is_tunable and n_bfs > 0:
                bad_tunable = h_idx[(h_idx >= offset) & (h_idx < offset + n_bfs)]
                if len(bad_tunable) > 0:
                    print(f"      - Tunable Class K: {bad_tunable - offset}")
                offset += n_bfs

            # CBF constraints
            if n_cbf_total > 0:
                bad_cbf = h_idx[(h_idx >= offset) & (h_idx < offset + n_cbf_total)]
                if len(bad_cbf) > 0:
                    print(f"      - CBF Constraints: {bad_cbf - offset}")
                offset += n_cbf_total

            # CLF constraints
            if n_clf_total > 0:
                bad_clf = h_idx[(h_idx >= offset) & (h_idx < offset + n_clf_total)]
                if len(bad_clf) > 0:
                    print(f"      - CLF Constraints: {bad_clf - offset}")

        @jit
        def controller(
            t: float, x: State, u_nom: Control, key: Key, data: ControllerData
        ) -> ControllerCallableReturns:
            """JIT-compatible portion of the CBF-CLF-QP control law.

            Args:
                t (float): time (in sec)
                x (State): state vector
                u_nom (Array): nominal control input
                key: for random number generation if needed
                data: additional data for controller to use
            Returns:
                ControllerCallableReturns: tuple consisting of control solution (Array) and auxiliary data (Dict)
            """
            nonlocal complete

            # Ensure u_nom is 1D to prevent broadcasting errors (e.g., (N,) + (N,1) -> (N,N))
            # and to handle scalar inputs for 1D systems.
            u_nom = u_nom.ravel()

            if u_nom.shape[0] != n_con:
                raise ValueError(
                    f"Nominal control input 'u_nom' has shape {u_nom.shape} (after ravel), "
                    f"but expected ({n_con},) based on 'control_limits'."
                )

            if n_bfs > 0:
                if "tunable_class_k" in kwargs and kwargs["tunable_class_k"]:
                    u_nom = jnp.hstack([u_nom, jnp.ones((n_bfs,))])
                else:
                    u_nom = jnp.hstack([u_nom, jnp.zeros((n_bfs,))])
            if n_lfs > 0:
                u_nom = jnp.hstack([u_nom, jnp.zeros((n_lfs,))])

            # Compute QP cost, constraint functions
            # Keep vectors 1D to avoid JAX broadcasting overhead in solver (prevents (N,1) vs (N,) mismatch)
            q_vec = jnp.matmul(-2 * p_mat, u_nom)
            g_mat_c, h_vec_c, sub_data = compute_cbf_clf_constraints(t, x)
            if "complete" in sub_data:
                complete = sub_data["complete"]

            # Stack input and certificate function constraints
            # Scaling of slack columns is now handled within compute_cbf_clf_constraints (via kwargs)
            # This avoids large array copies and multiply operations inside the JIT loop.

            # Normalize CBF/CLF constraint rows to improve numerical stability
            # Avoid normalizing noise vectors. If norm < tol, do NOT scale up.
            # Input constraints (box limits) are already normalized (row norm = 1).
            impossible_constraints = jnp.array(False)
            autodiff_safety_override = jnp.array(False)
            if auto_p_mat:
                # Compute norms with overflow/underflow-safe scaling.
                row_max = jnp.max(jnp.abs(g_mat_c), axis=1)
                # Keep autodiff guard only for truly degenerate rows while still
                # allowing very small-but-real constraints to be normalized.
                zero_rows = row_max < 1e-30
                row_max_safe = jnp.where(zero_rows, 1.0, row_max)
                g_mat_c_scaled = g_mat_c / row_max_safe[:, None]
                row_sumsq_scaled = jnp.sum(g_mat_c_scaled * g_mat_c_scaled, axis=1)
                row_norms_scaled = jnp.sqrt(row_sumsq_scaled)
                row_norms_c = row_norms_scaled * row_max_safe

                # Normalize with clamped reciprocal for numeric stability.
                inv_row_norms = 1.0 / jnp.where(zero_rows, 1.0, row_norms_c)
                safe_scales_c = jnp.where(zero_rows, 1.0, jnp.minimum(inv_row_norms, 1e30))

                g_mat_c = g_mat_c * safe_scales_c[:, None]
                h_vec_c = h_vec_c * safe_scales_c

                # Degenerate rows (all-zero coefficients) can produce NaN reverse-mode
                # sensitivities around norm(0) and singular KKT structure.
                infeasible_zero_rows = lax.stop_gradient(zero_rows & (h_vec_c < -1e-12))
                impossible_constraints = jnp.any(infeasible_zero_rows)
                feasible_zero_rows = lax.stop_gradient(zero_rows & (~infeasible_zero_rows))
                autodiff_safety_override = lax.stop_gradient(jnp.any(feasible_zero_rows))
                h_vec_c = jnp.where(feasible_zero_rows, 1.0, h_vec_c)

                # Add an inactive regularization row for feasible degenerate constraints
                # so reverse-mode differentiation avoids singular sensitivities.
                reg_rows = jnp.zeros_like(g_mat_c)
                reg_rows = reg_rows.at[:, 0].set(1e-6)
                g_mat_c = jnp.where(feasible_zero_rows[:, None], reg_rows, g_mat_c)

            # Use pre-computed static constraints (input limits + tunable non-negativity)
            g_mat = jnp.vstack([g_mat_static, g_mat_c])
            h_vec = jnp.hstack([h_vec_static, h_vec_c])

            # Detect NaNs in QP inputs
            mask_q = jnp.isnan(q_vec) | jnp.isinf(q_vec)
            mask_g = jnp.isnan(g_mat) | jnp.isinf(g_mat)
            mask_h = jnp.isnan(h_vec) | jnp.isinf(h_vec)

            nan_q = jnp.any(mask_q)
            nan_g = jnp.any(mask_g)
            nan_h = jnp.any(mask_h)
            nan_in_inputs = nan_q | nan_g | nan_h

            # Solve QP
            solver_params = None

            # Cast sub_data to typed version for safe access
            controller_sub_data = cast(
                CbfClfQpData, data.sub_data if data.sub_data is not None else {}
            )

            if "solver_params" in controller_sub_data:
                solver_params = controller_sub_data["solver_params"]

            safe_nominal_u = jnp.clip(
                u_nom[:n_con], -control_limits[:n_con], control_limits[:n_con]
            )

            def _solve_with(p_local: Array, q_local: Array, g_local: Array, h_local: Array):
                sol_local, status_local, new_params_local = solve_qp(
                    p_local,
                    q_local,
                    g_local,
                    h_local,
                    init_params=solver_params,
                )
                # Scout: Extract solver iterations for diagnostics
                # new_params is (KKTSolution, OSQPState)
                _, state_local = new_params_local
                iter_local = state_local.iter_num

                # Explicitly catch NaN solutions even if solver claims success.
                # Also catch if inputs were NaN (solver might return 0 and UNSOLVED status 0).
                status_local = jnp.where(
                    jnp.any(jnp.isnan(sol_local)) | jnp.any(jnp.isinf(sol_local)),
                    -1,
                    status_local,
                )
                status_local = jnp.where(nan_in_inputs, -2, status_local)
                status_local = jnp.where(impossible_constraints, 3, status_local)

                # Rescale solution back to physical units.
                if auto_p_mat:
                    sol_local = sol_local * sol_scaling_vector

                return sol_local, status_local, new_params_local, iter_local

            # Keep solver values in the primal computation while detaching them from reverse-mode.
            # This avoids NaN gradients around singular KKT points without changing control behavior.
            sol, status, new_params, iter_num = _solve_with(
                lax.stop_gradient(p_mat),
                lax.stop_gradient(q_vec),
                lax.stop_gradient(g_mat),
                lax.stop_gradient(h_vec),
            )
            sol = lax.stop_gradient(sol)
            status = lax.stop_gradient(status)
            new_params = tree_util.tree_map(lax.stop_gradient, new_params)
            iter_num = lax.stop_gradient(iter_num)

            success = status == 1
            solved_u = lax.cond(
                success,
                lambda _fake: sol[:n_con],
                lambda _fake: jnp.full_like(u_nom[:n_con], jnp.nan),
                0,
            )
            u = lax.cond(
                autodiff_safety_override,
                lambda _fake: lax.stop_gradient(safe_nominal_u),
                lambda _fake: solved_u,
                0,
            )

            def _print_failure(status, iter_num, sub_data):
                # Map status codes to human-readable strings
                def print_status_msg(msg):
                    jdebug.print(
                        "⚠️ CBF-CLF-QP Failed! Status: {status} ({msg}) (Iter: {iter}). Output set to NaN.\n"
                        "   Config: relax_cbf={relax_cbf}, relax_clf={relax_clf}",
                        status=status,
                        msg=msg,
                        iter=iter_num,
                        relax_cbf=relaxable_cbf,
                        relax_clf=relaxable_clf,
                    )

                lax.switch(
                    status + 2,  # Map -2 to index 0
                    [
                        lambda: (
                            jdebug.print(
                                "⚠️ CBF-CLF-QP Failed! Status: -2 (NAN_INPUT_DETECTED) (Iter: {iter}). Output set to NaN.\n"
                                "   Sources: q_vec={q}, g_mat={g}, h_vec={h}\n"
                                "   Config: relax_cbf={relax_cbf}, relax_clf={relax_clf}",
                                iter=iter_num,
                                q=nan_q,
                                g=nan_g,
                                h=nan_h,
                                relax_cbf=relaxable_cbf,
                                relax_clf=relaxable_clf,
                            ),
                            jdebug.callback(_log_nan_indices, mask_q, mask_g, mask_h),
                        )[
                            0
                        ],  # -2
                        lambda: print_status_msg("NAN_DETECTED"),  # -1
                        lambda: print_status_msg("UNSOLVED"),  # 0
                        lambda: jdebug.print(
                            "⚠️ CBF-CLF-QP Succeeded (Unexpected failure call). Status: {status}",
                            status=status,
                        ),  # 1 (Should not happen)
                        lambda: print_status_msg("MAX_ITER_REACHED"),  # 2
                        lambda: print_status_msg("PRIMAL_INFEASIBLE"),  # 3
                        lambda: print_status_msg("DUAL_INFEASIBLE"),  # 4
                        lambda: print_status_msg("MAX_ITER_UNSOLVED"),  # 5
                    ],
                )

                if "bfs" in sub_data:
                    h_val = sub_data["bfs"]
                    jdebug.print("   -> Barrier Values (h): {h}", h=h_val)
                    lax.cond(
                        jnp.any(h_val < 0.0),
                        lambda: jdebug.print(
                            "      (Warning: h < 0 detected. System is strictly unsafe.)"
                        ),
                        lambda: None,
                    )

                if "lfs" in sub_data:
                    jdebug.print("   -> Lyapunov Values (V): {V}", V=sub_data["lfs"])

            # Only print failure if we weren't already in error state
            prev_error = data.error if data.error is not None else jnp.array(False)
            should_print = jnp.logical_not(success) & jnp.logical_not(prev_error)

            # Debug hook: Print failure details if solver failed AND it's a new failure
            lax.cond(
                should_print,
                _print_failure,
                lambda *_: None,
                status,
                iter_num,
                sub_data,
            )

            error = lax.cond(success, lambda _fake: False, lambda _fake: True, 0)

            # logging data
            final_sub_data = cast(CbfClfQpData, sub_data or {})
            final_sub_data["solver_params"] = new_params
            final_sub_data["solver_iter"] = iter_num
            # Explicitly log solver status for diagnostics/warnings
            final_sub_data["solver_status"] = status

            detached_sub_data = cast(
                Dict[str, Any], tree_util.tree_map(lax.stop_gradient, final_sub_data)
            )

            data = ControllerData(
                error=error,
                error_data=status,
                complete=complete,
                sol=lax.stop_gradient(jnp.array(sol)),
                u=lax.stop_gradient(u),
                u_nom=lax.stop_gradient(u_nom),
                sub_data=detached_sub_data,
            )

            return u, data

        return controller

    return generate_cbf_clf_controller
