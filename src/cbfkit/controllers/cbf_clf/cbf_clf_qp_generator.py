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
from jax import Array, jit, lax

from cbfkit.certificates import concatenate_certificates
from cbfkit.optimization.quadratic_program.qp_solver_jaxopt import (
    solve_with_details as solve_qp,
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
    # Aegis: Ensure all component lists have the same length to prevent obscure JAX errors downstream.
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

        # Bolt: Normalize slack variables to improve QP conditioning (Condition number ~ 1)
        # delta_phys = delta_raw / sqrt(penalty). Cost: delta_raw^2. Constraint: ... + delta_raw/sqrt(penalty)
        scale_cbf = 1.0
        scale_clf = 1.0
        auto_p_mat = p_mat is None

        # Precompute scaling vector for solution recovery
        sol_scaling_vector = jnp.ones((n_con + n_bfs + n_lfs,))

        if auto_p_mat:
            penalty_cbf = slack_penalty_cbf
            penalty_clf = slack_penalty_clf

            # Janus: Robust scaling. Clamp penalty to avoid tiny scale factors (leading to ill-conditioned A).
            # If penalty > 1e6, scale factor saturates at 1e-3.
            # Transfer excess penalty to P matrix diagonal: weight = penalty * scale^2
            max_penalty = 1e6
            eff_penalty_cbf = jnp.minimum(penalty_cbf, max_penalty)
            eff_penalty_clf = jnp.minimum(penalty_clf, max_penalty)

            scale_cbf = 1.0 / jnp.sqrt(eff_penalty_cbf + 1e-8)
            scale_clf = 1.0 / jnp.sqrt(eff_penalty_clf + 1e-8)

            # Janus: Clamp weight to avoid catastrophic ill-conditioning in P matrix.
            # Even with robust scaling, extremely large penalties (>1e14) create huge condition numbers
            # for float32 QPs, leading to solver failure (MAX_ITER_UNSOLVED).
            # A max weight of 1e4 preserves 4 orders of magnitude relative to control cost (u^2),
            # which is sufficient for "hard" constraints while retaining numerical solvability (float32).
            # Note: With tol=1e-3 and float32 eps=1e-7, values > 1e4 cause residuals > tol purely due to precision loss.
            MAX_WEIGHT = 1.0e4
            weight_cbf = jnp.minimum(penalty_cbf * (scale_cbf**2), MAX_WEIGHT)
            weight_clf = jnp.minimum(penalty_clf * (scale_clf**2), MAX_WEIGHT)

            p_mat = jnp.diag(
                jnp.hstack(
                    [
                        jnp.ones((n_con,)),
                        weight_cbf * jnp.ones((n_bfs,)),  # Normalized weight
                        weight_clf * jnp.ones((n_lfs,)),  # Normalized weight
                    ]
                )
            )
            # Scale slack bounds (delta_raw = delta_phys * sqrt(penalty) = delta_phys / scale)
            # control_limits is already [u_lim, slack_cbf_lim, slack_clf_lim]
            limit_u = control_limits[:n_con]
            limit_cbf = control_limits[n_con : n_con + n_bfs] / scale_cbf
            limit_clf = control_limits[n_con + n_bfs :] / scale_clf
            control_limits = jnp.hstack([limit_u, limit_cbf, limit_clf])

            # Populate scaling vector
            if n_bfs > 0:
                sol_scaling_vector = sol_scaling_vector.at[n_con : n_con + n_bfs].set(scale_cbf)
            if n_lfs > 0:
                sol_scaling_vector = sol_scaling_vector.at[n_con + n_bfs :].set(scale_clf)

        # Bolt: Pass scales to constraint generators to avoid post-hoc scaling in loop
        if auto_p_mat:
            kwargs["scale_cbf"] = scale_cbf
            kwargs["scale_clf"] = scale_clf

        # Bolt: Pre-compute solution scaling vector to avoid repetitive slice updates in JIT loop
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

        # Bolt: Pre-compute static input constraints (box limits + tunable non-negativity)
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

            # Janus: Ensure u_nom is 1D to prevent broadcasting errors (e.g., (N,) + (N,1) -> (N,N))
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
            # Bolt: Keep vectors 1D to avoid JAX broadcasting overhead in solver (prevents (N,1) vs (N,) mismatch)
            q_vec = jnp.matmul(-2 * p_mat, u_nom)
            g_mat_c, h_vec_c, sub_data = compute_cbf_clf_constraints(t, x)
            if "complete" in sub_data:
                complete = sub_data["complete"]

            # Stack input and certificate function constraints
            # Bolt: Scaling of slack columns is now handled within compute_cbf_clf_constraints (via kwargs)
            # This avoids large array copies and multiply operations inside the JIT loop.

            # Bolt: Normalize CBF/CLF constraint rows to improve numerical stability
            # Janus: Avoid normalizing noise vectors. If norm < tol, do NOT scale up.
            # Input constraints (box limits) are already normalized (row norm = 1).
            if auto_p_mat:
                # Janus: Compute norm safely to avoid overflow for large gradients.
                # Naive sqrt(sum(sq)) overflows if elements > 1e19 (float32).
                # We scale by max(abs(x)) to keep squares in range.
                row_max = jnp.max(jnp.abs(g_mat_c), axis=1)
                # Avoid division by zero
                row_max_safe = jnp.where(row_max > 0, row_max, 1.0)

                # Check if row is effectively zero (to avoid nan gradients at 0)
                is_zero_row = row_max < 1e-20

                # Scale the matrix
                g_mat_c_scaled = g_mat_c / row_max_safe[:, None]

                # Compute norm of scaled matrix
                # Fix: Ensure we don't compute norm of zero vector, which has undefined gradient
                g_mat_c_safe_norm = jnp.where(
                    is_zero_row[:, None],
                    jnp.ones_like(g_mat_c_scaled),
                    g_mat_c_scaled,
                )

                row_norms_scaled = jnp.linalg.norm(g_mat_c_safe_norm, axis=1)

                # Recover true norm
                row_norms_c = row_norms_scaled * row_max_safe

                # Add epsilon for safety in division
                row_norms_c = row_norms_c + 1e-20

                # Janus: Normalize constraints robustly.
                # Use a clamped scaling factor (max 1e15) to:
                # 1. Enforce constraints with small gradients (e.g. 1e-13) to catch gross infeasibility (Safety).
                # 2. Allow normalization of small signals (e.g. 1e-15) for solver convergence.
                safe_scales_c = jnp.where(
                    is_zero_row, 1.0, jnp.minimum(1.0 / row_norms_c, 1e15)
                )

                g_mat_c = g_mat_c * safe_scales_c[:, None]
                h_vec_c = h_vec_c * safe_scales_c

            # Bolt: Use pre-computed static constraints (input limits + tunable non-negativity)
            g_mat = jnp.vstack([g_mat_static, g_mat_c])
            h_vec = jnp.hstack([h_vec_static, h_vec_c])

            # Sentinel: Detect NaNs in QP inputs
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

            sol, status, new_params = solve_qp(
                p_mat, q_vec, g_mat, h_vec, init_params=solver_params
            )

            # Scout: Extract solver iterations for diagnostics
            # new_params is (KKTSolution, OSQPState)
            _, state = new_params
            iter_num = state.iter_num

            # Sentinel: Explicitly catch NaN solutions even if solver claims success
            # Also catch if inputs were NaN (solver might return 0 and UNSOLVED status 0)
            status = jnp.where(
                jnp.any(jnp.isnan(sol)) | jnp.any(jnp.isinf(sol)), -1, status
            )
            status = jnp.where(nan_in_inputs, -2, status)

            # Bolt: Rescale solution back to physical units
            if auto_p_mat:
                sol = sol * sol_scaling_vector
            # QP solution already respects control limits via input constraints.
            # Only clip the fallback u_nom (which may exceed limits) to avoid
            # inadvertently violating CBF constraints on the QP-solved path.
            # Sentinel: Only accept SOLVED (1) as success.
            # Status 2 (MAX_ITER) or 5 (MAX_ITER_UNSOLVED) means potentially unconverged/unsafe solution.
            success = status == 1

            def _print_failure(status, iter_num, sub_data):
                # Sentinel: Map status codes to human-readable strings
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
                        )[0],  # -2
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
                        lambda: jdebug.print("      (Warning: h < 0 detected. System is strictly unsafe.)"),
                        lambda: None,
                    )

                if "lfs" in sub_data:
                    jdebug.print("   -> Lyapunov Values (V): {V}", V=sub_data["lfs"])

            # Sentinel: Only print failure if we weren't already in error state
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

            u = lax.cond(
                success,
                # Bolt: Avoid jnp.array copy and reshape (sol is already 1D)
                lambda _fake: sol[:n_con],
                # Aegis: Return NaN if QP fails to make failure mode explicit.
                # Returning u_nom is unsafe as it likely violates constraints.
                lambda _fake: jnp.full_like(u_nom[:n_con], jnp.nan),
                0,
            )

            if "ra_params" in kwargs:
                #! To Do: integrate RA-PI states
                pass

            error = lax.cond(success, lambda _fake: False, lambda _fake: True, 0)

            # logging data
            final_sub_data = cast(CbfClfQpData, sub_data or {})
            final_sub_data["solver_params"] = new_params
            final_sub_data["solver_iter"] = iter_num
            # Sentinel: Explicitly log solver status for diagnostics/warnings
            final_sub_data["solver_status"] = status

            data = ControllerData(
                error=error,
                error_data=status,
                complete=complete,
                sol=jnp.array(sol),
                u=u,
                u_nom=u_nom,
                sub_data=cast(Dict[str, Any], final_sub_data),
            )

            return u, data

        return controller

    return generate_cbf_clf_controller
