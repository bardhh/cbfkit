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
>>> from cbfkit.controllers.model_based.cbf_clf_controllers.cbf_clf_qp_generator import cbf_clf_qp_generator
>>> from cbfkit.controllers.model_based.cbf_clf_controllers.generate_constraints import (
>>>     generate_compute_zeroing_cbf_constraints,
>>>     generate_compute_vanilla_clf_constraints,
>>> )
>>> vanilla_cbf_clf_qp_controller = cbf_clf_qp_generator(
>>>     generate_compute_zeroing_cbf_constraints,
>>>     generate_compute_vanilla_clf_constraints,
>>> )
"""

from typing import Any, Dict, Optional, Tuple, Union

import jax.numpy as jnp
from jax import Array, jit, lax

from cbfkit.optimization.quadratic_program.qp_solver_jaxopt import (
    solve_with_details as solve_qp,
)
from cbfkit.utils.user_types import (
    EMPTY_CERTIFICATE_COLLECTION,
    CbfClfQpGenerator,
    CertificateCollection,
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
        barriers: Optional[CertificateCollection] = EMPTY_CERTIFICATE_COLLECTION,
        lyapunovs: Optional[CertificateCollection] = EMPTY_CERTIFICATE_COLLECTION,
        p_mat: Optional[Union[Array, None]] = None,
        **kwargs: Dict[str, Any],
    ) -> ControllerCallable:
        """Produces the function to deploy a CBF-CLF-QP control law.

        Args:
            control_limits (Array): symmetric actuation constraints [u1_bar, u2_bar, etc.]
            nominal_input (ControllerCallable): function to compute the nominal input
            dynamics_func (DynamicsCallable): function to compute dynamics based on current state
            barriers (CertificateCollection = ([], [], [], [], [])): collection of barrier functions,
                gradients, hessians, dh/dt, conditions
            lyapunovs (CertificateCollection = ([], [], [], [], [])): collection of lyapunov functions,
                gradients, hessians, dV/dt,  conditions
            p_mat (Optional[Union[Array, None]] = None): objective function matrix (quadratic term)
            **kwargs (Dict[str, Any]): keyword arguments, e.g., RiskAwareParams for RA-CBF-CLF-QP.
                relaxable_clf (bool): whether to treat CLF as a soft constraint (default: True).

        Returns
        -------
            ControllerCallable: function for computing control input based on CBF-CLF-QP
        """
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

        # Ensure barriers and lyapunovs are not None
        if barriers is None:
            barriers = EMPTY_CERTIFICATE_COLLECTION
        if lyapunovs is None:
            lyapunovs = EMPTY_CERTIFICATE_COLLECTION

        assert barriers is not None
        assert lyapunovs is not None

        if "tunable_class_k" in kwargs and kwargs["tunable_class_k"]:
            b_funcs, _, _, _, _ = barriers
            n_bfs = len(b_funcs)
            slack_cbf = kwargs.get("slack_bound_cbf", 100.0)
            control_limits = jnp.hstack([control_limits, slack_cbf * jnp.ones((n_bfs,))])
        elif "relaxable_cbf" in kwargs and kwargs["relaxable_cbf"]:
            b_funcs, _, _, _, _ = barriers
            n_bfs = len(b_funcs)
            slack_cbf = kwargs.get("slack_bound_cbf", 1e4)
            control_limits = jnp.hstack([control_limits, slack_cbf * jnp.ones((n_bfs,))])
        else:
            n_bfs = 0

        # Default to relaxable CLF for robustness unless explicitly disabled
        relaxable_clf = kwargs.get("relaxable_clf", True)
        if relaxable_clf:
            l_funcs, _, _, _, _ = lyapunovs
            n_lfs = len(l_funcs)
            slack_clf = kwargs.get("slack_bound_clf", 1e9)
            control_limits = jnp.hstack([control_limits, slack_clf * jnp.ones((n_lfs,))])
        else:
            n_lfs = 0

        # Bolt: Normalize slack variables to improve QP conditioning (Condition number ~ 1)
        # delta_phys = delta_raw / sqrt(penalty). Cost: delta_raw^2. Constraint: ... + delta_raw/sqrt(penalty)
        scale_cbf = 1.0
        scale_clf = 1.0
        auto_p_mat = p_mat is None

        if auto_p_mat:
            penalty_cbf = kwargs.get("slack_penalty_cbf", 2e3)
            penalty_clf = kwargs.get("slack_penalty_clf", 2e3)
            scale_cbf = 1.0 / jnp.sqrt(penalty_cbf + 1e-8)
            scale_clf = 1.0 / jnp.sqrt(penalty_clf + 1e-8)
            p_mat = jnp.diag(
                jnp.hstack(
                    [
                        jnp.ones((n_con,)),
                        jnp.ones((n_bfs,)),  # Normalized weight
                        jnp.ones((n_lfs,)),  # Normalized weight
                    ]
                )
            )
            # Scale slack bounds (delta_raw = delta_phys * sqrt(penalty) = delta_phys / scale)
            # control_limits is already [u_lim, slack_cbf_lim, slack_clf_lim]
            limit_u = control_limits[:n_con]
            limit_cbf = control_limits[n_con : n_con + n_bfs] / scale_cbf
            limit_clf = control_limits[n_con + n_bfs :] / scale_clf
            control_limits = jnp.hstack([limit_u, limit_cbf, limit_clf])

        # Bolt: Pass scales to constraint generators to avoid post-hoc scaling in loop
        if auto_p_mat:
            kwargs["scale_cbf"] = scale_cbf
            kwargs["scale_clf"] = scale_clf

        compute_input_constraints = generate_compute_input_constraints(control_limits)
        compute_cbf_clf_constraints = generate_compute_cbf_clf_constraints(
            generate_compute_cbf_constraints,
            generate_compute_clf_constraints,
            control_limits,
            verified_dynamics_func,
            barriers,
            lyapunovs,
            **kwargs,
        )

        # def controller(
        #     t: float, x: State, u_nom: Control, key: Key, data: list
        # ) -> ControllerCallableReturns:
        #     """CBF-CLQ-QP control law.

        #     Args:
        #         t (float): time (in sec)
        #         x (State): state vector

        #     Returns:
        #         ControllerCallableReturns: tuple consisting of control solution (Array) and auxiliary data (Dict)
        #     """
        #     return jittable_controller(t, x, u_nom, key, data)

        # @jit
        # def jittable_controller(
        #     t: float, x: State, u_nom: Array, key: Key, data: list
        # ) -> ControllerCallableReturns:
        @jit
        def controller(
            t: float, x: State, u_nom: Array, key: Key, data: ControllerData
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

            if u_nom.shape[0] != n_con:
                raise ValueError(
                    f"Nominal control input 'u_nom' has shape {u_nom.shape}, "
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
            g_mat_u, h_vec_u = compute_input_constraints(t, x)
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
                row_norms_c = jnp.linalg.norm(g_mat_c, axis=1)
                safe_norms_c = jnp.where(row_norms_c > 1e-8, row_norms_c, 1.0)
                g_mat_c = g_mat_c / safe_norms_c[:, None]
                h_vec_c = h_vec_c / safe_norms_c

            g_mat = jnp.vstack([g_mat_u, g_mat_c])
            h_vec = jnp.hstack([h_vec_u, h_vec_c])

            # Solve QP
            solver_params = None
            if data.sub_data is not None and "solver_params" in data.sub_data:
                solver_params = data.sub_data["solver_params"]

            sol, status, new_params = solve_qp(
                p_mat, q_vec, g_mat, h_vec, init_params=solver_params
            )

            # Bolt: Rescale solution back to physical units
            if auto_p_mat:
                if n_bfs > 0:
                    sol = sol.at[n_con : n_con + n_bfs].multiply(scale_cbf)
                if n_lfs > 0:
                    sol = sol.at[n_con + n_bfs : n_con + n_bfs + n_lfs].multiply(scale_clf)
            # QP solution already respects control limits via input constraints.
            # Only clip the fallback u_nom (which may exceed limits) to avoid
            # inadvertently violating CBF constraints on the QP-solved path.
            u = lax.cond(
                status == 1,
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

            error = lax.cond(status == 1, lambda _fake: False, lambda _fake: True, 0)

            # logging data
            final_sub_data = sub_data or {}
            final_sub_data["solver_params"] = new_params

            data = ControllerData(
                error=error,
                error_data=status,
                complete=complete,
                sol=jnp.array(sol),
                u=u,
                u_nom=u_nom,
                sub_data=final_sub_data,
            )

            return u, data

        return controller

    return generate_cbf_clf_controller
