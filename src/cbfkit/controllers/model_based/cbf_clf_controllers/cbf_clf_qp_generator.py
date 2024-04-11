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

from typing import Union, Dict, Any, Optional
import jax.numpy as jnp
from jax import Array, lax, jit

from cbfkit.utils.user_types import (
    CbfClfQpGenerator,
    CertificateCollection,
    ControllerCallable,
    ControllerCallableReturns,
    DynamicsCallable,
    GenerateComputeCertificateConstraintCallable,
    State,
)
from cbfkit.optimization.quadratic_program import solve as solve_qp
from .generate_constraints import (
    generate_compute_input_constraints,
    generate_compute_cbf_clf_constraints,
)


def cbf_clf_qp_generator(
    generate_compute_cbf_constraints: GenerateComputeCertificateConstraintCallable,
    generate_compute_clf_constraints: GenerateComputeCertificateConstraintCallable,
) -> CbfClfQpGenerator:
    """Function for producing a generating function for CBF-CLF-QP control laws of various forms.

    Args:
        generate_compute_cbf_constraints (GenerateComputeCertificateConstraintCallable)
        generate_compute_clf_constraints (GenerateComputeCertificateConstraintCallable)

    Returns:
        (CbfClfQpGenerator): function for generating CBF-CLF-QP control law
    """

    def generate_cbf_clf_controller(
        control_limits: Array,
        nominal_input: ControllerCallable,
        dynamics_func: DynamicsCallable,
        barriers: Optional[CertificateCollection] = ([], [], [], [], []),
        lyapunovs: Optional[CertificateCollection] = ([], [], [], [], []),
        p_mat: Optional[Union[Array, None]] = None,
        **kwargs: Dict[str, Any],
    ) -> ControllerCallable:
        """Produces the function to deploy a CBF-CLF-QP control law.

        Args:
            control_limits (Array): symmetric actuation constraints [u1_bar, u2_bar, etc.]
            nominal_input (ControllerCallable): function to compute the nominal input
            dynamics_func (DynamicsCallable): function to compute dynamics based on current state
            barriers (Optional[CertificateCollection] = ([], [], [], [], [])): collection of barrier functions,
                gradients, hessians, dh/dt, conditions
            lyapunovs (Optional[CertificateCollection] = ([], [], [], [], [])): collection of lyapunov functions,
                gradients, hessians, dV/dt,  conditions
            p_mat (Optional[Union[Array, None]] = None): objective function matrix (quadratic term)
            **kwargs (Dict[str, Any]): keyword arguments, e.g., RiskAwareParams for RA-CBF-CLF-QP

        Returns:
            ControllerCallable: function for computing control input based on CBF-CLF-QP
        """
        complete = False
        n_con = len(control_limits)

        if p_mat is None:
            if "tunable_class_k" not in kwargs:
                n_bfs = 0
            elif kwargs["tunable_class_k"]:
                b_funcs, _, _, _, _ = barriers
                n_bfs = len(b_funcs)
                control_limits = jnp.hstack([control_limits, 100 * jnp.ones((n_bfs,))])

            if "relaxable_clf" not in kwargs:
                n_lfs = 0
            elif kwargs["relaxable_clf"]:
                l_funcs, _, _, _, _ = lyapunovs
                n_lfs = len(l_funcs)
                control_limits = jnp.hstack([control_limits, 1e9 * jnp.ones((n_lfs,))])

            p_mat = jnp.diag(
                jnp.hstack([jnp.ones((n_con,)), 2e3 * jnp.ones((n_bfs,)), 2e3 * jnp.ones((n_lfs,))])
            )

        compute_input_constraints = generate_compute_input_constraints(control_limits)
        compute_cbf_clf_constraints = generate_compute_cbf_clf_constraints(
            generate_compute_cbf_constraints,
            generate_compute_clf_constraints,
            control_limits,
            dynamics_func,
            barriers,
            lyapunovs,
            **kwargs,
        )

        def controller(t: float, x: State) -> ControllerCallableReturns:
            """CBF-CLQ-QP control law.

            Args:
                t (float): time (in sec)
                x (State): state vector

            Returns:
                ControllerCallableReturns: tuple consisting of control solution (Array) and auxiliary data (Dict)
            """
            # Compute nominal control input
            u_nom, _ = nominal_input(t, x)

            return jittable_controller(t, x, u_nom)

        @jit
        def jittable_controller(t: float, x: State, u_nom: Array) -> ControllerCallableReturns:
            """JIT-compatible portion of the CBF-CLF-QP control law.

            Args:
                t (float): time (in sec)
                x (State): state vector
                u_nom (Array): nominal control input

            Returns:
                ControllerCallableReturns: tuple consisting of control solution (Array) and auxiliary data (Dict)
            """
            nonlocal complete

            if n_bfs > 0:
                u_nom = jnp.hstack([u_nom, jnp.ones((n_bfs,))])
            if n_lfs > 0:
                u_nom = jnp.hstack([u_nom, jnp.zeros((n_lfs,))])

            # Compute QP cost, constraint functions
            q_vec = jnp.expand_dims(jnp.matmul(-2 * p_mat, u_nom), axis=-1)
            g_mat_u, h_vec_u = compute_input_constraints(t, x)
            g_mat_c, h_vec_c, sub_data = compute_cbf_clf_constraints(t, x)
            if "complete" in sub_data:
                complete = sub_data["complete"]

            # Stack input and certificate function constraints
            g_mat = jnp.vstack([g_mat_u, g_mat_c])
            h_vec = jnp.expand_dims(jnp.hstack([h_vec_u, h_vec_c]), axis=-1)

            # Solve QP
            sol, status = solve_qp(p_mat, q_vec, g_mat, h_vec)
            u = lax.cond(
                status,
                lambda _fake: jnp.array(sol[:n_con]).reshape((n_con,)),
                lambda _fake: jnp.zeros((n_con,)),
                0,
            )

            # Saturate the solution if necessary
            u = jnp.clip(u, -control_limits[:n_con], control_limits[:n_con]).reshape((n_con,))

            if "ra_params" in kwargs:
                #! To Do: integrate RA-PI states
                pass

            error = lax.cond(status, lambda _fake: False, lambda _fake: True, 0)

            # logging data
            data = {
                "error": error,
                "complete": complete,
                "sol": jnp.array(sol),
                "u": u,
                "u_nom": u_nom,
                "sub_data": sub_data,
            }

            return u, data

        return controller

    return generate_cbf_clf_controller
