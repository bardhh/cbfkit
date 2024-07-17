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
>>> from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.cbf_clf_qp_generator import cbf_clf_qp_generator
>>> from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.generate_constraints import (
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
    ControllerCallable,
    ControllerCallableReturns,
    DynamicsCallable,
    State,
    Control,
    Key,
)


def waypoint_generator(
    # generate_compute_stage_cost: GenerateComputeStageCostCallable,
    # generate_compute_terminal_cost: GenerateComputeTerminalCostCallable,
) -> float:
    """Function for producing a generating function for MPPI laws of various forms.

    Args:
        generate_compute_stage_cost (GenerateComputeCertificateConstraintCallable)
        generate_compute_terminal_cost (GenerateComputeCertificateConstraintCallable)

    Returns:
        (WaypointGenerator): function for generating MPPI control law
    """

    def generate_waypoint(
        target_state: Array,
        **kwargs: Dict[str, Any],
    ) -> ControllerCallable:
        """Produces the function to deploy a CBF-CLF-QP control law.

        Args:
            control_limits (Array): symmetric actuation constraints [u1_bar, u2_bar, etc.]
            dynamics_func (DynamicsCallable): function to compute dynamics based on current state
            stage_cost (DynamicsCallable): function to compute dynamics based on current state
            terminal-cost (DynamicsCallable): function to compute dynamics based on current state

            **kwargs (Dict[str, Any]): keyword arguments, e.g., RiskAwareParams for RA-CBF-CLF-QP

        Returns:
            ControllerCallable: function for computing control input based on CBF-CLF-QP
        """
        complete = False

        # TODO: define State, Control Trajectory types??
        def process(
            t: float, x: State, u_nom: Control, key: Key, data: list
        ) -> ControllerCallableReturns:
            """MPPI control law.

            Args:
                t (float): time (in sec)
                x (State): state vector

            Returns:
                ControllerCallableReturns: tuple consisting of control solution (Array) and auxiliary data (Dict)
            """

            return jittable_process(t, x, key, data)

        @jit
        def jittable_process(t: float, x: State, key: Key, data: list) -> ControllerCallableReturns:
            """JIT-compatible portion of the CBF-CLF-QP control law.

            Args:
                t (float): time (in sec)
                x (State): state vector
                u (Array): previous control input trajectory

            Returns:
                ControllerCallableReturns: tuple consisting of control solution (Array) and auxiliary data (Dict)
            """
            nonlocal complete

            return None, {"u_traj": None, "x_traj": target_state.reshape(-1, 1)}

        return process

    return generate_waypoint
