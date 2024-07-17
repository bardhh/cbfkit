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
from .mppi_source import setup_mppi

from cbfkit.utils.user_types import (
    MppiGenerator,
    ControllerCallable,
    ControllerCallableReturns,
    DynamicsCallable,
    GenerateComputeStageCostCallable,
    GenerateComputeTerminalCostCallable,
    State,
    Control,
    TrajectoryCostCallable,
    StageCostCallable,
    TerminalCostCallable,
    Key,
)

# from cbfkit.optimization.quadratic_program import solve as solve_qp
# from .generate_constraints import (
#     generate_compute_input_constraints,
#     generate_compute_cbf_clf_constraints,
# )


def mppi_generator(
    # generate_compute_stage_cost: GenerateComputeStageCostCallable,
    # generate_compute_terminal_cost: GenerateComputeTerminalCostCallable,
) -> MppiGenerator:
    """Function for producing a generating function for MPPI laws of various forms.

    Args:
        generate_compute_stage_cost (GenerateComputeCertificateConstraintCallable)
        generate_compute_terminal_cost (GenerateComputeCertificateConstraintCallable)

    Returns:
        (MppiGenerator): function for generating MPPI control law
    """

    def generate_mppi(
        control_limits: Array,
        dynamics_func: DynamicsCallable,
        trajectory_cost: TrajectoryCostCallable,
        stage_cost: StageCostCallable,
        terminal_cost: TerminalCostCallable,
        mppi_args: list,
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
        n_con = len(control_limits)

        mppi = setup_mppi(
            dyn_func=dynamics_func,
            trajectory_cost_func=trajectory_cost,
            stage_cost_func=stage_cost,
            terminal_cost_func=terminal_cost,
            robot_state_dim=mppi_args["robot_state_dim"],
            robot_control_dim=mppi_args["robot_control_dim"],
            horizon=mppi_args["prediction_horizon"],
            samples=mppi_args["num_samples"],
            control_bound=control_limits,
            dt=mppi_args["time_step"],
            use_GPU=mppi_args["use_GPU"],
            costs_lambda=mppi_args["costs_lambda"],
            cost_perturbation_coeff=mppi_args["cost_perturbation"],
        )

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

            # Solve MPPI
            (
                robot_sampled_states,
                robot_selected_states,
                robot_action,
                action_trajectory,
            ) = mppi(key, data["u_traj"], x.reshape(-1, 1), t, data["xs"], data["prev_robustness"])

            # Saturate the solution if necessary
            u = jnp.clip(robot_action, -control_limits[:n_con], control_limits[:n_con]).reshape(
                (n_con,)
            )

            # logging data
            data = {
                "complete": complete,  # TODO: what is complete
                "u": u,
                "robot_sampled_states": robot_sampled_states,
                "x_traj": robot_selected_states,
                "u_traj": action_trajectory,
                "xs": data["xs"],
            }

            return u, data

        return process

    return generate_mppi
