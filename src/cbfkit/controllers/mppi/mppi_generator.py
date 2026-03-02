"""MPPI planner-law generator."""

from typing import Any, Dict, Optional, cast

import jax.numpy as jnp
from jax import Array, jit

from cbfkit.utils.user_types import (
    Control,
    DynamicsCallable,
    Key,
    MppiGenerator,
    MppiParameters,
    PlannerCallable,
    PlannerCallableReturns,
    PlannerData,
    StageCostCallable,
    State,
    TerminalCostCallable,
    Time,
    TrajectoryCostCallable,
)

from .mppi_source import setup_mppi


def mppi_generator() -> MppiGenerator:
    """Return a factory that builds MPPI planner callables."""

    def generate_mppi(
        control_limits: Array,
        dynamics_func: DynamicsCallable,
        stage_cost: Optional[StageCostCallable] = None,
        terminal_cost: Optional[TerminalCostCallable] = None,
        trajectory_cost: Optional[TrajectoryCostCallable] = None,
        mppi_args: Optional[MppiParameters] = None,
        **kwargs: Dict[str, Any],
    ) -> PlannerCallable:
        """Produces the function to deploy a MPPI control law.

        Args:
            control_limits (Array): symmetric actuation constraints [u1_bar, u2_bar, etc.]
            dynamics_func (DynamicsCallable): function to compute dynamics based on current state
            stage_cost (DynamicsCallable): function to compute dynamics based on current state
            terminal-cost (DynamicsCallable): function to compute dynamics based on current state

            **kwargs (Dict[str, Any]): keyword arguments

        Returns
        -------
            PlannerCallable: function for computing control input based on MPPI
        """
        n_con = len(control_limits)

        # Cast mppi_args to MppiParameters to avoid mypy errors, as the code assumes it is not None.
        # This preserves the original runtime behavior (crashing if None is passed) while enforcing types.
        mppi_args = cast(MppiParameters, mppi_args)

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

        def process(
            t: Time, x: State, u_nom: Optional[Control], key: Key, data: PlannerData
        ) -> PlannerCallableReturns:
            """MPPI control law.

            Args:
                t (float): time (in sec)
                x (State): state vector

            Returns
            -------
                PlannerCallableReturns: tuple consisting of control solution (Array) and auxiliary data (Dict)
            """
            t_float = cast(float, t)
            return jittable_process(t_float, x, key, data)

        @jit
        def jittable_process(
            t: float, x: State, key: Key, data: PlannerData
        ) -> PlannerCallableReturns:
            """JIT-compatible portion of the MPPI control law.

            Args:
                t (float): time (in sec)
                x (State): state vector
                u (Array): previous control input trajectory

            Returns
            -------
                PlannerCallableReturns: tuple consisting of control solution (Array) and auxiliary data (Dict)
            """
            # Solve MPPI
            xs = data.xs if data.xs is not None else x.reshape(-1, 1)
            prev_robustness = data.prev_robustness

            (
                robot_sampled_states,
                robot_selected_states,
                robot_action,
                action_trajectory,
            ) = mppi(key, data.u_traj, x.reshape(-1, 1), t, xs, prev_robustness)

            # Saturate the solution if necessary
            u = jnp.clip(robot_action, -control_limits[:n_con], control_limits[:n_con]).reshape(
                (n_con,)
            )

            # Explicitly catch NaN solutions
            is_nan = jnp.isnan(u).any()

            # logging data
            data = data._replace(
                sampled_x_traj=robot_sampled_states,
                x_traj=robot_selected_states,
                u_traj=action_trajectory,
                prev_robustness=prev_robustness,
                error=is_nan,
            )

            return u, data

        return process

    return generate_mppi
