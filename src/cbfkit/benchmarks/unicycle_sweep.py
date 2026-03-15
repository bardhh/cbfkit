"""Unicycle obstacle avoidance sweep benchmark.

Registers the unicycle_obstacle_avoidance_sweep scenario for CLI use.
Sweeps over CBF alpha, controller gains (Kp_pos, Kp_theta), and barrier type.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
from cbfkit.benchmarks.registry import register_sweepable_scenario
from cbfkit.certificates import concatenate_certificates, rectify_relative_degree
from cbfkit.certificates.barrier_functions import ellipsoidal_barrier_factory
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.integration import forward_euler
from cbfkit.simulation.monte_carlo_gpu import (
    MonteCarloSetup,
    MonteCarloGPUResults,
    conduct_monte_carlo_gpu,
    conduct_monte_carlo_gpu_multiseed,
    _default_sensor,
    _default_estimator,
    _default_perturbation,
)
from cbfkit.simulation.safety_verification import compute_safety_statistics
from cbfkit.utils.user_types import ControllerData, PlannerData

# State dimension: [x, y, v, theta]
STATE_DIM = 4
GOAL = jnp.array([2.0, 4.0, 0.0, 0.0])
N_TRIALS = 10
NUM_STEPS = 500
DT = 0.05
GOAL_TOL = 0.25

# Ellipsoidal barrier factory for unicycle (position indices 0,1)
_cbf, _cbf_grad, _cbf_hess = ellipsoidal_barrier_factory(
    system_position_indices=(0, 1),
    obstacle_position_indices=(0, 1),
    ellipsoid_axis_indices=(0, 1),
)

_BARRIER_TYPES = {
    "linear_class_k": zeroing_barriers.linear_class_k,
    "cubic_class_k": zeroing_barriers.cubic_class_k,
}

# Static obstacles matching examples/unicycle/reach_goal/vanilla_cbf_accel_unicycle.py
OBSTACLES = [
    jnp.array([1.0, 2.0, 0.0]),
    jnp.array([3.0, 2.0, 0.0]),
    jnp.array([2.0, 5.0, 0.0]),
    jnp.array([-1.0, 1.0, 0.0]),
    jnp.array([0.5, -1.0, 0.0]),
]
ELLIPSOIDS = [
    jnp.array([0.5, 1.5]),
    jnp.array([0.75, 2.0]),
    jnp.array([2.0, 0.25]),
    jnp.array([1.0, 0.75]),
    jnp.array([0.75, 0.5]),
]


def _build_setup(
    seed: int,
    alpha: float = 1.0,
    Kp_pos: float = 1.0,
    Kp_theta: float = 5.0,
    barrier_type: str = "linear_class_k",
    control_limit: float = 100.0,
):
    """Build a MonteCarloSetup for unicycle obstacle avoidance."""

    # Dynamics
    dynamics = unicycle.plant(l=1.0)
    dynamics.a_max = control_limit
    dynamics.omega_max = control_limit
    dynamics.v_max = 2.0
    dynamics.goal_tol = GOAL_TOL

    # Barrier condition
    if barrier_type not in _BARRIER_TYPES:
        raise ValueError(
            f"Unknown barrier_type '{barrier_type}'. "
            f"Must be one of: {list(_BARRIER_TYPES.keys())}"
        )
    barrier_fn = _BARRIER_TYPES[barrier_type]

    # Build barriers using rectify_relative_degree (handles high relative degree)
    barriers_list = [
        rectify_relative_degree(
            function=_cbf(obs, ell),
            system_dynamics=dynamics,
            state_dim=STATE_DIM,
            form="high-order",
            certificate_conditions=barrier_fn(alpha),
        )
        for obs, ell in zip(OBSTACLES, ELLIPSOIDS)
    ]
    barrier_packages = concatenate_certificates(*barriers_list)

    # Controller
    controller = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([control_limit, control_limit]),
        dynamics_func=dynamics,
        barriers=barrier_packages,
        relaxable_cbf=False,
        relaxable_clf=True,
    )

    # Nominal controller
    nominal_controller = unicycle.controllers.proportional_controller(
        dynamics=dynamics,
        Kp_pos=Kp_pos,
        Kp_theta=Kp_theta,
    )

    def initial_state_sampler(key):
        k1, k2 = random.split(key)
        pos = random.uniform(k1, (2,), minval=-1.0, maxval=1.0)
        theta = random.uniform(k2, (), minval=-jnp.pi, maxval=jnp.pi)
        return jnp.array([pos[0], pos[1], 0.0, theta])

    # Prime controller data
    prime_key = random.PRNGKey(42)
    x0_probe = jnp.zeros(STATE_DIM)
    u_nom_probe = jnp.zeros(2)
    _, c_data = controller(0.0, x0_probe, u_nom_probe, prime_key, ControllerData())

    # Planner data: constant desired state trajectory
    x_traj = jnp.tile(GOAL.reshape(-1, 1), (1, NUM_STEPS + 1))

    return MonteCarloSetup(
        dt=DT,
        num_steps=NUM_STEPS,
        dynamics=dynamics,
        integrator=forward_euler,
        initial_state_sampler=initial_state_sampler,
        nominal_controller=nominal_controller,
        controller=controller,
        sensor=_default_sensor,
        estimator=_default_estimator,
        perturbation=_default_perturbation,
        sigma=jnp.zeros(0),
        controller_data=c_data,
        planner=None,
        planner_data=PlannerData(x_traj=x_traj),
    )


def _process_results(results: MonteCarloGPUResults) -> dict:
    """Extract metrics from a single seed's Monte Carlo results."""
    stats = compute_safety_statistics(results)
    platform = jax.devices()[0].platform
    final_states = results.states[:, -1, :]
    final_goal_distance = float(
        jnp.mean(jnp.linalg.norm(final_states[:, :2] - GOAL[:2], axis=-1))
    )

    distances = jnp.linalg.norm(
        results.states[:, :, :2] - GOAL[:2], axis=-1
    )
    reached = distances < GOAL_TOL
    step_indices = jnp.arange(reached.shape[1])
    first_arrival = jnp.where(reached, step_indices[None, :], reached.shape[1])
    first_arrival_step = jnp.min(first_arrival, axis=1)
    time_to_goal = float(jnp.mean(first_arrival_step)) * DT

    return {
        "platform": platform,
        "n_trials": N_TRIALS,
        "wall_time_s": results.wall_time_s,
        "trials_per_sec": N_TRIALS / results.wall_time_s,
        "violation_rate": stats.violation_rate,
        "min_barrier_value": stats.min_barrier_value,
        "success": 1,
        "safety_violations": int(stats.violation_rate > 0),
        "solver_failures": 0,
        "avg_step_ms": results.wall_time_s / NUM_STEPS * 1000.0,
        "final_goal_distance": final_goal_distance,
        "time_to_goal": time_to_goal,
    }


def _unicycle_batch_runner(seeds: list[int], params: dict) -> list[dict]:
    """Run all seeds in a single vmap call (one JIT compilation)."""
    alpha = params.get("alpha", 1.0)
    Kp_pos = params.get("Kp_pos", 1.0)
    Kp_theta = params.get("Kp_theta", 5.0)
    barrier_type = params.get("barrier_type", "linear_class_k")

    setup = _build_setup(
        0,
        alpha=alpha,
        Kp_pos=Kp_pos,
        Kp_theta=Kp_theta,
        barrier_type=barrier_type,
    )

    results_list = conduct_monte_carlo_gpu_multiseed(setup, n_trials=N_TRIALS, seeds=seeds)
    return [_process_results(r) for r in results_list]


@register_sweepable_scenario(
    "unicycle_obstacle_avoidance_sweep",
    sweepable_params=["alpha", "Kp_pos", "Kp_theta", "barrier_type"],
    description="Unicycle obstacle avoidance with sweepable CBF alpha, controller gains, and barrier type.",
    batch_runner=_unicycle_batch_runner,
)
def unicycle_sweep(seed: int, params: dict) -> dict:
    alpha = params.get("alpha", 1.0)
    Kp_pos = params.get("Kp_pos", 1.0)
    Kp_theta = params.get("Kp_theta", 5.0)
    barrier_type = params.get("barrier_type", "linear_class_k")

    setup = _build_setup(
        seed,
        alpha=alpha,
        Kp_pos=Kp_pos,
        Kp_theta=Kp_theta,
        barrier_type=barrier_type,
    )

    results = conduct_monte_carlo_gpu(setup, n_trials=N_TRIALS, seed=seed)
    return _process_results(results)
