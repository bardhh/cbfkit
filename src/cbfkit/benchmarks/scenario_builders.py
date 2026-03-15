"""Shared scenario builders for single-integrator benchmark scenarios.

Provides reusable helpers so that ``monte_carlo_gpu_benchmark`` and
``single_integrator_sweep`` (and any future single-integrator scenarios)
don't duplicate dynamics, barrier, and setup logic.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import random

from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.controllers.cbf_clf.generate_constraints import (
    generate_compute_vanilla_clf_constraints,
    generate_compute_zeroing_cbf_constraints,
)
from cbfkit.integration import forward_euler
from cbfkit.simulation.monte_carlo_gpu import (
    MonteCarloSetup,
    _default_estimator,
    _default_perturbation,
    _default_sensor,
)
from cbfkit.utils.user_types import CertificateCollection, ControllerData, PlannerData

__all__ = [
    "DEFAULT_GOAL",
    "compute_sweep_metrics",
    "resolve_circular_obstacles",
    "resolve_ellipsoidal_obstacles",
    "si_dynamics",
    "make_circular_cbf",
    "build_single_integrator_setup",
]

DEFAULT_GOAL = jnp.array([10.0, 10.0])


def si_dynamics(x):
    """Single-integrator dynamics: dx/dt = u."""
    return jnp.zeros(2), jnp.eye(2)


def make_circular_cbf(center, radius, alpha=1.0):
    """Create a circular CBF barrier tuple for a single obstacle."""
    return (
        lambda _t, x: jnp.sum((x - center) ** 2) - radius**2,
        lambda _t, x: 2 * (x - center),
        lambda _t, _x: 2 * jnp.eye(2),
        lambda _t, _x: 0.0,
        lambda h, _a=alpha: _a * h,
    )


def build_single_integrator_setup(
    seed: int,
    *,
    n_obstacles: int = 5,
    alpha: float = 1.0,
    control_limit: float = 5.0,
    relaxable_cbf: bool = False,
    dt: float = 0.01,
    num_steps: int = 200,
    goal: jnp.ndarray | None = None,
    obstacles: list[tuple] | None = None,
) -> MonteCarloSetup:
    """Build a ``MonteCarloSetup`` for a single-integrator with circular obstacles.

    Parameters
    ----------
    seed : int
        PRNG seed used to generate obstacle positions/radii.
    n_obstacles : int
        Number of random circular obstacles (ignored when *obstacles* is given).
    alpha : float
        CBF class-K function gain.
    control_limit : float
        Symmetric control input bound.
    relaxable_cbf : bool
        Whether CBF constraints are relaxable.
    dt : float
        Integration time step.
    num_steps : int
        Number of simulation steps.
    goal : array, optional
        Goal position (default ``[10, 10]``).
    obstacles : list of (center, radius) tuples, optional
        Explicit obstacle layout.  When *None*, obstacles are randomly
        generated from the seed.
    """
    if goal is None:
        goal = DEFAULT_GOAL

    if obstacles is not None:
        barrier_tuples = [
            make_circular_cbf(jnp.array(c), float(r), alpha=alpha)
            for c, r in obstacles
        ]
    else:
        key = random.PRNGKey(seed)
        key_c, key_r = random.split(key)
        centers = random.uniform(key_c, (n_obstacles, 2), minval=2.0, maxval=8.0)
        radii = random.uniform(key_r, (n_obstacles,), minval=0.5, maxval=1.0)
        barrier_tuples = [
            make_circular_cbf(centers[i], radii[i], alpha=alpha)
            for i in range(n_obstacles)
        ]
    barriers = CertificateCollection(*[list(x) for x in zip(*barrier_tuples)])

    controller = cbf_clf_qp_generator(
        generate_compute_zeroing_cbf_constraints,
        generate_compute_vanilla_clf_constraints,
    )(
        control_limits=jnp.array([control_limit, control_limit]),
        dynamics_func=si_dynamics,
        barriers=barriers,
        relaxable_cbf=relaxable_cbf,
        relaxable_clf=True,
    )

    _goal = goal

    def nominal_controller(t, x, _k, _r):
        return 2.0 * (_goal - x), None

    def initial_state_sampler(key):
        return random.uniform(key, (2,), minval=-1.0, maxval=1.0)

    # Prime controller data by calling controller once
    prime_key = random.PRNGKey(42)
    x0_probe = jnp.zeros(2)
    u_nom_d = jnp.zeros(2)
    _, c_data = controller(0.0, x0_probe, u_nom_d, prime_key, ControllerData())

    return MonteCarloSetup(
        dt=dt,
        num_steps=num_steps,
        dynamics=si_dynamics,
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
        planner_data=PlannerData(),
    )


def compute_sweep_metrics(
    results,
    stats,
    setup: MonteCarloSetup,
    *,
    goal: jnp.ndarray | None = None,
    position_slice: slice = slice(None),
) -> dict[str, float | int]:
    """Build the common result dict shared by single-integrator sweep scenarios.

    Parameters
    ----------
    results : MonteCarloResults
        Output of ``conduct_monte_carlo_gpu``.
    stats : SafetyStatistics
        Output of ``compute_safety_statistics``.
    setup : MonteCarloSetup
        The setup used for the simulation.
    goal : array, optional
        Goal position (default ``DEFAULT_GOAL``).
    position_slice : slice
        Slice to extract position components from state (default: all).
    """
    if goal is None:
        goal = DEFAULT_GOAL
    final_states = results.states[:, -1, position_slice]
    final_goal_distance = float(jnp.mean(jnp.linalg.norm(final_states - goal, axis=-1)))

    return {
        "success": 1,
        "safety_violations": int(stats.violation_rate > 0),
        "solver_failures": 0,
        "avg_step_ms": results.wall_time_s / setup.num_steps * 1000.0,
        "violation_rate": stats.violation_rate,
        "min_barrier_value": stats.min_barrier_value,
        "wall_time_s": results.wall_time_s,
        "final_goal_distance": final_goal_distance,
    }


# ---------------------------------------------------------------------------
# Obstacle resolution helpers
# ---------------------------------------------------------------------------


def resolve_circular_obstacles(
    params: dict,
) -> list[tuple[jnp.ndarray, float]] | None:
    """Resolve circular obstacle layout from params dict.

    Returns a list of ``(center_array, radius)`` tuples, or *None* if no
    obstacle config was injected into *params*.
    """
    count = params.get("_obstacle_count")
    if count is None:
        return None
    result = []
    for i in range(count):
        # Swept values use plain keys; fixed values use underscore-prefixed keys
        center = params.get(f"obstacle_{i}_center", params.get(f"_obstacle_{i}_center"))
        radius = params.get(f"obstacle_{i}_radius", params.get(f"_obstacle_{i}_radius"))
        if center is None or radius is None:
            raise ValueError(f"Obstacle {i} missing center or radius in params.")
        result.append((jnp.array(center), float(radius)))
    return result


def resolve_ellipsoidal_obstacles(
    params: dict,
) -> tuple[list[jnp.ndarray], list[jnp.ndarray]] | None:
    """Resolve ellipsoidal obstacle layout from params dict.

    Returns ``(centers, semi_axes_list)`` or *None* if no obstacle config
    was injected into *params*.
    """
    count = params.get("_obstacle_count")
    if count is None:
        return None
    centers = []
    semi_axes_list = []
    for i in range(count):
        center = params.get(f"obstacle_{i}_center", params.get(f"_obstacle_{i}_center"))
        semi_axes = params.get(
            f"obstacle_{i}_semi_axes", params.get(f"_obstacle_{i}_semi_axes")
        )
        if center is None or semi_axes is None:
            raise ValueError(f"Obstacle {i} missing center or semi_axes in params.")
        centers.append(jnp.array(center))
        semi_axes_list.append(jnp.array(semi_axes))
    return centers, semi_axes_list
