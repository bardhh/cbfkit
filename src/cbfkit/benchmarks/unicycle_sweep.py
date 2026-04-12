"""Unicycle obstacle avoidance sweep benchmark.

Registers the unicycle_obstacle_avoidance_sweep scenario for CLI use.
Sweeps over CBF alpha, controller gains (Kp_pos, Kp_theta), and barrier type.

Performance note
----------------
The batch runner uses a parameterized simulation kernel where ``alpha``,
``Kp_pos``, ``Kp_theta``, and ``control_limit`` are dynamic JAX inputs.
Barrier Lie derivatives are computed inline via ``jax.grad`` (autodiff),
avoiding the closure-based ``rectify_relative_degree`` pipeline that
forces JIT recompilation on every combo.  JAX compiles once on the first
combo and reuses the XLA kernel for all subsequent combos.
"""

from __future__ import annotations

import time
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax import lax, random

import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
from cbfkit.benchmarks.registry import register_sweepable_scenario
from cbfkit.benchmarks.scenario_builders import resolve_ellipsoidal_obstacles
from cbfkit.certificates import concatenate_certificates, rectify_relative_degree
from cbfkit.certificates.barrier_functions import ellipsoidal_barrier_factory
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.integration import forward_euler
from cbfkit.optimization.quadratic_program.qp_solver_jaxopt import (
    solve_with_details as _solve_qp,
)
from cbfkit.simulation.monte_carlo_gpu import (
    MonteCarloGPUResults,
    MonteCarloSetup,
    _default_estimator,
    _default_perturbation,
    _default_sensor,
    conduct_monte_carlo_gpu,
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


# ---------------------------------------------------------------------------
# Original setup builder (used by per-seed fallback runner)
# ---------------------------------------------------------------------------


def _build_setup(
    seed: int,
    alpha: float = 1.0,
    Kp_pos: float = 1.0,
    Kp_theta: float = 5.0,
    barrier_type: str = "linear_class_k",
    control_limit: float = 100.0,
    obstacles: list | None = None,
    ellipsoids: list | None = None,
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

    obs_list = obstacles if obstacles is not None else OBSTACLES
    ell_list = ellipsoids if ellipsoids is not None else ELLIPSOIDS

    # Build barriers using rectify_relative_degree (handles high relative degree)
    barriers_list = [
        rectify_relative_degree(
            function=_cbf(obs, ell),
            system_dynamics=dynamics,
            state_dim=STATE_DIM,
            form="high-order",
            certificate_conditions=barrier_fn(alpha),
        )
        for obs, ell in zip(obs_list, ell_list)
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


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _compute_metrics(xs, psis, wall_time, n_trials):
    """Build result dict from raw simulation arrays."""
    platform = jax.devices()[0].platform
    final_states = xs[:, -1, :]
    final_goal_distance = float(jnp.mean(jnp.linalg.norm(final_states[:, :2] - GOAL[:2], axis=-1)))
    distances = jnp.linalg.norm(xs[:, :, :2] - GOAL[:2], axis=-1)
    reached = distances < GOAL_TOL
    step_indices = jnp.arange(reached.shape[1])
    first_arrival = jnp.where(reached, step_indices[None, :], reached.shape[1])
    first_arrival_step = jnp.min(first_arrival, axis=1)
    time_to_goal = float(jnp.mean(first_arrival_step)) * DT

    # Safety stats from barrier values
    min_barrier = float(jnp.min(psis))
    violated_steps = jnp.any(psis < -1e-4, axis=-1)  # (n_trials, num_steps)
    per_trial_violated = jnp.any(violated_steps, axis=1)
    violation_rate = float(jnp.mean(per_trial_violated))

    return {
        "platform": platform,
        "n_trials": n_trials,
        "wall_time_s": wall_time,
        "trials_per_sec": n_trials / wall_time if wall_time > 0 else 0.0,
        "violation_rate": violation_rate,
        "min_barrier_value": min_barrier,
        "success": 1,
        "safety_violations": int(violation_rate > 0),
        "solver_failures": 0,
        "avg_step_ms": wall_time / NUM_STEPS * 1000.0,
        "final_goal_distance": final_goal_distance,
        "time_to_goal": time_to_goal,
    }


def _process_results(results: MonteCarloGPUResults) -> dict:
    """Extract metrics from a single seed's Monte Carlo results (legacy path)."""
    stats = compute_safety_statistics(results)
    platform = jax.devices()[0].platform
    final_states = results.states[:, -1, :]
    final_goal_distance = float(jnp.mean(jnp.linalg.norm(final_states[:, :2] - GOAL[:2], axis=-1)))

    distances = jnp.linalg.norm(results.states[:, :, :2] - GOAL[:2], axis=-1)
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


# ---------------------------------------------------------------------------
# Optimised sweep path: compile once, run many combos
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _get_unicycle_sim_fn(n_obstacles: int, num_steps: int):
    """Build and cache a JIT-compiled unicycle simulation function.

    Barrier Lie derivatives are computed inline via ``jax.grad``.
    All sweep parameters (alpha, Kp_pos, Kp_theta, control_limit,
    barrier_type_flag) are dynamic JAX inputs — the compiled XLA kernel
    is reused across all combos with the same obstacle count.
    """

    def _simulate(
        key,
        x0,
        alpha,
        Kp_pos,
        Kp_theta,
        control_limit,
        barrier_type_flag,
        obs_centers,
        obs_semi_axes,
        goal,
        dt,
    ):
        # ── Pure functions (no captured sweep params) ────────────

        def _dynamics(x):
            v, theta = x[2], x[3]
            f = jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), 0.0, 0.0])
            g = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
            return f, g

        def _barrier(state, center, semi_axes):
            diff = state[:2] - center
            return jnp.sum((diff / semi_axes) ** 2) - 1.0

        def _rectified_barrier(state, center, semi_axes):
            """High-order CBF: psi = Lf(h) + h."""
            h = _barrier(state, center, semi_axes)
            grad_h = jax.grad(_barrier)(state, center, semi_axes)
            f_x, _ = _dynamics(state)
            return grad_h @ f_x + h

        # ── Simulation step ──────────────────────────────────────

        def _step(carry, _step_idx):
            key, x = carry
            key, _ = random.split(key)
            f_x, g_x = _dynamics(x)

            # Nominal controller (proportional)
            err = goal[:2] - x[:2]
            theta_d = jnp.arctan2(err[1], err[0])
            dist = jnp.linalg.norm(err)
            theta_err = jnp.mod((theta_d - x[3]) + jnp.pi, 2 * jnp.pi) - jnp.pi
            v_d = jnp.minimum(Kp_pos * dist, 2.0)
            u_nom = jnp.array([Kp_pos * (v_d - x[2]), Kp_theta * theta_err])

            # CBF constraints (vmapped over obstacles)
            def _single_cbf(center, semi_axes):
                psi = _rectified_barrier(x, center, semi_axes)
                grad_psi = jax.grad(_rectified_barrier)(x, center, semi_axes)
                Lf_psi = grad_psi @ f_x
                Lg_psi = grad_psi @ g_x  # (2,)
                alpha_psi = lax.cond(
                    barrier_type_flag == 0,
                    lambda p: alpha * p,
                    lambda p: alpha * p**3,
                    psi,
                )
                return -Lg_psi, Lf_psi + alpha_psi, psi

            a_rows, b_vals, psi_vals = jax.vmap(_single_cbf)(
                obs_centers,
                obs_semi_axes,
            )

            # Assemble QP: min ||u - u_nom||^2  s.t.  G u <= h_vec
            I2 = jnp.eye(2)
            G = jnp.concatenate([a_rows, I2, -I2], axis=0)
            h_vec = jnp.concatenate([b_vals, control_limit * jnp.ones(4)])
            sol = _solve_qp(2.0 * I2, -2.0 * u_nom, G, h_vec)
            u = sol.primal

            # Forward Euler
            x_next = x + dt * (f_x + g_x @ u)
            return (key, x_next), (x_next, u, psi_vals)

        _, (xs, us, psis) = lax.scan(_step, (key, x0), jnp.arange(num_steps))
        return xs, us, psis

    # vmap over (key, x0); sweep params broadcast
    return jax.jit(
        jax.vmap(
            _simulate,
            in_axes=(0, 0, None, None, None, None, None, None, None, None, None),
        )
    )


def _unicycle_batch_runner(seeds: list[int], params: dict) -> list[dict]:
    """Optimised batch runner — one JIT compilation across all combos.

    The first combo triggers JIT compilation of a parameterized kernel.
    Subsequent combos with different alpha / Kp / control_limit values
    reuse the compiled kernel (same function object + same input shapes).
    """
    alpha = float(params.get("alpha", 1.0))
    Kp_pos = float(params.get("Kp_pos", 1.0))
    Kp_theta = float(params.get("Kp_theta", 5.0))
    barrier_type = params.get("barrier_type", "linear_class_k")
    control_limit = float(params.get("control_limit", 100.0))
    resolved = resolve_ellipsoidal_obstacles(params)

    # Obstacle arrays
    if resolved is not None:
        obs_centers = jnp.stack([c[:2] for c in resolved[0]])
        obs_semi_axes = jnp.stack(resolved[1])
    else:
        obs_centers = jnp.stack([o[:2] for o in OBSTACLES])
        obs_semi_axes = jnp.stack(ELLIPSOIDS)

    n_obstacles = obs_centers.shape[0]
    barrier_type_flag = 0 if barrier_type == "linear_class_k" else 1

    sim_fn = _get_unicycle_sim_fn(n_obstacles, NUM_STEPS)

    # Generate keys and initial states for all seeds × trials
    n_seeds = len(seeds)
    total = n_seeds * N_TRIALS
    all_keys = []
    all_sampler_keys = []
    for seed in seeds:
        mk = random.PRNGKey(seed)
        all_keys.append(random.split(mk, N_TRIALS))
        all_sampler_keys.append(random.split(random.fold_in(mk, 1), N_TRIALS))

    keys = jnp.concatenate(all_keys, axis=0)  # (total, 2)
    sampler_keys = jnp.concatenate(all_sampler_keys, axis=0)

    def _sample_x0(k):
        k1, k2 = random.split(k)
        pos = random.uniform(k1, (2,), minval=-1.0, maxval=1.0)
        theta = random.uniform(k2, (), minval=-jnp.pi, maxval=jnp.pi)
        return jnp.array([pos[0], pos[1], 0.0, theta])

    x0s = jax.vmap(_sample_x0)(sampler_keys)  # (total, 4)

    # Run — first combo compiles, subsequent combos reuse
    start = time.perf_counter()
    xs, us, psis = sim_fn(
        keys,
        x0s,
        alpha,
        Kp_pos,
        Kp_theta,
        control_limit,
        barrier_type_flag,
        obs_centers,
        obs_semi_axes,
        GOAL,
        DT,
    )
    jax.block_until_ready(xs)
    wall_time = time.perf_counter() - start

    # Split into per-seed results
    per_seed_time = wall_time / n_seeds
    results = []
    for i in range(n_seeds):
        sl = slice(i * N_TRIALS, (i + 1) * N_TRIALS)
        results.append(_compute_metrics(xs[sl], psis[sl], per_seed_time, N_TRIALS))
    return results


# ---------------------------------------------------------------------------
# Scenario registration
# ---------------------------------------------------------------------------


@register_sweepable_scenario(
    "unicycle_obstacle_avoidance_sweep",
    sweepable_params=["alpha", "Kp_pos", "Kp_theta", "barrier_type"],
    description="Unicycle obstacle avoidance with sweepable CBF alpha, controller gains, and barrier type.",
    batch_runner=_unicycle_batch_runner,
)
def unicycle_sweep(seed: int, params: dict) -> dict:
    """Per-seed runner (fallback for falsifier mode)."""
    alpha = params.get("alpha", 1.0)
    Kp_pos = params.get("Kp_pos", 1.0)
    Kp_theta = params.get("Kp_theta", 5.0)
    barrier_type = params.get("barrier_type", "linear_class_k")
    resolved = resolve_ellipsoidal_obstacles(params)
    obs_kw = {} if resolved is None else {"obstacles": resolved[0], "ellipsoids": resolved[1]}

    setup = _build_setup(
        seed,
        alpha=alpha,
        Kp_pos=Kp_pos,
        Kp_theta=Kp_theta,
        barrier_type=barrier_type,
        **obs_kw,
    )

    results = conduct_monte_carlo_gpu(setup, n_trials=N_TRIALS, seed=seed)
    return _process_results(results)
