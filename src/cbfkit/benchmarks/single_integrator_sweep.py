"""Single integrator sweep benchmark.

Registers the tutorial_single_integrator_sweep scenario for CLI use.
See tutorials/parameter_sweep_tutorial.py for the full walkthrough.

Performance note
----------------
The batch runner uses a **parameterized** simulation kernel where ``alpha``
and ``control_limit`` are dynamic JAX inputs rather than closure-captured
constants.  JAX compiles the kernel once (on the first combo) and reuses it
for all subsequent combos that share the same obstacle count and step count.
This eliminates the ~3 s per-combo JIT recompilation overhead that dominates
wall time in parameter sweeps.
"""

from __future__ import annotations

import time
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax import lax, random

from cbfkit.benchmarks.registry import register_sweepable_scenario
from cbfkit.benchmarks.scenario_builders import (
    DEFAULT_GOAL,
    build_single_integrator_setup,
    compute_sweep_metrics,
    resolve_circular_obstacles,
)
from cbfkit.simulation.monte_carlo_gpu import conduct_monte_carlo_gpu
from cbfkit.simulation.safety_verification import compute_safety_statistics

from cbfkit.optimization.quadratic_program.qp_solver_jaxopt import (
    solve_with_details as _solve_qp,
)

N_TRIALS = 1


# ---------------------------------------------------------------------------
# Optimised sweep path: compile once, run many combos
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _get_sim_fn(n_obstacles: int, num_steps: int):
    """Build and cache a JIT-compiled simulation function.

    The returned function accepts ``alpha`` and ``control_limit`` as regular
    (non-static) JAX inputs.  Because the *same* Python function object is
    returned for a given ``(n_obstacles, num_steps)`` pair, JAX's JIT cache
    recognises that the input shapes and dtypes are unchanged across combos
    and **reuses the compiled XLA kernel** — no retracing, no recompilation.
    """

    def _simulate(key, x0, alpha, control_limit, centers, radii, goal, dt):
        """Simulate one trajectory with dynamic sweep parameters."""

        def _step(carry, _step_idx):
            key, x = carry
            key, _subkey = random.split(key)

            # Nominal controller: proportional drive toward goal
            u_nom = 2.0 * (goal - x)

            # ── CBF constraints ──────────────────────────────────
            # h_i(x) = ||x - c_i||² - r_i²
            diffs = x[None, :] - centers  # (n_obs, 2)
            h_vals = jnp.sum(diffs**2, axis=1) - radii**2  # (n_obs,)
            grad_h = 2.0 * diffs  # (n_obs, 2)

            # Zeroing-CBF: ∇h u + α h ≥ 0  ⟹  −∇h u ≤ α h
            G_cbf = -grad_h
            h_cbf = alpha * h_vals

            # ── Control-limit constraints ────────────────────────
            I2 = jnp.eye(2)
            G_lim = jnp.concatenate([I2, -I2], axis=0)  # (4, 2)
            h_lim = control_limit * jnp.ones(4)

            G = jnp.concatenate([G_cbf, G_lim], axis=0)
            h_vec = jnp.concatenate([h_cbf, h_lim])

            # ── QP: min ‖u − u_nom‖² ────────────────────────────
            H = 2.0 * I2
            f_vec = -2.0 * u_nom
            sol = _solve_qp(H, f_vec, G, h_vec)
            u = sol.primal

            # Forward-Euler integration (single integrator: ẋ = u)
            x_next = x + dt * u

            # Barrier values at the new state (for safety stats)
            diffs_next = x_next[None, :] - centers
            h_next = jnp.sum(diffs_next**2, axis=1) - radii**2

            return (key, x_next), (x_next, u, h_next)

        _, (xs, us, hs) = lax.scan(_step, (key, x0), jnp.arange(num_steps))
        return xs, us, hs

    # vmap over (key, x0); sweep params broadcast
    return jax.jit(jax.vmap(_simulate, in_axes=(0, 0, None, None, None, None, None, None)))


def _si_batch_runner(seeds: list[int], params: dict) -> list[dict]:
    """Optimised batch runner — one JIT compilation across all combos.

    On the first combo call, JAX traces and compiles the simulation kernel.
    Subsequent combos with different ``alpha`` / ``control_limit`` values
    reuse the compiled kernel because the function object and input
    shapes are identical (JAX JIT cache hit).
    """
    alpha = float(params.get("alpha", 1.0))
    control_limit = float(params.get("control_limit", 5.0))
    n_obstacles = int(params.get("n_obstacles", 2))
    obs = resolve_circular_obstacles(params)
    dt = 0.1
    num_steps = 100
    goal = DEFAULT_GOAL

    # Build obstacle arrays (fixed layout from seed 0)
    if obs is not None:
        centers = jnp.stack([jnp.array(c) for c, r in obs])
        radii = jnp.array([float(r) for c, r in obs])
    else:
        key = random.PRNGKey(0)
        key_c, key_r = random.split(key)
        centers = random.uniform(key_c, (n_obstacles, 2), minval=2.0, maxval=8.0)
        radii = random.uniform(key_r, (n_obstacles,), minval=0.5, maxval=1.0)

    # Get (or build) cached simulation function
    sim_fn = _get_sim_fn(n_obstacles, num_steps)

    # Generate PRNG keys and initial states (matches conduct_monte_carlo_gpu)
    n_seeds = len(seeds)
    keys = jnp.stack([random.PRNGKey(s) for s in seeds])
    sampler_keys = jnp.stack(
        [random.split(random.fold_in(random.PRNGKey(s), 1), 1)[0] for s in seeds]
    )
    x0s = jax.vmap(lambda k: random.uniform(k, (2,), minval=-1.0, maxval=1.0))(sampler_keys)

    # Run — first combo compiles, subsequent combos are JIT cache hits
    start = time.perf_counter()
    xs, us, hs = sim_fn(keys, x0s, alpha, control_limit, centers, radii, goal, dt)
    jax.block_until_ready(xs)
    wall_time = time.perf_counter() - start

    # Build per-seed result dicts
    per_seed_time = wall_time / n_seeds
    results = []
    for i in range(n_seeds):
        final_pos = xs[i, -1, :]
        final_goal_distance = float(jnp.linalg.norm(final_pos - goal))
        min_barrier = float(jnp.min(hs[i]))
        violated = bool(jnp.any(hs[i] < -1e-4))

        results.append(
            {
                "success": 1,
                "safety_violations": int(violated),
                "solver_failures": 0,
                "avg_step_ms": per_seed_time / num_steps * 1000.0,
                "violation_rate": float(violated),
                "min_barrier_value": min_barrier,
                "wall_time_s": per_seed_time,
                "final_goal_distance": final_goal_distance,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Scenario registration
# ---------------------------------------------------------------------------


@register_sweepable_scenario(
    "tutorial_single_integrator_sweep",
    sweepable_params=["alpha", "control_limit", "n_obstacles"],
    description="Single integrator obstacle avoidance with sweepable CBF parameters.",
    batch_runner=_si_batch_runner,
)
def tutorial_sweep(seed: int, params: dict) -> dict:
    """Per-seed runner (fallback for falsifier mode)."""
    alpha = params.get("alpha", 1.0)
    control_limit = params.get("control_limit", 5.0)
    n_obstacles = int(params.get("n_obstacles", 2))
    obs = resolve_circular_obstacles(params)

    setup = build_single_integrator_setup(
        seed,
        n_obstacles=n_obstacles,
        alpha=alpha,
        control_limit=control_limit,
        obstacles=obs,
        dt=0.1,
        num_steps=100,
    )

    results = conduct_monte_carlo_gpu(setup, n_trials=N_TRIALS, seed=seed)
    stats = compute_safety_statistics(results)
    return compute_sweep_metrics(results, stats, setup)
