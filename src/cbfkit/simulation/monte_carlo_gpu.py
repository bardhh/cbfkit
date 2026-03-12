"""GPU-accelerated Monte Carlo simulation via JAX vmap.

Runs N trajectories in parallel on a single device (CPU or GPU) by
vectorizing the JIT-compiled simulation loop with ``jax.vmap``.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax, random

from cbfkit.simulation.simulator_jit import _make_scan_step
from cbfkit.utils.user_types import (
    ControllerCallable,
    ControllerData,
    Covariance,
    DynamicsCallable,
    EstimatorCallable,
    IntegratorCallable,
    NominalControllerCallable,
    PerturbationCallable,
    PlannerCallable,
    PlannerData,
    SensorCallable,
    State,
)


class MonteCarloSetup(NamedTuple):
    """Configuration for GPU-accelerated Monte Carlo simulation."""

    dt: float
    num_steps: int
    dynamics: DynamicsCallable
    integrator: IntegratorCallable
    initial_state_sampler: Callable[[jax.Array], State]
    nominal_controller: Optional[NominalControllerCallable]
    controller: Optional[ControllerCallable]
    sensor: SensorCallable
    estimator: EstimatorCallable
    perturbation: PerturbationCallable
    sigma: jax.Array
    controller_data: ControllerData
    planner: Optional[PlannerCallable] = None
    planner_data: PlannerData = PlannerData()


class MonteCarloGPUResults(NamedTuple):
    """Results from GPU-accelerated Monte Carlo simulation."""

    states: jax.Array  # (n_trials, num_steps, state_dim)
    controls: jax.Array  # (n_trials, num_steps, control_dim)
    controller_datas: Any  # batched NamedTuple pytree
    planner_datas: Any  # batched NamedTuple pytree
    wall_time_s: float
    n_trials: int


def _default_sensor(t, x, *, sigma=None, key=None):
    return x


def _default_estimator(t, y, z, u, c):
    return y, c if c is not None else jnp.zeros((len(y), len(y)))


def _default_perturbation(x, u, f, g):
    def p(key):
        return jnp.zeros_like(x)

    return p


def conduct_monte_carlo_gpu(
    setup: MonteCarloSetup,
    n_trials: int,
    seed: int = 0,
) -> MonteCarloGPUResults:
    """Run Monte Carlo simulation in parallel via ``jax.vmap``.

    Each trial gets a unique PRNG key and initial state (sampled via
    ``setup.initial_state_sampler``).  All trajectories execute as a
    single vectorized kernel, achieving high GPU utilization.

    Args:
        setup: Simulation components and parameters.
        n_trials: Number of parallel trajectories.
        seed: Base PRNG seed.

    Returns:
        ``MonteCarloGPUResults`` with batched trajectory data and timing.
    """
    master_key = random.PRNGKey(seed)
    keys = random.split(master_key, n_trials)

    # Sample initial states for all trials
    sampler_keys = random.split(random.fold_in(master_key, 1), n_trials)
    initial_states = jax.vmap(setup.initial_state_sampler)(sampler_keys)

    # Build the vmap-safe scan step (no debug callbacks)
    scan_step = _make_scan_step(
        dynamics=setup.dynamics,
        integrator=setup.integrator,
        planner=setup.planner,
        nominal_controller=setup.nominal_controller,
        controller=setup.controller,
        sensor=setup.sensor,
        estimator=setup.estimator,
        perturbation=setup.perturbation,
        sigma=setup.sigma,
        dt=setup.dt,
        num_steps=setup.num_steps,
        enable_debug=False,
        progress_callback=None,
        progress_interval=1,
    )

    # Probe dimensions from dynamics
    probe_state = initial_states[0]
    _, g_probe = setup.dynamics(probe_state)
    control_dim = g_probe.shape[1]
    state_dim = probe_state.shape[0]

    # Build initial covariance
    c0 = jnp.zeros((state_dim, state_dim))

    def single_trajectory(key, x0):
        u0 = jnp.zeros((control_dim,))
        z0 = x0

        carry_init = (
            key,
            0.0,  # t=0
            x0,
            u0,
            z0,
            c0,
            setup.controller_data,
            setup.planner_data,
        )

        _final_carry, trajectory = lax.scan(
            scan_step, carry_init, jnp.arange(setup.num_steps)
        )
        return trajectory

    # JIT-compile the vmapped function
    batched_fn = jax.jit(jax.vmap(single_trajectory, in_axes=(0, 0)))

    # Warmup: compile without timing
    _ = batched_fn(keys, initial_states)
    # Block until compilation + execution finishes
    jax.block_until_ready(_)

    # Timed run
    start = time.perf_counter()
    trajectory = batched_fn(keys, initial_states)
    # Force synchronization for accurate timing
    jax.block_until_ready(trajectory)
    wall_time = time.perf_counter() - start

    xs, us, zs, cs, c_datas, p_datas = trajectory

    return MonteCarloGPUResults(
        states=xs,
        controls=us,
        controller_datas=c_datas,
        planner_datas=p_datas,
        wall_time_s=wall_time,
        n_trials=n_trials,
    )
