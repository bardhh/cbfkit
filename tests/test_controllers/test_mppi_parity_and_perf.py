import time

import jax
import jax.numpy as jnp
import pytest

from cbfkit.controllers.mppi.mppi_generator import mppi_generator
from cbfkit.utils.user_types import PlannerData


def _dynamics(x):
    return jnp.zeros_like(x), jnp.eye(2)


def _stage_cost(x, u):
    goal = jnp.array([0.5, -0.25])
    return jnp.sum((x - goal) ** 2) + 0.05 * jnp.sum(u**2)


def _terminal_cost(x):
    goal = jnp.array([0.5, -0.25])
    return jnp.sum((x - goal) ** 2)


def _trajectory_cost(_t, state_traj, control_traj, prev_robustness):
    prev = 0.0 if prev_robustness is None else jnp.sum(prev_robustness)
    return 0.1 * jnp.sum(state_traj**2) + 0.01 * jnp.sum(control_traj**2) - prev


def _make_controller(use_trajectory_cost: bool):
    args = {
        "robot_state_dim": 2,
        "robot_control_dim": 2,
        "prediction_horizon": 10,
        "num_samples": 128,
        "plot_samples": 0,
        "time_step": 0.05,
        "use_GPU": False,
        "costs_lambda": 0.1,
        "cost_perturbation": 0.2,
    }
    gen = mppi_generator()
    controller = gen(
        control_limits=jnp.array([2.0, 2.0]),
        dynamics_func=_dynamics,
        stage_cost=None if use_trajectory_cost else _stage_cost,
        terminal_cost=None if use_trajectory_cost else _terminal_cost,
        trajectory_cost=_trajectory_cost if use_trajectory_cost else None,
        mppi_args=args,
    )
    return controller, args


@pytest.mark.parametrize("use_trajectory_cost", [False, True])
def test_mppi_deterministic_for_fixed_key(use_trajectory_cost: bool):
    controller, args = _make_controller(use_trajectory_cost=use_trajectory_cost)

    x0 = jnp.array([1.0, -0.25])
    t0 = 0.0
    key = jax.random.PRNGKey(7)
    planner_data = PlannerData(
        u_traj=jnp.zeros((args["prediction_horizon"], args["robot_control_dim"])),
        prev_robustness=jnp.zeros((1,)) if use_trajectory_cost else None,
    )

    u1, d1 = controller(t0, x0, None, key, planner_data)
    u2, d2 = controller(t0, x0, None, key, planner_data)

    assert jnp.allclose(u1, u2)
    assert d1.u_traj is not None and d2.u_traj is not None
    assert jnp.allclose(d1.u_traj, d2.u_traj)


def test_mppi_runtime_budget_regression_guard():
    """Coarse runtime guard to catch major MPPI regressions in CPU CI."""
    controller, args = _make_controller(use_trajectory_cost=False)

    x0 = jnp.array([0.25, -0.75])
    t0 = 0.0
    key = jax.random.PRNGKey(0)
    planner_data = PlannerData(
        u_traj=jnp.zeros((args["prediction_horizon"], args["robot_control_dim"])),
    )

    # Warm-up compile
    u_warm, d_warm = controller(t0, x0, None, key, planner_data)
    u_warm.block_until_ready()
    assert d_warm.u_traj is not None
    d_warm.u_traj.block_until_ready()

    n_iters = 8
    start = time.perf_counter()
    for _ in range(n_iters):
        key, subkey = jax.random.split(key)
        u, d = controller(t0, x0, None, subkey, planner_data)
        u.block_until_ready()
        assert d.u_traj is not None
        d.u_traj.block_until_ready()
    avg_seconds = (time.perf_counter() - start) / n_iters

    # Wide threshold to avoid flakiness while still catching catastrophic slowdowns.
    assert avg_seconds < 0.25, f"MPPI average step time too high: {avg_seconds:.3f}s"
