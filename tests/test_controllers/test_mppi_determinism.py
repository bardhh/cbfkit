import jax
import jax.numpy as jnp
import pytest
from cbfkit.controllers.mppi.mppi_generator import mppi_generator
from cbfkit.utils.user_types import PlannerData


# Simple 1D dynamics: x_dot = u
def dynamics(x):
    # x is (1,), returns f(x), g(x)
    # f(x) = 0, g(x) = 1 (identity matrix of size 1x1)
    # resulting in x_dot = 0 + 1*u = u
    return jnp.zeros_like(x), jnp.eye(1)


# Stage cost: x^2 + u^2
def stage_cost(x, u):
    return jnp.sum(x**2) + jnp.sum(u**2)


# Terminal cost: x^2
def terminal_cost(x):
    return jnp.sum(x**2)


def test_mppi_determinism():
    """
    Verifies that the MPPI controller is:
    1. Deterministic given a fixed JAX PRNGKey (Regression prevention for stochastic logic).
    2. Sensitive to changes in the PRNGKey (Ensures randomness is actually used).
    """
    # Setup
    key_seed = 42

    mppi_args = {
        "robot_state_dim": 1,
        "robot_control_dim": 1,
        "prediction_horizon": 10,
        "num_samples": 100,
        "plot_samples": 0,
        "time_step": 0.1,
        "use_GPU": False,  # CPU for deterministic testing is usually safer/standard
        "costs_lambda": 1.0,
        "cost_perturbation": 1.0,
    }

    # Generate controller
    # control_limits needs to be an array. Based on mppi_generator, it expects shape (dim,)
    control_limits = jnp.array([10.0])

    generator = mppi_generator()
    controller = generator(
        control_limits=control_limits,
        dynamics_func=dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        trajectory_cost=None,
        mppi_args=mppi_args,
    )

    # Initial state
    t0 = 0.0
    x0 = jnp.array([1.0])

    # Initialize PlannerData with warm start (u_traj)
    # u_traj shape: (horizon, control_dim)
    u_traj_init = jnp.zeros((mppi_args["prediction_horizon"], mppi_args["robot_control_dim"]))
    data_init = PlannerData(u_traj=u_traj_init)

    # Run 1: Key A
    key1 = jax.random.PRNGKey(key_seed)
    u1, data1 = controller(t0, x0, None, key1, data_init)

    # Run 2: Key A (Same)
    # Note: We must pass data_init again, not data1, to ensure identical start state
    u2, data2 = controller(t0, x0, None, key1, data_init)

    # Run 3: Key B (Different)
    key2 = jax.random.PRNGKey(key_seed + 1)
    u3, data3 = controller(t0, x0, None, key2, data_init)

    # Assertions

    # 1. Determinism: Same Key -> Same Result
    # We check both the immediate control (u) and the updated plan (u_traj)
    assert jnp.allclose(u1, u2), "MPPI Output (u) must be identical for same key"
    assert jnp.allclose(
        data1.u_traj, data2.u_traj
    ), "MPPI Trajectory must be identical for same key"

    # 2. Sensitivity: Diff Key -> Diff Result
    # With 100 samples and perturbation=1.0, divergence is statistically guaranteed
    assert not jnp.allclose(u1, u3), "MPPI Output (u) should differ for different keys"
    assert not jnp.allclose(
        data1.u_traj, data3.u_traj
    ), "MPPI Trajectory should differ for different keys"


if __name__ == "__main__":
    test_mppi_determinism()


def test_mppi_control_std_scales_the_sampled_perturbations():
    """``control_std`` sets the sampling std (default 2.0 keeps the historical covariance 4 I):
    with a small std the sampled trajectories stay close to the warm start; with the default
    and a wide bound they spread out."""
    base = {
        "robot_state_dim": 1,
        "robot_control_dim": 1,
        "prediction_horizon": 10,
        "num_samples": 200,
        "time_step": 0.1,
        "use_GPU": False,
        "costs_lambda": 1.0,
        "cost_perturbation": 0.0,
    }
    spreads = {}
    for std in (0.05, 2.0):
        ctrl = mppi_generator()(
            control_limits=jnp.array([10.0]),
            dynamics_func=dynamics,
            stage_cost=stage_cost,
            terminal_cost=terminal_cost,
            mppi_args={**base, "control_std": std},
        )
        _, d = ctrl(
            0.0,
            jnp.array([1.0]),
            None,
            jax.random.PRNGKey(0),
            PlannerData(u_traj=jnp.zeros((10, 1))),
        )
        # sampled_x_traj: (state_dim * samples, horizon) -- spread of the first-step states
        spreads[std] = float(jnp.std(d.sampled_x_traj[:, 1]))
    assert spreads[0.05] < 0.1 * spreads[2.0]
    # the default equals the historical behaviour
    ctrl_default = mppi_generator()(
        control_limits=jnp.array([10.0]),
        dynamics_func=dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        mppi_args=base,
    )
    _, d_def = ctrl_default(
        0.0, jnp.array([1.0]), None, jax.random.PRNGKey(0), PlannerData(u_traj=jnp.zeros((10, 1)))
    )
    ctrl_two = mppi_generator()(
        control_limits=jnp.array([10.0]),
        dynamics_func=dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        mppi_args={**base, "control_std": 2.0},
    )
    _, d_two = ctrl_two(
        0.0, jnp.array([1.0]), None, jax.random.PRNGKey(0), PlannerData(u_traj=jnp.zeros((10, 1)))
    )
    assert jnp.allclose(d_def.u_traj, d_two.u_traj)
