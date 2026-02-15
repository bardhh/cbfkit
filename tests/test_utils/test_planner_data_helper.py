import jax.numpy as jnp
from cbfkit.utils.user_types import PlannerData


def test_planner_data_from_constant():
    goal = jnp.array([1.0, 2.0])

    # Test with 1D array
    pd = PlannerData.from_constant(goal)
    assert pd.x_traj is not None
    assert pd.x_traj.shape == (2, 1)
    assert jnp.array_equal(pd.x_traj[:, 0], goal)

    # Test with 2D array (already column)
    goal_col = goal.reshape(-1, 1)
    pd_col = PlannerData.from_constant(goal_col)
    assert pd_col.x_traj is not None
    assert pd_col.x_traj.shape == (2, 1)
    assert jnp.array_equal(pd_col.x_traj, goal_col)


def test_planner_data_from_constant_list():
    goal_list = [1.0, 2.0]
    pd = PlannerData.from_constant(goal_list)  # type: ignore
    assert pd.x_traj is not None
    assert pd.x_traj.shape == (2, 1)
    assert jnp.array_equal(pd.x_traj[:, 0], jnp.array(goal_list))


def test_planner_data_broadcasting_simulation():
    """Verify that simulated broadcasting logic (clipping) works with the single-column trajectory."""
    # Mimic stepper logic
    x_traj = PlannerData.from_constant([1.0, 2.0]).x_traj
    assert x_traj is not None

    dt = 0.1
    t_vals = [0.0, 0.1, 1.0, 100.0]

    for t in t_vals:
        timestep_idx = jnp.round(t / dt).astype(int)
        timestep_idx = jnp.clip(timestep_idx, 0, x_traj.shape[1] - 1)
        ref = x_traj[:, timestep_idx]
        assert jnp.array_equal(ref, jnp.array([1.0, 2.0]))
