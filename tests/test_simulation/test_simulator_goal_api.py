
import jax.numpy as jnp
import pytest
from cbfkit.simulation import simulator
from cbfkit.utils.user_types import PlannerData

def test_goal_sets_xtraj():
    """Test that passing 'goal' correctly populates planner_data.x_traj."""
    goal = jnp.array([1.0, 2.0, 3.0])

    # Run execute with just goal (mocking other required args with minimal values)
    # We don't need to run full simulation, just check the argument processing.
    # However, 'execute' runs the simulation. We can use a dummy dynamics/integrator
    # and 0 steps to just check initialization if we could inspect it, but execute returns results.
    # Instead, we can't easily inspect internal state of 'execute' without running it.
    # But we can verify that it runs without error and produces a result where
    # the planner data might be reflected if we had a planner.

    # Better approach: We can check the logic by calling execute with a dummy system
    # and checking if it raises error or runs.
    # But to verify x_traj is set, we might need to rely on the fact that if we provide
    # a nominal controller that uses x_des, it receives the goal.

    # Let's mock a nominal controller that asserts the reference state matches goal.

    goal_arr = jnp.array([10.0, 20.0])

    def dynamics(x):
        return jnp.zeros_like(x), jnp.zeros((x.shape[0], 1))

    def integrator(x, f, dt):
        return x

    def nominal_controller(t, x, key, ref):
        # ref should match goal (reshaped)
        assert ref is not None
        # ref comes from planner_data.x_traj slicing
        # Simulation logic: x_des = traj[:, idx]
        # Since traj is constant, ref should be goal
        assert jnp.allclose(ref.flatten(), goal_arr)
        return jnp.zeros(1), {}

    # 1. Test with goal only
    simulator.execute(
        x0=jnp.zeros(2),
        dt=0.1,
        num_steps=1,
        dynamics=dynamics,
        integrator=integrator,
        nominal_controller=nominal_controller,
        goal=goal_arr
    )

def test_goal_with_empty_planner_data():
    """Test goal works when planner_data is provided but empty."""
    goal_arr = jnp.array([5.0])

    def dynamics(x):
        return jnp.zeros_like(x), jnp.zeros((x.shape[0], 1))

    def integrator(x, f, dt):
        return x

    def nominal_controller(t, x, key, ref):
        assert jnp.allclose(ref.flatten(), goal_arr)
        return jnp.zeros(1), {}

    simulator.execute(
        x0=jnp.zeros(1),
        dt=0.1,
        num_steps=1,
        dynamics=dynamics,
        integrator=integrator,
        nominal_controller=nominal_controller,
        goal=goal_arr,
        planner_data=PlannerData() # Empty
    )

    # Also test with dict
    simulator.execute(
        x0=jnp.zeros(1),
        dt=0.1,
        num_steps=1,
        dynamics=dynamics,
        integrator=integrator,
        nominal_controller=nominal_controller,
        goal=goal_arr,
        planner_data={} # Empty dict
    )

def test_goal_conflict_raises_error():
    """Test that providing both goal and x_traj raises ValueError."""
    goal_arr = jnp.array([1.0])
    traj = jnp.array([[2.0]])

    def dynamics(x): return jnp.zeros_like(x), jnp.zeros((x.shape[0], 1))
    def integrator(x, f, dt): return x

    # Case 1: PlannerData object
    with pytest.raises(ValueError, match="Cannot specify both 'goal' and 'planner_data.x_traj'"):
        simulator.execute(
            x0=jnp.zeros(1),
            dt=0.1,
            num_steps=1,
            dynamics=dynamics,
            integrator=integrator,
            goal=goal_arr,
            planner_data=PlannerData(x_traj=traj)
        )

    # Case 2: Dict
    with pytest.raises(ValueError, match="Cannot specify both 'goal' and 'planner_data"):
        simulator.execute(
            x0=jnp.zeros(1),
            dt=0.1,
            num_steps=1,
            dynamics=dynamics,
            integrator=integrator,
            goal=goal_arr,
            planner_data={"x_traj": traj}
        )
