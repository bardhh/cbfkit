
import pytest
from cbfkit.controllers.mppi.mppi_source import setup_mppi

def test_mppi_setup_missing_dynamics():
    """Test that setup_mppi raises ValueError when dynamics function is missing."""
    with pytest.raises(ValueError, match="Dynamics function must be provided"):
        setup_mppi(dyn_func=None)

def test_mppi_setup_missing_costs():
    """Test that setup_mppi raises ValueError when costs are missing."""
    def dummy_dynamics(x):
        return x, x

    # Case 1: No trajectory cost and no stage cost
    with pytest.raises(ValueError, match="Either Trajectory Cost or Stage Cost function must be provided"):
        setup_mppi(
            dyn_func=dummy_dynamics,
            trajectory_cost_func=None,
            stage_cost_func=None,
            terminal_cost_func=lambda x: 0.0
        )

    # Case 2: No trajectory cost and no terminal cost
    with pytest.raises(ValueError, match="Either Trajectory Cost or Terminal Cost function must be provided"):
        setup_mppi(
            dyn_func=dummy_dynamics,
            trajectory_cost_func=None,
            stage_cost_func=lambda x, u: 0.0,
            terminal_cost_func=None
        )

def test_mppi_setup_success():
    """Test that setup_mppi succeeds with valid inputs."""
    def dummy_dynamics(x):
        return x, x

    # Case 1: Trajectory cost provided
    try:
        setup_mppi(
            dyn_func=dummy_dynamics,
            trajectory_cost_func=lambda *args: 0.0,
        )
    except Exception as e:
        pytest.fail(f"setup_mppi raised exception with valid trajectory cost: {e}")

    # Case 2: Stage and Terminal cost provided
    try:
        setup_mppi(
            dyn_func=dummy_dynamics,
            trajectory_cost_func=None,
            stage_cost_func=lambda x, u: 0.0,
            terminal_cost_func=lambda x: 0.0
        )
    except Exception as e:
        pytest.fail(f"setup_mppi raised exception with valid stage/terminal costs: {e}")
