import pytest
from cbfkit.simulation.simulator import _format_error_status

def test_format_error_status():
    assert _format_error_status(-1) == "NAN_DETECTED (Status: -1)"
    assert _format_error_status(-2) == "NAN_INPUT_DETECTED (Status: -2)"
    assert _format_error_status(0) == "UNSOLVED (Likely Infeasible) (Status: 0)"
    assert _format_error_status(1) == "SOLVED (Status: 1)"
    assert _format_error_status(2) == "MAX_ITER_REACHED (Status: 2)"
    assert _format_error_status(3) == "PRIMAL_INFEASIBLE (Status: 3)"
    assert _format_error_status(4) == "DUAL_INFEASIBLE (Status: 4)"
    assert _format_error_status(999) == "Status: 999"
    assert _format_error_status("Unknown") == "Unknown"

import jax.numpy as jnp
from cbfkit.simulation.simulator import _check_simulation_status

def test_check_simulation_status_suppresses_contradictory_warning(capsys):
    """Test that MAX_ITER warning is suppressed if error is True at the same step."""

    # Setup data where error=True and status=2 (MAX_ITER) at step 0
    controller_keys = ["error", "error_data"]
    controller_values = [
        jnp.array([True]),  # error=True
        jnp.array([2])      # status=2
    ]
    planner_keys = []
    planner_values = []

    _check_simulation_status(
        controller_keys, controller_values,
        planner_keys, planner_values
    )

    captured = capsys.readouterr()
    # Should warn about controller error
    assert "Simulation stopped early due to controller error" in captured.out or \
           "Simulation stopped early due to controller error" in captured.err

    # Should NOT warn about "Solutions were accepted"
    assert "Solutions were accepted" not in captured.out
    assert "Solutions were accepted" not in captured.err

def test_check_simulation_status_warns_on_accepted_max_iter(capsys):
    """Test that MAX_ITER warning is SHOWN if error is False."""

    # Setup data where error=False and status=2 (MAX_ITER) at step 0
    # Note: simulator.py looks for "sub_data_solver_status" or "error_data" for status code
    controller_keys = ["error", "error_data"]
    controller_values = [
        jnp.array([False]), # error=False
        jnp.array([2])      # status=2
    ]
    planner_keys = []
    planner_values = []

    _check_simulation_status(
        controller_keys, controller_values,
        planner_keys, planner_values
    )

    captured = capsys.readouterr()
    # Should Warn about "Solutions were accepted"
    assert "Solutions were accepted" in captured.out or \
           "Solutions were accepted" in captured.err
