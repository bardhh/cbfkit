import pytest
from cbfkit.simulation.simulator import _format_error_status

def test_format_error_status():
    assert _format_error_status(0) == "UNSOLVED (Likely Infeasible) (Status: 0)"
    assert _format_error_status(1) == "SOLVED (Status: 1)"
    assert _format_error_status(2) == "MAX_ITER_REACHED (Status: 2)"
    assert _format_error_status(3) == "PRIMAL_INFEASIBLE (Status: 3)"
    assert _format_error_status(4) == "DUAL_INFEASIBLE (Status: 4)"
    assert _format_error_status(999) == "Status: 999"
    assert _format_error_status("Unknown") == "Unknown"
