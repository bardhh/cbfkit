"""Tests for the rich-based ProgressCallback."""

from unittest.mock import MagicMock

from cbfkit.simulation.callbacks import ProgressCallback
from cbfkit.simulation.utils import SimulationStepData


def _mock_step_data() -> MagicMock:
    return MagicMock(spec=SimulationStepData)


def test_progress_callback_lifecycle():
    """ProgressCallback starts, advances, and stops without error."""
    cb = ProgressCallback()
    cb.on_start(total_steps=10, dt=0.1)

    mock_data = _mock_step_data()
    for i in range(10):
        cb.on_step(step_idx=i, time=i * 0.1, data=mock_data)

    cb.on_end(success=True, message="Done")


def test_progress_callback_early_stop():
    """ProgressCallback handles early termination gracefully."""
    cb = ProgressCallback()
    cb.on_start(total_steps=100, dt=0.01)

    mock_data = _mock_step_data()
    for i in range(5):
        cb.on_step(step_idx=i, time=i * 0.01, data=mock_data)

    cb.on_end(success=False, message="Controller error")


def test_progress_callback_success_no_message():
    """on_end with success=True and no message should be silent."""
    cb = ProgressCallback()
    cb.on_start(total_steps=5, dt=0.1)

    mock_data = _mock_step_data()
    for i in range(5):
        cb.on_step(step_idx=i, time=i * 0.1, data=mock_data)

    cb.on_end(success=True, message="")


def test_progress_callback_no_start():
    """on_step and on_end are safe even if on_start was never called."""
    cb = ProgressCallback()
    mock_data = _mock_step_data()
    cb.on_step(step_idx=0, time=0.0, data=mock_data)  # Should not raise
    cb.on_end(success=True)  # Should not raise
