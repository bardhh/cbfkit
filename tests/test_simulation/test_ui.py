"""Tests for simulation UI utilities."""

from cbfkit.simulation.ui import (
    console,
    create_progress,
    print_error,
    print_info,
    print_jit_status,
    print_simulation_end,
    print_success,
    print_trial_progress,
    print_warning,
)


def test_console_exists():
    assert console is not None


def test_create_progress_lifecycle():
    p = create_progress(total=100, description="Test")
    assert p is not None
    p.start()
    task_id = p.add_task("Test", total=100)
    p.update(task_id, advance=50)
    p.stop()


def test_create_progress_transient():
    p = create_progress(total=10, description="Transient", transient=True)
    p.start()
    task_id = p.add_task("Transient", total=10)
    for _ in range(10):
        p.update(task_id, advance=1)
    p.stop()


def test_print_info_does_not_raise():
    print_info("info message")


def test_print_warning_does_not_raise():
    print_warning("warning message")


def test_print_error_does_not_raise():
    print_error("error message")


def test_print_success_does_not_raise():
    print_success("success message")


def test_print_jit_status_does_not_raise():
    print_jit_status("jit message")


def test_print_simulation_end_success():
    print_simulation_end(success=True, message="Done")


def test_print_simulation_end_failure():
    print_simulation_end(success=False, message="Controller error")


def test_print_simulation_end_silent():
    """No output when success with no message."""
    print_simulation_end(success=True, message="")


def test_print_trial_progress():
    for i in range(1, 11):
        print_trial_progress(i, 10, prefix="Trial", suffix="complete")


def test_print_trial_progress_no_prefix_suffix():
    print_trial_progress(5, 10)


def test_print_trial_progress_zero_total():
    """Zero total should not raise (division by zero guard)."""
    print_trial_progress(0, 0)
