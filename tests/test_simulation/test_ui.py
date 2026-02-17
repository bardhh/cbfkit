"""Tests for simulation UI utilities."""

from io import StringIO

from rich.console import Console

import cbfkit.simulation.ui as ui
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


def test_non_tty_output_has_no_ansi(monkeypatch):
    """Ensure non-TTY/redirected output remains plain text."""
    sink = StringIO()
    plain_console = Console(
        file=sink,
        force_terminal=False,
        color_system=None,
        no_color=True,
        width=120,
    )
    monkeypatch.setattr(ui, "console", plain_console)

    ui.print_info("info message")
    ui.print_warning("warning message")
    ui.print_error("error message")
    ui.print_success("success message")
    ui.print_jit_status("jit message")
    ui.print_trial_progress(3, 10, prefix="Trial", suffix="done")

    output = sink.getvalue()
    assert "info message" in output
    assert "Warning:" in output
    assert "Error:" in output
    assert "success message" in output
    assert "jit message" in output
    assert "Trial [3/10] 30.0% done" in output
    assert "\x1b[" not in output
