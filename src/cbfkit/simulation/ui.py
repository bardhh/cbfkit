"""Simulation output utilities.

Centralizes console output and progress reporting using rich.
Supports TTY (live rendering), non-TTY/CI (plain fallback), and
file logging (no ANSI codes).
"""

from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.theme import Theme

# -- Singleton console --------------------------------------------------

_CBFKIT_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow bold",
        "error": "red bold",
        "success": "green bold",
        "jit": "magenta",
    }
)

# The console auto-detects TTY. When stderr is not a terminal (piped or
# redirected) force_terminal=False ensures plain text. When it IS a TTY,
# rich renders colors and live updates.
console = Console(stderr=True, theme=_CBFKIT_THEME)


# -- Progress bar factory -----------------------------------------------


def create_progress(
    total: int,
    description: str = "Simulating",
    transient: bool = False,
) -> Progress:
    """Create a rich Progress bar configured for simulation use.

    The progress bar is written to stderr so that stdout remains clean
    for programmatic consumers (e.g., piping JSON logs).

    Args:
        total: Total number of steps.
        description: Label shown to the left of the bar.
        transient: If True, the bar disappears on completion.

    Returns:
        A configured ``rich.progress.Progress`` instance.  The caller
        must use it as a context manager or call ``start()``/``stop()``
        manually, and add a task via ``add_task()``.
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
        # Throttle refresh to 10 Hz -- keeps overhead negligible even
        # for 10k+ step simulations where on_step fires every iteration.
        refresh_per_second=10,
    )
    return progress


# -- Formatted message helpers ------------------------------------------


def print_info(message: str) -> None:
    """Print an informational message."""
    console.print(f"[info]{message}[/info]")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[warning]Warning:[/warning] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[error]Error:[/error] {message}")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[success]{message}[/success]")


def print_jit_status(message: str) -> None:
    """Print a JIT-related status message."""
    console.print(f"[jit]{message}[/jit]")


def print_simulation_end(success: bool, message: str = "") -> None:
    """Print the simulation completion or failure banner."""
    if success and message:
        print_success(message)
    elif not success and message:
        print_error(f"Simulation stopped: {message}")


def print_trial_progress(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
) -> None:
    """Lightweight progress display for loops without a Progress context.

    Suitable for Monte Carlo trial loops or similar sequential iterations
    where a full ``Progress`` bar context manager is not used.
    """
    pct = 100.0 * iteration / total if total > 0 else 0.0
    label = f"{prefix} " if prefix else ""
    tail = f" {suffix}" if suffix else ""
    console.print(
        f"{label}[{iteration}/{total}] {pct:.1f}%{tail}",
        highlight=False,
    )
