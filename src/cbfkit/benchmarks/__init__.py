"""Benchmarking utilities for reproducible cbfkit experiments."""

from .registry import BenchmarkRegistry, register_scenario, register_sweepable_scenario, registry
from .runner import BenchmarkRun, compare_runs, run_scenario, write_artifacts
from .sweep import SweepRun, run_optuna_sweep, run_sweep, write_sweep_artifacts
from .sweep_config import SweepConfigError
from . import solver_stress
from . import qp_solver_stress


def _load_scenarios() -> None:
    """Import scenario modules that register themselves via decorators."""
    from . import monte_carlo_gpu_benchmark as _mcgb  # noqa: F401
    from . import single_integrator_sweep as _sis  # noqa: F401
    from . import unicycle_sweep as _us  # noqa: F401


registry.add_lazy_loader(_load_scenarios)

__all__ = [
    "BenchmarkRegistry",
    "BenchmarkRun",
    "SweepConfigError",
    "SweepRun",
    "register_scenario",
    "register_sweepable_scenario",
    "registry",
    "run_scenario",
    "run_optuna_sweep",
    "run_sweep",
    "compare_runs",
    "write_artifacts",
    "write_sweep_artifacts",
]
