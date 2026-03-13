"""Benchmark scenario registry for cbfkit-bench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence


BenchmarkResult = Mapping[str, float | int | bool | str]
BenchmarkScenario = Callable[[int], BenchmarkResult]
SweepableRunner = Callable[[int, Dict[str, Any]], BenchmarkResult]


@dataclass(frozen=True)
class ScenarioSpec:
    """Container describing a benchmark scenario."""

    name: str
    runner: BenchmarkScenario
    description: str = ""
    sweepable_params: tuple[str, ...] = ()
    sweep_runner: Optional[SweepableRunner] = None


class BenchmarkRegistry:
    """In-memory registry of benchmark scenarios."""

    def __init__(self) -> None:
        self._scenarios: Dict[str, ScenarioSpec] = {}

    def register(
        self,
        name: str,
        runner: BenchmarkScenario,
        description: str = "",
        *,
        overwrite: bool = False,
        sweepable_params: tuple[str, ...] = (),
        sweep_runner: Optional[SweepableRunner] = None,
    ) -> None:
        if not overwrite and name in self._scenarios:
            raise ValueError(f"Scenario '{name}' is already registered")
        self._scenarios[name] = ScenarioSpec(
            name=name,
            runner=runner,
            description=description,
            sweepable_params=sweepable_params,
            sweep_runner=sweep_runner,
        )

    def scenario(self, name: str) -> ScenarioSpec:
        if name not in self._scenarios:
            available = ", ".join(sorted(self._scenarios))
            raise KeyError(f"Unknown scenario '{name}'. Available: {available}")
        return self._scenarios[name]

    def names(self) -> Iterable[str]:
        return sorted(self._scenarios)


registry = BenchmarkRegistry()


def register_scenario(
    name: str,
    *,
    description: str = "",
    overwrite: bool = False,
) -> Callable[[BenchmarkScenario], BenchmarkScenario]:
    """Decorator registering a scenario in the global registry."""

    def _decorator(func: BenchmarkScenario) -> BenchmarkScenario:
        registry.register(name=name, runner=func, description=description, overwrite=overwrite)
        return func

    return _decorator


def register_sweepable_scenario(
    name: str,
    *,
    sweepable_params: Sequence[str],
    description: str = "",
    overwrite: bool = False,
) -> Callable[[SweepableRunner], SweepableRunner]:
    """Decorator registering a sweep-aware scenario.

    The decorated function must have signature ``(seed: int, params: dict) -> dict``.
    A default runner (with empty params) is also registered for non-sweep use.
    """

    def _decorator(func: SweepableRunner) -> SweepableRunner:
        def default_runner(seed: int) -> BenchmarkResult:
            return func(seed, {})

        registry.register(
            name=name,
            runner=default_runner,
            description=description,
            overwrite=overwrite,
            sweepable_params=tuple(sweepable_params),
            sweep_runner=func,
        )
        return func

    return _decorator


@register_scenario(
    "sanity_random_safety",
    description="Synthetic deterministic benchmark for validating bench plumbing.",
)
def sanity_random_safety(seed: int) -> BenchmarkResult:
    """Simple baseline scenario used for smoke tests and examples."""
    # deterministic pseudo-behavior solely from seed
    score = (seed * 7919) % 100
    return {
        "success": int(score > 20),
        "safety_violations": int(score < 10),
        "solver_failures": int(score in (0, 1, 2)),
        "avg_step_ms": float(1.0 + (score / 100.0)),
    }
