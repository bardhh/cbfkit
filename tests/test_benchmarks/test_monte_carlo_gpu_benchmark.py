"""Tests for the Monte Carlo GPU speedup benchmark scenario."""

from cbfkit.benchmarks.registry import registry


def test_monte_carlo_gpu_speedup_registered():
    """Verify the benchmark scenario is registered."""
    assert "monte_carlo_gpu_speedup" in registry.names()
