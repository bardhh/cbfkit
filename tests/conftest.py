import os

# Force JAX to use CPU backend before any jax import occurs.
# This prevents Metal/GPU initialization failures in CI and sandboxed environments.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.random
import pytest


@pytest.fixture
def prng_key():
    """Standard PRNG key for reproducible tests."""
    return jax.random.PRNGKey(0)


def pytest_sessionfinish(session, exitstatus):
    """Dedicated integration lanes must execute their promised tests."""
    if os.environ.get("CBFKIT_REQUIRE_NO_SKIPS") != "1":
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None and reporter.stats.get("skipped"):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        reporter.write_sep("=", "Required integration tests were skipped", red=True)
