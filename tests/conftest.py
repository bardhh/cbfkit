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


def _maybe_patch_default_qp_solver():
    """When CBFKIT_QP_SOLVER is set, reroute get_solver('jaxopt') to that solver.

    Used by the integration test suite to run every example/tutorial under
    both jaxopt and the fast solver without modifying the scripts themselves.
    """
    target = os.environ.get("CBFKIT_QP_SOLVER", "").strip().lower()
    if not target or target == "jaxopt":
        return

    from cbfkit.optimization.quadratic_program import solver_registry as _sr

    _orig = _sr.get_solver

    def _patched(name: str = "jaxopt", **kwargs):
        if name == "jaxopt":
            try:
                return _orig(target, **kwargs)
            except TypeError:
                return _orig(target)
        return _orig(name, **kwargs)

    _sr.get_solver = _patched
    try:
        from cbfkit.controllers.cbf_clf import cbf_clf_qp_generator as _gen

        _gen.get_solver = _patched
    except Exception:
        pass


_maybe_patch_default_qp_solver()
