# Gymnasium Safety Filter Wrapper — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone `SafetyFilter` and optional Gymnasium `SafetyFilterWrapper` to CBFKit for CBF-based safety filtering in RL workflows.

**Architecture:** Two-layer design — `SafetyFilter` (no gym dep) wraps a CBFKit `ControllerCallable` with per-step state management (time, PRNG key, solver warm-start). `SafetyFilterWrapper` is a thin `gymnasium.Wrapper` subclass that calls `SafetyFilter.filter()` inside `step()`. A custom Gymnasium env + barrier helper provide a self-contained demo.

**Tech Stack:** JAX, CBFKit controllers/certificates, Gymnasium >=1.0 (optional), matplotlib (for env rendering)

**Spec:** `docs/superpowers/specs/2026-05-16-gymnasium-wrapper-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/cbfkit/wrappers/__init__.py` | Export `SafetyFilter`, `__all__` |
| `src/cbfkit/wrappers/safety_filter.py` | Core `SafetyFilter` class — construction, filter(), reset(), barrier_values() |
| `src/cbfkit/wrappers/gymnasium.py` | `SafetyFilterWrapper` — Gymnasium `Wrapper` subclass, import guard |
| `src/cbfkit/envs/__init__.py` | Package marker |
| `src/cbfkit/envs/gymnasium.py` | Custom env, `register_envs()`, `circular_obstacle_barriers()` |
| `examples/gymnasium/safe_single_integrator.py` | Demo script: unsafe vs safe comparison with plots |
| `examples/gymnasium/README.md` | Quick-start snippet |
| `tests/test_wrappers/test_safety_filter.py` | All tests: filter, fallback, wrapper, optional dep |
| `pyproject.toml` | Add `gymnasium` optional dep |

---

## Chunk 1: SafetyFilter Core

### Task 1: Package scaffolding and pyproject.toml

**Files:**
- Create: `src/cbfkit/wrappers/__init__.py`
- Create: `src/cbfkit/envs/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create wrappers package**

```python
# src/cbfkit/wrappers/__init__.py
"""Safety filter wrappers for CBFKit controllers."""

from .safety_filter import SafetyFilter

__all__ = ["SafetyFilter"]
```

- [ ] **Step 2: Create envs package**

```python
# src/cbfkit/envs/__init__.py
"""CBFKit Gymnasium environments."""
```

- [ ] **Step 3: Add gymnasium optional dep to pyproject.toml**

Add to `[project.optional-dependencies]`:
```toml
gymnasium = [
    "gymnasium>=1.0",
]
```

Also add `"gymnasium>=1.0"` to the `dev` extras list so CI runs gymnasium tests.

- [ ] **Step 4: Commit**

```bash
git add src/cbfkit/wrappers/__init__.py src/cbfkit/envs/__init__.py pyproject.toml
git commit -m "feat(gymnasium): add package scaffolding and optional dep"
```

---

### Task 2: SafetyFilter — from_controller path + filter() + reset()

**Files:**
- Create: `src/cbfkit/wrappers/safety_filter.py`
- Create: `tests/test_wrappers/test_safety_filter.py`

- [ ] **Step 1: Write failing tests for from_controller and filter()**

```python
# tests/test_wrappers/test_safety_filter.py
import pytest
import jax
import jax.numpy as jnp
from cbfkit.utils.user_types import ControllerData
from cbfkit.wrappers import SafetyFilter


def _mock_controller(scale=1.0):
    """Controller that scales u_nom by `scale`."""
    def controller(t, x, u_nom, key, data):
        return u_nom * scale, ControllerData()
    return controller


def _failing_controller():
    """Controller that returns NaN and error."""
    def controller(t, x, u_nom, key, data):
        return jnp.full_like(u_nom, jnp.nan), ControllerData(error=True, error_data=1)
    return controller


class TestSafetyFilterFromController:
    def test_construction(self):
        sf = SafetyFilter.from_controller(_mock_controller(), dt=0.01, seed=0)
        assert sf is not None

    def test_filter_returns_tuple(self):
        sf = SafetyFilter.from_controller(_mock_controller(), dt=0.01, seed=0)
        state = jnp.array([1.0, 2.0])
        action = jnp.array([0.5, -0.3])
        u_applied, info = sf.filter(state, action)
        assert u_applied.shape == action.shape

    def test_filter_passthrough_when_no_modification(self):
        sf = SafetyFilter.from_controller(_mock_controller(scale=1.0), dt=0.01, seed=0)
        state = jnp.array([1.0, 2.0])
        action = jnp.array([0.5, -0.3])
        u_applied, info = sf.filter(state, action)
        assert jnp.allclose(u_applied, action)
        assert not info["intervened"]
        assert not info["fallback_used"]

    def test_filter_detects_intervention(self):
        sf = SafetyFilter.from_controller(_mock_controller(scale=0.5), dt=0.01, seed=0)
        state = jnp.array([1.0, 2.0])
        action = jnp.array([1.0, 1.0])
        u_applied, info = sf.filter(state, action)
        assert jnp.allclose(u_applied, action * 0.5)
        assert info["intervened"]

    def test_info_has_all_keys(self):
        sf = SafetyFilter.from_controller(_mock_controller(), dt=0.01, seed=0)
        _, info = sf.filter(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))
        required_keys = {"u_nom", "u_qp", "u_applied", "intervened",
                         "barrier_values", "solver_status", "controller_error", "fallback_used"}
        assert required_keys == set(info.keys())

    def test_info_scalars_are_python_types(self):
        sf = SafetyFilter.from_controller(_mock_controller(), dt=0.01, seed=0)
        _, info = sf.filter(jnp.array([0.0]), jnp.array([1.0]))
        assert isinstance(info["intervened"], bool)
        assert isinstance(info["controller_error"], bool)
        assert isinstance(info["fallback_used"], bool)

    def test_time_increments(self):
        sf = SafetyFilter.from_controller(_mock_controller(), dt=0.05, seed=0)
        assert sf.time == 0.0
        sf.filter(jnp.array([0.0]), jnp.array([1.0]))
        assert abs(sf.time - 0.05) < 1e-10
        sf.filter(jnp.array([0.0]), jnp.array([1.0]))
        assert abs(sf.time - 0.10) < 1e-10

    def test_reset_clears_state(self):
        sf = SafetyFilter.from_controller(_mock_controller(), dt=0.05, seed=0)
        sf.filter(jnp.array([0.0]), jnp.array([1.0]))
        sf.reset()
        assert sf.time == 0.0

    def test_reset_with_seed(self):
        sf = SafetyFilter.from_controller(_mock_controller(), dt=0.01, seed=0)
        sf.filter(jnp.array([0.0]), jnp.array([1.0]))
        sf.reset(seed=42)
        assert sf.time == 0.0

    def test_barrier_values_none_without_barriers(self):
        sf = SafetyFilter.from_controller(_mock_controller(), dt=0.01, seed=0)
        assert sf.barrier_values(jnp.array([0.0, 0.0])) is None

    def test_float32_input_accepted(self):
        sf = SafetyFilter.from_controller(_mock_controller(), dt=0.01, seed=0)
        state = jnp.array([1.0, 2.0], dtype=jnp.float32)
        action = jnp.array([0.5, -0.3], dtype=jnp.float32)
        u_applied, info = sf.filter(state, action)
        assert u_applied.shape == action.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_wrappers/test_safety_filter.py -v`
Expected: ImportError — `safety_filter` module doesn't exist yet

- [ ] **Step 3: Implement SafetyFilter**

```python
# src/cbfkit/wrappers/safety_filter.py
"""Standalone CBF safety filter for action filtering."""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array, random

from cbfkit.utils.user_types import (
    ControllerCallable,
    ControllerData,
    CertificateCollection,
    DynamicsCallable,
    EMPTY_CERTIFICATE_COLLECTION,
)


class SafetyFilter:
    """CBF-based safety filter that wraps a ControllerCallable.

    Manages per-step state (time, PRNG key, solver warm-start) and provides
    fallback behavior on solver failure. Usable standalone or via the
    Gymnasium SafetyFilterWrapper.

    Not thread-safe. Not compatible with gymnasium.vector environments.
    """

    def __init__(
        self,
        controller: ControllerCallable,
        dt: float,
        seed: int,
        fallback: Union[str, Callable] = "passthrough",
        barriers: Optional[CertificateCollection] = None,
    ):
        self._controller = controller
        self._dt = dt
        self._fallback = fallback
        self._barriers = barriers
        self._t = 0.0
        self._key = random.PRNGKey(seed)
        self._data = ControllerData()
        self._initial_seed = seed

    @classmethod
    def from_controller(
        cls,
        controller: ControllerCallable,
        dt: float = 0.01,
        seed: int = 0,
        fallback: Union[str, Callable] = "passthrough",
        barriers: Optional[CertificateCollection] = None,
    ) -> "SafetyFilter":
        """Create a SafetyFilter from a pre-built ControllerCallable."""
        return cls(controller=controller, dt=dt, seed=seed, fallback=fallback, barriers=barriers)

    @classmethod
    def from_cbf_qp(
        cls,
        dynamics: DynamicsCallable,
        barriers,  # CertificateInput
        control_limits: Array,
        variant: str = "vanilla",
        lyapunovs=None,
        fallback: Union[str, Callable] = "passthrough",
        dt: float = 0.01,
        seed: int = 0,
        **kwargs: Any,
    ) -> "SafetyFilter":
        """Create a SafetyFilter from CBF-QP specification."""
        from cbfkit.controllers.cbf_clf import (
            vanilla_cbf_clf_qp_controller,
            robust_cbf_clf_qp_controller,
            stochastic_cbf_clf_qp_controller,
        )

        generators = {
            "vanilla": vanilla_cbf_clf_qp_controller,
            "robust": robust_cbf_clf_qp_controller,
            "stochastic": stochastic_cbf_clf_qp_controller,
        }
        if variant not in generators:
            raise ValueError(
                f"Unknown variant {variant!r}. Choose from: {list(generators.keys())}. "
                f"For risk-aware variants, use SafetyFilter.from_controller() with a "
                f"pre-built controller."
            )

        generator = generators[variant]
        controller = generator(
            control_limits=control_limits,
            dynamics_func=dynamics,
            barriers=barriers,
            lyapunovs=lyapunovs if lyapunovs is not None else EMPTY_CERTIFICATE_COLLECTION,
            **kwargs,
        )

        # Normalize barriers to CertificateCollection for barrier_values()
        normalized_barriers = None
        if barriers is not None and barriers != EMPTY_CERTIFICATE_COLLECTION:
            if isinstance(barriers, tuple) and hasattr(barriers, 'functions'):
                normalized_barriers = barriers
            elif isinstance(barriers, (list, tuple)):
                from cbfkit.certificates import concatenate_certificates
                normalized_barriers = concatenate_certificates(*barriers)
            else:
                normalized_barriers = barriers

        return cls(
            controller=controller, dt=dt, seed=seed,
            fallback=fallback, barriers=normalized_barriers,
        )

    @property
    def time(self) -> float:
        """Current internal time."""
        return self._t

    def filter(self, state, action) -> Tuple[Array, Dict[str, Any]]:
        """Filter an action through the CBF safety controller.

        Args:
            state: Current system state.
            action: Proposed action (u_nom).

        Returns:
            (u_applied, info) where u_applied is the safe action and info
            contains diagnostics.
        """
        state = jnp.asarray(state, dtype=jnp.float64)
        u_nom = jnp.asarray(action, dtype=jnp.float64)

        self._key, subkey = random.split(self._key)
        u_qp, updated_data = self._controller(
            self._t, state, u_nom, subkey, self._data
        )

        # Detect solver failure (explicit error flag or NaN output)
        solver_failed = bool(updated_data.error) or bool(jnp.any(jnp.isnan(u_qp)))

        if solver_failed:
            u_applied = self._apply_fallback(state, u_nom)
            fallback_used = True
        else:
            u_applied = u_qp
            fallback_used = False

        intervened = bool(not jnp.allclose(u_nom, u_applied, atol=1e-4))

        # Extract barrier values from controller sub_data or recompute
        barrier_values = None
        if updated_data.sub_data and "bfs" in updated_data.sub_data:
            barrier_values = updated_data.sub_data["bfs"]
        elif self._barriers is not None:
            barrier_values = self.barrier_values(state)

        info = {
            "u_nom": u_nom,
            "u_qp": u_qp,
            "u_applied": u_applied,
            "intervened": intervened,
            "barrier_values": barrier_values,
            "solver_status": int(updated_data.error_data) if updated_data.error_data is not None else None,
            "controller_error": bool(updated_data.error),
            "fallback_used": fallback_used,
        }

        self._t += self._dt
        self._data = updated_data

        return u_applied, info

    def _apply_fallback(self, state: Array, u_nom: Array) -> Array:
        """Apply fallback strategy on solver failure."""
        if self._fallback == "passthrough":
            return u_nom
        elif self._fallback == "zero":
            return jnp.zeros_like(u_nom)
        elif callable(self._fallback):
            return self._fallback(state, u_nom)
        else:
            raise ValueError(f"Unknown fallback strategy: {self._fallback!r}")

    def barrier_values(self, state, t=None) -> Optional[Array]:
        """Evaluate barrier functions at the given state.

        Args:
            state: System state.
            t: Time (defaults to internal time counter).

        Returns:
            Array of barrier values, or None if no barriers available.
        """
        if self._barriers is None:
            return None
        if t is None:
            t = self._t
        state = jnp.asarray(state, dtype=jnp.float64)
        return jnp.array([f(t, state) for f in self._barriers[0]])

    def reset(self, seed=None):
        """Reset filter state for a new episode.

        Args:
            seed: Optional new PRNG seed. If None, re-seeds with original seed.
        """
        self._t = 0.0
        self._data = ControllerData()
        if seed is not None:
            self._key = random.PRNGKey(seed)
        else:
            self._key = random.PRNGKey(self._initial_seed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_wrappers/test_safety_filter.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/cbfkit/wrappers/safety_filter.py tests/test_wrappers/test_safety_filter.py
git commit -m "feat(gymnasium): add SafetyFilter core with from_controller, filter, reset"
```

---

### Task 3: SafetyFilter — fallback tests

**Files:**
- Modify: `tests/test_wrappers/test_safety_filter.py`

- [ ] **Step 1: Write failing fallback tests**

Add to `tests/test_wrappers/test_safety_filter.py`:

```python
class TestSafetyFilterFallback:
    def test_passthrough_on_error(self):
        sf = SafetyFilter.from_controller(
            _failing_controller(), dt=0.01, seed=0, fallback="passthrough"
        )
        action = jnp.array([1.0, 2.0])
        u_applied, info = sf.filter(jnp.array([0.0, 0.0]), action)
        assert jnp.allclose(u_applied, action)
        assert info["fallback_used"]
        assert info["controller_error"]

    def test_zero_on_error(self):
        sf = SafetyFilter.from_controller(
            _failing_controller(), dt=0.01, seed=0, fallback="zero"
        )
        action = jnp.array([1.0, 2.0])
        u_applied, info = sf.filter(jnp.array([0.0, 0.0]), action)
        assert jnp.allclose(u_applied, jnp.zeros(2))
        assert info["fallback_used"]

    def test_callable_fallback(self):
        sf = SafetyFilter.from_controller(
            _failing_controller(), dt=0.01, seed=0,
            fallback=lambda state, action: action * 0.1,
        )
        action = jnp.array([1.0, 2.0])
        u_applied, info = sf.filter(jnp.array([0.0, 0.0]), action)
        assert jnp.allclose(u_applied, action * 0.1)
        assert info["fallback_used"]

    def test_u_qp_preserves_nan_on_failure(self):
        sf = SafetyFilter.from_controller(
            _failing_controller(), dt=0.01, seed=0, fallback="passthrough"
        )
        _, info = sf.filter(jnp.array([0.0, 0.0]), jnp.array([1.0, 2.0]))
        assert jnp.any(jnp.isnan(info["u_qp"]))

    def test_nan_without_error_flag_triggers_fallback(self):
        """Controller returns NaN but doesn't set error=True."""
        def nan_controller(t, x, u_nom, key, data):
            return jnp.full_like(u_nom, jnp.nan), ControllerData(error=False)

        sf = SafetyFilter.from_controller(nan_controller, dt=0.01, seed=0)
        action = jnp.array([1.0])
        u_applied, info = sf.filter(jnp.array([0.0]), action)
        assert jnp.allclose(u_applied, action)  # passthrough fallback
        assert info["fallback_used"]

    def test_solver_status_preserved(self):
        sf = SafetyFilter.from_controller(
            _failing_controller(), dt=0.01, seed=0, fallback="passthrough"
        )
        _, info = sf.filter(jnp.array([0.0, 0.0]), jnp.array([1.0, 2.0]))
        assert info["solver_status"] == 1
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_wrappers/test_safety_filter.py::TestSafetyFilterFallback -v`
Expected: All PASS (implementation already handles these cases)

- [ ] **Step 3: Commit**

```bash
git add tests/test_wrappers/test_safety_filter.py
git commit -m "test(gymnasium): add SafetyFilter fallback tests"
```

---

### Task 4: SafetyFilter — from_cbf_qp path with real controller

**Files:**
- Modify: `tests/test_wrappers/test_safety_filter.py`

- [ ] **Step 1: Write integration test for from_cbf_qp**

Add to `tests/test_wrappers/test_safety_filter.py`:

```python
from cbfkit.certificates import generate_certificate, concatenate_certificates
from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
from cbfkit.systems.single_integrator.dynamics import two_dimensional_single_integrator


def _make_circle_barrier(cx, cy, radius, alpha=1.0):
    """Create a barrier for a circular obstacle: h(x) = ||x - c||^2 - r^2."""
    def h(x):
        return (x[0] - cx) ** 2 + (x[1] - cy) ** 2 - radius ** 2
    return generate_certificate(h, linear_class_k(alpha), input_style="state")


class TestSafetyFilterFromCbfQp:
    def test_construction(self):
        barriers = _make_circle_barrier(2.0, 0.0, 0.5)
        sf = SafetyFilter.from_cbf_qp(
            dynamics=two_dimensional_single_integrator(),
            barriers=barriers,
            control_limits=jnp.array([1.0, 1.0]),
        )
        assert sf is not None

    def test_filter_modifies_unsafe_action(self):
        """Action pointing toward obstacle should be modified."""
        barriers = _make_circle_barrier(1.0, 0.0, 0.5)
        sf = SafetyFilter.from_cbf_qp(
            dynamics=two_dimensional_single_integrator(),
            barriers=barriers,
            control_limits=jnp.array([1.0, 1.0]),
        )
        # State near obstacle, action pointing into it
        state = jnp.array([0.6, 0.0])
        action = jnp.array([1.0, 0.0])  # heading straight toward obstacle center
        u_applied, info = sf.filter(state, action)
        assert info["intervened"]
        assert info["barrier_values"] is not None

    def test_filter_preserves_safe_action(self):
        """Action pointing away from obstacle should not be modified."""
        barriers = _make_circle_barrier(2.0, 0.0, 0.5)
        sf = SafetyFilter.from_cbf_qp(
            dynamics=two_dimensional_single_integrator(),
            barriers=barriers,
            control_limits=jnp.array([1.0, 1.0]),
        )
        state = jnp.array([0.0, 0.0])
        action = jnp.array([-1.0, 0.0])  # heading away from obstacle
        u_applied, info = sf.filter(state, action)
        assert not info["intervened"]

    def test_barrier_values_with_from_cbf_qp(self):
        barriers = _make_circle_barrier(2.0, 0.0, 0.5)
        sf = SafetyFilter.from_cbf_qp(
            dynamics=two_dimensional_single_integrator(),
            barriers=barriers,
            control_limits=jnp.array([1.0, 1.0]),
        )
        bv = sf.barrier_values(jnp.array([0.0, 0.0]))
        assert bv is not None
        assert bv.shape == (1,)  # one barrier
        assert float(bv[0]) > 0  # far from obstacle -> positive

    def test_concatenated_barriers(self):
        b1 = _make_circle_barrier(2.0, 0.0, 0.5)
        b2 = _make_circle_barrier(-2.0, 0.0, 0.5)
        barriers = concatenate_certificates(b1, b2)
        sf = SafetyFilter.from_cbf_qp(
            dynamics=two_dimensional_single_integrator(),
            barriers=barriers,
            control_limits=jnp.array([1.0, 1.0]),
        )
        bv = sf.barrier_values(jnp.array([0.0, 0.0]))
        assert bv.shape == (2,)

    def test_unknown_variant_raises(self):
        barriers = _make_circle_barrier(2.0, 0.0, 0.5)
        with pytest.raises(ValueError, match="Unknown variant"):
            SafetyFilter.from_cbf_qp(
                dynamics=two_dimensional_single_integrator(),
                barriers=barriers,
                control_limits=jnp.array([1.0, 1.0]),
                variant="nonexistent",
            )
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_wrappers/test_safety_filter.py::TestSafetyFilterFromCbfQp -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_wrappers/test_safety_filter.py
git commit -m "test(gymnasium): add SafetyFilter from_cbf_qp integration tests"
```

---

## Chunk 2: Gymnasium Wrapper + Custom Environment

### Task 5: SafetyFilterWrapper

**Files:**
- Create: `src/cbfkit/wrappers/gymnasium.py`
- Modify: `tests/test_wrappers/test_safety_filter.py`

- [ ] **Step 1: Write failing gymnasium wrapper tests**

Add to `tests/test_wrappers/test_safety_filter.py`:

```python
gymnasium = pytest.importorskip("gymnasium")
import numpy as np
from gymnasium import spaces


class _SimpleBoxEnv(gymnasium.Env):
    """Minimal continuous env for testing."""
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-10, 10, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self._state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._state = np.zeros(2, dtype=np.float32)
        return self._state.copy(), {}

    def step(self, action):
        self._state = self._state + np.asarray(action, dtype=np.float32) * 0.1
        return self._state.copy(), -1.0, False, False, {}


class TestSafetyFilterWrapper:
    def test_step_reset_flow(self):
        from cbfkit.wrappers.gymnasium import SafetyFilterWrapper
        env = _SimpleBoxEnv()
        sf = SafetyFilter.from_controller(_mock_controller(), dt=0.1, seed=0)
        safe_env = SafetyFilterWrapper(env, safety_filter=sf)
        obs, info = safe_env.reset()
        assert obs.shape == (2,)
        obs, reward, terminated, truncated, info = safe_env.step(np.array([0.5, 0.5]))
        assert "safety_filter" in info

    def test_step_before_reset_raises(self):
        from cbfkit.wrappers.gymnasium import SafetyFilterWrapper
        env = _SimpleBoxEnv()
        sf = SafetyFilter.from_controller(_mock_controller(), dt=0.1, seed=0)
        safe_env = SafetyFilterWrapper(env, safety_filter=sf)
        with pytest.raises(RuntimeError, match="reset"):
            safe_env.step(np.array([0.5, 0.5]))

    def test_output_dtype_matches_env(self):
        from cbfkit.wrappers.gymnasium import SafetyFilterWrapper
        env = _SimpleBoxEnv()
        sf = SafetyFilter.from_controller(_mock_controller(), dt=0.1, seed=0)
        safe_env = SafetyFilterWrapper(env, safety_filter=sf)
        safe_env.reset()
        obs, _, _, _, info = safe_env.step(np.array([0.5, 0.5]))
        assert obs.dtype == np.float32

    def test_action_clipped_to_space(self):
        from cbfkit.wrappers.gymnasium import SafetyFilterWrapper
        # Controller that doubles the action (exceeds [-1, 1])
        sf = SafetyFilter.from_controller(_mock_controller(scale=2.0), dt=0.1, seed=0)
        env = _SimpleBoxEnv()
        safe_env = SafetyFilterWrapper(env, safety_filter=sf)
        safe_env.reset()
        obs, _, _, _, info = safe_env.step(np.array([0.8, 0.8]))
        # The action sent to env should be clipped to [-1, 1]
        u_applied = info["safety_filter"]["u_applied"]
        assert float(jnp.max(jnp.abs(u_applied))) <= 1.0 + 1e-6  # pre-clip value

    def test_info_nested_under_safety_filter_key(self):
        from cbfkit.wrappers.gymnasium import SafetyFilterWrapper
        env = _SimpleBoxEnv()
        sf = SafetyFilter.from_controller(_mock_controller(), dt=0.1, seed=0)
        safe_env = SafetyFilterWrapper(env, safety_filter=sf)
        safe_env.reset()
        _, _, _, _, info = safe_env.step(np.array([0.5, 0.5]))
        sf_info = info["safety_filter"]
        assert "u_nom" in sf_info
        assert "u_applied" in sf_info
        assert "intervened" in sf_info

    def test_obs_to_state_adapter(self):
        from cbfkit.wrappers.gymnasium import SafetyFilterWrapper
        env = _SimpleBoxEnv()
        sf = SafetyFilter.from_controller(_mock_controller(), dt=0.1, seed=0)
        safe_env = SafetyFilterWrapper(
            env, safety_filter=sf,
            obs_to_state=lambda obs: obs[:1],  # only use first element
        )
        safe_env.reset()
        # Should not crash — adapter extracts subset of obs
        safe_env.step(np.array([0.5, 0.5]))

    def test_reset_forwards_seed(self):
        from cbfkit.wrappers.gymnasium import SafetyFilterWrapper
        env = _SimpleBoxEnv()
        sf = SafetyFilter.from_controller(_mock_controller(), dt=0.1, seed=0)
        safe_env = SafetyFilterWrapper(env, safety_filter=sf)
        safe_env.reset(seed=42)
        assert sf.time == 0.0

    def test_discrete_action_space_raises(self):
        from cbfkit.wrappers.gymnasium import SafetyFilterWrapper

        class DiscreteEnv(gymnasium.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Box(-1, 1, shape=(2,))
                self.action_space = spaces.Discrete(3)
            def reset(self, **kwargs): return np.zeros(2), {}
            def step(self, a): return np.zeros(2), 0, False, False, {}

        sf = SafetyFilter.from_controller(_mock_controller(), dt=0.1, seed=0)
        with pytest.raises(ValueError, match="Box"):
            SafetyFilterWrapper(DiscreteEnv(), safety_filter=sf)

    def test_from_cbf_qp_convenience(self):
        from cbfkit.wrappers.gymnasium import SafetyFilterWrapper
        barriers = _make_circle_barrier(5.0, 0.0, 0.5)
        env = _SimpleBoxEnv()
        safe_env = SafetyFilterWrapper.from_cbf_qp(
            env,
            dynamics=two_dimensional_single_integrator(),
            barriers=barriers,
            control_limits=jnp.array([1.0, 1.0]),
        )
        obs, _ = safe_env.reset()
        obs, _, _, _, info = safe_env.step(np.array([0.5, 0.5]))
        assert "safety_filter" in info
```

- [ ] **Step 2: Implement SafetyFilterWrapper**

```python
# src/cbfkit/wrappers/gymnasium.py
"""Gymnasium wrapper for CBFKit safety filtering."""

try:
    import gymnasium
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "SafetyFilterWrapper requires gymnasium. "
        "Install with: pip install cbfkit[gymnasium]"
    ) from exc

import numpy as np
from typing import Any, Callable, Optional

from jax import Array

from .safety_filter import SafetyFilter


class SafetyFilterWrapper(gymnasium.Wrapper):
    """Gymnasium wrapper that filters actions through a CBF safety filter.

    Requires continuous (Box) action spaces. Discrete actions are not supported.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        safety_filter: SafetyFilter,
        obs_to_state: Optional[Callable] = None,
        action_to_control: Optional[Callable] = None,
        control_to_action: Optional[Callable] = None,
    ):
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Box):
            raise ValueError(
                f"SafetyFilterWrapper requires a continuous (Box) action space, "
                f"got {type(env.action_space).__name__}."
            )
        self.safety_filter = safety_filter
        self._obs_to_state = obs_to_state or (lambda obs: obs)
        self._action_to_control = action_to_control or (lambda a: a)
        self._control_to_action = control_to_action or (lambda c: c)
        self._last_obs = None

    @classmethod
    def from_cbf_qp(cls, env, dynamics, barriers, control_limits, **kwargs):
        """One-liner convenience constructor."""
        obs_to_state = kwargs.pop("obs_to_state", None)
        action_to_control = kwargs.pop("action_to_control", None)
        control_to_action = kwargs.pop("control_to_action", None)
        sf = SafetyFilter.from_cbf_qp(
            dynamics=dynamics, barriers=barriers,
            control_limits=control_limits, **kwargs,
        )
        return cls(
            env, safety_filter=sf,
            obs_to_state=obs_to_state,
            action_to_control=action_to_control,
            control_to_action=control_to_action,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self.safety_filter.reset(seed=kwargs.get("seed"))
        return obs, info

    def step(self, action):
        if self._last_obs is None:
            raise RuntimeError("Call reset() before step().")

        state = self._obs_to_state(self._last_obs)
        control = self._action_to_control(action)
        safe_control, filter_info = self.safety_filter.filter(state, control)
        safe_action = self._control_to_action(safe_control)

        # Cast to env dtype and clip to action space
        safe_action = np.clip(
            np.asarray(safe_action, dtype=self.env.action_space.dtype),
            self.env.action_space.low,
            self.env.action_space.high,
        )

        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        self._last_obs = obs
        info["safety_filter"] = filter_info
        return obs, reward, terminated, truncated, info
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_wrappers/test_safety_filter.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/cbfkit/wrappers/gymnasium.py tests/test_wrappers/test_safety_filter.py
git commit -m "feat(gymnasium): add SafetyFilterWrapper with Gymnasium integration"
```

---

### Task 6: Custom environment + barrier helper

**Files:**
- Create: `src/cbfkit/envs/gymnasium.py`

- [ ] **Step 1: Write the custom env test**

Add to `tests/test_wrappers/test_safety_filter.py`:

```python
class TestCustomEnv:
    def test_env_creation(self):
        from cbfkit.envs.gymnasium import register_envs
        register_envs()
        env = gymnasium.make("CBFKit/SafeSingleIntegratorObstacles-v0")
        assert env.observation_space.shape == (4,)
        assert env.action_space.shape == (2,)

    def test_reset_step_loop(self):
        from cbfkit.envs.gymnasium import register_envs
        register_envs()
        env = gymnasium.make("CBFKit/SafeSingleIntegratorObstacles-v0")
        obs, info = env.reset(seed=42)
        assert obs.shape == (4,)
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                break

    def test_circular_obstacle_barriers(self):
        from cbfkit.envs.gymnasium import circular_obstacle_barriers
        obstacles = [(2.0, 0.0, 0.5), (3.0, 1.0, 0.3)]
        barriers = circular_obstacle_barriers(obstacles, alpha=1.0)
        assert len(barriers[0]) == 2  # 2 barrier functions

    def test_deterministic_with_seed(self):
        from cbfkit.envs.gymnasium import register_envs
        register_envs()
        env1 = gymnasium.make("CBFKit/SafeSingleIntegratorObstacles-v0")
        env2 = gymnasium.make("CBFKit/SafeSingleIntegratorObstacles-v0")
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        assert np.allclose(obs1, obs2)
```

- [ ] **Step 2: Implement custom env and barrier helper**

```python
# src/cbfkit/envs/gymnasium.py
"""CBFKit Gymnasium environments and barrier construction helpers."""

try:
    import gymnasium
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "CBFKit environments require gymnasium. "
        "Install with: pip install cbfkit[gymnasium]"
    ) from exc

import numpy as np
import jax.numpy as jnp
from typing import List, Optional, Tuple

from cbfkit.certificates import generate_certificate, concatenate_certificates
from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
from cbfkit.utils.user_types import CertificateCollection

DEFAULT_START = np.array([0.0, 0.0])
DEFAULT_GOAL = np.array([4.0, 0.0])
DEFAULT_OBSTACLES = [(2.0, 0.3, 0.5), (3.0, -0.2, 0.4), (2.5, -0.5, 0.3)]
DEFAULT_DT = 0.05
DEFAULT_MAX_STEPS = 200


def circular_obstacle_barriers(
    obstacles: List[Tuple[float, float, float]],
    alpha: float = 1.0,
) -> CertificateCollection:
    """Build CBF barriers for circular obstacles.

    Args:
        obstacles: list of (cx, cy, radius) tuples.
        alpha: class-K gain for the zeroing CBF condition.

    Returns:
        CertificateCollection containing one barrier per obstacle.
    """
    collections = []
    for cx, cy, r in obstacles:
        def _make_h(cx_=cx, cy_=cy, r_=r):
            def h(x):
                return (x[0] - cx_) ** 2 + (x[1] - cy_) ** 2 - r_ ** 2
            return h
        cert = generate_certificate(_make_h(), linear_class_k(alpha), input_style="state")
        collections.append(cert)
    return concatenate_certificates(*collections)


class SingleIntegratorObstaclesEnv(gymnasium.Env):
    """2D single-integrator navigation with circular obstacles.

    Observation: [x, y, goal_x, goal_y]
    Action: [vx, vy] clipped to [-1, 1]
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    def __init__(
        self,
        obstacles=None,
        goal=None,
        start=None,
        dt=DEFAULT_DT,
        max_steps=DEFAULT_MAX_STEPS,
        render_mode=None,
    ):
        super().__init__()
        self.obstacles = list(obstacles or DEFAULT_OBSTACLES)
        self._default_goal = np.array(goal) if goal is not None else DEFAULT_GOAL.copy()
        self._default_start = np.array(start) if start is not None else DEFAULT_START.copy()
        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._state = None
        self._goal = None
        self._step_count = 0
        self._trajectory = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._state = self._default_start.copy()
        self._goal = self._default_goal.copy()
        self._step_count = 0
        self._trajectory = [self._state.copy()]
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._state = self._state + action * self.dt
        self._step_count += 1
        self._trajectory.append(self._state.copy())

        dist_to_goal = np.linalg.norm(self._state - self._goal)
        collision = any(
            np.linalg.norm(self._state - np.array([cx, cy])) < r
            for cx, cy, r in self.obstacles
        )
        goal_reached = dist_to_goal < 0.1

        reward = -dist_to_goal
        if collision:
            reward -= 100.0
        if goal_reached:
            reward += 100.0

        terminated = collision or goal_reached
        truncated = self._step_count >= self.max_steps

        return self._get_obs(), float(reward), terminated, truncated, {
            "collision": collision,
            "goal_reached": goal_reached,
        }

    def _get_obs(self):
        return np.concatenate([self._state, self._goal]).astype(np.float32)

    def render(self):
        if self.render_mode is None:
            return None
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            if self.render_mode == "rgb_array":
                matplotlib.use("Agg")
        except ImportError:
            return None

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        # Draw obstacles
        for cx, cy, r in self.obstacles:
            circle = plt.Circle((cx, cy), r, color="red", alpha=0.4)
            ax.add_patch(circle)
        # Draw goal
        ax.plot(*self._goal, "g*", markersize=15)
        # Draw trajectory
        traj = np.array(self._trajectory)
        ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=1.5)
        ax.plot(*self._state, "bo", markersize=8)
        ax.set_xlim(-1, 5)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        if self.render_mode == "rgb_array":
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            plt.close(fig)
            return data[:, :, :3]
        else:
            plt.show()
            plt.close(fig)
            return None


def register_envs():
    """Register CBFKit Gymnasium environments."""
    gymnasium.register(
        id="CBFKit/SafeSingleIntegratorObstacles-v0",
        entry_point="cbfkit.envs.gymnasium:SingleIntegratorObstaclesEnv",
    )
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_wrappers/test_safety_filter.py::TestCustomEnv -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/cbfkit/envs/gymnasium.py tests/test_wrappers/test_safety_filter.py
git commit -m "feat(gymnasium): add single-integrator obstacle env and barrier helper"
```

---

## Chunk 3: Example Script + README + Final Verification

### Task 7: Example script

**Files:**
- Create: `examples/gymnasium/safe_single_integrator.py`

- [ ] **Step 1: Write the example script**

Write `examples/gymnasium/safe_single_integrator.py` following the spec:
- Register env, create with deterministic defaults
- Build barriers with `circular_obstacle_barriers`
- Run unsafe (naive straight-to-goal policy) with `seed=42`
- Run safe (same policy + `SafetyFilterWrapper.from_cbf_qp`) with `seed=42`
- Save `gymnasium_safe_vs_unsafe.png` (side-by-side matplotlib plot)
- Print terminal summary (collision, goal_reached, min barrier, intervention rate)
- Gate plots behind `CBFKIT_TEST_MODE` env var

The script should be ~120 lines and runnable as `python examples/gymnasium/safe_single_integrator.py`.

- [ ] **Step 2: Run the example**

Run: `CBFKIT_TEST_MODE=1 python examples/gymnasium/safe_single_integrator.py`
Expected: Terminal output showing unsafe collision, safe avoidance, summary stats

- [ ] **Step 3: Commit**

```bash
git add examples/gymnasium/safe_single_integrator.py
git commit -m "feat(gymnasium): add safe vs unsafe single-integrator demo"
```

---

### Task 8: Example README

**Files:**
- Create: `examples/gymnasium/README.md`

- [ ] **Step 1: Write README**

Write `examples/gymnasium/README.md` with the ~25-line quick-start snippet from the spec.
Include complete barrier construction — no undefined variables.

- [ ] **Step 2: Commit**

```bash
git add examples/gymnasium/README.md
git commit -m "docs(gymnasium): add quick-start README for gymnasium examples"
```

---

### Task 9: Optional dependency test

**Files:**
- Modify: `tests/test_wrappers/test_safety_filter.py`

- [ ] **Step 1: Add optional dep test**

Add at the top of the test file (before gymnasium-dependent tests):

```python
class TestOptionalDependency:
    def test_safety_filter_importable_always(self):
        """SafetyFilter should be importable without gymnasium."""
        from cbfkit.wrappers import SafetyFilter
        assert SafetyFilter is not None
```

The gymnasium wrapper tests already use `pytest.importorskip("gymnasium")` which
provides the "skip without gymnasium" behavior.

- [ ] **Step 2: Commit**

```bash
git add tests/test_wrappers/test_safety_filter.py
git commit -m "test(gymnasium): add optional dependency import test"
```

---

### Task 10: Full test suite verification

- [ ] **Step 1: Run all wrapper tests**

Run: `pytest tests/test_wrappers/ -v`
Expected: All PASS

- [ ] **Step 2: Run full test suite to check for regressions**

Run: `pytest tests/ -m "not slow" -x -q`
Expected: All tests PASS, no regressions

- [ ] **Step 3: Run linter**

Run: `ruff check src/cbfkit/wrappers/ src/cbfkit/envs/`
Expected: Clean

- [ ] **Step 4: Run example**

Run: `CBFKIT_TEST_MODE=1 python examples/gymnasium/safe_single_integrator.py`
Expected: Completes without error, prints summary

- [ ] **Step 5: Final commit if any fixes needed**

---

### Task 11: Push and create PR

- [ ] **Step 1: Push branch**

```bash
git push -u origin feat/gymnasium-wrapper
```

- [ ] **Step 2: Create PR**

Title: `feat: add Gymnasium safety filter wrapper for safe RL`

Body should include:
- Summary of SafetyFilter + SafetyFilterWrapper
- The 20-line quick-start snippet
- Test plan checklist
- Link to spec document
