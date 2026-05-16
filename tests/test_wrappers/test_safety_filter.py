"""Tests for SafetyFilter, SafetyFilterWrapper, and custom Gymnasium env."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from cbfkit.utils.user_types import ControllerData
from cbfkit.wrappers import SafetyFilter


# ---------------------------------------------------------------------------
# Mock controllers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Task 2: SafetyFilter core — from_controller + filter + reset
# ---------------------------------------------------------------------------


class TestSafetyFilterFromController:
    def test_construction(self):
        sf = SafetyFilter.from_controller(_mock_controller(), dt=0.01, seed=0)
        assert sf is not None

    def test_filter_returns_tuple(self):
        sf = SafetyFilter.from_controller(_mock_controller(), dt=0.01, seed=0)
        u_applied, info = sf.filter(jnp.array([1.0, 2.0]), jnp.array([0.5, -0.3]))
        assert u_applied.shape == (2,)

    def test_filter_passthrough_when_no_modification(self):
        sf = SafetyFilter.from_controller(_mock_controller(scale=1.0), dt=0.01, seed=0)
        action = jnp.array([0.5, -0.3])
        u_applied, info = sf.filter(jnp.array([1.0, 2.0]), action)
        assert jnp.allclose(u_applied, action)
        assert not info["intervened"]
        assert not info["fallback_used"]

    def test_filter_detects_intervention(self):
        sf = SafetyFilter.from_controller(_mock_controller(scale=0.5), dt=0.01, seed=0)
        action = jnp.array([1.0, 1.0])
        u_applied, info = sf.filter(jnp.array([1.0, 2.0]), action)
        assert jnp.allclose(u_applied, action * 0.5)
        assert info["intervened"]

    def test_info_has_all_keys(self):
        sf = SafetyFilter.from_controller(_mock_controller(), dt=0.01, seed=0)
        _, info = sf.filter(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))
        required_keys = {
            "u_nom",
            "u_qp",
            "u_applied",
            "intervened",
            "barrier_values",
            "solver_status",
            "controller_error",
            "fallback_used",
        }
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
        assert u_applied.shape == (2,)


# ---------------------------------------------------------------------------
# Task 3: Fallback tests
# ---------------------------------------------------------------------------


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
        sf = SafetyFilter.from_controller(_failing_controller(), dt=0.01, seed=0, fallback="zero")
        action = jnp.array([1.0, 2.0])
        u_applied, info = sf.filter(jnp.array([0.0, 0.0]), action)
        assert jnp.allclose(u_applied, jnp.zeros(2))
        assert info["fallback_used"]

    def test_callable_fallback(self):
        sf = SafetyFilter.from_controller(
            _failing_controller(),
            dt=0.01,
            seed=0,
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

    def test_u_applied_matches_return_value(self):
        sf = SafetyFilter.from_controller(_failing_controller(), dt=0.01, seed=0, fallback="zero")
        u_applied, info = sf.filter(jnp.array([0.0]), jnp.array([1.0]))
        assert jnp.allclose(u_applied, info["u_applied"])


# ---------------------------------------------------------------------------
# Task 4: from_cbf_qp integration tests
# ---------------------------------------------------------------------------

from cbfkit.certificates import generate_certificate, concatenate_certificates
from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
from cbfkit.systems.single_integrator.dynamics import two_dimensional_single_integrator


def _make_circle_barrier(cx, cy, radius, alpha=1.0):
    """Create a barrier for a circular obstacle: h(x) = ||x - c||^2 - r^2."""

    def h(x):
        return (x[0] - cx) ** 2 + (x[1] - cy) ** 2 - radius**2

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
        state = jnp.array([0.6, 0.0])
        action = jnp.array([1.0, 0.0])
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
        action = jnp.array([-1.0, 0.0])
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
        assert bv.shape == (1,)
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


# ---------------------------------------------------------------------------
# Task 5: Gymnasium wrapper tests (skipped if gymnasium not installed)
# ---------------------------------------------------------------------------

try:
    import gymnasium
    from gymnasium import spaces

    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gymnasium = None
    spaces = None

requires_gymnasium = pytest.mark.skipif(not HAS_GYMNASIUM, reason="gymnasium not installed")


if HAS_GYMNASIUM:

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


@requires_gymnasium
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
        obs, _, _, _, _ = safe_env.step(np.array([0.5, 0.5]))
        assert obs.dtype == np.float32

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
            env,
            safety_filter=sf,
            obs_to_state=lambda obs: obs[:1],
        )
        safe_env.reset()
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

            def reset(self, **kwargs):
                return np.zeros(2), {}

            def step(self, a):
                return np.zeros(2), 0, False, False, {}

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


# ---------------------------------------------------------------------------
# Task 6: Custom env tests
# ---------------------------------------------------------------------------


@requires_gymnasium
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


# ---------------------------------------------------------------------------
# Optional dependency test
# ---------------------------------------------------------------------------


class TestOptionalDependency:
    def test_safety_filter_importable_always(self):
        """SafetyFilter should be importable without gymnasium."""
        from cbfkit.wrappers import SafetyFilter

        assert SafetyFilter is not None
