"""Gymnasium wrapper for CBFKit safety filtering."""

try:
    import gymnasium
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "SafetyFilterWrapper requires gymnasium. " "Install with: pip install cbfkit[gymnasium]"
    ) from exc

import numpy as np
from typing import Any, Callable, Optional

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
        """One-liner convenience constructor.

        Args:
            env: Gymnasium environment.
            dynamics: DynamicsCallable (x) -> (f, g).
            barriers: Barrier certificate(s).
            control_limits: Symmetric actuation limits.
            **kwargs: Forwarded to SafetyFilter.from_cbf_qp and wrapper
                (obs_to_state, action_to_control, control_to_action extracted).
        """
        obs_to_state = kwargs.pop("obs_to_state", None)
        action_to_control = kwargs.pop("action_to_control", None)
        control_to_action = kwargs.pop("control_to_action", None)
        sf = SafetyFilter.from_cbf_qp(
            dynamics=dynamics,
            barriers=barriers,
            control_limits=control_limits,
            **kwargs,
        )
        return cls(
            env,
            safety_filter=sf,
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
