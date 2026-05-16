"""CBFKit Gymnasium environments and barrier construction helpers."""

try:
    import gymnasium
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "CBFKit environments require gymnasium. " "Install with: pip install cbfkit[gymnasium]"
    ) from exc

import numpy as np
from typing import List, Optional, Tuple

import jax.numpy as jnp

from cbfkit.certificates import concatenate_certificates, generate_certificate
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

    Each barrier is h(x) = (x0 - cx)^2 + (x1 - cy)^2 - r^2 with a zeroing
    CBF condition using linear class-K gain alpha.

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
                return (x[0] - cx_) ** 2 + (x[1] - cy_) ** 2 - r_**2

            return h

        cert = generate_certificate(_make_h(), linear_class_k(alpha), input_style="state")
        collections.append(cert)
    return concatenate_certificates(*collections)


class SingleIntegratorObstaclesEnv(gymnasium.Env):
    """2D single-integrator navigation with circular obstacles.

    Observation: [x, y, goal_x, goal_y]
    Action: [vx, vy] clipped to [-1, 1]
    Reward: -dist_to_goal, -100 on collision, +100 on goal reached.
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
            np.linalg.norm(self._state - np.array([cx, cy])) < r for cx, cy, r in self.obstacles
        )
        goal_reached = dist_to_goal < 0.1

        reward = -dist_to_goal
        if collision:
            reward -= 100.0
        if goal_reached:
            reward += 100.0

        terminated = collision or goal_reached
        truncated = self._step_count >= self.max_steps

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            {"collision": collision, "goal_reached": goal_reached},
        )

    def _get_obs(self):
        return np.concatenate([self._state, self._goal]).astype(np.float32)

    def render(self):
        if self.render_mode is None:
            return None
        try:
            import matplotlib
            import matplotlib.pyplot as plt

            if self.render_mode == "rgb_array":
                matplotlib.use("Agg")
        except ImportError:
            return None

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        for cx, cy, r in self.obstacles:
            circle = plt.Circle((cx, cy), r, color="red", alpha=0.4)
            ax.add_patch(circle)
        ax.plot(*self._goal, "g*", markersize=15)
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
