"""Reactive pedestrians for the reduced-order CBF layer.

:class:`SocialForceCrowd` is the ``agents`` object for
:func:`cbfkit.systems.mujoco.reduced_order.safe_locomotion_controller_di`: ``N``
goal-seeking pedestrians driven by the social-force model of
:mod:`cbfkit.systems.pedestrian.behaviors` (goal attraction, repulsion from the
robot, from each other and from static obstacles). They are kinematic -- not MuJoCo
bodies -- and live in the controller's carry; the CBF-QP sees their current
position/velocity through the augmented state and predicts them at constant
velocity, so their *accelerations* (the reactions) are the unmodelled part.
"""

from typing import Sequence

import jax.numpy as jnp
from jax import Array

from cbfkit.systems.pedestrian.behaviors import social_force_policy


class SocialForceCrowd:
    """``N`` social-force pedestrians; see the module docstring.

    Args:
        starts, goals: ``(N, 2)`` positions; ``speeds``: desired speed per pedestrian.
        obstacles: static ``(M, 2)`` points (pillars) the pedestrians also avoid.
        ped_radius / agent_radius: interaction radii (the robot counts as an agent).
        repulsion_strength / repulsion_range / relaxation_time: social-force parameters.
        speed_cap: hard cap on a pedestrian's speed (multiple of its desired speed).
    """

    def __init__(
        self,
        starts: Sequence[Sequence[float]],
        goals: Sequence[Sequence[float]],
        speeds: Sequence[float],
        *,
        obstacles: Sequence[Sequence[float]] = (),
        ped_radius: float = 0.3,
        agent_radius: float = 0.35,
        repulsion_strength: float = 2.0,
        repulsion_range: float = 0.6,
        relaxation_time: float = 0.5,
        speed_cap: float = 1.5,
    ) -> None:
        starts_a = jnp.asarray(starts, dtype=float).reshape(-1, 2)
        goals_a = jnp.asarray(goals, dtype=float).reshape(-1, 2)
        speeds_a = jnp.asarray(speeds, dtype=float).reshape(-1)
        if not (starts_a.shape[0] == goals_a.shape[0] == speeds_a.shape[0]):
            raise ValueError("starts, goals and speeds must have the same length")
        self.n = int(starts_a.shape[0])
        self.goals = goals_a
        self.speeds = speeds_a
        self.obstacles = jnp.asarray(obstacles, dtype=float).reshape(-1, 2)
        d = goals_a - starts_a
        unit = d / jnp.maximum(jnp.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        self.x0: Array = jnp.concatenate([starts_a, speeds_a[:, None] * unit], axis=1)
        self._policies = [
            social_force_policy(
                goal=goals_a[i],
                desired_speed=float(speeds_a[i]),
                relaxation_time=relaxation_time,
                repulsion_strength=repulsion_strength,
                repulsion_range=repulsion_range,
                pedestrian_radius=ped_radius,
                agent_radius=agent_radius,
            )
            for i in range(self.n)
        ]
        self._cap = speed_cap

    def step(self, t, robot_xy: Array, states: Array, dt: float) -> Array:
        """One Euler step of all pedestrians given the robot's planar position."""
        states = jnp.asarray(states, dtype=float).reshape(self.n, 4)
        robot_xy = jnp.asarray(robot_xy, dtype=float).reshape(1, 2)
        out = []
        for i, pol in enumerate(self._policies):
            others = jnp.concatenate(
                [robot_xy, jnp.delete(states[:, :2], i, axis=0), self.obstacles], axis=0
            )
            acc = pol(t, states[i], {"others_states": others})
            v = states[i, 2:] + acc * dt
            speed = jnp.linalg.norm(v)
            vmax = self._cap * self.speeds[i]
            v = jnp.where(speed > vmax, v * (vmax / (speed + 1e-9)), v)
            out.append(jnp.concatenate([states[i, :2] + v * dt, v]))
        return jnp.stack(out)


__all__ = ["SocialForceCrowd"]
