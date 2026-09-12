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

import jax
import jax.numpy as jnp
from jax import Array


def social_force_accelerations(
    states: Array,
    goals: Array,
    speeds: Array,
    others: Array,
    *,
    relaxation_time: float,
    repulsion_strength: float,
    repulsion_range: float,
    ped_radius: float,
    agent_radius: float,
    arrive_radius: float = 0.0,
) -> Array:
    """Vectorised social-force accelerations for ``N`` pedestrians.

    Same model as :func:`cbfkit.systems.pedestrian.behaviors.social_force_policy` (desired
    force toward the goal + exponential repulsion from every other agent inside
    ``ped_radius + agent_radius + repulsion_range``), evaluated for all pedestrians at once:
    ``states`` ``(N, 4)``, ``goals`` ``(N, 2)``, ``speeds`` ``(N,)``, ``others`` ``(M, 2)``
    extra agents/obstacles (robot, pillars). Pedestrians repel each other too.
    ``arrive_radius > 0`` scales the desired speed down linearly inside that distance of
    the goal so pedestrians stop there instead of oscillating through it (0 = the library
    policy's behaviour).
    """
    pos, vel = states[:, :2], states[:, 2:]
    to_goal = goals - pos
    dist = jnp.linalg.norm(to_goal, axis=1, keepdims=True)
    unit = jnp.where(dist > 1e-6, to_goal / jnp.maximum(dist, 1e-9), 0.0)
    gain = jnp.minimum(1.0, dist / arrive_radius) if arrive_radius > 0 else 1.0
    f_desired = (speeds[:, None] * unit * gain - vel) / relaxation_time
    sum_radii = ped_radius + agent_radius
    cutoff = sum_radii + repulsion_range

    def repulsion(p, q, active):  # force on p from q
        d_vec = p - q
        d = jnp.linalg.norm(d_vec)
        mag = jnp.where(
            active & (d < cutoff),
            repulsion_strength * jnp.exp((sum_radii - d) / repulsion_range),
            0.0,
        )
        return mag * jnp.where(d > 1e-6, d_vec / jnp.maximum(d, 1e-9), 0.0)

    n = pos.shape[0]
    not_self = ~jnp.eye(n, dtype=bool)
    f_peds = jax.vmap(
        lambda i: jax.vmap(lambda j: repulsion(pos[i], pos[j], not_self[i, j]))(jnp.arange(n)).sum(
            0
        )
    )(jnp.arange(n))
    if others.shape[0] > 0:
        f_others = jax.vmap(
            lambda i: jax.vmap(lambda q: repulsion(pos[i], q, True))(others).sum(0)
        )(jnp.arange(n))
    else:
        f_others = jnp.zeros_like(pos)
    return f_desired + f_peds + f_others


class SocialForceCrowd:
    """``N`` social-force pedestrians; see the module docstring.

    Args:
        starts, goals: ``(N, 2)`` positions; ``speeds``: desired speed per pedestrian.
        obstacles: static ``(M, 2)`` points (pillars) the pedestrians also avoid.
        ped_radius / agent_radius: interaction radii (the robot counts as an agent).
        repulsion_strength / repulsion_range / relaxation_time: social-force parameters.
        react_to_robot: if False the pedestrians ignore the robot entirely (social forces
            act only among pedestrians and from the static obstacles), so all of the
            avoidance has to come from the robot's side.
        speed_cap: hard cap on a pedestrian's speed (multiple of its desired speed).
        arrive_radius: slow down and stop within this distance of the goal (0 = keep walking).
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
        arrive_radius: float = 0.0,
        react_to_robot: bool = True,
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
        self._params = dict(
            relaxation_time=relaxation_time,
            repulsion_strength=repulsion_strength,
            repulsion_range=repulsion_range,
            ped_radius=ped_radius,
            agent_radius=agent_radius,
            arrive_radius=arrive_radius,
        )
        self._cap = speed_cap
        self.react_to_robot = bool(react_to_robot)

    def step(self, t, robot_xy: Array, states: Array, dt: float) -> Array:
        """One Euler step of all pedestrians given the robot's planar position."""
        states = jnp.asarray(states, dtype=float).reshape(self.n, 4)
        robot_xy = jnp.asarray(robot_xy, dtype=float).reshape(1, 2)
        others = (
            jnp.concatenate([robot_xy, self.obstacles], axis=0)
            if self.react_to_robot
            else self.obstacles
        )
        acc = social_force_accelerations(states, self.goals, self.speeds, others, **self._params)
        v = states[:, 2:] + acc * dt
        speed = jnp.linalg.norm(v, axis=1, keepdims=True)
        vmax = (self._cap * self.speeds)[:, None]
        v = jnp.where(speed > vmax, v * (vmax / jnp.maximum(speed, 1e-9)), v)
        return jnp.concatenate([states[:, :2] + v * dt, v], axis=1)


__all__ = ["SocialForceCrowd", "social_force_accelerations"]
