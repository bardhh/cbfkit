"""Socially aware cost terms for MPPI among pedestrians.

The terms operate on the planar *augmented* layout used by the reduced-order CoM models,
``[px, py, vx, vy | (p_i, v_i) x N]`` (robot position and velocity, then ``N`` tracked agents
with ``px, py, vx, vy`` each -- see :func:`pack_state`), so an MPPI rollout that predicts the
agents at constant velocity (``embedded_double_integrator(2, (0, 1), n_agents=N)``) carries
everything the costs need. They encode what makes robot motion *non-intrusive* to people
rather than merely collision-free:

* **proxemics** -- Kirby's asymmetric-Gaussian personal space (``asymmetric_gaussian``):
  wide in front of a walking pedestrian (and wider the faster she walks), narrow behind, so
  crossing *behind* someone is cheap and cutting *in front* is expensive;
* **ttc** -- a time-to-collision power law (Karamouzas et al. 2014, ``k / ttc^2 exp(-ttc/tau0)``)
  on the relative constant-velocity motion: closing fast is penalised, being near is not;
* **collision** -- a hinge on the contact distance (the CBF-QP downstream handles the hard part);
* **goal** -- terminal distance to the goal plus a *small* stage term: waiting a second costs
  only the progress not made, so "let her pass first" is a first-class plan;
* **jerk / turn / back / speed / slow** -- legibility: smooth accelerations, no zig-zag, no
  backing away from the goal, stay under the tracked speed, walk slowly when close to people;
* **pass_side** -- optional convention for head-on encounters (``"left"`` = keep left, as in
  Japan; ``"right"``; ``None`` = no preference).

:func:`social_trajectory_cost` returns a ``TrajectoryCostCallable`` for
:func:`cbfkit.controllers.mppi.vanilla_mppi`; :func:`social_cost_terms` returns the per-term
breakdown of the same quantities (for ablations and for scoring realised trajectories).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import jax.numpy as jnp
from jax import Array

__all__ = [
    "SocialCostWeights",
    "asymmetric_gaussian",
    "pack_state",
    "social_cost_terms",
    "social_trajectory_cost",
    "time_to_collision",
]


@dataclass(frozen=True)
class SocialCostWeights:
    """Weights (and shape parameters) of the social cost. Defaults are the ones tuned on the
    G1 scramble's 2-D proxy over 5 crowd seeds (config "G_balanced" of
    ``examples/mujoco/g1_scramble_social_eval.py``): vs the goal-seeking baseline they cut the
    intimate-zone rate 4.3 -> 1.4 and front intrusions 2.2 -> 1.0 ped-s per 10 s (the crowd's
    own human norm is 3.5 / 2.5) for +13 s crossing time; confirmed on the MJX G1
    (intimate 4.2/6.2 -> 1.7/2.5 on seeds 0/1)."""

    goal: float = 10.0  # terminal |p_H - goal|
    progress: float = 0.4  # stage |p_k - goal| (per second) -- keeps "waiting" from being free
    proxemics: float = 15.0  # asymmetric-Gaussian personal-space intrusion (per agent-second)
    ttc: float = 5.0  # time-to-collision power law (per agent-second)
    collision: float = 200.0  # hinge^2 on the contact distance (per agent-second)
    jerk: float = 5.0  # |a_k - a_{k-1}|^2 (per step)
    turn: float = 5.0  # heading change, speed-weighted (per step)
    back: float = 2.0  # velocity component *away* from the goal (per second)
    speed: float = 200.0  # relu(|v| - v_max)^2 (per second)
    slow: float = 2.0  # |v|^2 weighted by proximity to people (per agent-second)
    pass_side: float = 0.0  # wrong-side penalty for oncoming pedestrians (per agent-second)
    # shape parameters
    sigma_front: float = 0.7  # m, personal space ahead of a standing pedestrian
    sigma_front_per_speed: float = 0.6  # s, extra front sigma per m/s of pedestrian speed
    sigma_side: float = 0.45  # m
    sigma_rear: float = 0.3  # m
    ttc_tau0: float = 3.0  # s, decay of the TTC power law
    ttc_min: float = (
        0.2  # s, TTC floor (avoids the 1/ttc^2 blow-up at contact; `collision` covers it)
    )
    collision_margin: float = 0.2  # m, hinge starts this far outside contact
    slow_sigma: float = 1.0  # m, range of the proximity-weighted speed penalty
    side_range: float = 3.0  # m, range of the pass-side preference
    v_max: float = 0.5  # m/s, speed the tracking layer can follow


def pack_state(robot: Array, agents: Array) -> Array:
    """``[px, py, vx, vy | agents.reshape(-1)]`` -- the planar augmented layout."""
    agents = jnp.asarray(agents, dtype=float).reshape(-1, 4)
    return jnp.concatenate([jnp.asarray(robot, dtype=float).reshape(4), agents.reshape(-1)])


def asymmetric_gaussian(
    offset: Array,
    heading: Array,
    speed: Array,
    *,
    sigma_front: float = 0.7,
    sigma_front_per_speed: float = 0.6,
    sigma_side: float = 0.45,
    sigma_rear: float = 0.3,
) -> Array:
    """Kirby-style personal space of a pedestrian, evaluated at ``offset`` (robot minus
    pedestrian, any leading batch dims).

    ``heading`` is the pedestrian's velocity direction (unnormalised is fine; a zero vector
    means standing, which makes the space isotropic with ``sigma_side``), ``speed`` her speed.
    The longitudinal sigma is ``sigma_front + sigma_front_per_speed * speed`` ahead of her and
    ``sigma_rear`` behind; the lateral sigma is ``sigma_side``. Returns ``exp(-q/2)`` in (0, 1].
    """
    hn = jnp.linalg.norm(heading, axis=-1, keepdims=True)
    walking = hn[..., 0] > 1e-3
    e_long = jnp.where(hn > 1e-3, heading / jnp.maximum(hn, 1e-9), 0.0)
    e_lat = jnp.stack([-e_long[..., 1], e_long[..., 0]], axis=-1)
    lon = jnp.sum(offset * e_long, axis=-1)
    lat = jnp.sum(offset * e_lat, axis=-1)
    s_front = sigma_front + sigma_front_per_speed * jnp.asarray(speed)
    s_long = jnp.where(lon >= 0.0, s_front, sigma_rear)
    q_walk = lon**2 / s_long**2 + lat**2 / sigma_side**2
    q_still = jnp.sum(offset**2, axis=-1) / sigma_side**2
    return jnp.exp(-0.5 * jnp.where(walking, q_walk, q_still))


def time_to_collision(rel: Array, v_rel: Array, radius: float) -> Array:
    """Time until ``|rel + v_rel t| = radius`` (``rel`` = other minus self, ``v_rel`` = other's
    velocity minus self's). ``0`` if already overlapping, ``inf`` if receding or missing.
    Works on any leading batch dims."""
    a = jnp.sum(v_rel * v_rel, axis=-1)
    b = 2.0 * jnp.sum(rel * v_rel, axis=-1)
    c = jnp.sum(rel * rel, axis=-1) - radius**2
    disc = b * b - 4.0 * a * c
    safe_a = jnp.maximum(a, 1e-9)
    root = (-b - jnp.sqrt(jnp.maximum(disc, 0.0))) / (2.0 * safe_a)
    hits = (disc > 0.0) & (b < 0.0) & (a > 1e-9)
    ttc = jnp.where(hits, jnp.maximum(root, 0.0), jnp.inf)
    return jnp.where(c < 0.0, 0.0, ttc)


def _unpack(states: Array, n_agents: int):
    """(dim, H) -> robot P (H,2), V (H,2), agents positions (H,N,2), velocities (H,N,2)."""
    st = jnp.asarray(states)
    H = st.shape[1]
    P = st[0:2, :].T
    V = st[2:4, :].T
    A = st[4 : 4 + 4 * n_agents, :].T.reshape(H, n_agents, 4)
    return P, V, A[..., :2], A[..., 2:]


def social_cost_terms(
    states: Array,
    controls: Array,
    *,
    n_agents: int,
    goal: Array,
    weights: SocialCostWeights,
    dt: float,
    robot_radius: float = 0.35,
    ped_radius: float = 0.30,
    pass_side: Optional[str] = None,
) -> Dict[str, Array]:
    """Per-term costs (already multiplied by their weights) of one trajectory.

    ``states`` ``(4 + 4 n_agents, H)`` in the :func:`pack_state` layout (agents at their
    predicted positions per step), ``controls`` ``(2, H)`` accelerations, ``dt`` the step.
    Keys: ``goal, progress, proxemics, ttc, collision, jerk, turn, back, speed, slow, pass_side``.
    """
    w = weights
    goal = jnp.asarray(goal, dtype=float)[:2]
    P, V, Pp, Vp = _unpack(states, n_agents)
    U = jnp.asarray(controls)[:2, :].T  # (H, 2)
    R = robot_radius + ped_radius

    d_goal = jnp.linalg.norm(P - goal, axis=-1)  # (H,)
    terms: Dict[str, Array] = {}
    terms["goal"] = w.goal * d_goal[-1]
    terms["progress"] = w.progress * jnp.sum(d_goal) * dt

    # --- agents
    offset = P[:, None, :] - Pp  # robot minus pedestrian (H, N, 2)
    dist = jnp.linalg.norm(offset, axis=-1)  # (H, N)
    ped_speed = jnp.linalg.norm(Vp, axis=-1)
    prox = asymmetric_gaussian(
        offset,
        Vp,
        ped_speed,
        sigma_front=w.sigma_front,
        sigma_front_per_speed=w.sigma_front_per_speed,
        sigma_side=w.sigma_side,
        sigma_rear=w.sigma_rear,
    )
    terms["proxemics"] = w.proxemics * jnp.sum(prox) * dt

    ttc = time_to_collision(-offset, Vp - V[:, None, :], R)
    ttc_c = jnp.maximum(ttc, w.ttc_min)
    ttc_cost = jnp.where(jnp.isfinite(ttc), jnp.exp(-ttc_c / w.ttc_tau0) / ttc_c**2, 0.0)
    terms["ttc"] = w.ttc * jnp.sum(ttc_cost) * dt

    hinge = jnp.maximum(R + w.collision_margin - dist, 0.0) / max(w.collision_margin, 1e-6)
    terms["collision"] = w.collision * jnp.sum(hinge**2) * dt

    near = jnp.exp(-0.5 * dist**2 / w.slow_sigma**2)  # (H, N)
    speed2 = jnp.sum(V * V, axis=-1)  # (H,)
    terms["slow"] = w.slow * jnp.sum(speed2[:, None] * near) * dt

    # --- legibility
    dU = U[1:] - U[:-1]
    terms["jerk"] = w.jerk * jnp.sum(dU * dU)
    sp = jnp.linalg.norm(V, axis=-1)
    cos = jnp.sum(V[1:] * V[:-1], axis=-1) / jnp.maximum(sp[1:] * sp[:-1], 1e-6)
    terms["turn"] = w.turn * jnp.sum((1.0 - cos) * jnp.minimum(sp[1:], sp[:-1]))
    to_goal = goal - P
    g_hat = to_goal / jnp.maximum(jnp.linalg.norm(to_goal, axis=-1, keepdims=True), 1e-6)
    terms["back"] = w.back * jnp.sum(jnp.maximum(-jnp.sum(V * g_hat, axis=-1), 0.0)) * dt
    terms["speed"] = w.speed * jnp.sum(jnp.maximum(sp - w.v_max, 0.0) ** 2) * dt

    # --- pass-side convention (head-on encounters only)
    if pass_side is None or w.pass_side == 0.0:
        terms["pass_side"] = jnp.zeros(())
    else:
        if pass_side not in ("left", "right"):
            raise ValueError("pass_side must be 'left', 'right' or None")
        v_hat = V / jnp.maximum(sp[:, None], 1e-6)  # (H, 2)
        rel = -offset  # pedestrian minus robot (H, N, 2)
        ahead = jnp.sum(rel * v_hat[:, None, :], axis=-1)  # (H, N)
        side = (
            v_hat[:, None, 0] * rel[..., 1] - v_hat[:, None, 1] * rel[..., 0]
        )  # > 0: on robot's left
        closing = -jnp.sum(
            Vp * v_hat[:, None, :], axis=-1
        )  # > 0: pedestrian coming toward the robot
        gate = (
            jnp.clip(closing / 0.3, 0.0, 1.0)
            * jnp.clip(ahead / 0.5, 0.0, 1.0)
            * jnp.clip((w.side_range - dist) / 1.0, 0.0, 1.0)
        )
        wrong = jnp.maximum(side, 0.0) if pass_side == "left" else jnp.maximum(-side, 0.0)
        terms["pass_side"] = w.pass_side * jnp.sum(gate * wrong) * dt
    return terms


def social_trajectory_cost(
    *,
    n_agents: int,
    goal: Array,
    weights: SocialCostWeights,
    dt: float,
    robot_radius: float = 0.35,
    ped_radius: float = 0.30,
    pass_side: Optional[str] = None,
):
    """``TrajectoryCostCallable`` ``(time, states (dim, H), controls (m, H), prev_robustness)``
    summing :func:`social_cost_terms`; pass as ``trajectory_cost=`` to ``vanilla_mppi``."""
    goal = jnp.asarray(goal, dtype=float)[:2]

    def cost(time, states, controls, prev_robustness=None):
        terms = social_cost_terms(
            states,
            controls,
            n_agents=n_agents,
            goal=goal,
            weights=weights,
            dt=dt,
            robot_radius=robot_radius,
            ped_radius=ped_radius,
            pass_side=pass_side,
        )
        return sum(terms.values())

    return cost
