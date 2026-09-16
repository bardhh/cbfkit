"""Trajectory costs for reduced-order MuJoCo locomotion planners."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import jax.numpy as jnp


@dataclass(frozen=True)
class EllipseCostWeights:
    """Weights of :func:`ellipse_trajectory_cost`. The clearance term is a *preference*,
    not a wall: quadratic in the violation ``clearance_margin - h`` in absolute h-units,
    saturated at ``clearance_cap``. Random shooting cannot thread the exact centimetres of
    a tight squeeze -- the hard QP downstream owns h >= 0 and does the threading -- so a
    slightly sloppy pass must cost a scrape, a facing-forward push ~10x more (depth is the
    rotation signal), and a deep push through a person the saturated maximum (worse than
    waiting). Normalising the violation by the margin instead (the obvious hinge) makes
    every violation effectively infinite and the planner freezes in front of the gap
    (measured). ``align``/``spin`` are mild legibility terms: face the direction of travel
    when it costs nothing, do not thrash the heading."""

    goal: float = 10.0  # terminal |p_H - goal|
    progress: float = 1.0  # stage |p_k - goal| (per second) -- waiting costs only progress
    clearance: float = 200.0  # x violation^2 (absolute h-units, per agent-second)
    clearance_margin: float = 0.05  # h-units above 0 where the violation starts
    clearance_cap: float = 0.6  # h-units: saturation depth of the violation
    lane: float = 100.0  # (|y| - lane_halfwidth)^2 outside the lane (per second)
    align: float = 0.5  # (1 - cos(th - travel)) |v| -- prefer facing the walk (per second)
    spin: float = 0.5  # omega^2 (per second)
    overturn: float = 40.0  # (|wrap(th - travel)| - pi/2)^2 beyond the +-90 deg band (per second)
    speed: float = 200.0  # relu(|v| - v_max)^2 (per second)


def ellipse_trajectory_cost(
    n_agents: int,
    goal: Any,
    axes: Tuple[float, float],
    ped_radius: float,
    dt: float,
    *,
    heading: bool = True,
    lane_halfwidth: Optional[float] = None,
    v_max: float = 0.5,
    weights: Optional[EllipseCostWeights] = None,
):
    """``TrajectoryCostCallable`` for MPPI over the compact heading-augmented DI state
    ``[p v th om | agents]`` (``heading=True``) or the plain DI state ``[p v | agents]``
    (``heading=False``, circular ``axes``): goal progress, a clearance hinge on the same
    rotating-ellipse ``h`` as :func:`com_agent_ellipse_hocbfs`, an optional lane cost
    (keeps the plan inside ``|y| <= lane_halfwidth`` -- a corridor the planner respects
    without a wall barrier), and mild align/spin legibility terms. With a horizon of a
    few seconds the planner *discovers* that rotating early pays -- the myopic QP cannot
    (measured: it parks facing forward), and a hand-coded suggestion dithers."""
    w = weights if weights is not None else EllipseCostWeights()
    goal = jnp.asarray(goal, dtype=float)[:2]
    a_lon = float(axes[0]) + float(ped_radius)
    a_lat = float(axes[1]) + float(ped_radius)
    head = 6 if heading else 4

    def cost(time, states, controls, prev_robustness=None):
        st = jnp.asarray(states)
        H = st.shape[1]
        P = st[0:2, :].T  # (H, 2)
        V = st[2:4, :].T
        Pp = st[head : head + 4 * n_agents, :].T.reshape(H, n_agents, 4)[..., :2]
        d_goal = jnp.linalg.norm(P - goal, axis=-1)
        c = w.goal * d_goal[-1] + w.progress * jnp.sum(d_goal) * dt
        diff = P[:, None, :] - Pp  # (H, N, 2)
        if heading:
            th = st[4, :]
            cth, sth = jnp.cos(th)[:, None], jnp.sin(th)[:, None]
            lon = (cth * diff[..., 0] + sth * diff[..., 1]) / a_lon
            lat = (-sth * diff[..., 0] + cth * diff[..., 1]) / a_lat
            h = jnp.sqrt(lon**2 + lat**2 + 1e-12) - 1.0
        else:
            h = jnp.linalg.norm(diff, axis=-1) / a_lon - 1.0
        viol = jnp.clip(w.clearance_margin - h, 0.0, w.clearance_cap)
        c = c + w.clearance * jnp.sum(viol**2) * dt
        if lane_halfwidth is not None:
            over = jnp.maximum(jnp.abs(P[:, 1]) - lane_halfwidth, 0.0)
            c = c + w.lane * jnp.sum(over**2) * dt
        sp = jnp.linalg.norm(V, axis=-1)
        c = c + w.speed * jnp.sum(jnp.maximum(sp - v_max, 0.0) ** 2) * dt
        if heading:
            # Heading reference: velocity blended with a small goal-direction bias. Gating
            # by raw speed instead lets the heading wander whenever the robot slows -- the
            # G1 pirouetted a full turn mid-corridor, and in the crowd every stop wound the
            # heading further (measured 5-7 revolutions). With the blend the reference is
            # always defined: standing, prefer facing the goal; walking, face the travel.
            to_goal = goal - P
            gdir = to_goal / (jnp.linalg.norm(to_goal, axis=-1, keepdims=True) + 1e-6)
            refv = V + 0.15 * gdir
            ref = jnp.arctan2(refv[:, 1], refv[:, 0])
            c = c + w.align * jnp.sum(1.0 - jnp.cos(th - ref)) * dt
            c = c + w.spin * jnp.sum(st[5, :] ** 2) * dt
            # +-90 deg heading band around the reference: by the ellipse's pi-symmetry
            # every slimming profile already exists inside the band, so leaving it buys
            # nothing -- it is exactly how the heading winds into a pirouette when threats
            # alternate sides. The fold is 2pi-periodic, so a wound plan is priced until
            # it unwinds.
            fold = jnp.abs(jnp.arctan2(jnp.sin(th - ref), jnp.cos(th - ref)))
            over = jnp.maximum(fold - jnp.pi / 2, 0.0)
            c = c + w.overturn * jnp.sum(over**2) * dt
        return c

    return cost


def heading_social_trajectory_cost(
    n_agents: int,
    goal: Any,
    weights: Any,
    dt: float,
    axes: Tuple[float, float],
    ped_radius: float,
    *,
    robot_radius: float = 0.35,
    pass_side: Optional[str] = None,
    ellipse_weights: Optional[EllipseCostWeights] = None,
):
    """The social MPPI cost re-hosted on the heading-augmented layout ``[p v th om | peds]``.

    The person-centred terms of :func:`cbfkit.controllers.mppi.social_costs.social_cost_terms`
    (proxemics, ttc, goal/progress, legibility, pass side) are evaluated on the sliced DI
    sub-state -- they know nothing about the robot's shape. The circular collision hinge is
    zeroed and replaced by the rotating-ellipse clearance preference of
    :func:`ellipse_trajectory_cost` (saturated quadratic in absolute h-units), so *rotating
    to slim the profile pays inside the plan* -- the planner can aim a shoulder-turn at a gap
    in the crowd before it opens. The ttc term keeps the circular ``robot_radius`` as a
    conservative closing-speed shaping. ``ellipse_weights.goal``/``progress``/``lane``/
    ``speed`` are ignored here (the social cost owns those); only ``clearance*``, ``align``
    and ``spin`` are used, with the align term speed-gated as in
    :func:`ellipse_trajectory_cost`.
    """
    import dataclasses

    from cbfkit.controllers.mppi.social_costs import social_cost_terms

    ew = ellipse_weights if ellipse_weights is not None else EllipseCostWeights()
    w_social = dataclasses.replace(weights, collision=0.0)
    goal = jnp.asarray(goal, dtype=float)[:2]
    a_lon = float(axes[0]) + float(ped_radius)
    a_lat = float(axes[1]) + float(ped_radius)

    def cost(time, states, controls, prev_robustness=None):
        st = jnp.asarray(states)
        H = st.shape[1]
        di = jnp.concatenate([st[0:4], st[6:]], axis=0)
        terms = social_cost_terms(
            di,
            jnp.asarray(controls)[:2],
            n_agents=n_agents,
            goal=goal,
            weights=w_social,
            dt=dt,
            robot_radius=robot_radius,
            ped_radius=ped_radius,
            pass_side=pass_side,
        )
        c = sum(terms.values())
        P = st[0:2, :].T
        V = st[2:4, :].T
        th = st[4, :]
        Pp = st[6 : 6 + 4 * n_agents, :].T.reshape(H, n_agents, 4)[..., :2]
        diff = P[:, None, :] - Pp
        cth, sth = jnp.cos(th)[:, None], jnp.sin(th)[:, None]
        lon = (cth * diff[..., 0] + sth * diff[..., 1]) / a_lon
        lat = (-sth * diff[..., 0] + cth * diff[..., 1]) / a_lat
        h = jnp.sqrt(lon**2 + lat**2 + 1e-12) - 1.0
        viol = jnp.clip(ew.clearance_margin - h, 0.0, ew.clearance_cap)
        c = c + ew.clearance * jnp.sum(viol**2) * dt
        to_goal = goal - P
        gdir = to_goal / (jnp.linalg.norm(to_goal, axis=-1, keepdims=True) + 1e-6)
        refv = V + 0.15 * gdir
        ref = jnp.arctan2(refv[:, 1], refv[:, 0])
        c = c + ew.align * jnp.sum(1.0 - jnp.cos(th - ref)) * dt
        c = c + ew.spin * jnp.sum(st[5, :] ** 2) * dt
        fold = jnp.abs(jnp.arctan2(jnp.sin(th - ref), jnp.cos(th - ref)))
        over = jnp.maximum(fold - jnp.pi / 2, 0.0)
        c = c + ew.overturn * jnp.sum(over**2) * dt
        return c

    return cost
