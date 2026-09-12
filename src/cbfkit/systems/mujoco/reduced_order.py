"""Reduced-order CBF safety on top of a MuJoCo plant.

The certificate lives on a 2-D single integrator on the robot's centre of mass,
``d/dt com_xy = v``, and filters the planar velocity *command* that a
locomotion controller (sampling MPC, learned policy, ...) tracks. Nothing here
knows about the humanoid's 71 states -- the QP sees the plant's flat state only
through ``plant.com_indices``.

Two tricks make this work with CBFKit's stock CBF-QP generator and barriers:

* :func:`embedded_single_integrator` writes the reduced dynamics in the *full*
  state's coordinates (``f = 0``, ``g`` = unit columns at the CoM indices), so
  ``∂h/∂x · g`` in the QP is exactly the 2-D barrier gradient at the CoM and no
  projection hook is needed.
* :func:`com_obstacle_barriers` points ``ellipsoidal_barrier_factory`` at the
  CoM entries of the flat state (``[qpos | qvel | com_xyz]``) instead of its
  default ``(0, 1)`` -- which on a floating-base robot would be the pelvis
  free-joint position, not the CoM.
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, cast

import jax.numpy as jnp
from jax import Array

from cbfkit.certificates import certificate_package, concatenate_certificates
from cbfkit.certificates.barrier_functions import ellipsoidal_barrier_factory
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.utils.user_types import CertificateCollection, ControllerCallable, DynamicsCallable


def embedded_single_integrator(state_dim: int, indices: Tuple[int, int]) -> DynamicsCallable:
    """``ẋ = g u`` with ``g`` the two unit columns at ``indices``; ``f = 0``. Control is (vx, vy)."""
    g = jnp.zeros((state_dim, 2)).at[indices[0], 0].set(1.0).at[indices[1], 1].set(1.0)
    f = jnp.zeros(state_dim)

    def dynamics(x: Array):
        return f, g

    return dynamics


def com_obstacle_barriers(
    plant: Any,
    obstacles: Sequence[Sequence[float]],
    ellipsoids: Sequence[Sequence[float]],
    class_k_gain: float = 1.0,
):
    """Ellipsoidal keep-out barriers around planar obstacles, evaluated at the plant's CoM.

    ``obstacles`` are ``(x, y)`` centres, ``ellipsoids`` are ``(a, b)`` semi-axes
    (already inflated by the robot's planar radius). ``h = ((cx-x)/a)^2 +
    ((cy-y)/b)^2 - 1 >= 0`` is safe. Returns a ``CertificateCollection`` for
    ``vanilla_cbf_clf_qp_controller(barriers=...)``.
    """
    cbf, cbf_grad, cbf_hess = ellipsoidal_barrier_factory(
        system_position_indices=tuple(plant.com_indices),
        obstacle_position_indices=(0, 1),
        ellipsoid_axis_indices=(0, 1),
    )
    package = certificate_package(cbf, cbf_grad, cbf_hess, plant.state_dim)
    conditions = zeroing_barriers.linear_class_k(class_k_gain)
    return concatenate_certificates(
        *[
            package(
                certificate_conditions=conditions,
                obstacle=jnp.asarray(o, dtype=float),
                ellipsoid=jnp.asarray(e, dtype=float),
            )
            for o, e in zip(obstacles, ellipsoids)
        ]
    )


def safe_locomotion_controller(
    cbf_qp: ControllerCallable, locomotion: ControllerCallable
) -> ControllerCallable:
    """Compose ``v_safe = cbf_qp(x, v_nom)`` then ``u = locomotion(x, v_safe)``.

    Both parts are ``ControllerCallable``s. The CBF-QP acts on the 2-D velocity
    command; the locomotion controller receives it as its ``u_nom`` (for
    ``SamplingMpc.as_controller()`` that is the ``aux`` its costs track) and
    returns the actuator command. Per-step state of both lives in one
    ``ControllerData.sub_data``: the CBF's entries plus the locomotion
    controller's carry-only ``"_..."`` keys; ``"v_nom"``/``"v_safe"`` are logged.
    """

    def controller(t, x, v_nom, key, data):
        prev = data.sub_data if data.sub_data is not None else {}
        v_safe, d1 = cbf_qp(t, x, v_nom, key, data)
        sub = dict(d1.sub_data) if d1.sub_data is not None else {}
        for k, v in prev.items():  # keep the locomotion controller's carry-only state
            if k.startswith("_") and k not in sub:
                sub[k] = v
        u, d2 = locomotion(t, x, v_safe, key, d1._replace(sub_data=sub))
        sub2 = dict(d2.sub_data) if d2.sub_data is not None else {}
        sub2["v_nom"] = jnp.asarray(v_nom)
        sub2["v_safe"] = jnp.asarray(v_safe)
        # A QP failure is a controller error; goal completion is the CBF's call.
        return u, d2._replace(
            sub_data=sub2, u=u, u_nom=jnp.asarray(v_nom), error=d1.error | d2.error
        )

    controller.__cbfkit_controller_adapter__ = True  # type: ignore[attr-defined]
    return controller


__all__ = ["com_obstacle_barriers", "embedded_single_integrator", "safe_locomotion_controller"]


# --------------------------------------------------------------------------- double integrator
def agent_slice(state_dim: int, i: int) -> slice:
    """Slots of tracked agent ``i`` -- ``(px, py, vx, vy)`` -- in the augmented DI state."""
    start = state_dim + 2 + 4 * i
    return slice(start, start + 4)


def embedded_double_integrator(
    state_dim: int, indices: Tuple[int, int], n_agents: int = 0
) -> DynamicsCallable:
    """Command-side double integrator on the CoM, in *augmented* coordinates ``[x | v | agents]``.

    ``x`` is the plant's flat state (``state_dim`` entries) and ``v`` (2 entries,
    appended) is the *commanded* planar CoM velocity that the locomotion layer
    tracks. Dynamics: ``d/dt com = v``, ``d/dt v = a`` (control), everything else 0.
    Position barriers then have relative degree 2 in ``a`` and go through
    ``rectify_relative_degree`` (high-order CBF).

    With ``n_agents > 0`` the state carries ``n_agents`` tracked agents
    (``px, py, vx, vy`` each, see :func:`agent_slice`) with ``d/dt p_i = v_i`` -- a
    constant-velocity prediction from the agent's *current* velocity, so a barrier on
    ``p_i`` sees the agent's motion in ``dh/dt`` (its acceleration is unmodelled).
    """
    n = state_dim + 2 + 4 * n_agents
    g = jnp.zeros((n, 2)).at[state_dim, 0].set(1.0).at[state_dim + 1, 1].set(1.0)

    def dynamics(xa: Array):
        f = jnp.zeros(n).at[indices[0]].set(xa[state_dim]).at[indices[1]].set(xa[state_dim + 1])
        for i in range(n_agents):
            sl = agent_slice(state_dim, i)
            f = f.at[sl.start : sl.start + 2].set(xa[sl.start + 2 : sl.stop])
        return f, g

    return dynamics


def _keepout(diff_over_axes: Array, shape: str) -> Array:
    """``h`` of a keep-out around a point from the axis-normalised offset ``(c - p) / (a, b)``.

    ``"ellipsoid"``: ``|d|^2 - 1`` (the stock barrier; gradient grows with distance).
    ``"distance"``: ``|d| - 1`` (unit-norm gradient, so a robust margin ``|dh/dx| * delta``
    does not grow with distance and the HOCBF approach limit is distance-proportional).
    """
    q = jnp.sum(diff_over_axes**2)
    if shape == "ellipsoid":
        return q - 1.0
    if shape == "distance":
        return jnp.sqrt(q + 1e-12) - 1.0
    raise ValueError(f"unknown barrier shape {shape!r}; use 'ellipsoid' or 'distance'")


def _com_static_barrier(plant: Any, p: Any, ellipsoid: Any, shape: str):
    ci = plant.com_indices
    p = jnp.asarray(p, dtype=float)
    axes = jnp.asarray(ellipsoid, dtype=float)

    def h(x):
        return _keepout((jnp.asarray(x)[ci[0] : ci[0] + 2] - p) / axes, shape)

    return h


def com_obstacle_hocbfs(
    plant: Any,
    obstacles: Sequence[Sequence[float]],
    ellipsoids: Sequence[Sequence[float]],
    class_k_gain: float = 1.0,
    roots: Any = None,
    *,
    shape: str = "ellipsoid",
    n_agents: int = 0,
):
    """High-order (relative-degree-2) CoM keep-out barriers for :func:`embedded_double_integrator`.

    Same ellipsoids as :func:`com_obstacle_barriers`, lifted to the augmented
    ``[x | v | agents]`` state with ``rectify_relative_degree(form="high-order")``.
    ``shape`` selects the barrier form (see ``_keepout``); ``n_agents`` must match the
    dynamics the QP is built on (the barriers are evaluated on the same augmented state).
    """
    from cbfkit.certificates import rectify_relative_degree

    n = plant.state_dim + 2 + 4 * n_agents
    dyn = embedded_double_integrator(plant.state_dim, plant.com_indices, n_agents)
    conditions = zeroing_barriers.linear_class_k(class_k_gain)
    return concatenate_certificates(
        *[
            cast(
                CertificateCollection,
                rectify_relative_degree(
                    function=_com_static_barrier(plant, o, e, shape),
                    system_dynamics=dyn,
                    state_dim=n,
                    roots=roots,
                    form="high-order",
                    certificate_conditions=conditions,
                    input_style="state",
                ),
            )
            for o, e in zip(obstacles, ellipsoids)
        ]
    )


def safe_locomotion_controller_di(
    cbf_qp: ControllerCallable,
    locomotion: ControllerCallable,
    plant: Any,
    dt: float,
    *,
    v_max: float = 0.5,
    velocity_gain: float = 2.0,
    agents: Any = None,
    local_planner: Any = None,
) -> ControllerCallable:
    """Double-integrator variant of :func:`safe_locomotion_controller`.

    The nominal ``v_nom`` (2-D velocity) is turned into a nominal acceleration
    ``a_nom = velocity_gain * (v_nom - v)`` toward it; ``cbf_qp`` (built on
    :func:`embedded_double_integrator` + :func:`com_obstacle_hocbfs`) filters it
    on the augmented state ``[x | v]``; the commanded velocity is integrated,
    ``v <- clip(v + a_safe dt, |v| <= v_max)``, carried in ``sub_data["_di_v"]``,
    and handed to the locomotion controller as its command. Logged:
    ``v_nom``, ``v_safe`` (= the integrated command), ``a_nom``, ``a_safe``.

    ``agents`` (optional) adds tracked agents to the augmented state -- an object with
    ``x0`` (``(N, 4)`` initial ``px, py, vx, vy``) and ``step(t, robot_xy, states, dt)
    -> states`` (e.g. :class:`cbfkit.systems.mujoco.crowd.SocialForceCrowd`). The QP sees
    the agents' *current* states (``embedded_double_integrator(n_agents=N)`` predicts
    them at constant velocity); they are stepped after the QP with the robot's CoM,
    carried in ``sub_data["_agents"]`` and the states the QP used are logged as
    ``"agents"``.

    ``local_planner`` (optional) replaces the P-law on ``v_nom`` by a planner that sees the
    *compact* augmented state ``[com | v | agents]`` (``4 + 4 N`` entries, the
    :func:`cbfkit.controllers.mppi.social_costs.pack_state` layout):
    ``local_planner(t, xa_compact, v_nom, key, sub) -> (a_nom, v_plan, sub_updates)`` --
    ``a_nom`` goes to the QP, ``v_plan`` is logged as ``v_nom`` and ``sub_updates`` is merged
    into the controller carry (``_``-prefixed keys persist across steps, others are logged).
    See :func:`mppi_local_planner`.
    """
    ci = plant.com_indices

    def controller(t, x, v_nom, key, data):
        prev = data.sub_data if data.sub_data is not None else {}
        v = prev.get("_di_v")
        if v is None:
            v = jnp.zeros(2)
        v_nom = jnp.asarray(v_nom, dtype=float)[:2]
        if agents is not None:
            ag = prev.get("_agents")
            if ag is None:
                ag = jnp.asarray(agents.x0, dtype=float)
            xa = jnp.concatenate([x, v, ag.reshape(-1)])
            ag_flat = ag.reshape(-1)
        else:
            xa = jnp.concatenate([x, v])
            ag_flat = jnp.zeros(0)
        sub_lp: dict = {}
        if local_planner is not None:
            xa_c = jnp.concatenate([x[ci[0] : ci[0] + 2], v, ag_flat])
            a_nom, v_nom, sub_lp = local_planner(t, xa_c, v_nom, key, prev)
            a_nom = jnp.asarray(a_nom, dtype=float)[:2]
            v_nom = jnp.asarray(v_nom, dtype=float)[:2]
        else:
            a_nom = velocity_gain * (v_nom - v)
        a_safe, d1 = cbf_qp(t, xa, a_nom, key, data)
        v_new = v + jnp.asarray(a_safe) * dt
        speed = jnp.linalg.norm(v_new)
        v_new = jnp.where(speed > v_max, v_new * (v_max / (speed + 1e-9)), v_new)
        sub = dict(d1.sub_data) if d1.sub_data is not None else {}
        for k, val in prev.items():
            if k.startswith("_") and k not in sub:
                sub[k] = val
        sub["_di_v"] = v_new
        sub.update(sub_lp)
        if agents is not None:
            sub["_agents"] = agents.step(t, x[ci[0] : ci[0] + 2], ag, dt)
        u, d2 = locomotion(t, x, v_new, key, d1._replace(sub_data=sub))
        sub2 = dict(d2.sub_data) if d2.sub_data is not None else {}
        sub2["v_nom"] = v_nom
        sub2["v_safe"] = v_new
        sub2["a_nom"] = a_nom
        sub2["a_safe"] = jnp.asarray(a_safe)
        if agents is not None:
            sub2["agents"] = ag
        return u, d2._replace(sub_data=sub2, u=u, u_nom=v_nom, error=d1.error | d2.error)

    controller.__cbfkit_controller_adapter__ = True  # type: ignore[attr-defined]
    return controller


__all__ += ["com_obstacle_hocbfs", "embedded_double_integrator", "safe_locomotion_controller_di"]


# --------------------------------------------------------------------------- moving obstacles
def moving_obstacle_position(p0: Any, v: Any, t: Any) -> Array:
    """Known constant-velocity obstacle: ``p(t) = p0 + v t``."""
    return jnp.asarray(p0, dtype=float) + jnp.asarray(v, dtype=float) * t


def _com_moving_barrier(plant: Any, p0: Any, v: Any, ellipsoid: Any, shape: str = "ellipsoid"):
    """``h(t, x)`` of a keep-out (``shape``, see ``_keepout``) around ``p(t) = p0 + v t``."""
    ci = plant.com_indices
    p0 = jnp.asarray(p0, dtype=float)
    v = jnp.asarray(v, dtype=float)
    axes = jnp.asarray(ellipsoid, dtype=float)

    def h(t, x):
        com = jnp.asarray(x)[ci[0] : ci[0] + 2]
        return _keepout((com - moving_obstacle_position(p0, v, t)) / axes, shape)

    return h


def com_moving_obstacle_barriers(
    plant: Any,
    positions: Sequence[Sequence[float]],
    velocities: Sequence[Sequence[float]],
    ellipsoids: Sequence[Sequence[float]],
    class_k_gain: float = 1.0,
):
    """Time-varying keep-out barriers around constant-velocity obstacles (single-integrator model).

    ``positions[i] + velocities[i] * t`` is obstacle ``i``'s centre at time ``t``; the QP receives
    ``dh/dt`` through the packaged partial (``input_style="separated"``). Companion of
    :func:`com_obstacle_barriers` for :func:`embedded_single_integrator`.
    """
    conditions = zeroing_barriers.linear_class_k(class_k_gain)
    packages = []
    for p0, v, e in zip(positions, velocities, ellipsoids):
        h = _com_moving_barrier(plant, p0, v, e)
        # certificate_package expects a *factory*; bind h now (default arg) to avoid late binding.
        factory = certificate_package(lambda h=h: h, n=plant.state_dim, input_style="separated")
        packages.append(factory(certificate_conditions=conditions))
    return concatenate_certificates(*packages)


def com_moving_obstacle_hocbfs(
    plant: Any,
    positions: Sequence[Sequence[float]],
    velocities: Sequence[Sequence[float]],
    ellipsoids: Sequence[Sequence[float]],
    class_k_gain: float = 1.0,
    roots: Any = None,
    *,
    shape: str = "ellipsoid",
    n_agents: int = 0,
):
    """High-order, time-varying keep-out barriers for :func:`embedded_double_integrator`.

    Same barrier as :func:`com_moving_obstacle_barriers`, on the augmented ``[x | v]`` state;
    ``rectify_relative_degree`` carries ``dh/dt`` into the lifted barrier, so an obstacle walking
    toward a standing robot makes the QP act even though the robot's own velocity is zero.
    """
    from cbfkit.certificates import rectify_relative_degree

    n = plant.state_dim + 2 + 4 * n_agents
    dyn = embedded_double_integrator(plant.state_dim, plant.com_indices, n_agents)
    conditions = zeroing_barriers.linear_class_k(class_k_gain)
    return concatenate_certificates(
        *[
            cast(
                CertificateCollection,
                rectify_relative_degree(
                    function=_com_moving_barrier(plant, p0, v, e, shape),
                    system_dynamics=dyn,
                    state_dim=n,
                    roots=roots,
                    form="high-order",
                    certificate_conditions=conditions,
                    input_style="separated",
                ),
            )
            for p0, v, e in zip(positions, velocities, ellipsoids)
        ]
    )


__all__ += [
    "com_moving_obstacle_barriers",
    "com_moving_obstacle_hocbfs",
    "moving_obstacle_position",
]


# --------------------------------------------------------------------------- tracked agents
def com_agent_hocbfs(
    plant: Any,
    n_agents: int,
    ellipsoids: Sequence[Sequence[float]],
    class_k_gain: float = 1.0,
    roots: Any = None,
    *,
    shape: str = "distance",
):
    """High-order keep-out barriers around the ``n_agents`` tracked agents of the augmented state.

    Agent ``i`` lives at :func:`agent_slice` of the ``[x | v | agents]`` state used by
    ``embedded_double_integrator(n_agents=...)``; its predicted motion ``d/dt p_i = v_i``
    enters ``dh/dt`` through the dynamics, so an agent walking at the robot makes the QP
    act. Default ``shape="distance"`` (unit-norm gradient) so robust margins stay
    distance-independent. Pair with :func:`safe_locomotion_controller_di(agents=...)`.
    """
    from cbfkit.certificates import rectify_relative_degree

    ci = plant.com_indices
    n = plant.state_dim + 2 + 4 * n_agents
    dyn = embedded_double_integrator(plant.state_dim, ci, n_agents)
    conditions = zeroing_barriers.linear_class_k(class_k_gain)

    def barrier(i, ellipsoid):
        sl = agent_slice(plant.state_dim, i)
        axes = jnp.asarray(ellipsoid, dtype=float)

        def h(xa):
            xa = jnp.asarray(xa)
            return _keepout((xa[ci[0] : ci[0] + 2] - xa[sl.start : sl.start + 2]) / axes, shape)

        return h

    return concatenate_certificates(
        *[
            cast(
                CertificateCollection,
                rectify_relative_degree(
                    function=barrier(i, e),
                    system_dynamics=dyn,
                    state_dim=n,
                    roots=roots,
                    form="high-order",
                    certificate_conditions=conditions,
                    input_style="state",
                ),
            )
            for i, e in zip(range(n_agents), ellipsoids)
        ]
    )


__all__ += ["agent_slice", "com_agent_hocbfs"]


# --------------------------------------------------------------------------- local planners
def mppi_local_planner(
    mppi: Any,
    n_agents: int,
    *,
    horizon: int,
    replan_every: int,
    velocity_gain: float = 2.0,
    control_dim: int = 2,
    state_head: int = 4,
    plan_smoothing: float = 0.0,
) -> Any:
    """Adapt a ``cbfkit.controllers.mppi`` planner to the ``local_planner`` hook of
    :func:`safe_locomotion_controller_di`.

    ``mppi`` is the ``PlannerCallable`` from ``vanilla_mppi(...)`` built on
    ``embedded_double_integrator(2, (0, 1), n_agents)`` (the compact layout) with a
    ``trajectory_cost`` such as :func:`cbfkit.controllers.mppi.social_costs.social_trajectory_cost`.
    It is re-solved every ``replan_every`` controller steps (``= mppi time step / controller
    dt``, so the warm-start shift of one MPPI step per solve is exact) and its first planned
    acceleration is held in between; the QP receives it as ``a_nom``. The warm start
    ``_mppi_u_traj`` ``(horizon, 2)``, the held acceleration ``_mppi_a``, the plan
    ``_mppi_x_traj`` ``(state_head + 4 N, horizon + 1)`` and a step counter ``_mppi_k`` ride
    in the controller carry; the plan is also logged as ``mppi_x_traj`` and the solver flag as
    ``mppi_error``. On an MPPI failure (NaN) the P-law ``velocity_gain (v_nom - v)`` is used
    (padded with zeros beyond the two acceleration channels).

    ``control_dim`` / ``state_head`` generalise the layout: the default ``(2, 4)`` is the
    planar DI hook of :func:`safe_locomotion_controller_di`; ``(3, 6)`` adapts an MPPI built
    on :func:`embedded_heading_double_integrator` (controls ``[ax, ay, alpha]``, compact
    state ``[p v th om | agents]``) to the hook of :func:`safe_locomotion_controller_hdi`.

    ``plan_smoothing`` in [0, 1) retains that fraction of the previous nominal
    acceleration at each successful replan. The first solve is unsmoothed; the QP
    still filters the resulting nominal. The MPPI warm start and logged trajectory
    remain the optimizer's raw plan. Zero preserves the original behavior.
    """
    from jax import lax

    from cbfkit.utils.user_types import PlannerData

    if not 0.0 <= plan_smoothing < 1.0:
        raise ValueError("plan_smoothing must be in [0, 1)")
    dim = state_head + 4 * n_agents

    def planner(t, xa, v_nom, key, sub):
        U = sub.get("_mppi_u_traj")
        if U is None:
            U = jnp.zeros((horizon, control_dim))
        a_hold = sub.get("_mppi_a")
        if a_hold is None:
            a_hold = jnp.zeros(control_dim)
        X_hold = sub.get("_mppi_x_traj")
        if X_hold is None:
            X_hold = jnp.zeros((dim, horizon + 1))
        k = sub.get("_mppi_k")
        if k is None:
            k = jnp.zeros((), dtype=jnp.int32)
        err_hold = sub.get("_mppi_error", jnp.asarray(False))

        def solve(_):
            u, d = mppi(t, xa, None, key, PlannerData(u_traj=U))
            a = jnp.asarray(u, dtype=float)[:control_dim]
            if plan_smoothing:
                a = jnp.where(k > 0, plan_smoothing * a_hold + (1 - plan_smoothing) * a, a)
            return (
                a,
                d.u_traj,
                d.x_traj,
                jnp.asarray(d.error),
            )

        def hold(_):
            return a_hold, U, X_hold, err_hold

        a, U_new, X_new, err = lax.cond(k % replan_every == 0, solve, hold, None)
        v = xa[2:4]
        fallback = jnp.concatenate(
            [velocity_gain * (jnp.asarray(v_nom, dtype=float)[:2] - v), jnp.zeros(control_dim - 2)]
        )
        a_nom = jnp.where(err, fallback, a)
        # A failed sample must not poison either the warm start or a later smoothed
        # nominal. Keep reporting the failure until a successful replan recovers.
        U_new = jnp.where(err, U, U_new)
        X_new = jnp.where(err, X_hold, X_new)
        v_plan = X_new[2:4, 1]
        return (
            a_nom,
            v_plan,
            {
                "_mppi_u_traj": U_new,
                "_mppi_a": a_nom,
                "_mppi_error": err,
                "_mppi_x_traj": X_new,
                "_mppi_k": k + 1,
                "mppi_x_traj": X_new,
                "mppi_error": err,
            },
        )

    return planner


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


__all__ += [
    "EllipseCostWeights",
    "ellipse_trajectory_cost",
    "heading_social_trajectory_cost",
    "mppi_local_planner",
]


# --------------------------------------------------------------------------- anisotropic footprint
# Adopted rotating-ellipse footprint of the G1 under the AMO whole-body policy, measured by
# examples/mujoco/g1_footprint_measure.py (MJX, stand / walk vx 0.4 / sidestep vy 0.3):
# upper-body (z > 0.6 m) extents from the CoM in the pelvis-yaw frame. The disc it replaces
# is 0.35 m. Legs exceed the longitudinal extent during forward stride (documented choice:
# pedestrian discs describe whole people too, and feet interleave when humans squeeze).
G1_FOOTPRINT = {"lon": 0.16, "lat": 0.28}


def heading_slice(state_dim: int) -> slice:
    """Slots of ``(theta, omega)`` in the heading-augmented DI state."""
    return slice(state_dim + 2, state_dim + 4)


def hd_agent_slice(state_dim: int, i: int) -> slice:
    """Agent ``i``'s ``(px, py, vx, vy)`` in the heading-augmented DI state."""
    start = state_dim + 4 + 4 * i
    return slice(start, start + 4)


def embedded_heading_double_integrator(
    state_dim: int, indices: Tuple[int, int], n_agents: int = 0
) -> DynamicsCallable:
    """Command-side double integrator *with a heading channel*: ``[x | v | theta omega | agents]``.

    ``d/dt com = v``, ``d/dt v = a``, ``d/dt theta = omega``, ``d/dt omega = alpha`` --
    controls ``u = [ax, ay, alpha]``. Making the heading second order keeps every barrier
    uniformly relative degree 2, so :func:`cbfkit.certificates.rectify_relative_degree`
    applies unchanged (a direct ``omega`` input would create mixed-degree barriers).
    Agents as in :func:`embedded_double_integrator` (constant-velocity prediction).
    """
    n = state_dim + 4 + 4 * n_agents
    g = (
        jnp.zeros((n, 3))
        .at[state_dim, 0]
        .set(1.0)
        .at[state_dim + 1, 1]
        .set(1.0)
        .at[state_dim + 3, 2]
        .set(1.0)
    )

    def dynamics(xa: Array):
        f = (
            jnp.zeros(n)
            .at[indices[0]]
            .set(xa[state_dim])
            .at[indices[1]]
            .set(xa[state_dim + 1])
            .at[state_dim + 2]
            .set(xa[state_dim + 3])
        )
        for i in range(n_agents):
            sl = hd_agent_slice(state_dim, i)
            f = f.at[sl.start : sl.start + 2].set(xa[sl.start + 2 : sl.stop])
        return f, g

    return dynamics


def com_agent_ellipse_hocbfs(
    plant: Any,
    n_agents: int,
    axes: Tuple[float, float],
    ped_radius: float,
    class_k_gain: float = 1.0,
    roots: Any = None,
):
    """Rotating-ellipse keep-out barriers around tracked agents (heading-augmented state).

    ``h_i = || diag(1/(lon + r), 1/(lat + r)) R(theta)^T (com - p_i) || - 1`` with
    ``axes = (lon, lat)`` the footprint semi-axes in the *body* frame (longitudinal =
    facing direction, lateral = shoulder line) and ``r`` the pedestrian radius. Rotating
    the body so the *longitudinal* (narrow) axis spans a gap raises ``h`` -- the QP can
    trade heading acceleration against braking and discovers sidestepping on its own.
    Distance shaping (norm - 1), relative degree 2 in ``[a, alpha]`` via the rectifier.
    """
    from cbfkit.certificates import rectify_relative_degree

    ci = plant.com_indices
    sd = plant.state_dim
    n = sd + 4 + 4 * n_agents
    dyn = embedded_heading_double_integrator(sd, ci, n_agents)
    conditions = zeroing_barriers.linear_class_k(class_k_gain)
    a_lon = float(axes[0]) + float(ped_radius)
    a_lat = float(axes[1]) + float(ped_radius)

    def barrier(i):
        sl = hd_agent_slice(sd, i)

        def h(xa):
            xa = jnp.asarray(xa)
            diff = xa[ci[0] : ci[0] + 2] - xa[sl.start : sl.start + 2]
            th = xa[sd + 2]
            c, s = jnp.cos(th), jnp.sin(th)
            lon = (c * diff[0] + s * diff[1]) / a_lon
            lat = (-s * diff[0] + c * diff[1]) / a_lat
            return jnp.sqrt(lon**2 + lat**2 + 1e-12) - 1.0

        return h

    return concatenate_certificates(
        *[
            cast(
                CertificateCollection,
                rectify_relative_degree(
                    function=barrier(i),
                    system_dynamics=dyn,
                    state_dim=n,
                    roots=roots,
                    form="high-order",
                    certificate_conditions=conditions,
                    input_style="state",
                ),
            )
            for i in range(n_agents)
        ]
    )


def safe_locomotion_controller_hdi(
    cbf_qp: ControllerCallable,
    locomotion: ControllerCallable,
    plant: Any,
    dt: float,
    *,
    v_max: float = 0.5,
    omega_max: float = 1.0,
    velocity_gain: float = 2.0,
    heading_gain: float = 2.0,
    agents: Any = None,
    face_velocity: bool = True,
    local_planner: Any = None,
) -> ControllerCallable:
    """Heading-augmented variant of :func:`safe_locomotion_controller_di`.

    The commanded state is ``[v (2) | theta | omega]``: the nominal acceleration is
    ``velocity_gain (v_nom - v)`` and the nominal heading acceleration turns toward the
    direction of travel (``face_velocity=True``; a 3-entry ``v_nom`` overrides the target
    heading with ``v_nom[2]``). ``cbf_qp`` -- built on
    :func:`embedded_heading_double_integrator` + :func:`com_agent_ellipse_hocbfs` --
    filters ``[a, alpha]`` jointly, so rotating the footprint is a *control choice* the
    QP makes only when it pays. The integrated ``[v, theta]`` command goes to the
    locomotion layer as ``[vx, vy, target_yaw]`` (the AMO adapter accepts the 3rd entry).
    Carry: ``_hdi_v`` (2), ``_hdi_th`` (2: theta, omega), ``_agents`` as in the DI
    wrapper; logged: ``v_nom``, ``v_safe``, ``theta_cmd``, ``a_nom``, ``a_safe`` (3).

    ``local_planner`` (optional) replaces the P-laws on ``v_nom`` by a planner with real
    lookahead over the *compact* heading-augmented state ``[com | v | theta omega | agents]``
    (``6 + 4 N`` entries): ``local_planner(t, xa_compact, v_nom, key, sub) ->
    (a_nom (3), v_plan, sub_updates)`` -- ``a_nom = [ax, ay, alpha]`` goes to the QP, so the
    *rotation itself is planned* (the myopic QP alone never invents it; measured in
    ``examples/mujoco/g1_corridor.py``). See :func:`mppi_local_planner` with
    ``control_dim=3, state_head=6`` and :func:`ellipse_trajectory_cost`.
    """
    ci = plant.com_indices

    def controller(t, x, v_nom, key, data):
        prev = data.sub_data if data.sub_data is not None else {}
        v = prev.get("_hdi_v")
        if v is None:
            v = jnp.zeros(2)
        th = prev.get("_hdi_th")
        if th is None:
            th = jnp.zeros(2)
        v_nom = jnp.asarray(v_nom, dtype=float)
        if agents is not None:
            ag = prev.get("_agents")
            if ag is None:
                ag = jnp.asarray(agents.x0, dtype=float)
            xa = jnp.concatenate([x, v, th, ag.reshape(-1)])
            ag_flat = ag.reshape(-1)
        else:
            xa = jnp.concatenate([x, v, th])
            ag_flat = jnp.zeros(0)
        sub_lp: dict = {}
        if local_planner is not None:
            xa_c = jnp.concatenate([x[ci[0] : ci[0] + 2], v, th, ag_flat])
            a_nom, v_nom, sub_lp = local_planner(t, xa_c, v_nom, key, prev)
            a_nom = jnp.asarray(a_nom, dtype=float)[:3]
            v_nom = jnp.asarray(v_nom, dtype=float)[:2]
        else:
            a_v = velocity_gain * (v_nom[:2] - v)
            if v_nom.shape[0] >= 3:
                th_des = v_nom[2]
            elif face_velocity:
                speed = jnp.linalg.norm(v_nom[:2])
                th_des = jnp.where(speed > 0.05, jnp.arctan2(v_nom[1], v_nom[0]), th[0])
            else:
                th_des = th[0]
            dth = jnp.arctan2(jnp.sin(th_des - th[0]), jnp.cos(th_des - th[0]))
            a_th = heading_gain * dth - 2.0 * jnp.sqrt(heading_gain) * th[1]  # critically damped
            a_nom = jnp.concatenate([a_v, jnp.array([a_th])])
        a_safe, d1 = cbf_qp(t, xa, a_nom, key, data)
        a_safe = jnp.asarray(a_safe)
        v_new = v + a_safe[:2] * dt
        speed = jnp.linalg.norm(v_new)
        v_new = jnp.where(speed > v_max, v_new * (v_max / (speed + 1e-9)), v_new)
        om_new = jnp.clip(th[1] + a_safe[2] * dt, -omega_max, omega_max)
        th_new = jnp.array([th[0] + om_new * dt, om_new])
        sub = dict(d1.sub_data) if d1.sub_data is not None else {}
        for k, val in prev.items():
            if k.startswith("_") and k not in sub:
                sub[k] = val
        sub["_hdi_v"] = v_new
        sub["_hdi_th"] = th_new
        sub.update(sub_lp)
        if agents is not None:
            sub["_agents"] = agents.step(t, x[ci[0] : ci[0] + 2], ag, dt)
        cmd = jnp.array([v_new[0], v_new[1], th_new[0]])
        u, d2 = locomotion(t, x, cmd, key, d1._replace(sub_data=sub))
        sub2 = dict(d2.sub_data) if d2.sub_data is not None else {}
        sub2["v_nom"] = v_nom[:2]
        sub2["v_safe"] = v_new
        sub2["theta_cmd"] = th_new[0]
        sub2["a_nom"] = a_nom
        sub2["a_safe"] = a_safe
        if agents is not None:
            sub2["agents"] = ag
        return u, d2._replace(sub_data=sub2, u=u, u_nom=v_nom[:2], error=d1.error | d2.error)

    controller.__cbfkit_controller_adapter__ = True  # type: ignore[attr-defined]
    return controller


__all__ += [
    "G1_FOOTPRINT",
    "com_agent_ellipse_hocbfs",
    "embedded_heading_double_integrator",
    "hd_agent_slice",
    "heading_slice",
    "safe_locomotion_controller_hdi",
]
