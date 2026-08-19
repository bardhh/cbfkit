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

from typing import Any, Sequence, Tuple

import jax.numpy as jnp
from jax import Array

from cbfkit.certificates import certificate_package, concatenate_certificates
from cbfkit.certificates.barrier_functions import ellipsoidal_barrier_factory
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.utils.user_types import ControllerCallable, DynamicsCallable


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
            rectify_relative_degree(
                function=_com_static_barrier(plant, o, e, shape),
                system_dynamics=dyn,
                state_dim=n,
                roots=roots,
                form="high-order",
                certificate_conditions=conditions,
                input_style="state",
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
) -> ControllerCallable:
    """Double-integrator variant of :func:`safe_locomotion_controller`.

    The nominal ``v_nom`` (2-D velocity) is turned into a nominal acceleration
    ``a_nom = velocity_gain * (v_nom - v)`` toward it; ``cbf_qp`` (built on
    :func:`embedded_double_integrator` + :func:`com_obstacle_hocbfs`) filters it
    on the augmented state ``[x | v]``; the commanded velocity is integrated,
    ``v <- clip(v + a_safe dt, |v| <= v_max)``, carried in ``sub_data["_di_v"]``,
    and handed to the locomotion controller as its command. Logged:
    ``v_nom``, ``v_safe`` (= the integrated command), ``a_safe``.

    ``agents`` (optional) adds tracked agents to the augmented state -- an object with
    ``x0`` (``(N, 4)`` initial ``px, py, vx, vy``) and ``step(t, robot_xy, states, dt)
    -> states`` (e.g. :class:`cbfkit.systems.mujoco.crowd.SocialForceCrowd`). The QP sees
    the agents' *current* states (``embedded_double_integrator(n_agents=N)`` predicts
    them at constant velocity); they are stepped after the QP with the robot's CoM,
    carried in ``sub_data["_agents"]`` and the states the QP used are logged as
    ``"agents"``.
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
        else:
            xa = jnp.concatenate([x, v])
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
        if agents is not None:
            sub["_agents"] = agents.step(t, x[ci[0] : ci[0] + 2], ag, dt)
        u, d2 = locomotion(t, x, v_new, key, d1._replace(sub_data=sub))
        sub2 = dict(d2.sub_data) if d2.sub_data is not None else {}
        sub2["v_nom"] = v_nom
        sub2["v_safe"] = v_new
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
            rectify_relative_degree(
                function=_com_moving_barrier(plant, p0, v, e, shape),
                system_dynamics=dyn,
                state_dim=n,
                roots=roots,
                form="high-order",
                certificate_conditions=conditions,
                input_style="separated",
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
            rectify_relative_degree(
                function=barrier(i, e),
                system_dynamics=dyn,
                state_dim=n,
                roots=roots,
                form="high-order",
                certificate_conditions=conditions,
                input_style="state",
            )
            for i, e in zip(range(n_agents), ellipsoids)
        ]
    )


__all__ += ["agent_slice", "com_agent_hocbfs"]
