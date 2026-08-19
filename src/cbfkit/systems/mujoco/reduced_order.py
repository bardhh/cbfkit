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
def embedded_double_integrator(state_dim: int, indices: Tuple[int, int]) -> DynamicsCallable:
    """Command-side double integrator on the CoM, in *augmented* coordinates ``[x | v]``.

    ``x`` is the plant's flat state (``state_dim`` entries) and ``v`` (2 entries,
    appended) is the *commanded* planar CoM velocity that the locomotion layer
    tracks. Dynamics: ``d/dt com = v``, ``d/dt v = a`` (control), everything else 0.
    Position barriers then have relative degree 2 in ``a`` and go through
    ``rectify_relative_degree`` (high-order CBF).
    """
    n = state_dim + 2
    g = jnp.zeros((n, 2)).at[state_dim, 0].set(1.0).at[state_dim + 1, 1].set(1.0)

    def dynamics(xa: Array):
        f = jnp.zeros(n).at[indices[0]].set(xa[state_dim]).at[indices[1]].set(xa[state_dim + 1])
        return f, g

    return dynamics


def com_obstacle_hocbfs(
    plant: Any,
    obstacles: Sequence[Sequence[float]],
    ellipsoids: Sequence[Sequence[float]],
    class_k_gain: float = 1.0,
    roots: Any = None,
):
    """High-order (relative-degree-2) CoM keep-out barriers for :func:`embedded_double_integrator`.

    Same ellipsoids as :func:`com_obstacle_barriers`, lifted to the augmented
    ``[x | v]`` state with ``rectify_relative_degree(form="high-order")``.
    """
    from cbfkit.certificates import rectify_relative_degree

    cbf, _, _ = ellipsoidal_barrier_factory(
        system_position_indices=tuple(plant.com_indices),
        obstacle_position_indices=(0, 1),
        ellipsoid_axis_indices=(0, 1),
    )
    n = plant.state_dim + 2
    dyn = embedded_double_integrator(plant.state_dim, plant.com_indices)
    conditions = zeroing_barriers.linear_class_k(class_k_gain)
    return concatenate_certificates(
        *[
            rectify_relative_degree(
                function=cbf(jnp.asarray(o, dtype=float), jnp.asarray(e, dtype=float)),
                system_dynamics=dyn,
                state_dim=n,
                roots=roots,
                form="high-order",
                certificate_conditions=conditions,
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
) -> ControllerCallable:
    """Double-integrator variant of :func:`safe_locomotion_controller`.

    The nominal ``v_nom`` (2-D velocity) is turned into a nominal acceleration
    ``a_nom = velocity_gain * (v_nom - v)`` toward it; ``cbf_qp`` (built on
    :func:`embedded_double_integrator` + :func:`com_obstacle_hocbfs`) filters it
    on the augmented state ``[x | v]``; the commanded velocity is integrated,
    ``v <- clip(v + a_safe dt, |v| <= v_max)``, carried in ``sub_data["_di_v"]``,
    and handed to the locomotion controller as its command. Logged:
    ``v_nom``, ``v_safe`` (= the integrated command), ``a_safe``.
    """

    def controller(t, x, v_nom, key, data):
        prev = data.sub_data if data.sub_data is not None else {}
        v = prev.get("_di_v")
        if v is None:
            v = jnp.zeros(2)
        v_nom = jnp.asarray(v_nom, dtype=float)[:2]
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
        u, d2 = locomotion(t, x, v_new, key, d1._replace(sub_data=sub))
        sub2 = dict(d2.sub_data) if d2.sub_data is not None else {}
        sub2["v_nom"] = v_nom
        sub2["v_safe"] = v_new
        sub2["a_safe"] = jnp.asarray(a_safe)
        return u, d2._replace(sub_data=sub2, u=u, u_nom=v_nom, error=d1.error | d2.error)

    controller.__cbfkit_controller_adapter__ = True  # type: ignore[attr-defined]
    return controller


__all__ += ["com_obstacle_hocbfs", "embedded_double_integrator", "safe_locomotion_controller_di"]


# --------------------------------------------------------------------------- moving obstacles
def moving_obstacle_position(p0: Any, v: Any, t: Any) -> Array:
    """Known constant-velocity obstacle: ``p(t) = p0 + v t``."""
    return jnp.asarray(p0, dtype=float) + jnp.asarray(v, dtype=float) * t


def _com_moving_barrier(plant: Any, p0: Any, v: Any, ellipsoid: Any):
    """``h(t, x) = ||(com - p(t)) / (a, b)||^2 - 1`` on the plant's flat state."""
    ci = plant.com_indices
    p0 = jnp.asarray(p0, dtype=float)
    v = jnp.asarray(v, dtype=float)
    axes = jnp.asarray(ellipsoid, dtype=float)

    def h(t, x):
        com = jnp.asarray(x)[ci[0] : ci[0] + 2]
        return jnp.sum(((com - moving_obstacle_position(p0, v, t)) / axes) ** 2) - 1.0

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
):
    """High-order, time-varying keep-out barriers for :func:`embedded_double_integrator`.

    Same barrier as :func:`com_moving_obstacle_barriers`, on the augmented ``[x | v]`` state;
    ``rectify_relative_degree`` carries ``dh/dt`` into the lifted barrier, so an obstacle walking
    toward a standing robot makes the QP act even though the robot's own velocity is zero.
    """
    from cbfkit.certificates import rectify_relative_degree

    n = plant.state_dim + 2
    dyn = embedded_double_integrator(plant.state_dim, plant.com_indices)
    conditions = zeroing_barriers.linear_class_k(class_k_gain)
    return concatenate_certificates(
        *[
            rectify_relative_degree(
                function=_com_moving_barrier(plant, p0, v, e),
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
