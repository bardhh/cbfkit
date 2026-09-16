"""Internal geometry helpers; public API is cbfkit.systems.mujoco.showcase."""

from typing import Optional, Sequence, Tuple, Union

import mujoco
import numpy as np

Vec = Union[Sequence[float], np.ndarray]


# --------------------------------------------------------------------------- colours
def _rgb(values: Vec) -> Tuple[float, float, float]:
    """A plain 3-tuple of floats (what matplotlib's colour setters accept)."""
    v = np.asarray(values, dtype=float).reshape(-1)
    return (float(v[0]), float(v[1]), float(v[2]))


_GREEN = np.array([0.20, 0.75, 0.35])
_AMBER = np.array([0.95, 0.65, 0.10])
_RED = np.array([0.85, 0.15, 0.20])


def h_rgba(h: float, alpha: float = 0.35) -> np.ndarray:
    """Barrier-value colour: green at ``h >= 0.5``, amber at ``h = 0``, red at ``h <= -0.5``.

    Returns a ``(4,)`` float32 RGBA array ready for ``mjv_initGeom``.
    """
    value = float(h)
    if value >= 0.0:
        frac = min(value / 0.5, 1.0)
        rgb = _AMBER + frac * (_GREEN - _AMBER)
    else:
        frac = min(-value / 0.5, 1.0)
        rgb = _AMBER + frac * (_RED - _AMBER)
    return np.asarray([*rgb, float(alpha)], dtype=np.float32)


# --------------------------------------------------------------------------- primitives
def _init_geom(
    scn: mujoco.MjvScene,
    geom_type: int,
    size: Vec,
    pos: Vec,
    mat: np.ndarray,
    rgba: Vec,
) -> Optional[mujoco.MjvGeom]:
    """Append a visual-only geom with an explicit rotation; None when the scene is full."""
    if scn.ngeom >= scn.maxgeom:
        return None
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        int(geom_type),
        np.asarray(size, dtype=float),
        np.asarray(pos, dtype=float),
        np.asarray(mat, dtype=float).reshape(9),
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1
    return geom


def _connector(
    scn: mujoco.MjvScene,
    geom_type: int,
    radius: float,
    p_from: Vec,
    p_to: Vec,
    rgba: Vec,
) -> int:
    """Append one connector geom (capsule/cylinder/arrow) between two 3-D points."""
    a = np.asarray(p_from, dtype=float)
    b = np.asarray(p_to, dtype=float)
    if float(np.linalg.norm(b - a)) < 1e-9:
        return 0
    geom = _init_geom(scn, geom_type, (radius, radius, radius), a, np.eye(3), rgba)
    if geom is None:
        return 0
    mujoco.mjv_connector(geom, int(geom_type), float(radius), a, b)
    return 1


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(float(theta)), np.sin(float(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def disc(
    scn: mujoco.MjvScene,
    pos_xy: Vec,
    r: float,
    rgba: Vec,
    z: float = 0.004,
) -> int:
    """Flat 4 mm floor disc of radius ``r`` centred ``z`` above the floor."""
    p = np.asarray(pos_xy, dtype=float)
    geom = _init_geom(
        scn,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        (float(r), 0.002, 0.0),
        (p[0], p[1], float(z)),
        np.eye(3),
        rgba,
    )
    return 0 if geom is None else 1


def ring(
    scn: mujoco.MjvScene,
    pos_xy: Vec,
    r: float,
    rgba: Vec,
    width: float = 0.02,
    n: int = 48,
    z: float = 0.006,
) -> int:
    """Keep-out ring of radius ``r`` as an ``n``-segment capsule polygon (MuJoCo has no torus)."""
    p = np.asarray(pos_xy, dtype=float)
    ang = np.linspace(0.0, 2.0 * np.pi, int(n) + 1)
    pts = np.stack(
        [p[0] + float(r) * np.cos(ang), p[1] + float(r) * np.sin(ang), np.full(ang.size, float(z))],
        axis=1,
    )
    added = 0
    for a, b in zip(pts[:-1], pts[1:]):
        added += _connector(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, float(width) / 2.0, a, b, rgba)
    return added


def arrow(
    scn: mujoco.MjvScene,
    p_from: Vec,
    p_to: Vec,
    rgba: Vec,
    width: float = 0.03,
) -> int:
    """3-D arrow from ``p_from`` to ``p_to``; 0 geoms for a degenerate (zero-length) arrow."""
    return _connector(scn, mujoco.mjtGeom.mjGEOM_ARROW, float(width) / 2.0, p_from, p_to, rgba)


def ellipse(
    scn: mujoco.MjvScene,
    pos_xy: Vec,
    a: float,
    b: float,
    theta: float,
    rgba: Vec,
    z: float = 0.005,
    thickness: float = 0.004,
) -> int:
    """Flattened ellipsoid footprint with semi-axes ``a``/``b``, rotated ``theta`` about z."""
    p = np.asarray(pos_xy, dtype=float)
    geom = _init_geom(
        scn,
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        (float(a), float(b), float(thickness) / 2.0),
        (p[0], p[1], float(z)),
        _rot_z(theta),
        rgba,
    )
    return 0 if geom is None else 1


def path(
    scn: mujoco.MjvScene,
    points_xy: np.ndarray,
    rgba: Vec,
    width: float = 0.015,
    z: float = 0.01,
) -> int:
    """Polyline on the floor through ``points_xy`` ``(N, 2)`` as capsule segments."""
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    if pts.shape[0] < 2:
        return 0
    added = 0
    for a, b in zip(pts[:-1], pts[1:]):
        added += _connector(
            scn,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            float(width) / 2.0,
            (a[0], a[1], float(z)),
            (b[0], b[1], float(z)),
            rgba,
        )
    return added


def trail(
    scn: mujoco.MjvScene,
    points_xy: np.ndarray,
    rgba: Vec,
    width: float = 0.012,
    z: float = 0.008,
) -> int:
    """Fading floor trail through ``points_xy``: alpha ramps 0 -> ``rgba[3]`` along the path."""
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    if pts.shape[0] < 2:
        return 0
    base = np.asarray(rgba, dtype=float)
    n_seg = pts.shape[0] - 1
    added = 0
    for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        seg_rgba = (base[0], base[1], base[2], base[3] * (i + 1) / n_seg)
        added += _connector(
            scn,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            float(width) / 2.0,
            (a[0], a[1], float(z)),
            (b[0], b[1], float(z)),
            seg_rgba,
        )
    return added


def pedestrian(
    scn: mujoco.MjvScene,
    p_xy: Vec,
    v_xy: Vec,
    r: float,
    rgba: Vec,
    height: float = 1.65,
    body_scale: float = 0.8,
) -> int:
    """A pedestrian: standing capsule body, heading arrow, and a translucent floor disc.

    The floor disc has radius ``r`` (the pedestrian's own footprint) and the body capsule
    ``body_scale * r``, so a keep-out boundary drawn separately at the inflated radius stays
    visibly wider than the person. The arrow is ``0.5 * |v|`` long, clipped to 0.6 m, and is
    omitted when ``v`` is ~zero.
    """
    p = np.asarray(p_xy, dtype=float)
    v = np.asarray(v_xy, dtype=float).reshape(-1)[:2]
    body_r = float(body_scale) * float(r)
    half_len = max(0.5 * (float(height) - 2.0 * body_r), 1e-3)
    added = 0
    geom = _init_geom(
        scn,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        (body_r, half_len, 0.0),
        (p[0], p[1], float(height) / 2.0),
        np.eye(3),
        rgba,
    )
    if geom is not None:
        added += 1
    speed = float(np.linalg.norm(v))
    if speed > 1e-6:
        length = min(0.5 * speed, 0.6)
        tip = p + v / speed * length
        added += arrow(scn, (p[0], p[1], 0.06), (tip[0], tip[1], 0.06), rgba, width=0.05)
    base = np.asarray(rgba, dtype=float)
    added += disc(scn, p, float(r), (base[0], base[1], base[2], 0.4))
    return added


BEACON_POLE_RADIUS = 0.01
BEACON_POLE_HEIGHT = 1.5
BEACON_POLE_ALPHA = 0.5  # of the sphere's own alpha


def beacon(scn: mujoco.MjvScene, pos_xy: Vec, r: float, rgba: Vec) -> int:
    """Goal beacon: translucent sphere of radius ``r`` plus a hairline column to 1.5 m.

    The sphere marks the spot and the pole makes it findable from across the scene; the
    goal *region* is drawn separately (as floor rings) so the sphere can stay small.
    """
    p = np.asarray(pos_xy, dtype=float)
    added = 0
    geom = _init_geom(
        scn,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        (float(r), 0.0, 0.0),
        (p[0], p[1], float(r)),
        np.eye(3),
        rgba,
    )
    if geom is not None:
        added += 1
    base = np.asarray(rgba, dtype=float)
    column = (base[0], base[1], base[2], BEACON_POLE_ALPHA * float(base[3]))
    added += _connector(
        scn,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        BEACON_POLE_RADIUS,
        (p[0], p[1], 0.0),
        (p[0], p[1], BEACON_POLE_HEIGHT),
        column,
    )
    return added
