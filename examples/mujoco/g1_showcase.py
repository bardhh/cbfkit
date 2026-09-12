"""Showcase driver for the four Unitree G1 examples: simulate once, render many times.

``simulate`` runs one of ``g1_navigate`` / ``g1_plaza`` / ``g1_corridor`` / ``g1_scramble``
in the configuration that produced the README clip, then dumps every per-step array a
renderer needs into ``results/showcase/g1_<example>[_unfiltered].npz``. Nothing is
re-simulated at render time.

    python examples/mujoco/g1_showcase.py simulate navigate [--unfiltered]
           [--unfiltered-mode planner|nominal] [--out DIR] [--seed N] [--duration S]

``render`` hands everything after it to ``g1_showcase_render.py`` (``render --help`` prints
that module's options), filling in ``--npz`` from ``results/showcase/`` and, under
``--side-by-side``, the comparison run -- ``_unfiltered_nominal`` when it exists, else
``_unfiltered``:

    python examples/mujoco/g1_showcase.py render navigate [--side-by-side] [--stills] ...

The four configurations (the README clips, see ``scripts/render_showcase.py``):

    navigate   --reduced-model di, robust 0.18 (the example default), 20 s
    plaza      defaults: distance barriers, robust 0.31, 45 s
    corridor   --g1 --planner mppi, ellipse footprint, vanilla, gap 1.25 m, 150 s
    scramble   --robot unitree --planner mppi, disc footprint, relaxed, vanilla, 100 s,
               26 non-yielding pedestrians (SCRAMBLE_N_PED; the example's own default is 40)

``--unfiltered`` is the comparison run: the identical stack with an **empty certificate
collection** in the CBF-QP, so the QP reduces to the projection of the nominal command
onto the control box and the obstacles (which are visual-only markers, not MuJoCo bodies)
are walked straight through. ``--unfiltered-mode`` chooses how much is stripped:

* ``planner`` (default) removes only the certificate. On ``corridor`` and ``scramble`` the
  MPPI local planner stays in the loop, so this is "planner without a certificate" -- and
  the planner alone already avoids nearly everything (measured on the full runs: scramble
  h_min +0.29 unfiltered vs +0.17 filtered).
* ``nominal`` removes the local planner too and leaves the bare goal-directed P-law:
  ``g1_<example>_unfiltered_nominal.npz``. This is the run with neither planner nor
  certificate, the honest left panel of a side-by-side. On ``navigate`` and ``plaza``
  there is no local planner to remove, so both modes are the same configuration.

The pass-through invariant is ``a_safe == clip(a_nom, +-limits)``, not ``v_safe == v_nom``:
the DI/HDI wrappers certify an *acceleration* and integrate it, so ``v_safe`` is an integral
of the nominal and never equals it. The single-integrator wrapper is the one where the
velocity command passes through untouched. The check runs on every unfiltered call, prints
the deviation and raises when it is real.

Two npz keys are *absent* from an unfiltered run, because the QP assembles no certificate
rows: ``bfs`` and ``violated``. A relaxed run (``scramble``) also loses its slack columns,
so ``sol`` is ``(T, n_u)`` there instead of ``(T, n_u + n_barriers)``. ``nominal`` mode
additionally drops ``mppi_x_traj`` and ``mppi_error``. Everything else, ``h`` included, is
written for every run, and ``unfiltered_mode`` in the npz says which run it was.

``sub_data_bfs`` is *not* the barrier value for any of these configurations (it is psi_1 of
the rectified high-order barrier), so ``h`` in the npz is recomputed offline from the logged
CoM and the logged agent states with each example's own geometry.
"""

import argparse
import contextlib
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import jax
import numpy as np

import cbfkit.simulation.simulator as sim
from cbfkit.certificates import concatenate_certificates

TEST_MODE = bool(os.getenv("CBFKIT_TEST_MODE"))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SHOWCASE_DIR = os.path.join(RESULTS_DIR, "showcase")
JAX_CACHE_DIR = os.path.expanduser("~/.cache/cbfkit/jax")

EXAMPLES = ("navigate", "plaza", "corridor", "scramble")
# Durations of the README configurations: long enough for the crossing to finish
# (measured: navigate 19.3 s, plaza 39.4 s, corridor 107.6 s, scramble 65 s).
DURATIONS = {"navigate": 20.0, "plaza": 45.0, "corridor": 150.0, "scramble": 100.0}
OVERRIDES: Dict[str, Any] = {}  # CLI overrides consumed by the per-example simulators
# README configuration of the scramble: 26 non-yielding pedestrians. At the example's own 40
# (and at 32) the crowd is a crush on every seed tried on the G1 (h_min -0.37 .. -0.99); 26 with
# seed 0 keeps h_min +0.59 with the filter and -0.49 without, so the certificate has something to
# certify. Override with --n-ped.
SCRAMBLE_N_PED = 26
PLANT_KINDS = {19: "unitree12", 30: "amo23", 36: "groot29"}
UNFILTERED_MODES = ("planner", "nominal")
# Examples whose nominal is already a bare goal P-law: they have no local planner, so
# --unfiltered-mode selects nothing and both modes run the identical configuration.
NO_LOCAL_PLANNER = ("navigate", "plaza")
# Accuracy tolerance of the unfiltered pass-through check, in control units (A_MAX = 1).
# It is a *solver* tolerance, not a modelling one: the in-repo PDIPM solves the box-only QP
# to ~3e-7 and jaxopt's OSQP (g1_navigate's default) to ~1e-4 (both measured). A run whose
# barriers were not removed sits 3-4 orders of magnitude above this (0.1-1 in these
# scenarios), so the check discriminates with a wide margin.
UNFILTERED_TOL = 1e-3


# --------------------------------------------------------------------------- helpers
def _empty_certificates(*_args, **_kwargs):
    """Stub for a barrier constructor: a collection with no certificate functions.

    The CBF-QP generator accepts it and assembles no certificate rows at all, so the
    wrappers, the logging keys and the control limits stay exactly as the filtered run
    built them and the QP reduces to the projection onto the control box.
    """
    return concatenate_certificates()


def _no_local_planner(*_args, **_kwargs):
    """Stub for an MPPI local-planner constructor: the wrapper falls back to its P-laws."""
    return None


@contextlib.contextmanager
def _patched(module: Any, **replacements: Any):
    """Temporarily rebind module-level names, restoring them on the way out."""
    saved = {n: getattr(module, n) for n in replacements}
    try:
        for n, value in replacements.items():
            setattr(module, n, value)
        yield
    finally:
        for n, value in saved.items():
            setattr(module, n, value)


def _plant_kind(plant: Any) -> str:
    nq = int(getattr(plant, "nq", -1))
    return PLANT_KINDS.get(nq, f"nq{nq}")


def _n_live(cd: Dict[str, Any], hit_index: Optional[int], n_steps: int) -> int:
    """First goal-hit index, clipped at the first controller error; ``n_steps`` if never."""
    n_live = int(hit_index) if hit_index is not None else n_steps
    err = np.flatnonzero(np.asarray(cd["error"]))
    if err.size:
        n_live = min(n_live, int(err[0]))
    return n_live


def _first_hit(com: np.ndarray, goal: np.ndarray, radius: float) -> Optional[int]:
    hit = np.flatnonzero(np.linalg.norm(com - np.asarray(goal), axis=1) < radius)
    return int(hit[0]) if hit.size else None


def _passthrough_dev(cd: Dict[str, Any], limits: np.ndarray, enforce: bool) -> float:
    """``max |u_safe - clip(u_nom)|`` over the non-error steps, in control units.

    This is zero exactly when the QP is a pass-through: with no certificates it minimises
    ``||u - u_nom||^2`` over the control box alone, so its solution is the clip of the
    nominal and the deviation is solver noise. With barriers it is the intervention
    magnitude. ``enforce`` (the ``--unfiltered`` run) turns a real deviation into an error
    and additionally requires ``bfs``/``violated`` to have left the log -- how an empty
    certificate collection announces itself.
    """
    if enforce and "sub_data_bfs" in cd:
        raise RuntimeError("unfiltered run still logged sub_data_bfs: barriers were not removed")
    if "sub_data_a_nom" in cd:
        nom, safe, what = cd["sub_data_a_nom"], cd["sub_data_a_safe"], "a"
    else:
        nom, safe, what = cd["sub_data_v_nom"], cd["sub_data_v_safe"], "v"
    nom = np.asarray(nom, dtype=float)
    safe = np.asarray(safe, dtype=float)
    live = ~np.asarray(cd["error"], dtype=bool)
    if not live.any():
        raise RuntimeError("the controller errored on every step")
    dev = float(np.abs(safe[live] - np.clip(nom[live], -limits, limits)).max())
    if enforce:
        print(f"unfiltered check: max |{what}_safe - clip({what}_nom)| = {dev:.2e}")
        if not np.isfinite(dev) or dev > UNFILTERED_TOL:
            raise RuntimeError(
                f"unfiltered run is still filtering: max |{what}_safe - clip({what}_nom)| = "
                f"{dev:.3e} > {UNFILTERED_TOL:.0e} over {int(live.sum())} non-error steps"
            )
    return dev


def _ellipse_h(
    com: np.ndarray, peds: np.ndarray, theta: np.ndarray, lon_ax: float, lat_ax: float
) -> np.ndarray:
    """Rotating-ellipse barrier ``h = |R(-theta)(com - p) / axes| - 1``, shape ``(T, N)``.

    The formula of ``g1_corridor.run`` and ``g1_scramble.run``: the axes are already
    inflated by the pedestrian radius, ``theta`` is the *commanded* heading.
    """
    rel = com[:, None, :] - peds  # (T, N, 2), peds may be (N, 2) or (T, N, 2)
    c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
    lon = (c * rel[..., 0] + s * rel[..., 1]) / lon_ax
    lat = (-s * rel[..., 0] + c * rel[..., 1]) / lat_ax
    return np.sqrt(lon**2 + lat**2) - 1.0


def _controller_arrays(res: Any) -> Dict[str, np.ndarray]:
    """Every logged controller array under its bare name (``sub_data_`` prefix stripped)."""
    out: Dict[str, np.ndarray] = {}
    for key, val in res.controller_data.items():
        name = key[len("sub_data_") :] if key.startswith("sub_data_") else key
        arr = np.asarray(val)
        if arr.dtype == object:  # nothing in these examples, but never write an object array
            continue
        out[name] = arr
    return out


def _payload(
    *,
    example: str,
    unfiltered: bool,
    mode: str,
    seed: int,
    duration: float,
    wall: float,
    plant: Any,
    res: Any,
    n_live: int,
    h: np.ndarray,
    h_names: Tuple[str, ...],
    pelvis_body: Optional[int],
    n_u: int,
    relax: bool,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the npz contents: logged arrays + the recomputed h + scenario metadata."""
    states = np.asarray(res["states"])
    ci = tuple(int(i) for i in plant.com_indices)
    out = _controller_arrays(res)
    x_traj = res.planner_data.get("x_traj")
    if x_traj is not None:
        out["x_traj"] = np.asarray(x_traj)
    out.update(
        states=states,
        com=states[:, ci[0] : ci[0] + 2],
        dt=np.float64(plant.dt),
        plant_kind=np.array(_plant_kind(plant)),
        nq=np.int64(getattr(plant, "nq", -1)),
        nv=np.int64(getattr(plant, "nv", -1)),
        com_indices=np.asarray(ci, dtype=np.int64),
        pelvis_body=np.int64(-1 if pelvis_body is None else pelvis_body),
        n_live=np.int64(n_live),
        h=np.asarray(h, dtype=float),
        h_names=np.asarray(h_names),
        example=np.array(example),
        unfiltered=np.bool_(unfiltered),
        # "planner" | "nominal" on an unfiltered run, "" on a filtered one
        unfiltered_mode=np.array(mode if unfiltered else ""),
        seed=np.int64(seed),
        duration=np.float64(duration),
        wall_s=np.float64(wall),
        n_u=np.int64(n_u),
        relax=np.bool_(relax),
    )
    out.update(meta)
    return out


def _npz_path(out_dir: str, example: str, unfiltered: bool = False, mode: str = "planner") -> str:
    """Where ``simulate`` writes (and ``render`` looks for) one run."""
    suffix = ""
    if unfiltered:
        suffix = "_unfiltered" + ("_nominal" if mode == "nominal" else "")
    return os.path.join(out_dir, f"g1_{example}{suffix}.npz")


def _write(out_dir: str, example: str, unfiltered: bool, mode: str, payload: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = _npz_path(out_dir, example, unfiltered, mode)
    np.savez_compressed(path, **payload)
    return path


def _summary(path: str, payload: Dict[str, Any]) -> None:
    n_live = int(payload["n_live"])
    h = payload["h"]
    t_end = max(n_live, 1)
    h_min = float(h[:t_end].min()) if h.size else float("nan")
    # Intervention = the QP moving the *filtered* variable: the acceleration for the DI/HDI
    # wrappers (v_safe is its integral, so |v_safe - v_nom| is nonzero even with no barriers),
    # the velocity for the single-integrator wrapper.
    if "a_nom" in payload:
        nom, safe, what = payload["a_nom"], payload["a_safe"], "a"
    else:
        nom, safe, what = payload["v_nom"], payload["v_safe"], "v"
    active = float(np.mean(np.linalg.norm(safe[:t_end] - nom[:t_end], axis=1) > 1e-3))
    label = payload["example"]
    if bool(payload["unfiltered"]):
        label = f"{label} (unfiltered, {payload['unfiltered_mode']})"
    print(
        f"{label}: "
        f"T={len(payload['states'])} steps ({len(payload['states']) * float(payload['dt']):.1f} s), "
        f"n_live={n_live} ({n_live * float(payload['dt']):.1f} s), h_min(live)={h_min:+.3f}, "
        f"intervention (|{what}_safe - {what}_nom| > 1e-3) {active * 100:.0f}% of live steps, "
        f"wall {float(payload['wall_s']):.1f} s -> {path}"
    )


# --------------------------------------------------------------------------- navigate
def simulate_navigate(seed: int, duration: float, unfiltered: bool, mode: str) -> Dict[str, Any]:
    """``g1_navigate.py --reduced-model di`` with its measured robust bound (the default)."""
    from examples.mujoco import g1_navigate as ex

    robust_bound = 0.18  # the example's measured default for the di model
    num_samples, iterations = (16, 1) if TEST_MODE else (256, 2)
    # No local planner in this scenario, so `mode` selects nothing: see NO_LOCAL_PLANNER.
    patches = {"com_obstacle_hocbfs": _empty_certificates} if unfiltered else {}
    with _patched(ex, **patches):
        plant, _loco, x0, pelvis_body, (controller, nominal) = ex.build(
            "policy", num_samples, iterations, seed, robust_bound, "di"
        )
    steps = 5 if TEST_MODE else int(round(duration / plant.dt))
    t0 = time.time()
    res = sim.execute(
        x0=x0,
        dt=plant.dt,
        num_steps=steps,
        plant=plant,
        nominal_controller=nominal,
        controller=controller,
        key=jax.random.PRNGKey(seed),
        use_jit=True,
        verbose=not TEST_MODE,
    )
    wall = time.time() - t0
    cd = res.controller_data
    dev = _passthrough_dev(cd, np.array([ex.A_MAX, ex.A_MAX]), unfiltered)
    states = np.asarray(res["states"])
    ci = plant.com_indices
    com = states[:, ci[0] : ci[0] + 2]
    obstacle = np.asarray(ex.OBSTACLE, dtype=float).reshape(1, 2)
    keepout = ex.OBSTACLE_RADIUS + ex.ROBOT_RADIUS
    # The di model uses the stock quadratic ellipsoid barrier (g1_navigate.main).
    h = (((com - obstacle[0]) / keepout) ** 2).sum(1)[:, None] - 1.0
    goal = np.asarray(ex.GOAL, dtype=float)
    n_live = _n_live(cd, _first_hit(com, goal, ex.GOAL_RADIUS), len(com))
    meta = dict(
        qp_dev_max=np.float64(dev),
        obstacles=obstacle,
        obstacle_radii=np.array([ex.OBSTACLE_RADIUS]),
        keepout_radii=np.array([keepout]),
        goal=goal,
        waypoints=np.zeros((0, 2)),
        footprint_axes=np.array([ex.ROBOT_RADIUS, ex.ROBOT_RADIUS]),
        ped_radius=np.float64(0.0),
        robot_radius=np.float64(ex.ROBOT_RADIUS),
        goal_radius=np.float64(ex.GOAL_RADIUS),
        v_max=np.float64(ex.V_MAX),
        barrier_shape=np.array("ellipsoid"),
        robust_bound=np.float64(robust_bound),
        config=np.array("g1_navigate.py --reduced-model di --robust 0.18"),
    )
    return _payload(
        example="navigate",
        unfiltered=unfiltered,
        mode=mode,
        seed=seed,
        duration=duration,
        wall=wall,
        plant=plant,
        res=res,
        n_live=n_live,
        h=h,
        h_names=("obstacle",),
        pelvis_body=pelvis_body,
        n_u=2,
        relax=False,
        meta=meta,
    )


# --------------------------------------------------------------------------- plaza
def simulate_plaza(seed: int, duration: float, unfiltered: bool, mode: str) -> Dict[str, Any]:
    """``g1_plaza.py`` with its defaults: distance barriers, robust 0.31, waypoint route."""
    from cbfkit.utils.user_types import PlannerData
    from examples.mujoco import g1_plaza as ex

    robust_bound = ex.DEFAULT_ROBUST_BOUND
    # No local planner in this scenario, so `mode` selects nothing: see NO_LOCAL_PLANNER.
    patches = (
        {"com_obstacle_hocbfs": _empty_certificates, "com_agent_hocbfs": _empty_certificates}
        if unfiltered
        else {}
    )
    with _patched(ex, **patches):
        plant, x0, pelvis_body, planner, nominal, controller = ex.build(
            seed, robust_bound, ex.BARRIER_SHAPE
        )
    steps = 5 if TEST_MODE else int(round(duration / plant.dt))
    t0 = time.time()
    res = sim.execute(
        x0=x0,
        dt=plant.dt,
        num_steps=steps,
        plant=plant,
        planner=planner,
        planner_data=PlannerData.from_constant(ex.WAYPOINTS[0]),
        nominal_controller=nominal,
        controller=controller,
        key=jax.random.PRNGKey(seed),
        use_jit=True,
        verbose=not TEST_MODE,
    )
    wall = time.time() - t0
    cd = res.controller_data
    dev = _passthrough_dev(cd, np.array([ex.A_MAX, ex.A_MAX]), unfiltered)
    states = np.asarray(res["states"])
    ci = plant.com_indices
    com = states[:, ci[0] : ci[0] + 2]
    agents = np.asarray(cd["sub_data_agents"])  # (T, N, 4) as the QP saw them
    h = ex.barrier_values(com, agents, ex.BARRIER_SHAPE)  # the example's own formula
    arrivals = ex.waypoint_arrivals(com)
    n_live = _n_live(cd, arrivals[-1], len(com))
    pillars = np.asarray(ex.PILLARS, dtype=float)
    meta = dict(
        qp_dev_max=np.float64(dev),
        obstacles=pillars,
        obstacle_radii=np.full(len(pillars), ex.PILLAR_RADIUS),
        keepout_radii=np.full(len(pillars), ex.R_PILLAR),
        goal=np.asarray(ex.WAYPOINTS[-1], dtype=float),
        waypoints=np.asarray(ex.WAYPOINTS, dtype=float),
        waypoint_radius=np.float64(ex.WAYPOINT_RADIUS),
        waypoint_arrivals=np.asarray([-1 if a is None else a for a in arrivals], dtype=np.int64),
        footprint_axes=np.array([ex.ROBOT_RADIUS, ex.ROBOT_RADIUS]),
        ped_radius=np.float64(ex.PED_RADIUS),
        ped_keepout=np.float64(ex.R_PED),
        robot_radius=np.float64(ex.ROBOT_RADIUS),
        goal_radius=np.float64(ex.GOAL_RADIUS),
        v_max=np.float64(ex.V_MAX),
        barrier_shape=np.array(ex.BARRIER_SHAPE),
        robust_bound=np.float64(robust_bound),
        config=np.array("g1_plaza.py (defaults: distance barriers, robust 0.31)"),
    )
    return _payload(
        example="plaza",
        unfiltered=unfiltered,
        mode=mode,
        seed=seed,
        duration=duration,
        wall=wall,
        plant=plant,
        res=res,
        n_live=n_live,
        h=h,
        h_names=tuple(ex.OBSTACLE_NAMES),
        pelvis_body=pelvis_body,
        n_u=2,
        relax=False,
        meta=meta,
    )


# --------------------------------------------------------------------------- corridor
def simulate_corridor(seed: int, duration: float, unfiltered: bool, mode: str) -> Dict[str, Any]:
    """``g1_corridor.py --g1 --planner mppi`` (ellipse footprint, vanilla, gap 1.25 m).

    The clip's configuration: the README render is ``g1_corridor_ellipse_mppi.gif`` and the
    measured G1 + MPPI crossing is the *vanilla* one (robust 0.12 refuses the gap, see the
    example's docstring), so ``robust_bound = 0``.
    """
    import mujoco

    from cbfkit.utils.user_types import PlannerData
    from examples.mujoco import g1_corridor as ex

    gap = ex.GAP_G1
    # The builder offers two planners and neither is a bare goal P-law: "suggest" is the
    # hand-coded heading ramp that *anticipates* the gap, "mppi" is the P-law nominal plus
    # the MPPI lookahead. So `nominal` mode takes the "mppi" branch -- whose nominal is
    # exactly the plain saturated P-law toward the goal -- and stubs out the MPPI
    # constructor, leaving `local_planner=None`. The hdi wrapper then runs its own P-laws
    # with `face_velocity=True`: walk at the goal, face the direction of travel, no hint
    # about the gap anywhere.
    no_planner = unfiltered and mode == "nominal"
    patches: Dict[str, Any] = {}
    if unfiltered:
        patches["com_agent_ellipse_hocbfs"] = _empty_certificates
        if no_planner:
            patches["build_corridor_mppi"] = _no_local_planner
    with _patched(ex, **patches):
        plant, x0, nominal, controller, agents, goal = ex.build(
            footprint="ellipse", gap=gap, g1=True, robust_bound=0.0, planner="mppi"
        )
    steps = 5 if TEST_MODE else int(round(duration / plant.dt))
    t0 = time.time()
    res = sim.execute(
        x0=x0,
        dt=plant.dt,
        num_steps=steps,
        plant=plant,
        planner_data=PlannerData.from_constant(goal),
        nominal_controller=nominal,
        controller=controller,
        key=jax.random.PRNGKey(seed),
        use_jit=True,
        verbose=not TEST_MODE,
    )
    wall = time.time() - t0
    cd = res.controller_data
    dev = _passthrough_dev(cd, np.array([ex.A_MAX, ex.A_MAX, ex.ALPHA_MAX]), unfiltered)
    states = np.asarray(res["states"])
    ci = plant.com_indices
    com = states[:, ci[0] : ci[0] + 2]
    peds = np.asarray(agents.x0, dtype=float)[:, :2]  # two motionless pedestrians
    lon_ax = ex.G1_FOOTPRINT["lon"] + ex.PED_RADIUS
    lat_ax = ex.G1_FOOTPRINT["lat"] + ex.PED_RADIUS
    theta = np.asarray(cd["sub_data_theta_cmd"], dtype=float)
    h = _ellipse_h(com, peds[None], theta, lon_ax, lat_ax)  # (T, 2)
    n_live = _n_live(cd, _first_hit(com, np.asarray(goal), ex.GOAL_RADIUS), len(com))
    pelvis_body = int(mujoco.mj_name2id(plant.mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"))
    meta = dict(
        qp_dev_max=np.float64(dev),
        obstacles=peds,
        obstacle_radii=np.full(len(peds), ex.PED_RADIUS),
        keepout_radii=np.full(len(peds), ex.PED_RADIUS + ex.ROBOT_DISC),
        goal=np.asarray(goal, dtype=float),
        waypoints=np.zeros((0, 2)),
        footprint_axes=np.array([ex.G1_FOOTPRINT["lon"], ex.G1_FOOTPRINT["lat"]]),
        footprint_axes_inflated=np.array([lon_ax, lat_ax]),
        ped_radius=np.float64(ex.PED_RADIUS),
        robot_radius=np.float64(ex.ROBOT_DISC),
        goal_radius=np.float64(ex.GOAL_RADIUS),
        v_max=np.float64(ex.V_MAX_G1),
        gap=np.float64(gap),
        offset=np.float64(0.0),
        barrier_shape=np.array("ellipse"),
        robust_bound=np.float64(0.0),
        config=np.array(
            "g1_corridor.py --g1 --footprint ellipse --robust 0 --planner "
            + ("goal P-law, no local planner" if no_planner else "mppi")
        ),
    )
    if not no_planner:  # layout of mppi_x_traj; absent when no plan was logged
        meta.update(mppi_dt=np.float64(ex.MPPI_DT), mppi_state_head=np.int64(6))
    return _payload(
        example="corridor",
        unfiltered=unfiltered,
        mode=mode,
        seed=seed,
        duration=duration,
        wall=wall,
        plant=plant,
        res=res,
        n_live=n_live,
        h=h,
        h_names=tuple(f"pedestrian {i + 1}" for i in range(len(peds))),
        pelvis_body=pelvis_body,
        n_u=3,
        relax=False,
        meta=meta,
    )


# --------------------------------------------------------------------------- scramble
def simulate_scramble(seed: int, duration: float, unfiltered: bool, mode: str) -> Dict[str, Any]:
    """``g1_scramble.py --planner mppi`` on the Unitree policy: disc footprint, relaxed."""
    from cbfkit.utils.user_types import PlannerData
    from examples.mujoco import g1_scramble as ex

    n_ped = OVERRIDES.get("n_ped") or (ex.N_PED if ex.TEST_MODE else SCRAMBLE_N_PED)
    relax = ex.DEFAULT_RELAX
    # `nominal` mode is the builder's own "goal" planner: local_planner=None and the
    # saturated P-law toward the goal as the nominal.
    planner = "goal" if (unfiltered and mode == "nominal") else "mppi"
    patches = {"com_agent_hocbfs": _empty_certificates} if unfiltered else {}
    with _patched(ex, **patches):
        plant, x0, pelvis_body, nominal, controller, crowd = ex.build(
            seed=seed,
            robust_bound=0.0,
            n_ped=n_ped,
            relax=relax,
            planner=planner,
            weights=ex.DEFAULT_WEIGHTS,
            proxy=False,
            robot="unitree",
            footprint="disc",
        )
    steps = 5 if TEST_MODE else int(round(duration / plant.dt))
    t0 = time.time()
    res = sim.execute(
        x0=x0,
        dt=plant.dt,
        num_steps=steps,
        plant=plant,
        planner_data=PlannerData.from_constant(ex.GOAL),
        nominal_controller=nominal,
        controller=controller,
        key=jax.random.PRNGKey(seed),
        use_jit=True,
        verbose=not TEST_MODE,
    )
    wall = time.time() - t0
    cd = res.controller_data
    dev = _passthrough_dev(cd, np.array([ex.A_MAX, ex.A_MAX]), unfiltered)
    states = np.asarray(res["states"])
    ci = plant.com_indices
    com = states[:, ci[0] : ci[0] + 2]
    agents = np.asarray(cd["sub_data_agents"])  # (T, N, 4)
    # The disc barrier of g1_scramble.main: h = |com - p| / R_PED - 1.
    h = np.linalg.norm(com[:, None, :] - agents[:, :, :2], axis=2) / ex.R_PED - 1.0
    goal = np.asarray(ex.GOAL, dtype=float)
    n_live = _n_live(cd, _first_hit(com, goal, ex.GOAL_RADIUS), len(com))
    meta = dict(
        qp_dev_max=np.float64(dev),
        obstacles=np.zeros((0, 2)),  # no static obstacles: the crowd is in `agents`
        obstacle_radii=np.zeros(0),
        keepout_radii=np.zeros(0),
        goal=goal,
        waypoints=np.zeros((0, 2)),
        footprint_axes=np.array([ex.ROBOT_RADIUS, ex.ROBOT_RADIUS]),
        ped_radius=np.float64(ex.PED_RADIUS),
        ped_keepout=np.float64(ex.R_PED),
        robot_radius=np.float64(ex.ROBOT_RADIUS),
        goal_radius=np.float64(ex.GOAL_RADIUS),
        v_max=np.float64(ex.V_MAX),
        half=np.float64(ex.HALF),
        n_ped=np.int64(n_ped),
        crowd_goals=np.asarray(crowd.goals, dtype=float),
        crowd_speeds=np.asarray(crowd.speeds, dtype=float),
        barrier_shape=np.array(ex.BARRIER_SHAPE),
        robust_bound=np.float64(0.0),
        config=np.array(
            f"g1_scramble.py --robot unitree --planner {planner} --relax (disc, vanilla)"
        ),
    )
    if planner == "mppi":  # layout of mppi_x_traj; absent when no plan was logged
        meta.update(mppi_dt=np.float64(ex.MPPI_DT), mppi_state_head=np.int64(4))
    return _payload(
        example="scramble",
        unfiltered=unfiltered,
        mode=mode,
        seed=seed,
        duration=duration,
        wall=wall,
        plant=plant,
        res=res,
        n_live=n_live,
        h=h,
        h_names=tuple(f"pedestrian {i + 1}" for i in range(n_ped)),
        pelvis_body=pelvis_body,
        n_u=2,
        relax=relax,
        meta=meta,
    )


SIMULATORS = {
    "navigate": simulate_navigate,
    "plaza": simulate_plaza,
    "corridor": simulate_corridor,
    "scramble": simulate_scramble,
}


def simulate(
    example: str,
    out_dir: str,
    seed: int,
    duration: Optional[float],
    unfiltered: bool,
    mode: str = "planner",
):
    """Run one example and write ``g1_<example>[_unfiltered[_nominal]].npz`` into ``out_dir``.

    Sets the JAX compilation cache here rather than at import: it is a process-global
    side effect, and importing this module should not impose it on the caller.
    """
    jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
    if duration is None:
        duration = DURATIONS[example]
    if unfiltered and mode == "nominal" and example in NO_LOCAL_PLANNER:
        print(
            f"note: {example} has no local planner -- its nominal is already a bare goal "
            "P-law, so --unfiltered-mode nominal runs the same configuration as planner."
        )
    payload = SIMULATORS[example](seed, duration, unfiltered, mode)
    path = _write(out_dir, example, unfiltered, mode, payload)
    _summary(path, payload)
    return path


def render(argv: List[str]):
    """Delegate to ``g1_showcase_render.main``, defaulting the npz paths ``simulate`` wrote.

    Everything after ``render`` is handed through untouched (so ``render --help`` prints the
    renderer's own options); this only fills in ``--npz`` and, for ``--side-by-side``, the
    comparison run. The side-by-side default prefers ``_unfiltered_nominal`` -- neither
    planner nor certificate -- and falls back to ``_unfiltered`` when it has not been
    simulated.
    """
    from examples.mujoco import g1_showcase_render as renderer

    argv = list(argv)
    example = argv[0] if argv and not argv[0].startswith("-") else None
    if example in EXAMPLES:
        if "--npz" not in argv:
            argv += ["--npz", _npz_path(SHOWCASE_DIR, example)]
        if "--side-by-side" in argv and "--unfiltered-npz" not in argv:
            nominal = _npz_path(SHOWCASE_DIR, example, True, "nominal")
            planner = _npz_path(SHOWCASE_DIR, example, True, "planner")
            argv += ["--unfiltered-npz", nominal if os.path.exists(nominal) else planner]
    return renderer.main(argv)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # `render` is intercepted before argparse: its arguments belong to the renderer, and
    # argparse.REMAINDER cannot capture a leading option, so `render --help` would be
    # rejected here instead of reaching the module that can answer it.
    if argv and argv[0] == "render":
        return render(argv[1:])
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("simulate", help="run one example and write its npz")
    s.add_argument("example", choices=EXAMPLES)
    s.add_argument(
        "--unfiltered", action="store_true", help="empty certificate collection (no CBF filtering)"
    )
    s.add_argument(
        "--unfiltered-mode",
        choices=UNFILTERED_MODES,
        default="planner",
        help="what --unfiltered strips: 'planner' keeps the example's local planner (MPPI on "
        "corridor/scramble), 'nominal' also drops it and runs the bare goal P-law",
    )
    s.add_argument("--out", default=SHOWCASE_DIR, help=f"output directory (default {SHOWCASE_DIR})")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument(
        "--n-ped", type=int, default=None, help="scramble only: crowd size (default N_PED)"
    )
    s.add_argument(
        "--duration",
        type=float,
        default=None,
        help="simulated seconds (default: per example, see DURATIONS)",
    )
    # Registered only so it shows up in --help; the call above never reaches this branch.
    sub.add_parser(
        "render",
        help="render a simulated npz: everything after `render` goes to g1_showcase_render.py "
        "(`render --help` prints its options); --npz is filled in when omitted",
        add_help=False,
    )
    a = p.parse_args(argv)
    if a.command == "simulate":
        if a.unfiltered_mode != "planner" and not a.unfiltered:
            p.error("--unfiltered-mode only applies together with --unfiltered")
        if a.n_ped:
            OVERRIDES["n_ped"] = int(a.n_ped)
        simulate(a.example, a.out, a.seed, a.duration, a.unfiltered, a.unfiltered_mode)


if __name__ == "__main__":
    main()
