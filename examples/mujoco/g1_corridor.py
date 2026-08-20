"""Squeeze test: a gap between two pedestrians that a disc footprint must refuse and a
rotating-ellipse footprint can certify -- the QP discovers "turn sideways and sidestep".

Two static pedestrians stand ``GAP`` apart (centre to centre); the robot must reach a
goal on the far side. Safety runs on the command-side *heading-augmented* double
integrator (``[p | v | theta omega | agents]``, controls ``[ax, ay, alpha]``) with HARD
constraints -- this is a certificate demo, not a filter:

* ``--footprint disc``    : the usual 0.65 m keep-out discs (0.35 robot + 0.30 pedestrian).
  Minimum passable gap: 1.30 m. At ``GAP`` = 1.0 m the QP has no safe way through --
  the correct certified behavior is to STOP in front of the gap.
* ``--footprint ellipse`` : the measured G1 footprint under AMO
  (``reduced_order.G1_FOOTPRINT``: 0.11 m longitudinal x 0.22 m lateral, upper body --
  ``g1_footprint_measure.py``), rotated by the heading state. Facing forward the wide
  lateral axis spans the gap (needs 1.04 m); turned sideways the narrow longitudinal
  axis does (needs 0.82 m). Rotation enters ``h`` through ``theta``, so rotating is a
  control the QP can certify.

**Measured honesty note -- who discovers the rotation?** Not the bare QP: a pointwise
min-norm filter is myopic -- at the stall in front of the gap, rotating costs deviation
*now* and pays only *later*, so it parks facing forward (measured: theta_max 1.4 deg,
x stalls at -0.23). The rotation must come from a layer with lookahead. Two are on offer:

* ``--planner suggest`` (default): a hand-coded ramp turns the heading toward sideways
  near the gap line -- BOTH footprints receive the identical suggestion, so the
  comparison isolates the *certificate* (the disc refuses either way, 1.30 m needed).
* ``--planner mppi``: 6 s of MPPI over the *same* heading-augmented DI the QP certifies
  (``reduced_order.ellipse_trajectory_cost``, 5 Hz replan, 1024 samples) -- no heading
  hint anywhere; the planner *discovers* that rotating early pays and hands
  ``[ax, ay, alpha]`` to the QP as the nominal. Cost-shaping lesson (measured): the
  planner's clearance term must be a saturated quadratic in absolute h-units -- a
  *preference*. Normalising it by the margin makes every violation infinite, the best
  sample is always "wait", and the robot freezes in front of the gap; random shooting
  cannot thread a +-4 cm tube -- proposing "through, sideways-ish" and letting the hard
  QP do the threading is the split that works. ``--offset`` shifts the gap off the
  start-goal line: the scripted ramp does not know and squeezes at h ~ +0.04; the MPPI
  plans a diagonal through the actual gap with ~40% less heading travel and ~6x the
  margin -- lookahead replaces the tuned maneuver instead of imitating it.

The tracking layer is AMO (``--g1``; lateral gait realises ~0.13 m/s in MJX, so the
squeeze is slow) or the 2-D lagged proxy (default; fast, exercises the QP mechanism).

Measured (hard constraints; proxy gap 1.0 m, G1 gap 1.25 m):

    config                          result
    proxy  ellipse (vanilla)        crossed 20.5 s: rotates to 90 deg, min centre dist 0.50 m,
                                    h >= +0.08 the whole way -- exact model, certificate real
    proxy  disc     (same nominal)  refuses: parks at the 0.65 boundary (correct behavior)
    G1     ellipse (vanilla)        crossed 102 s SIDEWAYS, upright 0.997; h_min -0.16 = the
                                    measured lateral tracking droop (v realises ~0.43x, gait-alive
                                    shuffle +-0.12); min centre distance 0.40 m, no contact
    G1     ellipse (robust 0.05)    same behavior, h_min -0.18
    G1     ellipse (robust 0.12)    REFUSES: the robust margin is applied to the full-state
                                    row norm of the rectified psi (incl. the exactly-known agent
                                    channels), ~2x the h-level margin -- honest conservatism
                                    finding, the next modelling frontier (per-channel bounds)
    G1     disc                     refuses (needs 1.30 m; correct)

    proxy, --planner mppi (no suggestion anywhere; seeds 0/1/2):
    ellipse, centred gap 1.0       crossed 34.8/27.7/33.6 s, theta_max 84-94 deg
                                    (discovered), heading travel 203-259 deg, h >= +0.07
    disc, identical planner        refuses: parks at x -0.83 (the plan itself keeps the
                                    0.68 m preference distance; certificate isolation holds)
    ellipse, offset -0.35          suggest: crossed 22.6 s, travel 179 deg, h_min +0.04
                                    mppi:    crossed 27-28 s, theta_max 38-51 deg only,
                                    travel 108-116 deg, h_min +0.21..0.26 -- plans the
                                    diagonal: less rotation, 5-6x the margin; the ramp
                                    survives only because the hard QP drags it through
                                    at its scripted 90 deg

    G1, --planner mppi (vanilla, gap 1.25; seeds 0/1): crossed 107.8/111.8 s,
    theta_max 125/134 deg, upright 0.999, h_min +0.01/-0.14 on the measured com --
    bounded by the same measured tracking droop as the ramp (-0.16); on the good seed
    the lookahead's planned buffer absorbs the slop entirely. Residual honesty:
    ~1.2-1.5 deg/replan of heading jitter (smooth between replans by construction --
    successive replans disagree because the G1 does not track the plan exactly). The
    align term is speed-GATED (clip(|v|/0.2)): scaling by raw speed lets the heading
    wander freely on a slow robot -- the first G1 run pirouetted a full 388 deg
    mid-crossing.

So the anisotropic gain certified end-to-end on the real robot is 1.25 m vs the disc's
1.30 m -- modest, because the G1's *tracking*, not its geometry, is the binding
constraint; the proxy shows the geometric mechanism at its full 1.0 vs 1.3 m. GIF:
``results/g1_corridor_ellipse.gif`` (the G1 walks up, turns sideways, sidesteps through
the gap, turns back and walks on).

    python examples/mujoco/g1_corridor.py [--footprint disc|ellipse] [--planner suggest|mppi]
                                          [--gap G] [--offset Y] [--g1] [--duration T]
                                          [--gif] [--view]
"""

import argparse
import os
import sys
import time

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import jax
import jax.numpy as jnp
import numpy as np

import cbfkit.simulation.simulator as sim
from cbfkit.controllers.cbf_clf import robust_cbf_clf_qp_controller, vanilla_cbf_clf_qp_controller
from cbfkit.controllers.mppi import vanilla_mppi
from cbfkit.optimization.quadratic_program.solver_registry import get_solver
from cbfkit.systems.mujoco.reduced_order import (
    G1_FOOTPRINT,
    com_agent_ellipse_hocbfs,
    com_agent_hocbfs,
    ellipse_trajectory_cost,
    embedded_double_integrator,
    embedded_heading_double_integrator,
    mppi_local_planner,
    safe_locomotion_controller_di,
    safe_locomotion_controller_hdi,
)
from cbfkit.utils.user_types import ControllerData, PlannerData

TEST_MODE = bool(os.getenv("CBFKIT_TEST_MODE"))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

GAP = 1.0  # m, centre-to-centre between the two pedestrians
PED_RADIUS = 0.30
ROBOT_DISC = 0.35  # the disc the ellipse replaces
START = jnp.array([-3.0, 0.15])  # slight lateral offset: breaks the symmetric saddle
GOAL_X = 3.0
GOAL_RADIUS = 0.35
V_MAX = 0.3
V_MAX_G1 = 0.2  # slower squeeze on the real robot: smaller command-vs-realised error
A_MAX = 1.0
ALPHA_MAX = 2.0
DEFAULT_DURATION = 60.0
# On the G1 the certificate must cover the measured tracking error (lateral realisation
# ~0.43x, gait-alive shuffle +-0.12 m/s): robust bound, and the honest consequence is a
# wider minimum gap -- 1.15 m instead of the proxy's 1.0 m (the disc still needs 1.30 m).
# (the robust row norm spans the full state incl. the exactly-known agent channels, so the
# margin ~0.41 h-units at bound 0.12: gap 1.15 was measured *infeasible* -- honest price)
GAP_G1 = 1.25
ROBUST_BOUND_G1 = 0.12
# --planner mppi: lookahead over the heading-augmented DI instead of the hand-coded ramp.
LANE_HALF = 0.9  # planner-side corridor: plans stay inside |y| <= LANE_HALF (no wall barrier),
# so MPPI cannot dodge the question by detouring around the pedestrians.
MPPI_DT = 0.2  # s -- replanned at 5 Hz, held in between
MPPI_HORIZON = 30  # x 0.2 s = 6 s: long enough to see that rotating early pays
MPPI_SAMPLES = 64 if TEST_MODE else 1024
MPPI_LAMBDA = 5.0
MPPI_CONTROL_STD = 0.4  # shared std over [ax, ay, alpha] (bounds 1, 1, 2)


class Static:
    """Two motionless pedestrians as a tracked-agents object; ``offset`` shifts the gap
    centre off the straight start-goal line (the hand-coded suggestion does not know)."""

    def __init__(self, gap, offset=0.0):
        self.x0 = jnp.array([[0.0, offset + gap / 2, 0.0, 0.0], [0.0, offset - gap / 2, 0.0, 0.0]])

    def step(self, t, robot_xy, states, dt):
        return states


def build_corridor_mppi(footprint, v_max, dt_ctrl, weights=None):
    """MPPI lookahead for the ``local_planner`` hook: 6 s horizon over the compact
    heading-augmented DI (``ellipse``; it *discovers* the rotation) or the plain DI
    (``disc``; same costs minus the heading terms, so the comparison stays controlled)."""
    n = 2
    goal = jnp.array([GOAL_X, 0.0])
    if footprint == "ellipse":
        dyn = embedded_heading_double_integrator(2, (0, 1), n_agents=n)
        limits = jnp.array([A_MAX, A_MAX, ALPHA_MAX])
        axes = (G1_FOOTPRINT["lon"], G1_FOOTPRINT["lat"])
        cdim, shead, heading = 3, 6, True
    else:
        dyn = embedded_double_integrator(2, (0, 1), n_agents=n)
        limits = jnp.array([A_MAX, A_MAX])
        axes = (ROBOT_DISC, ROBOT_DISC)
        cdim, shead, heading = 2, 4, False
    cost = ellipse_trajectory_cost(
        n,
        goal,
        axes,
        PED_RADIUS,
        MPPI_DT,
        heading=heading,
        lane_halfwidth=LANE_HALF,
        v_max=v_max,
        weights=weights,
    )
    mppi = vanilla_mppi(
        control_limits=limits,
        dynamics_func=dyn,
        trajectory_cost=cost,
        mppi_args={
            "robot_state_dim": shead + 4 * n,
            "robot_control_dim": cdim,
            "prediction_horizon": MPPI_HORIZON,
            "num_samples": MPPI_SAMPLES,
            "time_step": MPPI_DT,
            "use_GPU": False,
            "costs_lambda": MPPI_LAMBDA,
            "cost_perturbation": 0.0,
            "control_std": MPPI_CONTROL_STD,
        },
    )
    return mppi_local_planner(
        mppi,
        n,
        horizon=MPPI_HORIZON,
        replan_every=int(round(MPPI_DT / dt_ctrl)),
        control_dim=cdim,
        state_head=shead,
    )


def build(
    footprint="ellipse",
    gap=GAP,
    g1=False,
    robust_bound=0.0,
    planner="suggest",
    mppi_weights=None,
    offset=0.0,
):
    if g1:
        from cbfkit.systems.mujoco import amo_policy as amo

        plant = amo.make_g1_23dof_plant()
        loco = amo.AmoWholeBodyPolicy().as_controller()
        x0 = amo.x0_standing(plant)
        x0 = x0.at[0:2].add(START).at[plant.com_indices[0] : plant.com_indices[0] + 2].add(START)
    else:
        from examples.mujoco.g1_scramble import ProxyPlant

        plant = ProxyPlant()
        loco = plant.locomotion()
        x0 = jnp.concatenate([START, jnp.zeros(2)])
    ci = plant.com_indices
    agents = Static(gap, offset)
    v_max = V_MAX_G1 if g1 else V_MAX
    lp = (
        build_corridor_mppi(footprint, v_max, plant.dt, mppi_weights) if planner == "mppi" else None
    )
    kw = dict(
        control_limits=(
            jnp.array([A_MAX, A_MAX, ALPHA_MAX])
            if footprint == "ellipse"
            else jnp.array([A_MAX, A_MAX])
        ),
        solver=get_solver("fast", max_iter=32),
    )
    if footprint == "ellipse":
        kw.update(
            dynamics_func=embedded_heading_double_integrator(plant.state_dim, ci, n_agents=2),
            barriers=com_agent_ellipse_hocbfs(
                plant, 2, (G1_FOOTPRINT["lon"], G1_FOOTPRINT["lat"]), ped_radius=PED_RADIUS
            ),
        )
        wrapper = lambda qp: safe_locomotion_controller_hdi(  # noqa: E731
            qp, loco, plant, plant.dt, v_max=v_max, agents=agents, local_planner=lp
        )
    elif footprint == "disc":
        r = ROBOT_DISC + PED_RADIUS
        kw.update(
            dynamics_func=embedded_double_integrator(plant.state_dim, ci, n_agents=2),
            barriers=com_agent_hocbfs(plant, 2, [(r, r)] * 2, shape="distance"),
        )
        wrapper = lambda qp: safe_locomotion_controller_di(  # noqa: E731
            qp, loco, plant, plant.dt, v_max=v_max, agents=agents, local_planner=lp
        )
    else:
        raise ValueError(f"unknown footprint {footprint!r}")
    if robust_bound > 0.0:
        qp = robust_cbf_clf_qp_controller(
            disturbance_norm=2, disturbance_norm_bound=float(robust_bound), **kw
        )
    else:
        qp = vanilla_cbf_clf_qp_controller(**kw)
    safe = wrapper(qp)
    goal = jnp.array([GOAL_X, 0.0])

    def controller(t, x, v_nom, key, data):
        u, d = safe(t, x, v_nom, key, data)
        reached = jnp.linalg.norm(x[ci[0] : ci[0] + 2] - goal) < GOAL_RADIUS
        return u, d._replace(complete=d.complete | reached)

    controller.__cbfkit_controller_adapter__ = True  # type: ignore[attr-defined]

    def nominal(t, x, key, ref):
        com = x[ci[0] : ci[0] + 2]
        v = 1.0 * (jnp.asarray(ref)[:2] - com)
        speed = jnp.linalg.norm(v)
        v = jnp.where(speed > V_MAX, v * (V_MAX / (speed + 1e-9)), v)
        if planner == "mppi":
            # No hand-coded heading: the MPPI local planner owns the lookahead (this
            # v is only its fallback P-law target on a solver failure).
            return v, ControllerData()
        # Heading suggestion (identical for BOTH footprints; the certificate decides):
        # face the travel direction far from the gap, rotate toward sideways (pi/2)
        # within ~1.5 m of the gap line. The hdi wrapper reads entry 2 as the target
        # heading; the di wrapper ignores it.
        engage = jnp.clip((1.5 - jnp.abs(com[0])) / 0.7, 0.0, 1.0)
        th_des = engage * (jnp.pi / 2)
        return jnp.concatenate([v, jnp.array([th_des])]), ControllerData()

    return plant, x0, nominal, controller, agents, goal


def run(
    footprint="ellipse",
    gap=GAP,
    g1=False,
    duration=DEFAULT_DURATION,
    seed=0,
    verbose=False,
    robust_bound=0.0,
    planner="suggest",
    mppi_weights=None,
    offset=0.0,
):
    plant, x0, nominal, controller, agents, goal = build(
        footprint, gap, g1, robust_bound, planner, mppi_weights, offset
    )
    steps = 5 if TEST_MODE else int(round(duration / plant.dt))
    t0 = time.time()
    res = sim.execute(
        x0=x0,
        dt=plant.dt,
        num_steps=steps,
        **(
            dict(plant=plant)
            if g1
            else dict(
                dynamics=plant.dynamics(),
                integrator=__import__(
                    "cbfkit.integration", fromlist=["forward_euler"]
                ).forward_euler,
            )
        ),
        planner_data=PlannerData.from_constant(goal),
        nominal_controller=nominal,
        controller=controller,
        key=jax.random.PRNGKey(seed),
        use_jit=True,
        verbose=verbose,
    )
    wall = time.time() - t0
    S = np.asarray(res["states"])
    ci = plant.com_indices
    com = S[:, ci[0] : ci[0] + 2]
    cd = res.controller_data
    dist_goal = np.linalg.norm(com - np.asarray(goal), axis=1)
    hit = np.flatnonzero(dist_goal < GOAL_RADIUS)
    n_live = int(hit[0]) if hit.size else len(com)
    err = np.flatnonzero(np.asarray(cd["error"]))
    if err.size:
        n_live = min(n_live, int(err[0]))
    peds = np.asarray(agents.x0)[:, :2]
    d = np.linalg.norm(com[:n_live, None, :] - peds[None], axis=2)
    m = {
        "footprint": footprint,
        "planner": planner,
        "gap": gap,
        "crossed": bool(hit.size),
        "time_s": n_live * plant.dt,
        "min_centre_dist": float(d.min()),
        "x_max": float(com[:n_live, 0].max()),
        "qp_errors": int(err.size),
        "wall_s": wall,
    }
    if "sub_data_mppi_error" in cd:
        m["mppi_errors"] = int(np.sum(np.asarray(cd["sub_data_mppi_error"])[:n_live]))
    if footprint == "ellipse" and "sub_data_theta_cmd" in cd:
        th = np.asarray(cd["sub_data_theta_cmd"])[:n_live]
        m["theta_cmd_max_deg"] = float(np.rad2deg(np.abs(th).max()))
        # Dithering metric: total heading travel. One clean rotate-in/rotate-out
        # maneuver costs ~2 x theta_max; every extra reversal adds to this.
        m["theta_travel_deg"] = float(np.rad2deg(np.abs(np.diff(th)).sum())) if th.size > 1 else 0.0
        # h on the *commanded* trajectory (theta_cmd): the certificate quantity
        lon_ax = G1_FOOTPRINT["lon"] + PED_RADIUS
        lat_ax = G1_FOOTPRINT["lat"] + PED_RADIUS
        rel = com[:n_live, None, :] - peds[None]
        c, s = np.cos(th)[:, None], np.sin(th)[:, None]
        lon = (c * rel[..., 0] + s * rel[..., 1]) / lon_ax
        lat = (-s * rel[..., 0] + c * rel[..., 1]) / lat_ax
        m["h_min"] = float((np.sqrt(lon**2 + lat**2) - 1.0).min())
    else:
        m["h_min"] = float((d / (ROBOT_DISC + PED_RADIUS) - 1.0).min())
    if g1:
        up = 1 - 2 * (S[:n_live, 4] ** 2 + S[:n_live, 5] ** 2)
        m["upright_min"] = float(up.min())
    return m, S[: n_live + 1], cd, plant


def main(
    footprint,
    gap,
    g1,
    duration,
    gif=False,
    view=False,
    robust_bound=None,
    planner="suggest",
    offset=0.0,
):
    if robust_bound is None:
        robust_bound = ROBUST_BOUND_G1 if g1 else 0.0
    m, S, cd, plant = run(
        footprint,
        gap,
        g1,
        duration,
        verbose=not TEST_MODE,
        robust_bound=robust_bound,
        planner=planner,
        offset=offset,
    )
    for k, v in m.items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"h(x) min over run: {m['h_min']:.3f}")
    if TEST_MODE:
        return m
    if g1 and (gif or view):
        from cbfkit.systems.mujoco.viewer_utils import add_marker, render_gif, replay_in_viewer

        import mujoco

        peds = np.asarray(Static(gap, offset).x0)[:, :2]

        def markers(scn, k, t):
            for p in peds:
                add_marker(
                    scn,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    [PED_RADIUS * 0.6, 0.55, 0],
                    [*p, 0.85],
                    [0.2, 0.4, 0.9, 1.0],
                )

        os.makedirs(RESULTS_DIR, exist_ok=True)
        tag = footprint if planner == "suggest" else f"{footprint}_{planner}"
        if gif:
            render_gif(
                plant,
                S,
                os.path.join(RESULTS_DIR, f"g1_corridor_{tag}.gif"),
                track_body=int(plant.mj_model.body("pelvis").id),
                markers=markers,
                distance=4.0,
                elevation=-25.0,
            )
        if view:
            replay_in_viewer(plant, S, markers=markers)
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--footprint", choices=("disc", "ellipse"), default="ellipse")
    p.add_argument(
        "--planner",
        choices=("suggest", "mppi"),
        default="suggest",
        help="heading source: hand-coded ramp (suggest) or MPPI lookahead (mppi)",
    )
    p.add_argument(
        "--gap",
        type=float,
        default=None,
        help=f"ped centre distance (default {GAP} proxy / {GAP_G1} G1)",
    )
    p.add_argument(
        "--robust",
        type=float,
        default=None,
        help=f"robust bound (default 0 proxy / {ROBUST_BOUND_G1} G1)",
    )
    p.add_argument("--g1", action="store_true", help="MJX G1 + AMO instead of the 2-D proxy")
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    p.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="shift the gap centre off the start-goal line (the suggestion does not know)",
    )
    p.add_argument("--gif", action="store_true")
    p.add_argument("--view", action="store_true")
    a = p.parse_args()
    if a.gap is None:
        a.gap = GAP_G1 if a.g1 else GAP
    main(a.footprint, a.gap, a.g1, a.duration, a.gif, a.view, a.robust, a.planner, a.offset)
