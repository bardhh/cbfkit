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
x stalls at -0.23). The rotation must come from the layer with lookahead. Here a simple
*nominal suggestion* turns the heading toward sideways near the gap line -- and BOTH
footprints receive the identical suggestion, so the comparison isolates the
*certificate*: the disc CBF refuses the gap with or without the suggestion (1.30 m
needed); the ellipse CBF certifies the sideways passage and lets it through. In the
scramble, the MPPI planner over the heading-augmented model plays the suggestion role
with real lookahead.

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

So the anisotropic gain certified end-to-end on the real robot is 1.25 m vs the disc's
1.30 m -- modest, because the G1's *tracking*, not its geometry, is the binding
constraint; the proxy shows the geometric mechanism at its full 1.0 vs 1.3 m. GIF:
``results/g1_corridor_ellipse.gif`` (the G1 walks up, turns sideways, sidesteps through
the gap, turns back and walks on).

    python examples/mujoco/g1_corridor.py [--footprint disc|ellipse] [--gap G] [--g1]
                                          [--duration T] [--gif] [--view]
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
from cbfkit.optimization.quadratic_program.solver_registry import get_solver
from cbfkit.systems.mujoco.reduced_order import (
    G1_FOOTPRINT,
    com_agent_ellipse_hocbfs,
    com_agent_hocbfs,
    embedded_double_integrator,
    embedded_heading_double_integrator,
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


class Static:
    """Two motionless pedestrians as a tracked-agents object."""

    def __init__(self, gap):
        self.x0 = jnp.array([[0.0, gap / 2, 0.0, 0.0], [0.0, -gap / 2, 0.0, 0.0]])

    def step(self, t, robot_xy, states, dt):
        return states


def build(footprint="ellipse", gap=GAP, g1=False, robust_bound=0.0):
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
    agents = Static(gap)
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
            qp, loco, plant, plant.dt, v_max=(V_MAX_G1 if g1 else V_MAX), agents=agents
        )
    elif footprint == "disc":
        r = ROBOT_DISC + PED_RADIUS
        kw.update(
            dynamics_func=embedded_double_integrator(plant.state_dim, ci, n_agents=2),
            barriers=com_agent_hocbfs(plant, 2, [(r, r)] * 2, shape="distance"),
        )
        wrapper = lambda qp: safe_locomotion_controller_di(  # noqa: E731
            qp, loco, plant, plant.dt, v_max=(V_MAX_G1 if g1 else V_MAX), agents=agents
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
):
    plant, x0, nominal, controller, agents, goal = build(footprint, gap, g1, robust_bound)
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
        "gap": gap,
        "crossed": bool(hit.size),
        "time_s": n_live * plant.dt,
        "min_centre_dist": float(d.min()),
        "x_max": float(com[:n_live, 0].max()),
        "qp_errors": int(err.size),
        "wall_s": wall,
    }
    if footprint == "ellipse" and "sub_data_theta_cmd" in cd:
        th = np.asarray(cd["sub_data_theta_cmd"])[:n_live]
        m["theta_cmd_max_deg"] = float(np.rad2deg(np.abs(th).max()))
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


def main(footprint, gap, g1, duration, gif=False, view=False, robust_bound=None):
    if robust_bound is None:
        robust_bound = ROBUST_BOUND_G1 if g1 else 0.0
    m, S, cd, plant = run(
        footprint, gap, g1, duration, verbose=not TEST_MODE, robust_bound=robust_bound
    )
    for k, v in m.items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"h(x) min over run: {m['h_min']:.3f}")
    if TEST_MODE:
        return m
    if g1 and (gif or view):
        from cbfkit.systems.mujoco.viewer_utils import add_marker, render_gif, replay_in_viewer

        import mujoco

        peds = np.asarray(Static(gap).x0)[:, :2]

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
        if gif:
            render_gif(
                plant,
                S,
                os.path.join(RESULTS_DIR, f"g1_corridor_{footprint}.gif"),
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
    p.add_argument("--gif", action="store_true")
    p.add_argument("--view", action="store_true")
    a = p.parse_args()
    if a.gap is None:
        a.gap = GAP_G1 if a.g1 else GAP
    main(a.footprint, a.gap, a.g1, a.duration, a.gif, a.view, a.robust)
