"""Unitree G1 crosses a Shibuya-style scramble: ~40 pedestrians released on green, flowing in
six streams, while the robot takes the diagonal -- tracked-agent HOCBFs at crowd scale, with
either a go-to-goal nominal or a *socially tuned MPPI* planner in front of the CBF filter.

Same stack as ``g1_plaza.py`` (command-side double integrator, distance-shaped keep-out
barriers on *tracked agents* with constant-velocity prediction, CBF-QP on the in-repo
PDIPM, Unitree's walking policy, MJX G1), scaled up: ``N_PED`` social-force pedestrians
(``SocialForceCrowd``) start 0-30 m behind the kerbs of a 12 x 12 m intersection (so they
arrive throughout the robot's crossing) and cross W<->E, S<->N and along both diagonals at
0.8-1.3 m/s -- faster than the robot (0.5 m/s) -- reacting to each other and to the pillars
but NOT to the robot (``CROWD_REACTS = False``; ``--reactive-crowd`` restores the old yielding
crowd), so every bit of avoidance has to come from the robot's side.
The robot crosses SW -> NE (17 m).

Two planners (``--planner``):

* ``goal`` -- P-law toward the goal at 0.5 m/s; the soft CBF-QP does all the avoiding.
* ``mppi`` -- ``cbfkit.controllers.mppi`` over the compact augmented state
  ``[p | v | pedestrians]`` (5 Hz, 5 s horizon, pedestrians predicted at constant velocity:
  the robot assumes nobody will yield to it), with the cost of
  ``cbfkit.controllers.mppi.social_costs`` -- asymmetric-Gaussian personal space (cheap to
  pass behind someone, expensive to cut in front), time-to-collision, progress as a
  *terminal* cost so waiting for a gap is a legitimate plan, smoothness/legibility terms and
  a keep-left convention for head-on encounters. The same soft CBF-QP stays downstream as
  the filter; a well-behaved planner should rarely trigger it.

What to look at: the *social* metrics -- pedestrian-seconds spent in people's intimate
(< 0.45 m between bodies) and personal (< 1.2 m) zones, *front intrusions* (robot within
1.5 m and inside the +-45 deg cone ahead of a walking pedestrian: the "cut in front" event),
how much the crowd had to deviate from its robot-free paths and slow down near the robot,
how often the CBF had to override the planner -- next to crossing time, waiting and
``h_min``. ``--proxy`` replaces the G1 by the identified reduced model (velocity command
through a 0.2 s first-order lag, see ``g1_model_distance.py``) for fast multi-seed tuning;
``g1_scramble_social_eval.py`` runs the comparison table. Measured (rates are ped-s per
10 s the agent spent inside the intersection; the "human norm" is the same statistic for
a pedestrian in the robot-free crowd: intimate 3.5 / front 2.5):

    proxy, 5 seeds (means)   crossed  time   intimate-rate  front-rate  crowd-dev  CBF active
    goal                     5/5      47 s   4.3            2.2         1.39 m     43 %
    mppi, social terms off   4/5      72 s   1.0            0.6         0.55 m     15 %
    mppi (default weights)   5/5      60 s   1.4            1.0         0.36 m     17 %

    MJX G1, seeds 0/1        crossed  time     intimate-rate  front-rate  h_min        CBF active
    goal                     2/2      49/44 s  4.2 / 6.2      2.0 / 3.0   +0.08/-0.23  47/34 %
    mppi                     2/2      65/56 s  1.7 / 2.5      0.6 / 1.1   +0.16/-0.11  19/23 %

``--robot amo`` swaps the tracking layer for AMO's 23-DoF whole-body policy
(``systems/mujoco/amo_policy.py``); ``--robot groot`` for NVIDIA's GEAR-WBC
(``systems/mujoco/groot_policy.py``); ``--torso`` (AMO only) adds a reactive
shoulder-turn/lean toward the pedestrian being passed. Measured (MJX, mppi planner, seeds 0/1, 100 s; "upper
clearance" = min distance of the shoulder/elbow/hand bodies to the pedestrian discs via
offline forward kinematics):

    robot        torso  crossed   time     intimate-rate  h_min        upper clearance min
    unitree      --     2/2       65/56 s  1.7 / 2.5      +0.16/-0.11  (no arm bodies)
    amo          off    2/2       82/80 s  1.2 / 0.8      -0.12/-0.02  0.11 / 0.16 m
    amo          on     2/2       86/81 s  1.6 / 1.0      -0.14/-0.02  0.08 / 0.14 m
    groot        --     2/2       74/60 s  1.5 / 0.2      -0.06/+0.45  0.21 / 0.52 m

AMO is the gentlest-but-slowest (intimate rate ~1.0; it realises the lowest speed in
MJX); GR00T is the best whole-body compromise -- faster than AMO with comparable
politeness, the largest upper-body clearances (its tracking is closer to the DI command,
so the CBF's model holds better: h stays near or above 0), and on seed 1 it threads the
crowd almost without disturbing it (deviation 0.03 m, intimate rate 0.19). The
reactive shoulder-turn is a measured NEGATIVE result kept as an off-by-default experiment:
in two tuning rounds (engage < 2.0 m / yaw 1.2, then < 1.2 m / yaw 0.6) the torso twist
perturbed the walking policy's tracking by more than the ~8 cm of profile it freed --
upper clearance and h_min got slightly worse, never better. Body agility helps where the
*command* needs it (duck under, turn in place, reach); mid-gait twisting is not free.

``--footprint ellipse`` runs the whole stack on the anisotropic footprint: the corridor's
rotating measured ellipse as the certificate (``com_agent_ellipse_hocbfs`` +
``safe_locomotion_controller_hdi``) and the social MPPI re-hosted on the heading-augmented
model (``reduced_order.heading_social_trajectory_cost`` -- circular collision hinge
replaced by the ellipse clearance preference, so the *plan* owns the rotation and can aim
a shoulder-turn at a gap before it opens). ``--proxy`` or ``--robot amo`` only (the hdi
wrapper hands an absolute target yaw). Measured (mppi planner, soft CBF, seeds 0/1):

    config          crossed   time       intimate   front      h_min        theta travel
    proxy disc      2/2       70/54 s    1.6 / 3.0  0.6 / 2.5  +0.19/+0.01  --
    proxy ellipse   2/2       52/108 s   1.1 / 0.4  0.5 / 0.3  +0.49/+0.82  483/2215 deg
    amo   disc      2/2       82/80 s    1.2 / 0.8  --         -0.12/-0.02  --
    amo   ellipse   1/2       98/(>120)  0.9 / 0.4  0.4 / 0.2  +0.12/+0.05  757/2523 deg

Reading: the rotating footprint is measurably POLITER at better-certified safety -- its
h stays positive where the disc AMO rows droop negative, intimate rates drop, slack is
~0, and the AMO crossing posts the branch's best upper-body clearance p05 (0.54 m). The
open failure mode is heading WIND-UP on the hard seed: threats alternating sides ratchet
the pi-symmetric profile around (theta travel 2200-2500 deg) and the robot politely
pirouettes instead of arriving (amo/s1 did not cross in 120 s; safe and gentle
throughout -- it fails by not getting there). The ``overturn`` +-90 deg band and the
goal-biased heading reference (``reduced_order.EllipseCostWeights``) tame it on seed 0,
but cost weights cannot reliably beat the warm-started rotation momentum at a replan
boundary; plan-commitment smoothing in ``mppi_local_planner`` is the designated
follow-up.

Read it this way: the goal-seeking baseline is *more* intrusive than an average pedestrian
(intimate rate 4.3 vs the human norm 3.5) and lives on the CBF; the social MPPI is ~2.5x
less intrusive than that norm, disturbs the crowd 4x less, hands the CBF an almost-feasible
plan (interventions halve) and even shrinks the worst-case keep-out penetration on the hard
seed -- for ~+13 s of crossing time. The ablation (social terms zeroed) shows the proxemics/
TTC/slow terms are what buy *reliability at speed*: without them the planner is similarly
polite but timid (4/5 crossings, +10 s, 14 % waiting).

    python examples/mujoco/g1_scramble.py [--planner goal|mppi] [--footprint disc|ellipse]
                                          [--proxy] [--robust B] [--pedestrians N]
                                          [--relax|--hard] [--duration T] [--seed S] [--gif] [--view]
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
from cbfkit.controllers.mppi.social_costs import SocialCostWeights, social_trajectory_cost
from cbfkit.integration import forward_euler as euler
from cbfkit.optimization.quadratic_program.solver_registry import get_solver
from cbfkit.systems.mujoco.amo_policy import _rpy
from cbfkit.systems.mujoco.crowd import SocialForceCrowd
from cbfkit.systems.mujoco.reduced_order import (
    G1_FOOTPRINT,
    com_agent_ellipse_hocbfs,
    com_agent_hocbfs,
    embedded_double_integrator,
    embedded_heading_double_integrator,
    heading_social_trajectory_cost,
    mppi_local_planner,
    safe_locomotion_controller_di,
    safe_locomotion_controller_hdi,
)
from cbfkit.systems.mujoco.unitree_policy import _wrap
from cbfkit.utils.user_types import ControllerData, PlannerData

TEST_MODE = bool(os.getenv("CBFKIT_TEST_MODE"))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# --------------------------------------------------------------------------- scenario
HALF = 6.0  # the intersection is [-HALF, HALF]^2; kerbs at +-HALF
START = jnp.array([-HALF, -HALF])
GOAL = jnp.array([HALF, HALF])
GOAL_RADIUS = 0.4
N_PED_FULL = 40
N_PED = 8 if TEST_MODE else N_PED_FULL  # the smoke run keeps the JIT short
PED_SPEED_RANGE = (0.8, 1.3)  # m/s -- real pedestrians, faster than the robot
PED_RADIUS = 0.30
PED_REPULSION_RANGE = 0.4
RELEASE_DEPTH = 30.0  # pedestrians start 0-30 m behind the kerbs -> arrivals spread over ~30 s
ARRIVE_RADIUS = 1.0  # they stop at their goals (6 m past the far kerb)
ROBOT_RADIUS = 0.35
R_PED = PED_RADIUS + ROBOT_RADIUS
V_MAX = 0.5
A_MAX = 1.0
ALPHA_MAX = 2.0  # heading acceleration bound of the --footprint ellipse variant
BARRIER_SHAPE = "distance"
DEFAULT_ROBUST_BOUND = 0.0  # in a crush robust margins are eaten by slack (see the eval table)
CROWD_REACTS = False  # pedestrians ignore the robot (social forces only among themselves)
DEFAULT_RELAX = True  # hard barrier constraints are infeasible within ~10 s in this crowd
DEFAULT_DURATION = 60.0
DEFAULT_PLANNER = "goal"
DEFAULT_ROBOT = "unitree"
# reactive torso layer (--torso, AMO only): shoulder-turn toward the pedestrian being
# passed (torso depth ~0.16 m presented toward them instead of shoulder half-width
# ~0.24 m) plus a slight lean away; pure body language, the CBF keep-out is unchanged.
# First round (engage < 2.0 m, yaw <= 1.2, roll 0.25) HURT: 23-27% engagement with large
# twists degraded the gait tracking more than the geometry bought (upper clearance and
# h_min got worse, +14 s crossing). Tuned: engage only in an actual close pass, gently.
TORSO_TURN_RANGE = 1.2  # m: engage when the closest pedestrian is nearer than this
TORSO_YAW_MAX = 0.6  # rad (AMO tolerates 1.57, but large yaw costs gait accuracy)
TORSO_ROLL_AWAY = 0.15  # rad lean away from the pedestrian
TORSO_SMOOTH = 0.95  # first-order smoothing of the torso command per 20 ms step
CONTROL_DT = 0.02  # the G1 plant's step (policy at 50 Hz); the proxy uses the same
_TAG = ["run"]

# --------------------------------------------------------------------------- social MPPI
MPPI_DT = 0.2  # s -- replanned at 5 Hz
MPPI_HORIZON = 25  # 5 s
MPPI_SAMPLES = 64 if TEST_MODE else 1024
MPPI_LAMBDA = 5.0  # temperature, relative to costs of O(10-100)
MPPI_CONTROL_STD = 0.3  # m/s^2 sampling std (bound A_MAX = 1)
PASS_SIDE = "left"  # Japan: keep left in head-on encounters
DEFAULT_WEIGHTS = SocialCostWeights(pass_side=0.5, v_max=V_MAX)
PROXY_TAU = 0.20  # s, identified first-order lag of the G1 + walking policy (g1_model_distance)

# zones for the social metrics (distance between body surfaces, Hall's proxemics)
INTIMATE = 0.45
PERSONAL = 1.2
FRONT_RANGE = 1.5  # m (centre-to-centre) and
FRONT_HALF_ANGLE = np.deg2rad(45.0)  # +-45 deg ahead of a walking pedestrian = "cutting in front"

# streams: (start side, goal side) in intersection coordinates; weights favour the straight ones
_STREAMS = [
    ("W", "E", 5),
    ("E", "W", 5),
    ("S", "N", 5),
    ("N", "S", 5),
    ("SW", "NE", 2),
    ("NE", "SW", 2),
    ("NW", "SE", 2),
    ("SE", "NW", 2),
]
_ANCHOR = {
    "W": (-1, 0),
    "E": (1, 0),
    "S": (0, -1),
    "N": (0, 1),
    "SW": (-1, -1),
    "NE": (1, 1),
    "NW": (-1, 1),
    "SE": (1, -1),
}


def make_crowd(n_ped: int, seed: int):
    """Starts/goals/speeds for ``n_ped`` pedestrians drawn from the streams (seeded)."""
    rng = np.random.default_rng(seed + 1000)
    sides = [s for s in _STREAMS for _ in range(s[2])]
    starts, goals, speeds = [], [], []
    for i in range(n_ped):
        a, b, _ = sides[i % len(sides)]
        ua = np.asarray(_ANCHOR[a], float)
        ub = np.asarray(_ANCHOR[b], float)
        ua_n = ua / np.linalg.norm(ua)
        ub_n = ub / np.linalg.norm(ub)
        perp = np.array([-ua_n[1], ua_n[0]])
        back = rng.uniform(0.0, RELEASE_DEPTH)  # metres behind the kerb at t = 0: staggers arrivals
        lateral = rng.uniform(-2.5, 2.5) if len(a) == 1 else rng.uniform(-1.0, 1.0)
        start = ua_n * (HALF * np.linalg.norm(ua) + back) + perp * lateral
        goal = ub_n * (HALF * np.linalg.norm(ub) + 6.0) + perp * (lateral + rng.uniform(-1.0, 1.0))
        starts.append(start)
        goals.append(goal)
        speeds.append(rng.uniform(*PED_SPEED_RANGE))
    return np.asarray(starts), np.asarray(goals), np.asarray(speeds)


class ProxyPlant:
    """2-D stand-in for the G1 + walking policy: the CoM follows the planar velocity command
    through a first-order lag ``tau`` (the model identified in ``g1_model_distance.py``).
    State ``[px, py, vx, vy]``, control = commanded velocity. Exposes what the reduced-order
    layer needs (``com_indices``, ``state_dim``, ``dt``, ``nu``)."""

    state_dim = 4
    com_indices = (0, 1)
    nu = 2

    def __init__(self, tau: float = PROXY_TAU, dt: float = CONTROL_DT):
        self.tau = float(tau)
        self.dt = float(dt)

    def dynamics(self):
        tau = self.tau
        g = jnp.zeros((4, 2)).at[2, 0].set(1.0 / tau).at[3, 1].set(1.0 / tau)

        def f_g(x):
            return jnp.array([x[2], x[3], -x[2] / tau, -x[3] / tau]), g

        return f_g

    def locomotion(self):
        def loco(t, x, cmd, key, data):
            return jnp.asarray(cmd, dtype=float)[:2], data

        return loco


def build_social_mppi(
    n_ped: int,
    weights: SocialCostWeights,
    seed: int = 0,
    *,
    costs_lambda: float = MPPI_LAMBDA,
    control_std: float = MPPI_CONTROL_STD,
    samples: int = MPPI_SAMPLES,
    footprint: str = "disc",
    ellipse_weights=None,
):
    """MPPI planner over the compact ``[p | v | pedestrians]`` state with the social cost, plus
    the adapter for :func:`safe_locomotion_controller_di`. ``footprint="ellipse"`` re-hosts the
    social cost on the heading-augmented layout (:func:`heading_social_trajectory_cost`): the
    plan then owns the rotation and can aim a shoulder-turn at a gap before it opens."""
    if footprint == "ellipse":
        dyn = embedded_heading_double_integrator(2, (0, 1), n_agents=n_ped)
        limits = jnp.array([A_MAX, A_MAX, ALPHA_MAX])
        cost = heading_social_trajectory_cost(
            n_ped,
            GOAL,
            weights,
            MPPI_DT,
            (G1_FOOTPRINT["lon"], G1_FOOTPRINT["lat"]),
            PED_RADIUS,
            robot_radius=ROBOT_RADIUS,
            pass_side=PASS_SIDE if weights.pass_side > 0 else None,
            ellipse_weights=ellipse_weights,
        )
        cdim, shead = 3, 6
    else:
        dyn = embedded_double_integrator(2, (0, 1), n_agents=n_ped)
        limits = jnp.array([A_MAX, A_MAX])
        cost = social_trajectory_cost(
            n_agents=n_ped,
            goal=GOAL,
            weights=weights,
            dt=MPPI_DT,
            robot_radius=ROBOT_RADIUS,
            ped_radius=PED_RADIUS,
            pass_side=PASS_SIDE if weights.pass_side > 0 else None,
        )
        cdim, shead = 2, 4
    mppi = vanilla_mppi(
        control_limits=limits,
        dynamics_func=dyn,
        trajectory_cost=cost,
        mppi_args={
            "robot_state_dim": shead + 4 * n_ped,
            "robot_control_dim": cdim,
            "prediction_horizon": MPPI_HORIZON,
            "num_samples": samples,
            "time_step": MPPI_DT,
            "use_GPU": False,
            "costs_lambda": costs_lambda,
            "cost_perturbation": 0.0,
            "control_std": control_std,
        },
    )
    return mppi_local_planner(
        mppi,
        n_ped,
        horizon=MPPI_HORIZON,
        replan_every=int(round(MPPI_DT / CONTROL_DT)),
        control_dim=cdim,
        state_head=shead,
    )


def shoulder_turn_torso(ci):
    """State-aware torso command for ``AmoWholeBodyPolicy.as_controller``: yaw the torso
    toward the closest tracked pedestrian while passing (slims the profile toward them),
    lean slightly away; upright when nobody is near or the robot stands. Smoothed and
    carried in ``sub["_torso"]``."""

    def torso(t, x, sub):
        prev = sub.get("_torso")
        if prev is None:
            prev = jnp.zeros(4)
        ag = sub.get("_agents")
        if ag is None:
            return prev
        com = x[ci[0] : ci[0] + 2]
        rel = jnp.asarray(ag)[:, :2] - com
        d = jnp.linalg.norm(rel, axis=1)
        i = jnp.argmin(d)
        yaw = _rpy(x[3:7])[2]
        bearing = _wrap(jnp.arctan2(rel[i, 1], rel[i, 0]) - yaw)
        gain = jnp.clip((TORSO_TURN_RANGE - d[i]) / 0.4, 0.0, 1.0)
        tyaw = gain * jnp.clip(bearing, -TORSO_YAW_MAX, TORSO_YAW_MAX)
        troll = -gain * TORSO_ROLL_AWAY * jnp.sign(bearing)
        target = jnp.array([0.0, tyaw, 0.0, troll])
        new = TORSO_SMOOTH * prev + (1.0 - TORSO_SMOOTH) * target
        sub["_torso"] = new
        return new

    return torso


def build(
    seed: int = 0,
    robust_bound: float = 0.0,
    n_ped: int = N_PED,
    relax: bool = False,
    planner: str = DEFAULT_PLANNER,
    weights: SocialCostWeights = DEFAULT_WEIGHTS,
    proxy: bool = False,
    mppi_kw: dict = None,
    robot: str = DEFAULT_ROBOT,
    torso: bool = False,
    footprint: str = "disc",
    crowd_reacts: bool = CROWD_REACTS,
):
    if torso and (proxy or robot != "amo"):
        raise ValueError("--torso needs the AMO robot (--robot amo, not proxy)")
    if footprint not in ("disc", "ellipse"):
        raise ValueError(f"unknown footprint {footprint!r} (disc | ellipse)")
    if footprint == "ellipse" and not (proxy or robot == "amo"):
        raise ValueError(
            "--footprint ellipse needs --proxy or --robot amo: the hdi wrapper hands an "
            "absolute target yaw and only the AMO adapter takes one (GR00T/Unitree take rates)"
        )
    # --robot groot: NVIDIA GEAR-WBC (see g1_walk_compare.py -- flattest ride of the three)
    if proxy:
        plant = ProxyPlant()
        loco = plant.locomotion()
        x0 = jnp.concatenate([START, jnp.zeros(2)])
        pelvis_body = None
    elif robot == "amo":
        from cbfkit.systems.mujoco import amo_policy as amo

        plant = amo.make_g1_23dof_plant()
        torso_cmd = shoulder_turn_torso(plant.com_indices) if torso else None
        loco = amo.AmoWholeBodyPolicy().as_controller(torso_command=torso_cmd)
        x0 = amo.x0_standing(plant)
        x0 = x0.at[0:2].add(START).at[plant.com_indices[0] : plant.com_indices[0] + 2].add(START)
        pelvis_body = int(plant.mj_model.body("pelvis").id)
    elif robot == "groot":
        from cbfkit.systems.mujoco import groot_policy as groot

        plant = groot.make_g1_29dof_plant()
        loco = groot.GrootGearWbcPolicy().as_controller()
        x0 = groot.x0_standing(plant)
        x0 = x0.at[0:2].add(START).at[plant.com_indices[0] : plant.com_indices[0] + 2].add(START)
        pelvis_body = int(plant.mj_model.body("pelvis").id)
    elif robot == "unitree":
        from cbfkit.systems.mujoco.unitree_policy import (
            UnitreeG1WalkPolicy,
            make_g1_12dof_plant,
            x0_standing,
        )

        plant = make_g1_12dof_plant()
        loco = UnitreeG1WalkPolicy().as_controller()
        x0 = x0_standing(plant)
        # Place the robot at START (the XML puts it at the origin): shift the pelvis x, y.
        x0 = x0.at[0:2].add(START).at[plant.com_indices[0] : plant.com_indices[0] + 2].add(START)
        pelvis_body = int(plant.mj_model.body("pelvis").id)
    else:
        raise ValueError(f"unknown robot {robot!r} (unitree | amo | groot)")
    ci = plant.com_indices
    starts, goals, speeds = make_crowd(n_ped, seed)
    crowd = SocialForceCrowd(
        starts,
        goals,
        speeds,
        ped_radius=PED_RADIUS,
        agent_radius=ROBOT_RADIUS,
        repulsion_range=PED_REPULSION_RANGE,
        arrive_radius=ARRIVE_RADIUS,
        react_to_robot=crowd_reacts,
    )
    if footprint == "ellipse":
        dyn = embedded_heading_double_integrator(plant.state_dim, ci, n_agents=n_ped)
        barriers = com_agent_ellipse_hocbfs(
            plant, n_ped, (G1_FOOTPRINT["lon"], G1_FOOTPRINT["lat"]), ped_radius=PED_RADIUS
        )
        control_limits = jnp.array([A_MAX, A_MAX, ALPHA_MAX])
    else:
        dyn = embedded_double_integrator(plant.state_dim, ci, n_agents=n_ped)
        barriers = com_agent_hocbfs(
            plant, n_ped, [(R_PED, R_PED)] * n_ped, class_k_gain=1.0, shape=BARRIER_SHAPE
        )
        control_limits = jnp.array([A_MAX, A_MAX])
    kw = dict(
        control_limits=control_limits,
        dynamics_func=dyn,
        barriers=barriers,
        # 32 PDIPM iterations (default 16): at an MPPI replan boundary a_nom jumps and the
        # warm-started active set can need a few extra iterations (measured: 16 fails on
        # seed 1 at t = 12.4 s, 32 converges; cold start also solves it in 16). tol 1e-5
        # (default 1e-6): with a slack penalty of 1e3 the combined KKT residual of a
        # degenerate-optimal QP stalls just above 1e-6 (measured 2.8e-6, seed 1 t = 13.2 s)
        # while the control is already exact to ~1e-6 -- 1e-5 accepts it and the solver's
        # freeze-on-converge then prevents the late-stage Mehrotra blow-up.
        solver=get_solver("fast", max_iter=32, tol=1e-5),
    )
    if relax:
        kw.update(relaxable_cbf=True, slack_penalty_cbf=1e3, slack_bound_cbf=10.0)
    if robust_bound > 0.0:
        cbf_qp = robust_cbf_clf_qp_controller(
            disturbance_norm=2, disturbance_norm_bound=float(robust_bound), **kw
        )
    else:
        cbf_qp = vanilla_cbf_clf_qp_controller(**kw)
    if planner == "mppi":
        local_planner = build_social_mppi(
            n_ped, weights, seed, footprint=footprint, **(mppi_kw or {})
        )
    elif planner == "goal":
        local_planner = None
    else:
        raise ValueError(f"unknown planner {planner!r} (goal | mppi)")
    wrapper = (
        safe_locomotion_controller_hdi if footprint == "ellipse" else safe_locomotion_controller_di
    )
    safe = wrapper(
        cbf_qp, loco, plant, plant.dt, v_max=V_MAX, agents=crowd, local_planner=local_planner
    )

    def controller(t, x, v_nom, key, data):
        u, d = safe(t, x, v_nom, key, data)
        reached = jnp.linalg.norm(x[ci[0] : ci[0] + 2] - GOAL) < GOAL_RADIUS
        return u, d._replace(complete=d.complete | reached)

    controller.__cbfkit_controller_adapter__ = True  # type: ignore[attr-defined]

    def nominal(t, x, key, ref):
        com = x[ci[0] : ci[0] + 2]
        v = 1.0 * (jnp.asarray(ref)[:2] - com)
        speed = jnp.linalg.norm(v)
        v = jnp.where(speed > V_MAX, v * (V_MAX / (speed + 1e-9)), v)
        return v, ControllerData()

    return plant, x0, pelvis_body, nominal, controller, crowd


# --------------------------------------------------------------------------- metrics
def upper_body_clearance(plant, states, agents):
    """True clearance between the robot's *upper body* and the pedestrian discs.

    CPU-mujoco forward kinematics on the logged qpos: min over the shoulder / elbow /
    wrist(hand) body positions (XY) of the distance to the closest pedestrian centre,
    minus ``PED_RADIUS``. This is where a shoulder-turn shows up: the CoM keep-out disc
    is unchanged, but the body carried through it clears the pedestrians by more.
    Returns ``upper_clearance_min`` and ``upper_clearance_p05`` (metres, surface-to-centre
    minus pedestrian radius; > 0 means no upper-body contact with the disc).
    """
    import mujoco

    m = plant.mj_model
    d = mujoco.MjData(m)
    names = []
    for side in ("left", "right"):
        for part in ("shoulder_roll_link", "elbow_link", "wrist_yaw_link", "rubber_hand"):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_{part}")
            if bid >= 0:
                names.append(bid)
        # the 12-DoF model has no arm bodies; fall back to the torso/pelvis
    if not names:
        for part in ("torso_link", "pelvis"):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, part)
            if bid >= 0:
                names.append(bid)
    S = np.asarray(states)
    A = np.asarray(agents)
    mins = np.empty(len(S))
    for k in range(len(S)):
        d.qpos[:] = S[k, : plant.nq]
        mujoco.mj_kinematics(m, d)
        pts = d.xpos[names][:, :2]  # (B, 2)
        dist = np.linalg.norm(pts[:, None, :] - A[k, None, :, :2], axis=2)  # (B, N)
        mins[k] = dist.min()
    clear = mins - PED_RADIUS
    return {
        "upper_clearance_min": float(clear.min()),
        "upper_clearance_p05": float(np.percentile(clear, 5)),
    }


def robot_free_crowd(crowd: SocialForceCrowd, n_steps: int, dt: float) -> np.ndarray:
    """The crowd's trajectory ``(n_steps, N, 4)`` with the robot absent (parked far away)."""
    far = jnp.array([1e4, 1e4])

    def step(states, k):
        new = crowd.step(k * dt, far, states, dt)
        return new, states

    _, traj = jax.lax.scan(jax.jit(step), jnp.asarray(crowd.x0), jnp.arange(n_steps))
    return np.asarray(traj)


def _in_square(p, margin=1.0):
    return np.all(np.abs(p) <= HALF + margin, axis=-1)


def human_norm(free_agents, dt):
    """How much pedestrians intrude on *each other* in the robot-free rollout ``(T, N, 4)``:
    per pedestrian, pedestrian-seconds in others' intimate / personal zones and in the front
    cone of a walking other, divided by that pedestrian's time inside the intersection, averaged
    over pedestrians with >= 2 s inside -- reported per 10 s inside. The robot's rates below
    are computed the same way, so "rate ~ human rate" means "as intrusive as a pedestrian"."""
    P, V = free_agents[:, :, :2], free_agents[:, :, 2:]
    T, N = P.shape[:2]
    if N < 2:
        return dict(human_intimate_rate=0.0, human_personal_rate=0.0, human_front_rate=0.0)
    rel = P[:, :, None, :] - P[:, None, :, :]  # i from j: (T, N, N, 2)
    d = np.linalg.norm(rel, axis=-1)
    eye = np.eye(N, dtype=bool)[None]
    surf = d - 2 * PED_RADIUS
    sp = np.linalg.norm(V, axis=-1)  # (T, N)
    cosang = np.sum(rel * V[:, None, :, :], axis=-1) / np.maximum(d * sp[:, None, :], 1e-9)
    front = (d < FRONT_RANGE) & (sp[:, None, :] > 0.2) & (cosang > np.cos(FRONT_HALF_ANGLE)) & ~eye
    intimate = (surf < INTIMATE) & ~eye
    personal = (surf >= INTIMATE) & (surf < PERSONAL) & ~eye
    inside = _in_square(P)  # (T, N)
    t_in = inside.sum(0) * dt
    ok = t_in >= 2.0

    def rate(mask):
        per = (mask & inside[:, :, None]).sum(axis=(0, 2)) * dt  # per pedestrian i (as intruder)
        return float(np.mean(per[ok] / t_in[ok]) * 10.0) if ok.any() else 0.0

    return dict(
        human_intimate_rate=rate(intimate),
        human_personal_rate=rate(personal),
        human_front_rate=rate(front),
    )


def social_metrics(com, agents, free_agents, crowd, dt, a_nom, a_safe, v_safe, slack=None):
    """Intrusiveness and efficiency of one run (arrays truncated to the live part already).

    ``com`` ``(T, 2)``, ``agents`` ``(T, N, 4)`` (what the robot saw), ``free_agents`` the same
    crowd without the robot, ``a_nom``/``a_safe`` ``(T, 2)`` the planner's acceleration and the
    CBF-QP's, ``v_safe`` the integrated command.
    Pedestrian-seconds are summed over pedestrians and time; "front intrusion" counts the time
    the robot spent within ``FRONT_RANGE`` and inside the ``+-FRONT_HALF_ANGLE`` cone ahead of a
    walking pedestrian (speed > 0.2 m/s) -- the "cut in front" event; "ped deviation" is the
    pedestrians' position difference to the robot-free rollout (same time index), "ped slowdown"
    the robot-attributable speed loss (vs the robot-free rollout) within 2.5 m of the robot.
    ``*_rate`` are the robot's intimate / personal / front pedestrian-seconds per 10 s the robot
    spent inside the intersection, comparable to :func:`human_norm`.
    """
    T = len(com)
    d = np.linalg.norm(com[:, None, :] - agents[:, :, :2], axis=2)  # centre-to-centre (T, N)
    surf = d - R_PED
    vp = agents[:, :, 2:]
    sp = np.linalg.norm(vp, axis=2)
    rel = com[:, None, :] - agents[:, :, :2]  # robot from the pedestrian
    cosang = np.sum(rel * vp, axis=2) / np.maximum(d * sp, 1e-9)
    front = (d < FRONT_RANGE) & (sp > 0.2) & (cosang > np.cos(FRONT_HALF_ANGLE))
    intimate = surf < INTIMATE
    personal = (surf >= INTIMATE) & (surf < PERSONAL)
    near = d < 2.5
    speed_free = np.linalg.norm(free_agents[:T, :, 2:], axis=2)
    slow = np.maximum(speed_free - sp, 0.0) / np.maximum(np.asarray(crowd.speeds)[None], 1e-9)
    dev = np.linalg.norm(agents[:, :, :2] - free_agents[:T, :, :2], axis=2)  # (T, N)
    v_com = np.gradient(com, dt, axis=0) if T > 1 else np.zeros_like(com)
    speed = np.linalg.norm(v_com, axis=1)
    moving = speed > 0.1
    th = np.arctan2(v_com[:, 1], v_com[:, 0])
    dth = np.abs(np.angle(np.exp(1j * np.diff(th))))
    path = float(np.sum(np.linalg.norm(np.diff(com, axis=0), axis=1))) if T > 1 else 0.0
    straight = float(np.linalg.norm(com[-1] - com[0]))
    a_cmd = np.gradient(v_safe, dt, axis=0) if T > 1 else np.zeros_like(v_safe)
    jerk = np.gradient(a_cmd, dt, axis=0) if T > 2 else np.zeros_like(a_cmd)
    t_in = max(float(_in_square(com).sum() * dt), 1e-9)
    out = {
        "crossing_time_s": T * dt,
        "waiting_frac": float(np.mean(~moving)) if T else 0.0,
        "path_ratio": path / max(straight, 1e-9),
        "closest_m": float(d.min()) if d.size else np.inf,
        "intimate_ped_s": float(np.sum(intimate) * dt),
        "personal_ped_s": float(np.sum(personal) * dt),
        "front_intrusion_ped_s": float(np.sum(front) * dt),
        "front_intrusions_n": int(np.sum(front.any(0))),
        "intimate_rate": float(np.sum(intimate) * dt / t_in * 10.0),
        "personal_rate": float(np.sum(personal) * dt / t_in * 10.0),
        "front_rate": float(np.sum(front) * dt / t_in * 10.0),
        "robot_in_square_s": t_in,
        "ped_deviation_mean_m": float(dev.max(0).mean()) if dev.size else 0.0,
        "ped_deviation_max_m": float(dev.max()) if dev.size else 0.0,
        "ped_slowdown_ped_s": float(np.sum(slow * near) * dt),
        "robot_turn_rad": float(np.sum(dth[moving[1:] & moving[:-1]])) if T > 1 else 0.0,
        "robot_jerk_rms": float(np.sqrt(np.mean(np.sum(jerk**2, axis=1)))) if T > 2 else 0.0,
        "cbf_active_frac": float(np.mean(np.linalg.norm(a_safe - a_nom, axis=1) > 1e-2)),
        "slack_frac": float(np.mean(slack.max(1) > 1e-3)) if slack is not None else 0.0,
        "h_min": float((d / R_PED - 1.0).min()) if d.size else np.inf,
    }
    out.update(human_norm(free_agents[:T], dt))
    return out


# --------------------------------------------------------------------------- run
def run(
    duration=DEFAULT_DURATION,
    seed=0,
    robust_bound=DEFAULT_ROBUST_BOUND,
    n_ped=N_PED,
    relax=DEFAULT_RELAX,
    planner=DEFAULT_PLANNER,
    weights=DEFAULT_WEIGHTS,
    proxy=False,
    verbose=False,
    mppi_kw=None,
    robot=DEFAULT_ROBOT,
    torso=False,
    footprint="disc",
    crowd_reacts=CROWD_REACTS,
):
    """Simulate one crossing; returns a dict with the metrics and the raw arrays."""
    plant, x0, pelvis_body, nominal, controller, crowd = build(
        seed,
        robust_bound,
        n_ped,
        relax,
        planner,
        weights,
        proxy,
        mppi_kw,
        robot,
        torso,
        footprint,
        crowd_reacts,
    )
    steps = int(round(duration / plant.dt))
    t0 = time.time()
    kw = dict(plant=plant) if not proxy else dict(dynamics=plant.dynamics(), integrator=euler)
    res = sim.execute(
        x0=x0,
        dt=plant.dt,
        num_steps=steps,
        planner_data=PlannerData.from_constant(GOAL),
        nominal_controller=nominal,
        controller=controller,
        key=jax.random.PRNGKey(seed),
        use_jit=True,
        verbose=verbose,
        **kw,
    )
    wall = time.time() - t0
    S = np.asarray(res["states"])
    ci = plant.com_indices
    com = S[:, ci[0] : ci[0] + 2]
    cd = res.controller_data
    agents = np.asarray(cd["sub_data_agents"])  # (T, N, 4)
    status = np.asarray(cd["sub_data_solver_status"])
    v_nom = np.asarray(cd["sub_data_v_nom"])
    v_safe = np.asarray(cd["sub_data_v_safe"])
    a_nom = np.asarray(cd["sub_data_a_nom"])
    a_safe = np.asarray(cd["sub_data_a_safe"])
    dist_goal = np.linalg.norm(com - np.asarray(GOAL), axis=1)
    hit = np.flatnonzero(dist_goal < GOAL_RADIUS)
    n_live = int(hit[0]) if hit.size else len(com)
    err_steps = np.flatnonzero(np.asarray(cd["error"]))
    if err_steps.size:  # the simulation latches after a controller error: score the live part only
        n_live = min(n_live, int(err_steps[0]))
    slack = None
    if relax and "sol" in cd:
        n_u = 3 if footprint == "ellipse" else 2
        slack = np.asarray(cd["sol"])[:n_live, n_u:]  # slack columns follow the controls
    free = robot_free_crowd(crowd, max(n_live, 1), plant.dt)
    m = social_metrics(
        com[:n_live],
        agents[:n_live],
        free,
        crowd,
        plant.dt,
        a_nom[:n_live],
        a_safe[:n_live],
        v_safe[:n_live],
        slack,
    )
    m.update(
        crossed=bool(hit.size),
        dist_goal_end=float(dist_goal[-1]),
        stopped_at_s=(float(err_steps[0] * plant.dt) if err_steps.size else None),
        qp_nonconverged=int(np.sum(status[:n_live] != 1)),
        wall_s=wall,
        steps=steps,
        n_ped=n_ped,
        planner=planner,
        proxy=proxy,
        robot=None if proxy else robot,
        torso=torso,
        footprint=footprint,
    )
    if footprint == "ellipse" and "sub_data_theta_cmd" in cd:
        thc = np.asarray(cd["sub_data_theta_cmd"])[:n_live]
        # The certificate quantity: rotating-ellipse h on the measured com + commanded theta
        # (replaces the disc h_min; closest_m and the person-centred rates stay comparable).
        a_lon = G1_FOOTPRINT["lon"] + PED_RADIUS
        a_lat = G1_FOOTPRINT["lat"] + PED_RADIUS
        rel = com[:n_live, None, :] - agents[:n_live, :, :2]
        c_, s_ = np.cos(thc)[:, None], np.sin(thc)[:, None]
        lon = (c_ * rel[..., 0] + s_ * rel[..., 1]) / a_lon
        lat = (-s_ * rel[..., 0] + c_ * rel[..., 1]) / a_lat
        m["h_min"] = float((np.sqrt(lon**2 + lat**2) - 1.0).min()) if rel.size else np.inf
        m["theta_cmd_max_deg"] = float(np.rad2deg(np.abs(thc).max())) if thc.size else 0.0
        m["theta_travel_deg"] = (
            float(np.rad2deg(np.abs(np.diff(thc)).sum())) if thc.size > 1 else 0.0
        )
    if not proxy:
        q = S[:n_live, 3:7]
        m["upright_min"] = float((1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)).min())
        m["pelvis_z_min"] = float(S[:n_live, 2].min())
    if "sub_data_mppi_error" in cd:
        m["mppi_errors"] = int(np.sum(np.asarray(cd["sub_data_mppi_error"])[:n_live]))
    if not proxy:
        m.update(upper_body_clearance(plant, S[:n_live], agents[:n_live]))
    if "sub_data_amo_cmd" in cd:
        ac = np.asarray(cd["sub_data_amo_cmd"])[:n_live]
        m["torso_yaw_p95"] = float(np.percentile(np.abs(ac[:, 4]), 95))
        m["torso_yaw_frac"] = float(np.mean(np.abs(ac[:, 4]) > 0.2))
    return dict(
        metrics=m,
        states=S,
        com=com,
        agents=agents,
        free_agents=free,
        v_nom=v_nom,
        v_safe=v_safe,
        n_live=n_live,
        plant=plant,
        crowd=crowd,
        pelvis_body=pelvis_body,
        mppi_plans=(
            np.asarray(cd["sub_data_mppi_x_traj"]) if "sub_data_mppi_x_traj" in cd else None
        ),
    )


def print_report(m):
    dt_txt = (
        f"crossed at t={m['crossing_time_s']:.1f}s"
        if m["crossed"]
        else (f"crossing NOT completed (distance to goal at the end {m['dist_goal_end']:.2f} m)")
    )
    print(
        f"{m['steps']} steps in {m['wall_s']:.1f}s  ({m['n_ped']} pedestrians, planner={m['planner']}"
        f"{', proxy' if m['proxy'] else ''})"
    )
    print(
        f"h(x) min over run: {m['h_min']:.3f}   (>= 0 means no pedestrian keep-out disc was entered)"
    )
    print(
        f"closest CoM-pedestrian distance: {m['closest_m']:.2f} m (keep-out {R_PED:.2f}); {dt_txt}"
    )
    print(
        f"social: intimate {m['intimate_ped_s']:.1f} ped-s, personal {m['personal_ped_s']:.1f} ped-s, "
        f"front intrusions {m['front_intrusion_ped_s']:.1f} ped-s ({m['front_intrusions_n']} pedestrians); "
        f"crowd deviation mean {m['ped_deviation_mean_m']:.2f} / max {m['ped_deviation_max_m']:.2f} m, "
        f"slowdown near robot {m['ped_slowdown_ped_s']:.1f} ped-s"
    )
    print(
        f"rates per 10 s inside the intersection -- robot: intimate {m['intimate_rate']:.2f}, personal "
        f"{m['personal_rate']:.2f}, front {m['front_rate']:.2f} ped-s; a pedestrian (robot-free crowd): "
        f"{m['human_intimate_rate']:.2f}, {m['human_personal_rate']:.2f}, {m['human_front_rate']:.2f}"
    )
    print(
        f"robot: waiting {m['waiting_frac']*100:.0f}%, path ratio {m['path_ratio']:.2f}, "
        f"turning {m['robot_turn_rad']:.1f} rad, jerk rms {m['robot_jerk_rms']:.2f} m/s^3; "
        f"CBF active {m['cbf_active_frac']*100:.0f}%"
        + (f", slack on {m['slack_frac']*100:.1f}%" if m["slack_frac"] else "")
    )
    if "upper_clearance_min" in m:
        print(
            f"upper-body clearance to the pedestrian discs: min {m['upper_clearance_min']:.2f} m, "
            f"p05 {m['upper_clearance_p05']:.2f} m"
            + (
                f"; torso yaw p95 {m['torso_yaw_p95']:.2f} rad, engaged {m['torso_yaw_frac']*100:.0f}%"
                if "torso_yaw_p95" in m
                else ""
            )
        )
    extra = []
    if "upright_min" in m:
        extra.append(
            f"pelvis height min {m['pelvis_z_min']:.2f}, upright min {m['upright_min']:.2f}"
        )
    extra.append(f"QP non-converged {m['qp_nonconverged']}")
    if m.get("mppi_errors") is not None:
        extra.append(f"MPPI errors {m['mppi_errors']}")
    if m.get("theta_travel_deg") is not None:
        extra.append(
            f"heading: theta max {m['theta_cmd_max_deg']:.0f} deg, travel {m['theta_travel_deg']:.0f} deg"
        )
    if m["stopped_at_s"] is not None:
        extra.append(f"SIMULATION STOPPED on controller error at t={m['stopped_at_s']:.1f}s")
    print("; ".join(extra))


def main(
    duration=DEFAULT_DURATION,
    seed=0,
    gif=False,
    view=False,
    robust_bound=None,
    n_ped=N_PED,
    relax=None,
    planner=DEFAULT_PLANNER,
    proxy=False,
    robot=DEFAULT_ROBOT,
    torso=False,
    footprint="disc",
    crowd_reacts=CROWD_REACTS,
):
    if robust_bound is None:
        robust_bound = DEFAULT_ROBUST_BOUND
    if relax is None:
        relax = DEFAULT_RELAX
    if TEST_MODE:
        duration = min(duration, 5 * CONTROL_DT)  # the smoke run keeps the JIT short
    r = run(
        duration,
        seed,
        robust_bound,
        n_ped,
        relax,
        planner,
        DEFAULT_WEIGHTS,
        proxy,
        not TEST_MODE,
        None,
        robot,
        torso,
        footprint,
        crowd_reacts,
    )
    m = r["metrics"]
    print_report(m)
    if TEST_MODE:
        return float(m["h_min"])
    _TAG[0] = (
        f"{planner}_{'relaxed_' if relax else ''}{'robust' if robust_bound > 0 else 'vanilla'}"
        + ("_proxy" if proxy else "")
        + (f"_{robot}" if not proxy and robot != "unitree" else "")
        + ("_torso" if torso else "")
        + ("_ellipse" if footprint == "ellipse" else "")
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plant, com, agents, n_live = r["plant"], r["com"], r["agents"], r["n_live"]
    t = np.arange(len(com)) * plant.dt
    n = min(len(com), n_live + int(1.0 / plant.dt))
    d = np.linalg.norm(com[:, None, :] - agents[:, :, :2], axis=2)
    H = d / R_PED - 1.0
    plans = r["mppi_plans"][:n] if r["mppi_plans"] is not None else None
    _plot(com[:n], t[:n], H[:n], agents[:n], r["v_nom"][:n], r["v_safe"][:n], plans)
    if proxy:
        return float(m["h_min"])
    from cbfkit.systems.mujoco.viewer_utils import render_gif, replay_in_viewer

    markers = _make_markers(agents)
    S = r["states"]
    if gif:
        path = os.path.join(RESULTS_DIR, f"g1_scramble_{_TAG[0]}.gif")
        render_gif(
            plant,
            S[:n],
            path,
            track_body=r["pelvis_body"],
            markers=markers,
            distance=7.0,
            elevation=-35.0,
        )
    if view:
        replay_in_viewer(plant, S[:n], markers=markers)
    return float(m["h_min"])


def _plot(com, t, H, agents, v_nom, v_safe, plans=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 4)
    snaps = [int(f * (len(t) - 1)) for f in (0.15, 0.4, 0.65, 0.9)]
    for j, k in enumerate(snaps):
        ax = fig.add_subplot(gs[0, j])
        ax.add_patch(
            plt.Rectangle((-HALF, -HALF), 2 * HALF, 2 * HALF, fill=False, ls="--", color="gray")
        )
        for i in range(agents.shape[1]):
            ax.add_patch(plt.Circle(agents[k, i, :2], PED_RADIUS, color="tab:blue", alpha=0.6))
            ax.arrow(
                *agents[k, i, :2],
                *(agents[k, i, 2:] * 0.6),
                head_width=0.15,
                color="tab:blue",
                alpha=0.6,
            )
        ax.plot(com[: k + 1, 0], com[: k + 1, 1], "-", color="tab:green", lw=2)
        if plans is not None:
            ax.plot(
                plans[k, 0, :], plans[k, 1, :], "-", color="tab:orange", lw=1.5, label="MPPI plan"
            )
        ax.add_patch(plt.Circle(com[k], ROBOT_RADIUS, color="tab:green"))
        ax.add_patch(plt.Circle(com[k], R_PED, color="tab:green", fill=False, ls=":"))
        ax.plot(*np.asarray(GOAL), "g*", ms=12)
        ax.set_aspect("equal")
        ax.set_xlim(-HALF - 3, HALF + 3)
        ax.set_ylim(-HALF - 3, HALF + 3)
        ax.set_title(f"t = {t[k]:.1f} s")
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(t, H.min(1), lw=1.5, label="min_i h_i (closest pedestrian)")
    ax.axhline(0, color="k", ls=":")
    ax.set_ylim(-0.3, 3)
    ax.set_xlabel("t [s]")
    ax.set_title("closest-pedestrian barrier (>= 0 safe)")
    ax.legend()
    ax = fig.add_subplot(gs[1, 2:])
    ax.plot(t, np.linalg.norm(v_nom, axis=1), "--", label="|v_nom| (planner)")
    ax.plot(t, np.linalg.norm(v_safe, axis=1), label="|v_safe| (after CBF)")
    v_com = np.gradient(com, t[1] - t[0], axis=0)
    ax.plot(t, np.linalg.norm(v_com, axis=1), lw=0.6, alpha=0.6, label="|v_com| (measured)")
    ax.set_xlabel("t [s]")
    ax.set_title("CoM speed: planner vs certified vs measured")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"g1_scramble_{_TAG[0]}.png")
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


def _make_markers(agents):
    import mujoco

    from cbfkit.systems.mujoco.viewer_utils import add_marker

    sph, cap, cyl = (
        mujoco.mjtGeom.mjGEOM_SPHERE,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
    )
    last = len(agents) - 1
    corners = [np.array(c, float) * HALF for c in ((1, 1), (1, -1), (-1, 1), (-1, -1))]

    def markers(scn, k, t):
        for p in agents[min(k, last), :, :2]:
            add_marker(scn, cap, [PED_RADIUS * 0.6, 0.55, 0], [*p, 0.85], [0.2, 0.4, 0.9, 1.0])
            add_marker(scn, sph, [PED_RADIUS * 0.5, 0, 0], [*p, 1.55], [0.2, 0.4, 0.9, 1.0])
        for c in corners:  # kerb markers
            add_marker(scn, cyl, [0.12, 0.3, 0], [*c, 0.3], [0.6, 0.6, 0.6, 1.0])
        add_marker(
            scn, sph, [GOAL_RADIUS, 0, 0], [*np.asarray(GOAL), GOAL_RADIUS], [0.1, 0.8, 0.2, 0.6]
        )

    return markers


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--planner", choices=("goal", "mppi"), default=DEFAULT_PLANNER)
    p.add_argument("--robot", choices=("unitree", "amo", "groot"), default=DEFAULT_ROBOT)
    p.add_argument(
        "--torso",
        action="store_true",
        help="reactive shoulder-turn/lean while passing (needs --robot amo)",
    )
    p.add_argument("--proxy", action="store_true", help="2-D lagged proxy instead of the G1")
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    p.add_argument("--robust", type=float, default=None, help="robust bound (m/s); 0 = vanilla")
    p.add_argument("--pedestrians", type=int, default=N_PED)
    p.add_argument(
        "--relax",
        dest="relax",
        action="store_true",
        default=None,
        help="soft barrier constraints (slack, penalised) -- the default here",
    )
    p.add_argument("--hard", dest="relax", action="store_false", help="hard barrier constraints")
    p.add_argument(
        "--reactive-crowd",
        dest="crowd_reacts",
        action="store_true",
        default=CROWD_REACTS,
        help="pedestrians also yield to the robot (default: they ignore it)",
    )
    p.add_argument(
        "--footprint",
        choices=("disc", "ellipse"),
        default="disc",
        help="robot keep-out: 0.35 m disc or the rotating measured ellipse (heading-augmented)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif", action="store_true")
    p.add_argument("--view", action="store_true")
    a = p.parse_args()
    if a.view:
        from cbfkit.systems.mujoco.viewer_utils import relaunch_under_mjpython_if_needed

        relaunch_under_mjpython_if_needed()
    main(
        a.duration,
        a.seed,
        a.gif,
        a.view,
        a.robust,
        a.pedestrians,
        a.relax,
        a.planner,
        a.proxy,
        a.robot,
        torso=a.torso,
        footprint=a.footprint,
        crowd_reacts=a.crowd_reacts,
    )
