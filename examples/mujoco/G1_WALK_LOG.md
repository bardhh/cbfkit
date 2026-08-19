# G1 walk — milestone 3 lab log (2026-08-18)

Goal: G1 tracks a planar velocity command (vx, vy) via `SamplingMpc` through `execute(plant=...)`,
without falling. Success = upright ≥ 0.9 throughout, mean pelvis speed ≥ ~60 % of the command,
and net displacement in the commanded direction; then a lateral/diagonal command to prove steerability.
Harness: `examples/mujoco/g1_walk_trial.py` (one JSON line per trial). Compile ≈ 100 s per config on CPU.

| # | config (deltas from harness defaults) | mean v (last 2 s) | dist (x, y) | min upright | fell | note |
|---|---|---|---|---|---|---|
| t01 | defaults (128 samples, no DR, zero spline 4 knots, H=0.6, vx=0.5) | (-0.04, -0.09); whole run (0.22, -0.06) | (0.87, -0.24) | 0.57 | **yes** | lurches 0.87 m forward in ~2 s, then falls; yaw-rate rms 1.06. 49 s wall (compile cache warm) |
| t02 | linear spline, 6 knots, H=0.8 | (0.07, 0.02) | (0.61, 0.40) | 0.77 | **yes** | collapses (min height 0.34) rather than tips; yaw rms 1.42 |
| t03 | w_vel 20, w_posture 0.02, noise 0.4 | (-0.03, -0.05) | (-0.01, -0.21) | 0.84 | **yes** | doesn't even move; collapses (0.31). Less posture prior + more noise = worse |
| t04 | + w_balance 5, w_yaw 5, w_height 10 | (-0.01, -0.06) | (0.15, -0.43) | 0.55 | **yes** | collapses at 0.86 s |
| t05 | t04 @ vx 0.3 | (0.36, 0.20) | (0.89, 0.13) | 0.73 | **yes** | collapses at 0.6 s, STANDS BACK UP at 2 s, moves 0.4-0.7 m/s for 1 s, collapses again |
| t06 | t04 + noise 0.15, 256 samples | (0.01, -0.01) | (0.76, 0.11) | 0.65 | **yes** | stands 1 s, accelerates to 0.74 m/s, collapses at 1.5 s |

**Diagnosis after wave 2:** every run over-accelerates past the command (0.6-0.9 m/s) then collapses onto the knees with torso upright (h≈0.35, up≈1.0) — a lunge, not a gait. Cost makes "collapsed at speed" (~3/step) about as cheap as "standing still" (~2.5/step). Wave 3: hinge on height (w_fall) + lower w_velocity + damped arm noise; horizon 1.0 vs 0.6; sim=planner diagnostic.
| t07 | w_fall 100 (h_min 0.75), w_vel 5, w_height 10, w_balance 5, w_yaw 5, arm noise ×0.3, H=1.0, 5 knots | (0.21, 0.02) | (0.50, 0.12) | 0.30 | **yes** | no kneeling now — TIPS OVER instead |
| t08 | t07 @ H=0.6, 4 knots | (0.17, 0.01) | (0.78, -0.37) | 0.44 | **yes** | same, tips |
| t09 | t07 with sim = planner model (no mismatch) | (-0.10, -0.10) | (-0.25, -0.09) | 0.20 | **yes** | falls too → model mismatch is NOT the primary cause |

**After wave 3:** the cost shaping moved the failure from kneeling to tipping; the search itself is the problem (128 samples, 1 iteration/step, long horizon). DIAL-MPC (the published MJX sampling-MPC legged result) uses H≈0.32 s, 4 nodes, 2 iterations/step, 2048 samples. Wave 4: MPPI-CMA; DIAL-shaped (short horizon, 2 iterations, 256 samples); slow+balanced.
| t10 | t08 cost + MPPI-CMA (min noise 0.2) | (0.24, -0.10) | (1.44, 0.27) | 0.38 | **yes** | fast lunge to 0.75 m/s at 1.5 s, falls, partial recovery |
| t11 | **DIAL-shaped**: H=0.32, 4 knots, 2 iterations/step, 256 samples (t08 cost) | (0.48, -0.20) | **(1.78, 0.22)** | 0.73 | stumble (h 0.46 @3.5 s) | **first run that doesn't fall**: h 0.66-0.94, alternating feet — but kicks to 0.9 m and lurches laterally. 82 s wall |
| t12 | t08 cost, vx 0.25, w_balance 20 | (0.34, -0.10) | (0.83, -0.70) | 0.64 | **yes** | slow start, then lurch/stumble |

**After wave 4:** short horizon + 2 iterations is the key (matches DIAL-MPC). Wave 5 on the t11 base: sustain over 8 s; foot-clearance hinge (z>0.15) + w_vel 10; 512 samples.
| t13 | t11 for 8 s | (0.39, -0.01) | (2.77, 0.34) | 0.37 | 3 stumbles, all recovered | sustained but scrappy; feet still to 0.9 m |
| t14 | t11 + w_feet 20 (z>0.15), w_vel 10 | (0.64, -0.10) | (1.75, -0.11) | 0.78 | 1 stumble @2.5 s (h 0.34), recovered | **first 2 s look like walking** (h 0.83-0.98, feet ≤0.57); over-speed spike 0.97 m/s → stumble → recovers to 0.87 m/s |
| t15 | t11 + 512 samples | (0.57, -0.24) | (2.12, -1.0) | 0.58 | **yes** | more samples is not the lever |

**After wave 5:** stumbles are preceded by over-speed spikes. Wave 6: w_vel 20, fall hinge 300 @0.78, foot cap 0.12 @50, w_qvel 0.02; iterations 3; a diagonal command to test steerability.
| t16 | w_vel 20, fall 300@0.78, feet cap 0.12@50, w_qvel 0.02 | (0.84, -0.35) | (3.17, -1.16) | 0.86 | 2 stumbles | vx spikes to 1.48 — the spikes are the pelvis toppling, i.e. a symptom of bad steps, not the cause |
| t17 | t16 + 3 iterations | (-0.02, 0.64) | (0.79, 1.23) | -0.77 | **flat on its back** | |
| t18 | t16 @ (0.3, 0.3) diagonal | (-0.10, -0.13) | (0.76, 0.27) | 0.12 | **yes** | |

**After wave 6:** cost tweaks on the free-form gait have plateaued. DIAL-MPC's published H1/Go2 walking uses a phase-based gait reward (periodic foot-height reference), not a free-form search. Wave 7 adds that prior (w_gait, freq, swing height, duty); rollout root now carries absolute time.
| t19 | gait prior w 20, 1.5 Hz, swing 0.08 (w_vel 10, fall 100, qvel 0.01) | (0.41, -0.04) | (1.11, -0.42) | 0.75 | crouch, stumble @3.5 s | feet now ≤0.4 (prior works); shuffles at h≈0.65 |
| t20 | gait w 50 | (0.77, 0.21) | (2.69, 0.23) | 0.86 | crouching 0.4-0.6 | moves, low and unsteady |
| t21 | gait w 30, 1.2 Hz, swing 0.06, duty 0.6 | (0.12, -0.06) | (1.21, -0.09) | 0.63 | collapses @1.5 s | |

**After wave 7:** gait prior fixes the kicking; failure moves to crouching/collapse. Wave 8: prior + stronger height/fall; 2 Hz cadence; friction DR ×4.
| t22 | gait 50 + w_height 30, fall 300@0.8 | (0.31, -0.14) | (1.44, 0.67) | -0.05 | **yes** | |
| t23 | gait 30 @2 Hz swing 0.06 + height 30, fall 300 | (0.60, -0.39) | (2.23, -0.12) | -0.02 | deep tip @1.5 s, recovers | moves at 0.5-0.9 m/s but leans to horizontal and back |
| t24 | t22 + friction DR ×4 (128 samples) | (0.52, 0.01) | (2.14, 0.33) | 0.07 | deep tips, recovers | 159 s wall |

**Checkpoint after 24 trials (2026-08-18):** forward locomotion at ~0.5 m/s is reproducible; a *clean, sustained, steerable* gait is not, at 128-256 samples / 2 iterations on CPU. Best runs: t14 (cleanest first 2 s), t11/t13 (sustained with stumbles), t20/t23 (fastest). Recommendation recorded in the session: move the search to GPU (2048 samples, DIAL-MPC regime) or accept the current gait for the CBF demo. Everything needed is in the harness; `--update cma`, `--iterations`, per-actuator noise, gait prior all available.

**End-to-end (g1_navigate.py, 8 s, CBF + t14/t20-class gait, 2026-08-18):** CBF active on 83 % of steps,
CoM path deflects left around the obstacle; but h(x) dips to -0.50 at t≈6 s (CoM 0.50 m from the obstacle
centre vs 0.70 m keep-out; physical clearance ≈0.15 m) — the gait's tracking error is far larger than any
small robust bound. Torso height min 0.33 (stumble); ends 2.5 m short of the goal. Qualitatively right,
quantitatively violated: a better gait (GPU-scale sampling or a trained policy) is what turns this into a demo.

## Milestone 3 closed by a pretrained policy (2026-08-18, Bardh's suggestion)

`unitree_rl_gym` ships a G1 walking checkpoint (`deploy/pre_train/g1/motion.pt`, LSTM 47→64→32→12,
12-DoF legs, upper body fixed; BSD-3). Integrated as `cbfkit.systems.mujoco.unitree_policy`: files fetched
from a pinned commit with SHA manifest, TorchScript read **without torch**, evaluated in JAX inside the scan,
PD torques at 500 Hz via `MujocoPlant(ctrl_map=...)`. MJX needed the collision set trimmed (no
cylinder–mesh pair; body meshes collide with the floor only).

| cmd (vx, vy) | mean v last 4 s | dist 6 s | pelvis z min | note |
|---|---|---|---|---|
| (0.5, 0) | (0.48, −0.02) | (2.68, −0.14) | 0.763 | clean, upright, sustained |
| (0.3, 0.3) | (0.32, 0.21) | (1.80, 1.27) | 0.762 | diagonal ok |
| (0, 0.4) | (0.04, 0.30) | (0.21, 1.80) | 0.766 | pure lateral ok (~75 % of cmd) |
| (−0.3, 0) | (−0.26, −0.01) | (−1.50, −0.05) | 0.775 | backward ok |

The MPC line (t01–t24) stays in the repo as the in-house alternative; the demo uses the policy.

## Heading follower + double-integrator reduced model (2026-08-18, late)

Bardh noticed the robot walked-sidestepped-walked instead of turning. Two causes: the CoM certificate is a
holonomic point (no heading), and the policy adapter fed the *world*-frame v_safe as a *body*-frame command
(right only while yaw ≈ 0 — a latent bug). Fixed both in `UnitreeG1WalkPolicy.as_controller`: world→body
rotation by pelvis yaw + heading follower `wz = 2·wrap(atan2(v) − yaw)`. Check: world cmd (0, 0.4) → yaw
reaches +90° in 2 s and the robot walks forward along +y at 0.38 m/s (was: strafing).

Added a command-side double integrator (`embedded_double_integrator`, `com_obstacle_hocbfs` via
`rectify_relative_degree(form="high-order")`, `safe_locomotion_controller_di`): the CBF filters an
acceleration; the velocity command is its integral. Its smooth commands cut the gait's *max* tracking error
from 0.49 to 0.17 m/s, so the robust bound can be set above the observed maximum (0.18) — the
self-consistent claim. Table of all runs is in `g1_navigate.py`'s docstring; DI + robust 0.18 is the default.

## Plaza crossing (2026-08-19)

`g1_plaza.py`: 3-waypoint route, 2 pillars, 3 constant-velocity pedestrians (kinematic, certificate-only),
DI HOCBFs incl. time-varying ones (`com_moving_obstacle_hocbfs`; `dh/dt` enters through the rectifier), a
stateless waypoint planner (`cbfkit.planners.waypoint_route`). Every obstacle is a real encounter (the
nominal legs pass inside each keep-out). All rows walk upright and complete the route (SI robust 0.25 did
not within 40 s — too conservative; SI uses the p95 bound 0.18). Full table in the example's docstring.

Found while measuring: the robust CBF-QP took the Frobenius norm of the *stacked* barrier Jacobians as the
margin for every constraint, so five barriers made the QP infeasible at δ = 0.25 and pushed the robot
backwards at 0.2, before it had moved. One barrier (g1_navigate) was unaffected. Fixed row-wise
(`robustness_terms.py`) with a regression test; `examples/unicycle/reach_goal/robust_cbf.py` (2 obstacles)
is less conservative than before as a result.

### Plaza, round 2: reactive pedestrians (2026-08-19, later)

Bardh: "the dynamic obstacles appear non-interacting." Two causes: open-loop pedestrians, and the quadratic
barrier's distance-growing gradient kept the robust robot ≥ 1.3 m from everything. Changes: (1) pedestrians are
social-force agents (`SocialForceCrowd`, reusing `systems/pedestrian/behaviors`) stepped in the safety wrapper;
the CBF sees them as tracked agents `[p_i, v_i]` in the augmented state with `ṗ_i = v_i` (`com_agent_hocbfs`) —
constant-velocity prediction, accelerations unmodelled (reported: up to 2.4 m/s²); (2) `shape="distance"` barrier
`h = |c − p|/r − 1` so robust margins are distance-independent; (3) encounters built to collide (head-on just past
W1, cut-across on leg 1, overtake on leg 3). Two more things surfaced: with the quadratic barrier + robust 0.31 the
QP is infeasible at 6 s (far pedestrians' margins), and jaxopt-OSQP stalls (10 000 it, NaN) on a *feasible*
2-variable QP when the robot is wedged between pillar 2 and the oncoming P1 with both margins active — the
in-repo PDIPM solves it in 16 iterations, so the example uses `get_solver("fast")`. A scenario fault also showed
up as a freezing-robot standoff (P1's intended line went through pillar 2 → both stopped facing each other for
30 s); fixed by moving P1's meeting point before pillar 2. Final: robust 0.31 (≥ max 0.298): h_min +0.59, closest
pedestrian 1.18 m, route 39.4 s, P1/P3 stop and step aside, the robot swings around the far side of pillar 2.

## Model distance: DI/SI reduced model vs the G1 + policy (2026-08-19)

`g1_model_distance.py`: 6 × 20 s open-loop command runs per class (DI: random |a| ≤ 1 accelerations with stops; SI:
piecewise-constant velocity jumps), no CBF. Results (`results/g1_model_distance.md`):

| | DI-class commands | SI-class commands |
|---|---|---|
| residual ‖v_com − v_cmd‖ mean | 0.130 m/s | 0.101 m/s |
| δ_0.05 / δ_0.01 (pooled quantile) | 0.283 / 0.361 | 0.268 / 0.479 |
| per-run max median / worst | 0.386 / 0.447 | 0.543 / 0.685 |
| ε(H) p95 at H = 0.02 / 0.5 / 1 / 2 s | 0.006 / 0.126 / 0.216 / 0.316 m | 0.005 / 0.109 / 0.169 / 0.246 m |
| identified v_com = k e^{−sL}/(τs+1) v_cmd | k 0.95, τ 0.20 s, L 0.12 s (RMS 0.066 vs 0.107 for v_com = v_cmd) | — |
| ν-gap(1/s, G_id/s) | 0.247 at 2.7 rad/s; \|G_id − 1\| > 0.3 above 1 rad/s | — |

Readings: (a) the in-scenario plaza bound 0.31 covers ~95 % of the general envelope but not its worst case
(0.45 under DI commands) — the residual grows with command turn rate (mean 0.11 → 0.19) and is worst at
0.2–0.35 m/s; (b) SI commands have a lower mean but a fat tail (jumps: worst 0.69) — the DI's smoothness buys
the tail, not the mean; (c) the reduced model is trustworthy for ≲ 0.25 s (ε p95 < 7 cm) and off by 0.2–0.3 m
at 1–2 s, which is the horizon a HOCBF with α = 1 implicitly reasons over; (d) the gait is a ~0.3 s lag
(τ + L); the DI assumption holds below ~1 rad/s. Natural next step: a first-order-lag reduced model
(v̇_com = (k v_cmd − v_com)/τ, relative degree 3) should shrink δ by ~40 % (RMS 0.107 → 0.066).

## Scramble crossing (2026-08-19, evening)

`g1_scramble.py`: Shibuya-style 12 × 12 m intersection, 40 social-force pedestrians (vectorised `SocialForceCrowd`,
now with `arrive_radius`) released 0–30 m behind the kerbs in six streams at 0.8–1.3 m/s, robot crosses the diagonal.
Findings: (1) hard barrier constraints are infeasible within ~10 s whatever the bound (LP-verified: two pedestrians
closing at ~1 m/s from two sides, |a| ≤ 1 can't satisfy both under CV prediction) — a crush is a true infeasibility;
(2) soft constraints (`relaxable_cbf`, penalty 1e3) get the robot across in 49 s with h_min +0.08, slack on 10 % of
steps, 16 pedestrians within 1.5 m, no contact — a safety *filter* outcome, not a certificate; (3) robust margins
are eaten by slack in a crush (robust 0.15 soft: more slack, h_min −0.05, 65 s). First attempt released everyone
at t = 0 and the square was empty before the 0.5 m/s robot reached it — release depth matters. Default: soft
vanilla.

## Social MPPI in the scramble (2026-08-19, night)

"Use our MPPI in the scramble, tuned so the behavior is not intrusive to humans" — planner swap, same CBF filter.
`controllers/mppi/social_costs.py` + `--planner mppi` in `g1_scramble.py`: `vanilla_mppi` over the compact
`[p | v | 40 pedestrians]` state (5 Hz, 5 s horizon, constant-velocity prediction = "assume nobody yields"),
cost = Kirby asymmetric-Gaussian proxemics (front of a walking pedestrian is expensive, behind is cheap),
Karamouzas TTC power law, **progress as a terminal cost** (waiting 1 s costs only the metre not walked — so
"let her pass" is a plan, not a failure), jerk/turn/back/speed/slow legibility terms, optional keep-left.
Wired through `safe_locomotion_controller_di(local_planner=mppi_local_planner(...))`; plan held between solves
(`lax.cond`), P-law fallback on MPPI NaN. Tuned on a 2-D proxy (0.2 s lag model from `g1_model_distance`),
4 configs × 5 seeds × 90 s ≈ 30 s/run; validated on the MJX G1 (~2 min/run).

Measured (intrusiveness rates = ped-s per 10 s inside the intersection; human norm for this crowd 3.5 intimate / 2.5 front):
proxy 5-seed means — goal: 5/5 crossings, 47 s, intimate 4.9, front 2.3, crowd-dev 1.33 m, CBF 43 %;
social MPPI: 5/5, 61 s, intimate 1.5, front 1.0, dev 0.35 m, CBF 19 %; ablation (social terms off): 4/5, 71 s,
intimate 0.9 — polite but timid; the social terms buy reliability at speed, not just politeness.
G1 seeds 0/1 — goal: 49/44 s, intimate 4.2/6.2, h_min +0.08/−0.23; MPPI: 65/56 s, intimate 1.7/2.5,
h_min +0.16/−0.11. The baseline is more intrusive than an average pedestrian; the MPPI is ~2.5× less.

Two solver defects found (both fixed + regression-tested with a captured QP fixture):
(1) a warm-started fast-PDIPM can exhaust max_iter at an MPPI replan boundary (a_nom jump → stale active set)
while a cold start solves in 16 — `get_solver("fast")` now cold-restarts when the warm solve fails or returns
non-finite; (2) on a degenerate-optimal QP (optimum on the control bound, 42 var/124 row relaxable scramble QP)
Mehrotra's late iterations *degrade*: residual 2.8e-6 at iter 15, NaN at 18 — the loop now tracks the best
iterate and rejects non-finite steps, so raising max_iter can no longer turn a good answer into NaN. The
scramble uses tol 1e-5 (slack penalty 1e3 sets the residual scale; freeze-on-converge then avoids the
blow-up region entirely).

## AMO whole-body policy port (2026-08-19, late night)

Bardh asked for "more agile policies that can move the torso to the side while moving forward" → ported UCSD's AMO
(RSS 2025, `OpenTeleVision/AMO`, Apache-2.0): 23-DoF G1, command = [vx, target_yaw, vy, Δheight, torso yaw, pitch,
roll]. `systems/mujoco/amo_policy.py`: weights read torch-free (TorchScript reader + restricted numpy unpickler for
the norm stats), JAX forwards parity-tested vs torch at 1e-4 (dev-only oracle; the TorchScript top-level graph has a
baked cuda zeros so the parity test drives the sub-modules); history conv strides (2,1) recovered numerically. Plant
= their deployment XML (already collision-trimmed to 8 foot spheres + pelvis) at 2 ms × 10 PD with play_amo's gains;
PGS→Newton for MJX. Two play_amo subtleties preserved: the 10-frame proprio history EXCLUDES the current frame while
the 25-frame extra history INCLUDES it, and dyaw is gated by the PREVIOUS step's stand flag.

`g1_amo_demo.py` (40 s, vx 0.4): torso yaw ±1.2 → waist ±1.05/1.17 rad, lean ±0.5 → ±0.34 (rest from hip lean),
duck to CoM 0.48 m, bow 0.8 → 0.56, upright min 0.97 — all *while walking*. Honest gap: forward speed realises
~0.3 of the commanded 0.4 m/s in MJX (trained IsaacGym, authors demo MuJoCo-CPU; gait transfers, speed calibration
partly doesn't). Next steps if wanted: measure the tracking bound δ for the robust CBF (g1_model_distance battery on
AMO), and use torso yaw/lean as CBF decision variables in the scramble (shoulder-turn ≈ 0.35 → 0.25 m swept radius).
