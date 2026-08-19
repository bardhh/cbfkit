# Social MPPI in the G1 scramble — Implementation Plan

**Goal:** Replace the go-to-goal nominal in `examples/mujoco/g1_scramble.py` with cbfkit's
MPPI (`cbfkit.controllers.mppi`) running a *socially tuned* cost, keep the soft CBF-QP as
the downstream filter, and measure intrusiveness against the current controller.

**Architecture:** MPPI plans over the command-side double integrator in the compact
augmented layout `[p(2) | v(2) | (p_i, v_i) × N]` (`embedded_double_integrator(2,(0,1),N)`),
pedestrians predicted at constant velocity (the robot assumes nobody yields for it — the
polite assumption). It runs *inside* the DI locomotion wrapper (the plant state has no
pedestrians, the controller carry does) through a new `local_planner` hook of
`safe_locomotion_controller_di`, at the MPPI time step (5 Hz, horizon 5 s), and hands the
QP its first planned acceleration. Cost terms live in
`cbfkit/controllers/mppi/social_costs.py` and are generic 2-D.

**Why these terms (what "not weird to humans" means here):**
1. *Proxemics* — Kirby's asymmetric Gaussian personal space: wide in front of a walking
   pedestrian, narrow behind → crossing behind is cheap, cutting in front is expensive.
   Front sigma grows with pedestrian speed.
2. *Time-to-collision* — Karamouzas power law `k/ttc² e^{-ttc/τ0}` on relative CV motion:
   penalises closing fast, not being near; approaching slowly is fine.
3. *Progress as terminal cost* (plus a tiny stage term) — waiting 1–2 s costs ~0.5–1 m of
   terminal progress, small against an intrusion; so "wait for the gap" is a first-class
   plan, not a failure of the avoidance term.
4. *Smoothness / legibility* — jerk, heading change, no backing away from the goal,
   speed cap, speed penalty weighted by proximity (walk slowly when close).
5. *Pass-side convention* (optional, `pass_side="left"` for Japan) — head-on encounters
   keep left; zero weight disables.

**Evaluation (both controllers, same crowd, same seeds):** crossing time, waiting %,
path ratio, min distance, pedestrian-seconds in the intimate (< 0.45 m surface) and personal
(< 1.2 m) zones, **front intrusions** (robot within 1.5 m and within ±45° of a pedestrian's
heading), **pedestrian disruption** (path deviation vs the robot-free crowd rollout; speed
ratio near the robot), CBF intervention %, slack %, robot jerk.

**Tuning protocol:** tune on a 2-D proxy (DI robot with the identified first-order lag
τ = 0.2 s from `g1_model_distance`, same crowd, same CBF, no MuJoCo — seconds per run, many
seeds), validate on the G1 (minutes per run, few seeds). Report both honestly.

**Tech stack:** JAX, cbfkit MPPI planner, cbfkit CBF-QP (fast PDIPM), MJX G1.

## Tasks

### Task 1: `social_costs.py` — cost terms + trajectory-cost factory (TDD)
- Create `src/cbfkit/controllers/mppi/social_costs.py`:
  `SocialCostWeights` (dataclass), `asymmetric_gaussian`, `time_to_collision`,
  `social_trajectory_cost(n_agents, goal, weights, *, robot_radius, ped_radius, dt,
  v_max, pass_side)` → `TrajectoryCostCallable(time, states (dim,H), controls (m,H), prev)`,
  plus `social_cost_terms(...)` returning the per-term breakdown (for ablation/eval).
- Tests `tests/test_controllers/test_social_costs.py`: front > behind at equal distance;
  TTC analytic check; cost prefers passing behind a crossing pedestrian over in front;
  waiting beats cutting in front when the gap is short; pass_side sign.

### Task 2: MPPI knobs — `control_std` in `MppiParameters`/`setup_mppi` (default keeps 2.0)
- `mppi_source.setup_mppi(control_std=2.0)`, generator passes `mppi_args.get("control_std", 2.0)`.
- Test: std=0 → perturbation zero; default unchanged (determinism test still passes).

### Task 3: `local_planner` hook in `safe_locomotion_controller_di` + MPPI adapter
- `reduced_order.safe_locomotion_controller_di(..., local_planner=None)`; signature
  `local_planner(t, xa_compact, v_nom, key, sub) -> (a_nom, v_plan, sub)` where
  `xa_compact = [com, v, agents]`; logs `v_nom` = `v_plan`.
- `reduced_order.mppi_local_planner(mppi, n_agents, horizon, replan_every)` — carries
  `_mppi_u_traj`, `_mppi_a`, logs `mppi_x_traj (dim,H)`; `lax.cond` on `step % replan_every`.
- Tests: hook overrides a_nom and logs; replan holds between calls; MPPI adapter runs on a
  2-agent toy and returns finite plan.

### Task 4: scramble example — `--planner goal|mppi`, social metrics, proxy mode
- `g1_scramble.py`: `build(..., planner="goal"|"mppi", weights=...)`; `main(..., planner=...)`;
  `social_metrics(com, agents, crowd, dt, ...)`; `--proxy` runs the 2-D lagged proxy
  (`lagged_velocity_plant(tau)`) instead of MJX; results tagged by planner.
- Smoke test in `tests/test_mujoco/test_examples.py` for `--planner mppi` (TEST_MODE).

### Task 5: tune on proxy (seeds 0–4), validate on G1 (seeds 0,1), write the table
- `examples/mujoco/results/g1_scramble_social_mppi.md`, docstring table, G1_WALK_LOG.md,
  CLAUDE.md bullet, README line; slow acceptance test `test_scramble_mppi_crosses_politely`.
