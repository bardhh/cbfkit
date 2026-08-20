# Corridor MPPI Lookahead (Footprint Phase C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Replace the hand-coded heading *suggestion* in `g1_corridor.py` with an MPPI
planner over the heading-augmented double integrator, so the sidestep rotation is
*planned* with real lookahead (one clean rotate → squeeze → derotate maneuver) instead
of dithering out of a myopic ramp.

**Architecture:** Same "suggestion proposes, certificate decides" split as the scramble:
MPPI (long horizon, soft costs) produces the nominal `[ax, ay, alpha]`; the hard
ellipse-HOCBF QP filters it. New pieces: (1) an ellipse-footprint trajectory cost over
the compact `[p v | th om | agents]` layout (disc variant over `[p v | agents]`),
(2) a `local_planner` hook on `safe_locomotion_controller_hdi` mirroring the DI wrapper,
(3) a generalized `mppi_local_planner` (control_dim / state_head parameters),
(4) `--planner suggest|mppi` in the corridor example with dithering metrics
(`theta_travel`), lane cost so MPPI cannot detour around the pedestrians.

**Tech stack:** JAX, `cbfkit.controllers.mppi.vanilla_mppi`
(`trajectory_cost(time, states (dim,H), controls (m,H), prev_robustness)` convention).

**Spec:** continuation of `docs/superpowers/plans/2026-08-20-anisotropic-footprint-cbf.md`
(Phase C: "MPPI over the heading-augmented model as the lookahead layer").

## Global constraints

- `.venv/bin/python` / `.venv/bin/pytest`; black only touched files.
- No torch/onnx in runtime or pytest process.
- Measured tables live in example docstrings + `examples/mujoco/G1_WALK_LOG.md`
  (`results/` is gitignored).
- Hard constraints in the corridor (certificate demo): h >= 0 on the proxy must survive.

### Task 1: cost + hooks in `reduced_order.py`

- `EllipseCostWeights` dataclass + `ellipse_trajectory_cost(n_agents, goal, axes,
  ped_radius, dt, *, heading=True, lane_halfwidth=None, v_max, weights)`:
  goal (terminal) + progress (stage) + clearance hinge on the rotating-ellipse h +
  lane + align (face travel direction) + spin (omega^2) + speed. `heading=False`
  drops th/om slots and the align/spin terms (disc baseline over the DI layout).
- `mppi_local_planner(..., control_dim=2, state_head=4)` — generalize dims; fallback
  pads the P-law with zeros for the alpha channel. Scramble call sites unchanged.
- `safe_locomotion_controller_hdi(..., local_planner=None)` — compact state
  `[com v th om agents]`, planner returns `(a_nom(3), v_plan, sub_updates)`; merged
  like the DI wrapper.

### Task 2: unit tests (`tests/test_mujoco/test_reduced_order.py`)

- Cost: sideways-through-gap trajectory cheaper than forward-through-gap (ellipse,
  gap 1.0); through-pedestrian dearer than around; lane term penalizes detours;
  disc variant runs on the 4+4N layout.
- hdi hook: local_planner's a_nom reaches the QP; sub_updates merged; `_`-keys persist.

### Task 3: corridor `--planner suggest|mppi`

- `build(planner=...)`: `mppi` drops the engage ramp (plain goal P-law nominal),
  builds `vanilla_mppi` on the compact heading model (`robot_state_dim=10`,
  `robot_control_dim=3`, horizon 30 x 0.2 s = 6 s) with the ellipse cost; disc gets
  the DI-layout variant through the existing hook. Lane half-width blocks the detour
  for both.
- `run()` adds `theta_travel_deg` (sum |dtheta_cmd|, the dithering metric) and
  `mppi_errors`.

### Task 4: proxy measurements + acceptance tests

- Measure: ellipse+mppi crosses with NO suggestion (h >= 0 hard), rotation discovered
  (theta_max), theta_travel ~ 2 x theta_max (one maneuver); disc+mppi refuses and does
  not detour; compare crossing time vs the suggestion variant.
- `tests/test_mujoco/test_corridor.py`: `test_mppi_discovers_rotation_without_suggestion`.
- Register `("examples/mujoco/g1_corridor.py", "--planner", "mppi")` in
  `test_examples.py`.

### Task 5: G1 run + docs + commit

- `--g1 --planner mppi --robust 0` (vanilla, the measured crossing config), GIF.
- Update corridor docstring table, `G1_WALK_LOG.md`, `CLAUDE.md`; commit; memory.
