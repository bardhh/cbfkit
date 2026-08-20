# Scramble Anisotropic Footprint + Heading MPPI (Phase C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Run the rotating-ellipse footprint + heading-augmented MPPI in the Shibuya
scramble, where lookahead actually pays: the planner aims a shoulder-turn at a gap in
the crowd *before* it opens, instead of the disc robot having to decline the gap.

**Architecture:** `g1_scramble.py --footprint disc|ellipse`. Ellipse path:
`embedded_heading_double_integrator` + `com_agent_ellipse_hocbfs` (40 pedestrians, soft
constraints as with disc) + `safe_locomotion_controller_hdi(agents=crowd,
local_planner=...)`; the MPPI cost is the *social* cost re-hosted on the heading layout —
new `heading_social_trajectory_cost` in `reduced_order.py`: social terms (proxemics, ttc,
goal/progress, legibility) on the sliced DI sub-state with the circular collision hinge
zeroed, replaced by the rotating-ellipse clearance preference (saturated absolute-h
quadratic, as tuned in the corridor), plus speed-gated align + spin.

**Supported robots:** `--proxy` (tuning) and `--robot amo` (absolute target-yaw command;
the GR00T adapter takes a yaw *rate* — follow-up shim if wanted). Clear error otherwise.

**Spec:** continuation of `docs/superpowers/plans/2026-08-20-corridor-mppi-lookahead.md`.

## Global constraints

- `.venv/bin/python` / `.venv/bin/pytest`; black only touched files; measured tables in
  docstrings + `G1_WALK_LOG.md`.
- Person-centred metrics (intimate/front/deviation, human norm) stay unchanged so the
  disc rows remain comparable; footprint-dependent fixes: slack slice `sol[:, n_u:]`,
  ellipse-aware `h_min` on `theta_cmd`, add theta travel/max metrics.

### Task 1: `heading_social_trajectory_cost` (`reduced_order.py`) + unit test

- Factory `(n_agents, goal, weights, dt, axes, ped_radius, *, robot_radius, pass_side,
  ellipse_weights, )`; slices `[p v | peds]` out of `[p v th om | peds]`, calls
  `social_cost_terms` with `dataclasses.replace(weights, collision=0)`, adds ellipse
  viol + align(gated) + spin from `EllipseCostWeights`.
- Test: near a pedestrian dead ahead, a sideways-profile trajectory costs less than a
  facing-forward one; with th = om = 0 the heading cost equals the collision-free social
  cost plus the ellipse terms computed on the same geometry.

### Task 2: scramble wiring

- `--footprint` flag; build(): ellipse branch (dyn/barriers/limits `[A, A, ALPHA_MAX]`,
  hdi wrapper), `build_social_mppi(footprint=...)` (heading cost, `control_dim=3,
  state_head=6`), robot guard.
- run(): slack slice by n_u; ellipse h_min + `theta_cmd_max_deg`/`theta_travel_deg` from
  `sub_data_theta_cmd`; print_report extras; GIF/plot tag gains `_ellipse`.
- TEST_MODE smoke: `(g1_scramble.py, --footprint ellipse --planner mppi --proxy)` in
  `test_examples.py`.

### Task 3: proxy measurement (seeds 0/1)

- disc+mppi vs ellipse+mppi (and ellipse+goal as the myopic baseline): crossing time,
  theta stats (does it actually rotate in the crowd?), intimate/front rates, h_min.

### Task 4: AMO runs (seeds 0/1) + docs + commit

- `--robot amo --planner mppi --footprint ellipse`, GIF; compare against the committed
  AMO disc rows (82/80 s, intimate 1.2/0.8, upper 0.11/0.16).
- Docstring table rows, `G1_WALK_LOG.md`, `CLAUDE.md`, memory; commit.
