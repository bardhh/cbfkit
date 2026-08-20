# AMO in the MPPI scramble — Implementation Plan

**Goal:** Run the social-MPPI scramble on AMO's whole-body policy and use the torso as an
extra degree of flexibility: shoulder-turn toward the pedestrian being passed (slims the
profile toward them: depth ~0.16 m presented instead of shoulder half-width ~0.24 m) and a
slight lean away. Measure the *true* upper-body clearance (forward kinematics on logged
qpos: shoulder/elbow/hand bodies vs pedestrian discs), not just the CoM disc.

**Architecture (division of labor):**
- MPPI (unchanged, 2-D) plans the path with the social cost.
- Soft CBF-QP (unchanged) filters the acceleration on the CoM DI. Certificate untouched:
  the keep-out disc stays 0.65 m — the torso layer only changes how the body is carried
  inside the space the planner/CBF already allow. Honest framing: clearance gain is an
  *outcome*, measured post-hoc.
- AMO replaces the Unitree walk policy as the tracking layer (native heading following).
- A reactive torso layer computes (height, yaw, pitch, roll) from the tracked agents in
  the controller carry: when the closest pedestrian is within R_TURN while the robot
  moves, torso-yaw toward their bearing (clipped ±1.2, first-order-smoothed) and roll
  0.25 rad away; upright otherwise.

## Tasks
1. **amo_policy**: `as_controller(torso_command=...)` accepts a 3-arg callable
   `(t, x, sub) -> (4,)` (state/carry-aware behaviors); smoothing helper carry key
   `_amo_torso`. Test: 3-arg hook sees `sub` and its output lands in `amo_cmd`.
2. **g1_scramble**: `--robot unitree|amo` (build(): AMO plant/x0/controller path;
   V_MAX unchanged 0.5 = AMO ID limit); `--torso` flag (amo only) enabling
   `shoulder_turn_torso()` (factory in the example, reads `sub["_agents"]`).
   Smoke tests: `--robot amo --planner mppi` and `--robot amo --torso` in TEST_MODE.
3. **Clearance metric**: `upper_body_clearance(plant, states, agents)` — mujoco (CPU)
   forward on logged qpos every step, min over {left/right shoulder-roll, elbow,
   wrist/hand} body xpos XY vs pedestrian centres, minus PED_RADIUS; report min +
   per-close-pass stats next to the existing social metrics.
4. **Measure** (MJX, seed 0/1, mppi planner): amo vs amo+torso (and unitree reference):
   crossing time, h_min, social rates, upper-body clearance min/p05, torso usage stats.
   Docstring table + walk log + CLAUDE.md; full suite; commit.
