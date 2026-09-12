# G1 showcase visuals — plan (2026-09-12)

Goal: replace the four Unitree G1 README clips (navigate, plaza, corridor sidestep, scramble)
with renders that show *what the CBF does*, look polished, and come with an unfiltered
comparison. Approved scope (Bardh, 2026-09-11): CBF overlays, scene and camera, HUD strip,
side-by-side unfiltered vs CBF, general polish. Existing clips stay in the README until the new
ones are reviewed.

Compute: home box `bardhh@192.168.0.154`, env `~/code/cbfkit-viz/.venv` (Python 3.12,
mujoco 3.11.0, **jax==0.6.2 pinned** — 0.11 silently breaks the policy, `MUJOCO_GL=egl`,
ffmpeg via `imageio_ffmpeg`). Rendering costs ~2 ms/frame at 1080p; simulation is the cost
(navigate ~2 min, scramble 40 peds + MPPI ~10–30 min). Sync code by `rsync` of the working
tree (branch is unpushed), never edit on the box.

## Architecture

```
examples/mujoco/g1_showcase.py  simulate <example> [--unfiltered] -> results/showcase/<example>[_unfiltered].npz
                                render   <example> [--side-by-side] -> media/videos/g1_<example>.mp4 + media/showcase/g1_<example>.gif
src/cbfkit/systems/mujoco/showcase.py   reusable: render-only model (MjSpec), oriented markers,
                                        overlay primitives, smooth camera, HUD + panel compositor, writers
```

Simulate once, render many times. The npz holds: `states (T, state_dim)`, `dt`, `plant_kind`
(`unitree12|amo23|groot29`), `com (T,2)`, every `res.controller_data` array (`v_nom`, `v_safe`,
`a_nom`, `a_safe`, `agents (T,N,4)`, `bfs`, `violated`, `solver_status`, `sol`, `mppi_x_traj`,
`mppi_error`, `theta_cmd`, `error`, `complete`), planner `x_traj` (plaza), `n_live`, the
recomputed `h (T, n_barriers)` (mirror the geometry: `g1_plaza.barrier_values`,
`g1_corridor` rotating-ellipse h, `g1_scramble` crowd h — `sub_data_bfs` is psi, not h), and
scenario metadata (obstacles, radii, goal, waypoints, footprint axes, ped radius, seed).

Data-flow facts (verified by the explorer, report in the session scratchpad
`g1_viz_dataflow.md`): the markers callback gets `(scn, k, t)` only — no MjData, no camera —
so body-derived overlays are precomputed per step by forward kinematics
(`d.qpos[:] = S[k,:nq]; mj_kinematics`) and the render loop is custom (not `render_gif`).
`add_marker` hardcodes identity rotation; oriented geoms call `mujoco.mjv_initGeom` with a
rotation matrix directly. `plant.mj_model` is mutable but new lights/textures need `MjSpec`
on the plant's source XML (12-DoF: `unitree_rl_gym_dir()/resources/robots/g1_description/scene.xml`;
AMO: `amo_dir()/g1.xml`; GR00T: the patched `g1_29dof_floor.xml`) — nq/nv must stay identical.
`ProxyPlant` has no MjModel; all renders use the MJX robot paths.

## Visual spec

**Scene** (`showcase.render_model(plant_kind)`): from the plant's XML via `MjSpec`: light
gray checker floor (rgb 0.52/0.46, subtle marks, texrepeat 8, reflectance 0.08), soft
sky gradient (0.70 0.78 0.90 → 0.95 0.97 1.0), haze 0.85/0.87/0.90, headlight 0.25/ambient
0.15, key directional light `pos 3 -4 6 dir -0.4 0.55 -0.75` castshadow, fill light
`-4 3 4`, `shadowsize 8192`, `offsamples 8`, `offwidth 1920 offheight 1080`. Reference:
`src/cbfkit/systems/mujoco/models/g1/scene_showcase.xml` (already tuned on the box; the
"studio" variant). Replace, don't stack, the source scene's lights/skybox/floor material.

**Camera**: tracking with exponential lag on the look-at (τ≈0.4 s) so the gait jitter is
filtered, distance/elevation per example (navigate 3.6/−18, plaza 5.0/−24, corridor 3.8/−20,
scramble 6.5/−32), azimuth drifting slowly (+6°/10 s) and biased so the robot's heading and
the goal are in frame (azimuth = heading-relative, e.g. behind-left of the walking direction).
Never cut. Camera is written in the custom loop (`mjCAMERA_FREE` with computed lookat, or
`mjCAMERA_TRACKING` with per-frame azimuth). 1280×720 render, 30 fps output for MP4 (sim dt
0.02 → render every step is 50 fps; render every step and resample, or every 2nd step at
25 fps — pick 25 fps for MP4 = real time, and the README GIF at 2× speed 20 fps = 10 frames
per sim second, as the current clips).

**Overlay primitives** (`showcase.py`, all thin wrappers over `mjv_initGeom` with `mat`):
`disc(pos, r, rgba)` (flat cylinder 4 mm on the floor), `ring(pos, r, rgba, width)` (thin
torus is not available — use two discs or a `mjGEOM_LINE` polygon of 48 segments via
`mjv_connector`), `arrow(from, to, rgba, width)` (`mjGEOM_ARROW` via `mjv_connector`),
`ellipse(pos, a, b, theta, rgba)` (flattened `mjGEOM_ELLIPSOID`, rotated about z),
`path(points, rgba, width)` (`mjGEOM_CAPSULE` connectors between consecutive points),
`trail(points, rgba)` (fading capsule path, alpha ramp), `pedestrian(p, v, r, rgba)`
(capsule body + short heading arrow + floor disc), `beacon(goal, r)` (translucent sphere +
vertical thin cylinder). Colour by barrier value: `h_rgba(h)` maps h ≥ 0.5 → green
(0.20 0.75 0.35), h → 0 → amber, h < 0 → red; alpha 0.35 for floor discs.

**Per example**
- navigate: red obstacle cylinder (as now) + keep-out ring coloured by h; goal beacon;
  arrows at the pelvis: `v_nom` gray, `v_safe` green (scaled 0.6 s); CoM trail (last 6 s,
  fading); render only the live part (`n_live + 1 s`), not the frozen tail after `complete`.
- plaza: 2 pillars + rings; 3 pedestrians as `pedestrian()` with per-agent ring coloured by
  their h column; waypoint route as a dashed floor path with the active waypoint (planner
  `x_traj`) as the beacon; arrows; trail.
- corridor (`--g1 --planner mppi`, ellipse): static pedestrians; the commanded footprint
  ellipse (`theta_cmd`, semi-axes `G1_FOOTPRINT + PED_RADIUS`) coloured by h, plus a thin
  ghost ellipse at the realised pelvis yaw (`qpos[3:7]`); MPPI plan as a path of the planned
  CoM (rows 0:2 of `mppi_x_traj`, drawn when it changes, faded between replans); arrows.
- scramble (`--robot unitree --planner mppi`, disc, relax on): 40 pedestrians with heading
  arrows; rings only for the 6 nearest (declutter) coloured by h; MPPI plan path; arrows;
  the HUD shows slack use (`sol[:, n_u:]`).

**HUD strip** (matplotlib → PIL, 1280×160 under the 3D view, dark translucent band):
left: `t = 12.3 s`, mode label, `h_min` numeric coloured by `h_rgba`; centre: a 6 s
scrolling plot of `h_min(t)` with the zero line and a playhead; right: intervention gauge
`|v_safe − v_nom|` as a bar and a small "CBF ACTIVE" pill when > 1e-3, `MPPI fallback` and
`QP not converged` flags when they fire, slack magnitude for the scramble. Font: DejaVu Sans,
sizes ≥ 22 px at 1280 wide so it survives the README downscale.

**Side-by-side** (`render --side-by-side`): unfiltered (left, label "Walking policy only")
and CBF (right, label "+ CBFKit safety filter") panels 960×540 each with the same camera
schedule (drive the camera from the CBF run's trajectory for both so they stay comparable),
a 1920×180 HUD showing both h_min traces, and the `h_rgba` disc turning red when the
unfiltered robot intrudes. Unfiltered = the same wrappers with an **empty certificate
collection** (the QP then returns `u_nom`; identical logging keys) — verify `sub_data_v_safe
== sub_data_v_nom` on that run. Obstacles/pedestrians are visual-only, so the unfiltered
robot walks through them; that is the intended contrast (the disc goes red).

**Outputs**: `media/videos/g1_<ex>.mp4` and `g1_<ex>_side_by_side.mp4` (1080p h264 crf 18,
25 fps, real time) for the project page; `media/showcase/g1_<ex>.gif` (single CBF panel +
HUD, 2× speed, 20 fps, 480 px wide, palette 64 colours, ≤ 6 MB each) as candidates — the
README keeps the current GIFs until Bardh approves the new ones. `scripts/render_showcase.py`
gets a `g1_showcase_new` job that shells out to `g1_showcase.py render`.

## Tasks

1. **Library** `src/cbfkit/systems/mujoco/showcase.py` — render model via MjSpec per plant
   kind; primitives; `h_rgba`; camera schedule; `FrameWriter` (mp4 via imageio_ffmpeg, gif via
   ffmpeg palette); HUD renderer; panel compositor with labels. Unit tests
   (`tests/test_mujoco/test_showcase.py`, skipped without mujoco): primitives add the
   expected geoms with correct rotation; `h_rgba` endpoints; HUD returns the right size;
   writer round-trips 5 frames.
2. **Simulate driver** `examples/mujoco/g1_showcase.py simulate` — reuse each example's
   builder (`g1_navigate._build…`, `g1_plaza` setup returning `plant, x0, pelvis_body,
   planner, nominal, controller`, `g1_corridor.run`, `g1_scramble.run`) with an
   `unfiltered` switch (empty barriers), run `sim.execute`, recompute h, write the npz.
   Smoke: `CBFKIT_TEST_MODE=1` 5-step runs on the Mac; full runs on the box.
3. **Render driver** `g1_showcase.py render` — per-example overlay functions, camera, HUD,
   side-by-side, outputs. Reviewed visually on stills first (frame 0, 25 %, 50 %, 75 %),
   then the full clips.
4. **Runs on the box** — 8 simulations (4 examples × filtered/unfiltered), renders, fetch.
5. **Wire up** — `scripts/render_showcase.py` job, README alt text if adopted, plan log
   below, memory note for the box workflow.

Review gates: code review after tasks 1–3 (separate lane); Bardh reviews the stills and
clips before anything replaces the README GIFs.

## Log

- 2026-09-11 23:40 env on the box built; jax 0.11.1 → robot never moves; pinned 0.6.2 →
  navigate reaches goal at 19.3 s (matches Mac). EGL ok. Render 0.9/1.5/2.3 ms per frame at
  360p/720p/1080p.
- 2026-09-12 00:10 scene variants rendered; "studio" chosen (floor 0.52/0.46, key 0.7,
  fill 0.2, headlight 0.25/0.15); shadows confirmed.
- 2026-09-12 00:55 Task 1 done (showcase.py, 33 tests; MjSpec handles every edit on all three
  plants, nq/nv preserved). Still on the box: all primitives + HUD + compositor render; arrows at
  pelvis height vanish inside the torso → floor arrows; pedestrian body 0.6×r. Task 2 in
  progress; Task 3 dispatched into examples/mujoco/g1_showcase_render.py.
- 2026-09-11 23:57 Task 2 done (g1_showcase.py simulate; unfiltered invariant is on the
  acceleration for DI/HDI wrappers, tol 1e-3; corridor config = --g1 --planner mppi ellipse at
  robust 0, the vanilla row that made the README clip). Launched the 8 simulations on the box
  (~/code/showcase_sim.sh, logs in ~/code/cbfkit-viz/logs/), pairs in parallel, monitor armed.
- 2026-09-12 00:15 all 8 simulations done (wall 80–283 s each). Filtered/unfiltered: navigate
  h_min +1.01 / −1.00 (unfiltered walks through the obstacle, goal in 8.5 s); plaza +0.59 /
  −0.77; corridor −0.06 (crossed 92.5 s) / +0.06 (MPPI alone crosses in 112 s — no collision,
  so that side-by-side reads "slower, uncertified"); scramble see log. npz in
  ~/code/cbfkit-viz/examples/mujoco/results/showcase/.
- 2026-09-12 00:35 Task 3 done (g1_showcase_render.py, 13 tests). First stills reviewed on the
  box: scene/HUD/shadows good. Fixes sent: floor moiré wedge = shadow-frustum edge → SHADOWCLIP
  10 (verified); pedestrians pole-thin → body 0.8·r; goal beacon used goal_radius (huge) → small
  sphere + floor rings; arrows too short; HUD y-axis autoscale → fixed per clip; per-example
  windows (corridor 20–56 s, scramble 6–42 s); camera a bit closer. Unfiltered-nominal sims
  (no planner, no certificate) requested for corridor/scramble since MPPI alone avoids.
- 2026-09-12 00:50 nominal-unfiltered sims (no planner, no certificate): corridor h_min −0.60
  (goal at 46.5 s, straight through the gap's pedestrians), scramble h_min −0.38 (35.7 s,
  through the crowd). These are the left panels for those two side-by-sides. Second stills:
  moiré gone, targets/arrows/pedestrians fixed; round 3 = slimmer scramble crowd + higher
  camera, navigate HUD in distance form, HUD ymax 1.0.
- 2026-09-12 01:05 full render pass on the box: MP4s 4.5–20 MB fine; README GIFs 7–25 MB
  (36–70 KB/frame: tracking camera + drift + scrolling HUD defeat frame diffing; gifsicle
  lossy only −40 %). Side-by-side defect: shared look-at loses the unfiltered robot once the
  trajectories diverge → per-panel cameras. Knobs requested for a size sweep (width, colours,
  no-HUD GIF, no drift, fps). Code review lane running. Code committed on
  feat/g1-showcase-visuals (b4402ef).
- 2026-09-12 01:30 GIF size sweep (scramble 8–36 s): 480/64/20+HUD 19 MB; 400/48/20 13 MB;
  HUD off / drift off ≈ no gain; 16 fps 11 MB; the *floor* is the entropy: plain floor 3.4 MB,
  flat + 1 m grid 5.1 MB (adopted as the default look; checker kept as an option). README GIF
  settings: 400 px, 48 colours, 16 fps (8 sim-frames/s, 3.2 per step, no phase lock), no HUD,
  no drift; MP4s keep HUD + drift. Code review: 3 MAJOR (intervention bar scale, camera path
  depends on gif flag, pills truncated) + minors, all routed to the render executor.
- 2026-09-12 02:05 final pass: README candidates media/showcase/g1_<ex>_showcase.gif
  (navigate 2.1, corridor 3.7, plaza 5.4, scramble 6.8 MB = 18 MB, vs 19 MB current);
  MP4s (single with HUD 3–11 MB; side-by-side 3–13 MB) and side-by-side GIFs (3–12 MB)
  kept OUT of git (repo tracks no .mp4) at ~/code/cbfkit-viz/examples/mujoco/results/showcase/final
  on the box and in the session scratchpad; review page opened for Bardh. 88 tests pass.
  Gotcha: `render --side-by-side` also writes the single-panel GIF, so run the README GIF
  render last. Awaiting Bardh's call on replacing the README GIFs and where MP4s live.
- 2026-09-12 02:30 Bardh: README GIFs approved in principle; scramble crowd must NOT yield to
  the robot (social forces only among pedestrians). SocialForceCrowd(react_to_robot=False),
  g1_scramble CROWD_REACTS = False (+ --reactive-crowd). Proxy 40 s: h_min 0.38 → 0.07.
  Re-simulating + re-rendering the scramble on the box (final2/).
- 2026-09-12 02:55 scramble re-run with the non-reactive crowd: filtered h_min −0.43 (relaxed
  constraints; a contact-level intrusion in the crush, 12 % intervention, goal at 67.4 s);
  unfiltered nominal h_min −0.99 (walks through people, 35.7 s). New candidate GIF 6.4 MB
  replaces the reactive-crowd one; MP4s in final2/ and the review page.
- 2026-09-12 03:30 non-reactive crowd is a crush at 40 peds on every proxy seed (h_min −0.36 to
  −0.99) and on most seeds at 20–32. Bardh chose "smaller crowd, best seed". The proxy does NOT
  rank the G1: G1 results n20/s1 −0.22, n20/s9 +0.43 (79.6 s), n26/s0 +0.59 (53.5 s, 13 %
  intervention). Trying n32/s0 and n32/s1 on the G1 before settling.
- 2026-09-12 04:05 n32 on the G1: s0 −0.37, s1 −0.68 → settled on 26 pedestrians, seed 0
  (filtered h_min +0.59, goal 53.5 s; unfiltered nominal −0.49, 35.7 s). SCRAMBLE_N_PED = 26 in
  the showcase driver. Rendered; candidate GIF 6.1 MB swapped in, review page refreshed.
- 2026-09-12 05:10 corridor: body-footprint ellipse (0.16×0.28 puck) + pedestrian keep-out discs
  replace the inflated ellipse in the drawing; gap widened for the showcase (G1: 1.4 m h_min
  +0.07 / 90.5 s, 1.5 m +0.06 / 82.7 s, both still sidestep; unfiltered −0.48 / −0.39) →
  CORRIDOR_GAP = 1.5. 4×2 README layout (no filter | CBFKit rows) drafted in the preview;
  side-by-side GIFs at 720 px: navigate 2.6, plaza 7.0, corridor 5.5, scramble 8.9 MB.
- 2026-09-12 corridor control follow-up (Bardh: stay sideways until clear, then turn).
  Development host is Darwin arm64; ground-truth runs are Linux x86_64 on
  `bardhh@192.168.0.154`, `~/code/cbfkit-viz/.venv`, JAX 0.6.2, CPU, AMO 23-DoF G1
  (the corridor uses AMO, not the Unitree 12-DoF adapter). All G1 measurements below
  use `g1_showcase.py simulate corridor --gap 1.5 --seed N`, 150 s maximum,
  two static pedestrians, ellipse, hard constraints, robust bound 0. No changes to
  barriers, class-K conditions, QP, or relaxation semantics. Outputs live beneath
  `examples/mujoco/results/showcase/commitment/` on the box; local copies and test
  logs are in `.omc/corridor-commitment/`. README and media untouched.
  - Reproduced baseline, seed 0 (`smooth-0/`): goal **82.70 s**, h_min **+0.064403**,
    intervention **13%**, wall **113.8 s**. Command heading drops from 67 degrees at
    x=-0.25 m to 18.5 degrees at x=0; h_min occurs at x=+0.250, y=+0.192 m,
    heading 13.4 degrees. Total heading travel 530 degrees. This confirms the early
    unwind in the screenshot; the recomputed h is positive on this configuration.
  - Acceleration plan smoothing, retained fraction 0.8, seed 0 (`smooth-0.8/`):
    goal **145.04 s**, h_min **+0.264210**, intervention **17%**, wall **164.8 s**.
    **Rejected as an improvement:** heading winds to 2623 degrees in magnitude,
    total travel 4409 degrees. Smoothing acceleration delays angular braking;
    it does not commit a heading. The opt-in option defaults to zero. Failed MPPI
    samples now retain a finite warm start and keep the P-law fallback active until
    a successful replan, preventing NaNs from contaminating a smoothed nominal.
  - New scenario-specific nominal (`--sidestep-commitment`): MPPI owns the approach
    and chooses turn direction. When |theta| exceeds 45 degrees within 1 m before
    the pedestrian line, a phase latch holds the sideways heading and tracks a
    waypoint through the gap midpoint. At x>0.68 m (maximum inflated footprint
    radius plus 0.10 m), it switches once to goal/heading tracking. Every nominal
    still passes through the same hard CBF-QP. This is a hybrid nominal with a
    scripted maneuver after MPPI selects the turn, **not** MPPI discovering the
    full maneuver. The exit distance is a nominal preference, not a new certificate.
  - First commitment trial, **90-degree** target (`hold-s0/`, `hold-s1/`):
    seed 0 goal **85.98 s**, h_min **+0.277274**, intervention **5%**, wall **114.1 s**;
    seed 1 goal **58.06 s**, h_min **+0.145**, intervention **1%**, wall **61.7 s**.
    Seed 0 holds 90 degrees across the line and until clear; total heading travel
    falls to 296 degrees. **Fails the seed-0 speed target:** sideways translation
    realises only ~0.032 m/s, spending 36.7 s between engagement and release.
  - Proxy screening only: commitment at gap 1.5, seed 0, goal 27.2 s, h_min +0.263,
    wall 2.07 s, max heading 30 degrees (the latch never engages). At gap 1.0 with
    the 90-degree target, seed 0, goal 27.9 s, h_min +0.0817, wall 2.02 s, max heading
    89.65 degrees. No G1 performance claim is inferred from either proxy result.
  - **Rejected 75-degree target**, keeping the original 0.68 m release
    (`hold75-s0/`, `hold75-s1/`): seed 0 goal **99.54 s**, h_min **-0.014248**,
    intervention **4%**, wall **83.4 s**; seed 1 goal **59.50 s**, h_min **-0.358**,
    intervention **0%** rounded, wall **55.5 s**. Seed 0 drifted toward the lower
    pedestrian (minimum at x=0.274, y=-0.374 m). The proxy at gap 1.0/seed 0 passed
    in 28.14 s, h_min +0.0649, wall 1.98 s: another reminder that proxy screening is
    not G1 validation. Restored the 90-degree target.
  - Next trial (`clear-sN/`): retain the 90-degree target and identical translation
    waypoint, release only when x>0.46 m (inflated longitudinal radius) **and** both
    centre distances exceed 0.68 m (largest inflated semi-axis plus 0.10 m). This
    prices the full rotation sweep geometrically rather than always traversing
    another 0.22 m at the slow lateral gait. The QP remains the safety authority.
  - **Accepted clearance-release nominal**, gap 1.5 m, two static pedestrians:

    | seed | goal time | h_min (live, recomputed) | intervention | wall time |
    | --- | --- | --- | --- | --- |
    | 0 | 80.88 s | +0.277274 | 5% | 68.8 s |
    | 1 | 57.78 s | +0.145 | 1% | 53.6 s |
    | 2 | 71.16 s | +0.203 | 9% | 63.5 s |

    All reach the goal with zero controller errors. Seeds 0/1 engage and release
    the latch; seed 2 remains in phase 0 throughout (MPPI does not request a
    >=45-degree turn before the gap), so it validates the unchanged approach path,
    not an additional committed-sidestep maneuver. Seed 0 meets the requested
    h_min>=+0.05 and time<82.7 s: 1.82 s faster (2.2%), with the main improvement
    being clearance and sustained sideways motion. At x=0 and x=0.25 m, heading
    is 90 degrees (baseline 18.5/13.4 degrees); heading travel is 300 vs 530 degrees.
    The geometric gate releases at x>0.46, after the entire turn sweep is clear.
    These are measured closed-loop results on the specified G1, not a new guarantee
    against arbitrary plant tracking errors or a claim for other gaps/seeds.
    The showcase corridor enables commitment by default; reproduce the old nominal
    with `--no-sidestep-commitment`. The standalone corridor keeps this mode opt-in:
    `g1_corridor.py --g1 --planner mppi --gap 1.5 --robust 0 --duration 150
    --sidestep-commitment`. All MPPI/control/CBF limits and costs remain unchanged
    when smoothing is zero. The command and raw MPPI plan are logged separately;
    the raw MPPI rollout is not the committed maneuver's future trajectory.
  - Validation: Linux `.venv/bin/pytest -m "not slow" -q`: **872 passed, 14 skipped,
    148 deselected**, 484.29 s. Mac full-suite attempts first hit a read-only home
    asset cache (resolved using `CBFKIT_ASSET_DIR=/tmp/cbfkit-corridor-test-assets`),
    then the pre-existing Torch/kvxopt duplicate-OpenMP abort. No solver or tests
    were weakened to bypass it; the full suite ran on Linux instead. Final local
    corridor/reduced-order regressions: 33 passed, plus 8 showcase option tests.
    Tests cover both turn directions, a pedestrian still inside the turn sweep,
    monotone release under gait jitter, actual hard-QP proxy crossing, smoothing
    validation/hold/fallback/recovery, and CLI routing/default/baseline selection.
    Black and Ruff (including import sorting, 100-column configuration) pass.
  - Final integrated check after enabling the default: **41/41 focused tests passed**
    on Linux (61.21 s). Ran `g1_showcase.py simulate corridor --gap 1.5 --seed 0`
    with no commitment flag (`final-default/`): goal **80.88 s**, h_min **+0.277**,
    intervention **5%**, wall **106.7 s**, reproducing the accepted run. For the
    original baseline/smoothing trials under today's defaults, explicitly pass
    `--no-sidestep-commitment`. All jobs finished with exit 0; watchdog entries closed.
  - README review preview requested: rendered the accepted `final-default` trajectory
    against `cor_gap1.5/g1_corridor_unfiltered_nominal.npz`, window **24–68 s** so
    the delayed turn-back is included. Existing MuJoCo renderer, grid floor, no
    camera drift, no GIF HUD, 720 px / 48 colours / 16 fps, 2x speed: comparison
    GIF **5,102,084 bytes**, **22.0 s**, 352 frames. Reviewed five sampled frames;
    opened the animated GitHub-style preview in Safari at `http://127.0.0.1:8766/`.
    Local files: `.omc/corridor-commitment/readme-preview/`; remote renders:
    `results/showcase/commitment/readme-preview/`. README and published media unchanged.
- 2026-09-12 06:30 Codex's committed-sidestep nominal accepted (G1 gap 1.5 seed 0: h_min +0.28,
  goal 80.9 s, intervention 5 %, vs +0.06 / 82.7 s / 13 % pure MPPI). Corridor clips
  re-rendered from commitment/final-default (window 24–68 s): README GIF 3.9 MB, side-by-side
  5.8 MB. Codex's changes committed together with the media.
