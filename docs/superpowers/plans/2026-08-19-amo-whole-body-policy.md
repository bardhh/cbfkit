# AMO whole-body G1 policy in cbfkit — Implementation Plan

**Goal:** Port UCSD's AMO (RSS 2025) whole-body policy — locomotion + commandable torso
height/yaw/pitch/roll — as a torch-free JAX `ControllerCallable` on `MujocoPlant`, with a
demo showing torso agility while walking, and the tracking bound measured for the CBF layer.

**Why AMO:** Bardh asked for "more agile policies that can move the torso to the side while
moving forward". AMO's command vector is exactly that: [vx, target-yaw, vy, Δheight,
torso-yaw, torso-pitch, torso-roll] on a 23-DoF G1 (12 legs + 3 waist + 8 arms).

**Scouted facts (from OpenTeleVision/AMO@main, Apache-2.0):**
- Files: `amo_jit.pt` (17.8 MB TorchScript), `adapter_jit.pt`, `adapter_norm_stats.pt`
  (torch.save of 4 numpy arrays: input/output mean/std, 12→15), `g1.xml` + STL meshes,
  `play_amo.py` = the reference MuJoCo deployment (sim_dt 0.002, decimation 10 → 50 Hz,
  Python-side PD: kp [150,150,150,300,80,20]×2 legs, [400×3] waist, [80,80,40,60]×2 arms;
  kd [2,2,2,4,2,1]×2, [15×3], [2,2,1,1]×2; torque limits [88,139,88,139,50,50]×2,
  [88,50,50], [25×4]×2; XML actuators are raw torque motors ±200).
- Architecture (all ELU MLPs unless noted): adapter 12→512→512→256→15 (BatchNorm1d eval
  + LeakyReLU); text_feat_encoder 93→128→16 per frame on the LAST 4 history frames;
  text_feat_merger 64→16; history_encoder per-frame 93→30 + Conv1d(30→20,k4)+ELU+
  Conv1d(20→10,k2)+ELU+Flatten→30 + Linear 30→20 (conv strides TBD numerically — parity
  test settles them); student backbone 2474→1024→1024→512→15.
- Obs (93 proprio): [ang_vel·0.25, roll, pitch, sin/cos(dyaw), dof_pos−default (23),
  dof_vel·0.05 with idx {4,5,10,11,13,14} zeroed, last_action (23 = 15 raw + arm displ/0.25),
  sin(2π gait), adapter_out (15)]. Full policy input: obs (93+17 demo+3 priv+930 hist=1043)
  + extra_hist (25×93). Histories: proprio hist EXCLUDES current frame, extra hist INCLUDES it.
  dyaw = yaw − target_yaw (wrapped), forced 0 when |vx| < 0.1 (in-place stand flag).
  Gait: freq 1.3, cycle latches to [0.25,0.25] standing / [0.25,0.75] walking near sync points.
- Demo obs (17): [arm dof (8), vx, vy, 0, torso yaw, pitch, roll, height×3]; priv = zeros(3).
- Top-level TorchScript forward has a baked cuda zeros(105) — port the graph, don't run it;
  golden outputs come from driving the SUB-modules on CPU (torch 2.13 is in .venv, dev-only).

## Global constraints
- Runtime stays torch-free: weights read via the existing torch-free TorchScript reader in
  `unitree_policy.py` (extend for the torch.save norm-stats pickle); JAX inference.
- Assets fetched pinned-by-commit + SHA-256 into `~/.cache/cbfkit/` via `assets.py`
  (CBFKIT_ASSETS_OFFLINE honored); AMO's Apache-2.0 LICENSE fetched alongside.
- MJX: override XML `solver='PGS'` (unsupported) with newton; timestep 0.002 like play_amo;
  collision trim as in `load_g1_12dof` if mesh-plane contacts misbehave.
- `.venv` binaries; black only touched files; no Co-Authored-By.

## Tasks
1. **Assets**: `assets.py` — pinned AMO commit, `amo_dir()` fetching g1.xml, referenced
   meshes, 3 weight files, LICENSE, each SHA-256-verified. Test: skipped-if-offline fetch.
2. **Torch-free readers + JAX modules** (`systems/mujoco/amo_policy.py`):
   read both TorchScript archives + norm stats without torch; JAX forwards (adapter with
   eval BatchNorm; ELU/LeakyReLU; conv history encoder; the exact top-level graph incl.
   last-4-frames text-feat path). Parity test vs torch sub-modules (skipif no torch),
   atol 1e-5, and conv-stride discovery pinned by the test.
3. **Plant + adapter**: `make_g1_23dof_plant()` (MujocoPlant, substeps=10, PD ctrl_map with
   play_amo gains/limits, nu=23), `x0_standing`, `AmoWholeBodyPolicy.as_controller(...)`:
   carries hists/last_action/gait in sub_data; command interface (v_cmd world 2-D +
   torso dict) with heading-follower mapping to AMO's (vx, vy, target-yaw); logs commands.
4. **MJX trial**: stands 10 s; walks at 0.4 m/s; torso yaw/roll/height commands tracked
   while walking (measure). Slow test + TEST_MODE smoke via example.
5. **Demo + measurement**: `examples/mujoco/g1_amo_demo.py` — straight walk with a torso
   command schedule (lean/turn/duck while walking), GIF/plots, printed tracking metrics;
   measure v_com tracking bound (for robust CBF use later). Docs: CLAUDE.md bullet,
   G1_WALK_LOG entry, README line. Full suite green.
