# NVIDIA GR00T GEAR-WBC policy port — Implementation Plan

**Goal:** Port the released GR00T-WholeBodyControl "decoupled WBC" G1 checkpoints
(NVlabs/GR00T-WholeBodyControl, sim2mujoco: Balance + Walk ONNX, NVIDIA Open Model
License) as a torch-free JAX ControllerCallable — for a naturalness comparison against
the Unitree and AMO gaits, with the same posture flexibility hooks (height + torso rpy).

**Scouted contract (run_mujoco_gear_wbc.py + g1_gear_wbc.yaml):** 29-DoF model, 15
actions (legs + waist) at 50 Hz (dt 0.005 x 4), PD kp [150,150,150,200,40,40]x2 +
[250x3], kd [2,2,2,4,2,2]x2 + [5x3]; arms PD-held at 0 (kp 100, kd 0.5). Obs 86 =
[cmd(7): loco(3)*[2,2,0.5], height(abs, 0.74), rpy(3) | omega*0.5 | gravity-in-body |
(qj-default)(29) | dqj*0.05(29) | last_action(15)]; history 6 (current frame INCLUDED),
input 516. Two stateless ONNX policies switched on |loco_cmd| <= 0.05 (Balance/Walk).
Architecture (recovered from the protobuf): estimator 516-256-256-35 (ELU), split
[0:3]=v-hat + [3:35]=latent L2-normalized (clip 1e-12); actor concat[last-86, v-hat,
latent] 121-512-256-256-15 (ELU).

## Tasks
1. **Assets**: extend `ensure_repo_files` with per-file LFS support
   (media.githubusercontent.com); pinned manifest: g1_gear_wbc.xml + referenced meshes +
   2 ONNX + license + yaml. 
2. **groot_policy.py**: minimal protobuf wire reader (`load_onnx_tensors`, no deps);
   JAX forward (estimator+actor+normalize); obs/history carry; Balance/Walk switch via
   jnp.where; `make_g1_gear_plant()` (0.005 x 4, PD 15 + arms-at-zero); `x0_standing`;
   `as_controller(world_frame, heading_gain, height=0.74, torso_rpy hook)` — wz is a
   yaw RATE here (unitree-style heading law, unlike AMO's absolute yaw).
   Parity test vs onnxruntime (dev-only oracle, skipif; installed in .venv).
3. **Validate + compare**: MJX stand + walk; `g1_walk_compare.py` — Unitree vs AMO vs
   GR00T at vx 0.4: realised speed, CoM height oscillation, base angular-vel RMS,
   command-accel jerk, GIFs. "More natural" gets numbers + eyeballs, honestly reported.
4. Docs (CLAUDE.md, walk log), smoke + slow tests, suite green.
