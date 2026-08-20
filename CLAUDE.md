# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CBFKit is a Python toolbox for safe robotics control using Control Barrier Functions (CBFs) built on JAX. It provides safety-guaranteed controllers, planners, and simulation infrastructure for robotic systems. The core computation stack is JAX (autodiff + JIT) with jaxopt/cvxopt/casadi for quadratic programming.

## Build & Install

```bash
pip install -e ".[dev]"               # Editable dev install (includes all extras)
pip install -e ".[codegen,vis]"       # Specific profiles only
```

Requires Python 3.10–3.12. Uses `uv` for dependency resolution (`uv.lock`). On Apple Silicon, `kvxopt` is installed in place of `cvxopt` automatically. Optional extras that are missing at runtime raise `ImportError` with install guidance (e.g., "pip install cbfkit[casadi]").

**Docker:** VS Code dev container at `.devcontainer/cbfkit-container`, or `docker compose -f .devcontainer/docker-compose.yml run --rm cbfkit bash`. GPU profile available on Linux (`--profile gpu`).

## Testing

```bash
pytest -m "not slow" tests                                     # Fast tests (default CI)
CBFKIT_TEST_MODE=1 pytest -m "slow" tests                      # Slow tests (CI: 3.10 only)
pytest tests/test_controllers/test_cbf_clf.py                  # Single file
pytest tests/test_controllers/test_cbf_clf.py::TestCbfClfQP::test_vanilla_cbf  # Single test
```

The root `conftest.py` (loaded first, since `pytest.ini` anchors the rootdir) puts `./src` ahead of site-packages, optionally loads `.env`, and defaults JAX to CPU; `tests/conftest.py` repeats the CPU default via `os.environ.setdefault` so tests run on CPU locally, and CI also sets it as an env var. Slow tests are gated by `CBFKIT_TEST_MODE=1`; examples/tutorials also use this env var to shorten simulations and skip plots. CI runs linting and mypy only on Python 3.10.

MuJoCo tests live in `tests/test_mujoco/` and are skipped when the extra is not installed; run them with `CBFKIT_TEST_MODE=1 pytest tests/test_mujoco` (the cart-pole swing-up acceptance test is `slow`). CI installs the extra on the 3.10 job only. MuJoCo examples live in `examples/mujoco/` and are smoke-run by `tests/test_mujoco/test_examples.py`, not by `test_examples_and_tutorials.py` (which runs on all Python versions).

## Linting & Formatting

```bash
ruff check src/cbfkit          # Lint (E, F rules; ignores F401, E501)
black src/cbfkit               # Format (line-length 100)
isort src/cbfkit               # Sort imports (black-compatible)
mypy src/cbfkit                # Type check (see exclusions below)
pre-commit run --all-files     # All hooks: trailing-whitespace, eof-fixer, black, isort, ruff, mypy
```

Config: line-length 100, target Python 3.10, ruff selects E/F but ignores F401 (unused imports), E402 (import order), E501 (line length — enforced by black instead), and all D-rules. Tutorials directory excluded from ruff.

black/isort/ruff are configured in `pyproject.toml`; mypy and pytest keep their own `mypy.ini` and `pytest.ini` (both take precedence over `pyproject.toml`). There are two mypy exclusion lists and they do not match: `mypy.ini` skips `tutorials/`, `examples/wip/`, `build/`, and the generated-model directories, while the pre-commit hook additionally skips all of `examples/`. Note that CI runs `mypy src/cbfkit || true`, so a type error never fails CI — pre-commit is the only gate that enforces mypy.

## Benchmarking CLI

```bash
cbfkit-bench list                         # List available scenarios
cbfkit-bench run <scenario>               # Run a benchmark scenario
```

The benchmarks module (`src/cbfkit/benchmarks/`) provides scenario registration, sweep configs, and comparison tools.

## Architecture

### Simulation Pipeline

```
Planner → Nominal Controller → Safety Controller (CBF-CLF-QP) → Plant Dynamics → Integrator → Sensor → Estimator
```

The simulation loop in `simulation/simulator.py` orchestrates this pipeline. Each component is a pure function returning `(output, updated_data)`. The `simulation/` module is split into submodules: `backend.py` (step logic), `simulator_jit.py` (JIT path), `callbacks.py`, `status.py`, `formatting.py`.

**Planner output determines the pipeline path:**
- Planner returns a **control trajectory** (`u_traj`) → nominal controller is skipped, safety controller receives it directly.
- Planner returns a **state trajectory** (`x_traj`) → nominal controller converts desired state to control input, then safety controller filters it.

`sim.execute()` returns a `SimulationResults` object that supports both legacy 8-tuple unpacking and key-based access:
```python
results = sim.execute(...)
states = results["states"]          # key-based
x, u, z, p, dk, dv, pk, pv = results  # legacy tuple unpacking
```

### Controller Adapter Utilities

Legacy controller signatures are automatically adapted by the simulator via `setup_controller` and `setup_nominal_controller` in `controllers/utils.py`. Supported input signatures:
- `(t, x)`, `(t, x, u_nom)`, `(t, x, u_nom, key)`, `(t, x, key, data)`, `(t, x, u_nom, key, data)` → all normalized to canonical `ControllerCallable`
- `(t, x)`, `(t, x, ref)`, `(t, x, key, ref)` → normalized to `NominalControllerCallable`

### Key Modules (`src/cbfkit/`)

- **simulation/** — Simulator engine (`execute()`), Monte Carlo batch runs (multiprocessing-based `conduct_monte_carlo` and GPU-accelerated `jax.vmap` path), JIT-compiled path
- **controllers/** — CBF-CLF-QP generator (vanilla/robust/stochastic/risk-aware), MPPI sampling-based control
- **certificates/** — Barrier functions, Lyapunov functions, relative-degree rectifiers, certificate packaging
- **systems/** — Pre-built models (unicycle, quadrotor, van der Pol, fixed-wing UAV, double integrator, single integrator, pedestrian, nonlinear 2D)
- **systems/mujoco/** — Optional MuJoCo/MJX plant backend (`pip install cbfkit[mujoco]`; `mujoco` and `mujoco-mjx` are pinned to the same version). `MujocoPlant` wraps an MJCF model; pass it as `plant=` to `sim.execute()` in place of `dynamics`+`integrator`. The logged flat state is `[qpos | qvel | com_xyz]` (`plant.com_indices` gives the CoM slots). Only the JIT path is supported for real use; `use_jit=False` is debug-only. `perturbation` and vectorised Monte Carlo are unsupported on the plant path. Not imported by `cbfkit.systems` so the core stays importable without MuJoCo. `systems/mujoco/g1.py` has the Unitree G1 helpers (`load_g1(sim=...)`, ids/keyframes, `standup_costs`, `friction_randomizer`); the G1 XML/LICENSE are vendored under `systems/mujoco/models/g1/` and its 51 meshes are fetched on first use from a pinned `mujoco_menagerie` commit into `~/.cache/cbfkit/` (`CBFKIT_ASSET_DIR` overrides; `CBFKIT_ASSETS_OFFLINE=1` forbids downloads) with SHA-256 verification (`systems/mujoco/assets.py`). MJX does not implement the `OVERRIDE` enable flag, so `load_g1(sim=True)` applies hydrax's contact overrides per geom.
- **systems/mujoco/unitree_policy.py** — Unitree's pretrained G1 walking policy (`unitree_rl_gym`, 12-DoF legs, LSTM) as a `ControllerCallable`: files fetched from a pinned commit (`assets.unitree_rl_gym_dir`), TorchScript read **without torch**, evaluated in JAX inside the scan, PD torques at 500 Hz via `MujocoPlant(ctrl_map=...)`. `systems/mujoco/reduced_order.py` — the CBF layer on the CoM (embedded single integrator + ellipsoidal barriers at `plant.com_indices` + `safe_locomotion_controller`). `examples/mujoco/g1_navigate.py` is the end-to-end demo: `--reduced-model di` (default; command-side double integrator, HOCBF via `rectify_relative_degree`, `safe_locomotion_controller_di`) or `si`; robust CBF with a per-model measured tracking bound by default (`--robust 0` = vanilla); the policy adapter rotates the world-frame command into the body frame and follows the heading (`as_controller(world_frame=..., heading_gain=...)`). `examples/mujoco/g1_plaza.py` is the richer demo: a 3-waypoint route (`cbfkit.planners.waypoint_route`, a stateless planner that publishes the current waypoint as `x_traj`) past 2 pillars and 3 *reactive* social-force pedestrians (`systems/mujoco/crowd.py::SocialForceCrowd`, stepped inside `safe_locomotion_controller_di(agents=...)`; the CBF sees them as tracked agents in the augmented state `[x | v | p_i v_i]` with `ṗ_i = v_i`, `com_agent_hocbfs`; their accelerations are the unmodelled part). Barriers use `shape="distance"` (`h = |c−p|/r − 1`, unit-norm gradient) because the quadratic ellipsoid's margins grow with distance and make the robust QP infeasible with many barriers; the example uses `get_solver("fast")` because jaxopt-OSQP stalls on the thin-wedge QP when two robust margins are active. Pedestrians are kinematic and certificate-only, drawn into the GIF via the `markers(scn, k, t)` callback of `viewer_utils.render_gif` / `replay_in_viewer`. `com_moving_obstacle_hocbfs` / `com_moving_obstacle_barriers` remain for known constant-velocity obstacles (time-varying barriers). `examples/mujoco/g1_scramble.py` scales the crowd to a Shibuya-style scramble (40 social-force pedestrians faster than the robot): hard barrier constraints are infeasible in a crush, so it defaults to soft constraints (`relaxable_cbf`) and reports slack use, h_min and near-misses as filter outcomes, not certificates. Its `--planner mppi` puts a *socially tuned MPPI* in front of the CBF filter — `controllers/mppi/social_costs.py` (asymmetric-Gaussian proxemics wide in front of a walking pedestrian, TTC power law, progress as a *terminal* cost so waiting for a gap is a first-class plan, legibility terms, optional keep-left `pass_side`) as `trajectory_cost` for `vanilla_mppi` over the compact `[p | v | pedestrians]` state, wired in via `safe_locomotion_controller_di(local_planner=reduced_order.mppi_local_planner(...))` (5 Hz replan held between solves via `lax.cond`; `MppiParameters.control_std` sets the sampling std). `--proxy` swaps the MJX G1 for the identified 2-D first-order-lag model (τ = 0.2 s) for multi-seed tuning; `examples/mujoco/g1_scramble_social_eval.py` writes the goal-vs-mppi intrusiveness table (Hall-zone pedestrian-seconds, front intrusions, crowd deviation/slowdown vs the robot-free rollout, per-10 s rates against the crowd's own human norm). Gotcha found here: a *warm-started* fast-PDIPM QP can exhaust max_iter at an MPPI replan boundary while a cold start converges — `get_solver("fast")` now cold-restarts on failure. `systems/mujoco/amo_policy.py` — UCSD's AMO whole-body policy (RSS 2025, `OpenTeleVision/AMO`, Apache-2.0, pinned via `assets.amo_dir()`): 23-DoF G1 (12 legs + 3 waist + 8 arms), command = `[vx, target_yaw, vy, height_delta, torso_yaw, torso_pitch, torso_roll]` — torso posture *while walking*. TorchScript + torch.save weights read without torch (`load_torch_numpy_pickle` for the norm stats); JAX forwards parity-tested against torch (dev-only oracle, atol 1e-4); `make_g1_23dof_plant()` (2 ms × 10 PD, play_amo gains; upstream XML wants PGS → switched to Newton for MJX), `x0_standing` (home keyframe), `AmoWholeBodyPolicy.as_controller(torso_command=...)` (native heading following via the absolute target-yaw command; carry in `sub_data["_amo"]`; the 10-frame history excludes the current frame while the 25-frame extra history includes it — play_amo subtlety). `examples/mujoco/g1_amo_demo.py` walks through a torso schedule (look/lean/duck/bow, measured table in its docstring); MJX sim2sim: torso tracking is good, forward speed realises ~0.3 of the commanded 0.4 m/s. `g1_scramble.py --robot amo` runs the scramble on AMO (gentlest measured config: intimate rate ~1.0 vs unitree ~2, +20 s crossing); `--torso` (off-by-default) is a *measured negative result* — the reactive shoulder-turn perturbs gait tracking more than the ~8 cm of profile it frees, across two tuning rounds (`upper_body_clearance` metric: offline FK of shoulder/elbow/hand bodies vs pedestrian discs). `torso_command` also accepts `(t, x, sub)` callables for state-aware behaviors. **Anisotropic footprint** (`reduced_order`): `embedded_heading_double_integrator` (`[p v θ ω | peds]`, controls `[a, α]` — heading second-order keeps barriers uniformly rel-deg 2), `com_agent_ellipse_hocbfs` (rotating-ellipse keep-out, `G1_FOOTPRINT` measured 0.11×0.22 m by `g1_footprint_measure.py`), `safe_locomotion_controller_hdi` (hands `[vx, vy, target_yaw]` down; AMO adapter takes `u_nom[2]` as absolute heading). `examples/mujoco/g1_corridor.py`: the certified-sidestep demo — disc refuses a 1.0/1.25 m gap, ellipse crosses it (proxy h ≥ +0.08 exact; G1 h droop −0.16 = measured lateral tracking error). Its `--planner mppi` replaces the hand-coded heading ramp with 6 s of MPPI over the same heading model (`ellipse_trajectory_cost` + `mppi_local_planner(control_dim=3, state_head=6)` + `safe_locomotion_controller_hdi(local_planner=...)`): the rotation is *discovered* (proxy 3/3 seeds, h ≥ +0.07; disc still refuses under the identical planner), and on an `--offset` gap the lookahead plans a diagonal with ~40% less heading travel at 6× the margin of the ramp. Cost-shaping gotcha (measured): the planner's clearance term must be a saturated quadratic in absolute h-units — a margin-normalised hinge makes every violation infinite, the best sample is always "wait", and the robot freezes and dithers in front of the gap; random shooting cannot thread a ±4 cm tube, so MPPI proposes the homotopy and the hard QP does the threading. Gotchas: the myopic QP never *invents* rotation (suggestion in the nominal, certificate decides); play_amo's stand flag keys on |vx| only so pure-vy commands stand still (`keep_gait_alive` bumps vx ±0.12 zero-mean — a constant bump is unmodelled drift that walked through a keep-out); robust bounds on rectified ψ rows are ~2× the h-level margin (over-conservative at 0.12). `systems/mujoco/groot_policy.py` — NVIDIA GR00T GEAR-WBC (NVlabs/GR00T-WholeBodyControl, Apache-2.0 + NVIDIA Open Model License, `assets.groot_dir()`; `ensure_repo_files` now supports per-file git-LFS entries via the media endpoint): Balance + Walk ONNX read by a dependency-free protobuf wire parser (`load_onnx_tensors`), estimator(516→35, v̂+normalized latent)+actor(121→15) in JAX, parity vs onnxruntime (subprocess oracle); 29-DoF plant from the repo's `g1_29dof_old.xml` (robot-only upstream — a ground plane is injected at load; the sim2mujoco XML has 43 joints and would overflow the 86-dim obs — repo drift, documented), 5 ms × 4 PD, arms held at zero; command `[vx, vy, ωz(rate) | height | torso rpy]`. `examples/mujoco/g1_walk_compare.py` races all three policies (numbers in its docstring: GR00T flattest CoM 2 mm + best-of-the-whole-body tracking; Unitree best pure tracker; AMO least roll). In the scramble (`--robot groot`) GR00T is the best whole-body compromise: 74/60 s crossings, h −0.06/+0.45, largest upper-body clearances — better tracking keeps the command-side CBF model honest. `examples/mujoco/g1_model_distance.py` measures how far the DI/SI reduced models are from the G1 + policy (battery of open-loop command runs: disturbance-bound quantiles, ε(H) approximate-simulation distance, identified lag and ν-gap; results in `examples/mujoco/results/g1_model_distance.md`). `examples/mujoco/G1_WALK_LOG.md` records the in-house sampling-MPC gait attempts and the measured tables.
- **controllers/mjx_sampling_mpc/** — MPPI over spline knots with MJX rollouts and optional domain randomisation. Independent of `controllers/mppi` (which is a planner over control-affine ODEs). `SamplingMpc(...).as_controller()` returns a `ControllerCallable`; its state rides in `ControllerData.sub_data["mpc"]`.
- **optimization/** — QP solvers (jaxopt, cvxopt, casadi), MPC, dynamically-defined programs
- **integration/** — ODE integrators (RK4, forward Euler, solve_ivp)
- **modeling/** — System augmentation, disturbance models, neural network components
- **codegen/** — Jinja2-based code generation for new system models (plant dynamics, controllers, certificates) and ROS2 nodes. Templates live in `codegen/templates/*.j2`
- **sensors/** and **estimators/** — Sensor models and state estimation
- **benchmarks/** — Scenario-based benchmarking framework with sweep configs and comparison tools
- **utils/user_types/** — Central type definitions: `callables.py` (function signatures), `data.py` (ControllerData, PlannerData, SimulationResults)

### Design Patterns

- **Functional composition, not OOP inheritance.** Controllers, planners, and dynamics are factory functions returning callables. No class hierarchies.
- **JAX idioms throughout.** Immutable arrays, `jax.jit` for performance, `jax.vmap` for batching, 64-bit floats enabled globally in `__init__.py` for QP numerical stability.
- **Canonical function signatures:**
  - Dynamics: `(x) → (f(x), g(x))` for ẋ = f(x) + g(x)u
  - Controller: `(t, x, u_nom, key, data) → (u, ControllerData)`
  - Planner: `(t, x, u_prev, key, data) → (u_traj | None, PlannerData)`
  - Nominal controller: `(t, x, key, reference) → (u, ControllerData)`
- **CBF-CLF-QP** is the core safety filter pattern: minimize ‖u − u_nom‖² subject to barrier/Lyapunov constraints. Generated via `cbf_clf_qp_generator`. Five variants: `vanilla`, `robust`, `stochastic`, `risk_aware`, and `risk_aware_path_integral`.
- **Data passing via NamedTuples:** `ControllerData` and `PlannerData` are NamedTuples passed through the pipeline. Controllers like CBF-QP use minimal data; stateful planners like MPPI store solution trajectories in `PlannerData` fields (`u_traj`, `x_traj`). Use `PlannerData.from_constant(goal_state)` for fixed-setpoint simulations, or pass `goal=state` directly to `sim.execute()` as a shortcut.

### JAX Configuration

JAX 64-bit precision is enabled at import time in `src/cbfkit/__init__.py`. CI sets `JAX_PLATFORM_NAME=cpu` as an environment variable. JIT compilation is opt-in per simulation via `use_jit=True` in `sim.execute()`. Version is stored in `src/cbfkit/VERSION` and read dynamically.
