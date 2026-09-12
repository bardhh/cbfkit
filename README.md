# CBFKit: A Control Barrier Function Toolbox for Robotics Applications

[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://github.com/bardhh/cbfkit)
[![CI](https://github.com/bardhh/cbfkit/actions/workflows/ci.yml/badge.svg)](https://github.com/bardhh/cbfkit/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://github.com/bardhh/cbfkit/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2404.07158-b31b1b.svg)](https://arxiv.org/abs/2404.07158)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bardhh/cbfkit/blob/main/examples/gymnasium/safe_single_integrator.ipynb)

**JAX-based Control Barrier Function (CBF) safety filters for robotics.** CBFKit wraps a
CBF-QP around any nominal controller or learned policy, so the commands that reach the robot
keep the state inside the safe set of the barrier you supply (see
[Scope and assumptions](#scope-and-assumptions) for what that guarantee rests on). The
constraints come from automatic differentiation, the filter JIT-compiles for the control loop,
and a modular simulator lets you test the closed loop.

<p align="center">
  <img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/warehouse-safety.gif" width="100%" alt="ANYmal-C warehouse delivery in Isaac Lab: the unfiltered walking policy contacts a crossing cart; with CBFKit's batched safety filter the robot adjusts and reaches the goal">
</p>
<p align="center">
  <em>Isaac Lab, frozen ANYmal-C walking policy, moving cart. Left: unfiltered. Right: velocity commands filtered by a CBF-QP.</em>
</p>

[Quick start](#quick-start) · [Examples](examples/README.md) · [Tutorials](tutorials/README.md) · [Paper](https://arxiv.org/abs/2404.07158) · [Project page](https://bardhh.github.io/cbfkit/)

- **Safety filters** — CBF-CLF-QP controllers for control-affine systems in vanilla, robust (bounded disturbance) and stochastic (SDE) variants, an estimate-feedback risk-aware variant with a Gaussian chance-constraint margin, and an adaptive CVaR-CBF; high-order barriers via relative-degree rectification; a fast interior-point QP solver.
- **Planning** — Model Predictive Path Integral (MPPI) control with reach-avoid and STL-style costs, and a receding-horizon MPC.
- **Simulation and estimation** — a functional planner → controller → plant → sensor → estimator pipeline, EKF/UKF estimators, Monte Carlo rollouts with `jax.vmap`, and matplotlib/Plotly/Manim rendering.
- **Integrations** — a standalone `SafetyFilter`, a Gymnasium wrapper, a batched PyTorch bridge for Isaac Lab, a MuJoCo/MJX plant with Unitree G1 examples, and ROS2 node generation.

## Installation

Requires **Python 3.10–3.12**. There is no PyPI release yet; install from GitHub:

```bash
pip install "cbfkit @ git+https://github.com/bardhh/cbfkit.git"
```

The core install covers the safety filters, planners, simulator, estimators, code generation
(Jinja2), and matplotlib plotting. Optional features live behind extras, e.g.

```bash
pip install "cbfkit[gymnasium] @ git+https://github.com/bardhh/cbfkit.git"
```

<details>
<summary><strong>Optional extras</strong></summary>

| Extra | Adds | Used by |
|-------|------|---------|
| `gymnasium` | Gymnasium ≥ 1.0 | `cbfkit.wrappers.gymnasium.SafetyFilterWrapper` |
| `neural` | Flax, Optax | learned barrier functions (`examples/neural_cbf/`) |
| `casadi` | CasADi | `get_solver("casadi")` |
| `cvxopt` | CVXOPT (kvxopt on Apple Silicon) | `get_solver("cvxopt")` |
| `solvers` | `casadi` + `cvxopt` | |
| `plotly` | Plotly | interactive plots |
| `vis` | alias of `plotly` | |
| `optuna` | Optuna | parameter sweeps (`examples/parameter_sweep/`) |
| `codegen` | Black | formatting generated model code (generation itself needs no extra) |
| `manim` | Manim (also needs ffmpeg and LaTeX on the system) | `CBFAnimator(backend="manim")`, 3D renders |
| `torch` | PyTorch ≥ 2.8, JAX ≥ 0.6.2 | the Isaac Lab / PyTorch bridge |
| `mujoco` | MuJoCo + MJX (pinned to one version) | `MujocoPlant`, Unitree G1 examples |
| `all` | `gymnasium`, `neural`, `solvers`, `vis`, `optuna`, `codegen` (**not** `manim`, `torch`, `mujoco`) | |
| `dev` | `all` plus pytest, ruff, black, mypy, jupyter | development |

`manim`, `torch` and `mujoco` are excluded from `all` and `dev` because they pull in large or
platform-specific runtimes; select them explicitly, e.g.
`pip install "cbfkit[mujoco] @ git+https://github.com/bardhh/cbfkit.git"`.
</details>

For development, clone and install editable:

```bash
git clone https://github.com/bardhh/cbfkit.git && cd cbfkit
pip install -e ".[dev]"
pytest -m "not slow" tests
```

<details>
<summary><strong>Docker</strong></summary>

**VS Code Dev Container.** Open the project in VS Code and reopen in container, choosing the
**CBFKit CPU Dev Container** at `.devcontainer/cbfkit-container`.

**Docker Compose**
```bash
docker compose -f .devcontainer/docker-compose.yml build cbfkit
docker compose -f .devcontainer/docker-compose.yml run --rm cbfkit bash
docker compose -f .devcontainer/docker-compose.yml down
```

**GPU (Linux only)**
```bash
docker compose -f .devcontainer/docker-compose.yml --profile gpu build cbfkit_gpu
docker compose -f .devcontainer/docker-compose.yml --profile gpu run --rm cbfkit_gpu bash
```
</details>

## Quick start

A safety filter needs three things: dynamics in control-affine form, a barrier function
`h(x)` whose zero super-level set is the safe set, and the commands you want to keep safe.
Here a naive "drive straight at the goal" command is filtered around a keep-out disc:

```python
import jax.numpy as jnp
from cbfkit.certificates import generate_certificate
from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
from cbfkit.wrappers import SafetyFilter

# Single integrator on the plane: x_dot = u, with x = [px, py]
dynamics = lambda x: (jnp.zeros(2), jnp.eye(2))

# Keep-out disc of radius 0.5 at (2, 0); the safe set is {x : h(x) >= 0}
obstacle, radius = jnp.array([2.0, 0.0]), 0.5
barrier = generate_certificate(
    certificate=lambda x: jnp.sum((x - obstacle) ** 2) - radius**2,
    certificate_conditions=linear_class_k(1.0),  # h_dot >= -1.0 * h
)

# Minimally modify each commanded velocity so the disc is never entered
dt = 0.05  # the filter keeps its own clock; you still integrate the plant yourself
shield = SafetyFilter.from_cbf_qp(
    dynamics=dynamics, barriers=barrier, control_limits=jnp.array([2.0, 2.0]), dt=dt
)

x, goal = jnp.array([0.0, 0.1]), jnp.array([4.0, 0.0])
closest = jnp.inf
for _ in range(150):
    u_nom = (goal - x) / jnp.linalg.norm(goal - x)  # naive command, aimed through the obstacle
    u, info = shield.filter(x, u_nom)               # info["fallback_used"] flags a QP failure
    x = x + dt * u
    closest = jnp.minimum(closest, jnp.linalg.norm(x - obstacle))
print(f"final position {x.round(2)}, closest approach {closest:.2f} m (radius {radius})")
```

which prints

```
final position [ 3.96 -0.01], closest approach 0.56 m (radius 0.5)
```

The unfiltered command would drive straight through the disc; the filter bends the path
around it and lets the robot continue to the goal. `SafetyFilter.filter` works with any
control loop. It keeps its own clock, PRNG key and solver warm start, and on a QP failure it
falls back to the nominal command by default (`fallback=` accepts other strategies).

The same controllers plug into CBFKit's simulator for closed-loop studies with planners,
sensors and estimators. A unicycle version of this example, with the CBF-QP running inside
the simulator loop, is
[`examples/unicycle/reach_goal/unicycle_reach_avoid_cbf.py`](examples/unicycle/reach_goal/unicycle_reach_avoid_cbf.py).

## Scope and assumptions

CBFKit implements CBF-based controllers whose safety properties hold under the conditions of
the underlying theory, not unconditionally. In particular:

- **Certificate validity.** The filter enforces `h_dot(x, u) >= -alpha(h(x))` at each step. Forward invariance of the safe set follows only if `h` is a valid CBF for the model you supply and the QP stays feasible. Learned barriers (see [Neural CBF](#neural-cbf-learn-a-candidate-barrier-from-data)) are candidates fitted to samples, not certificates.
- **Model mismatch.** The certificate covers the model the filter sees. For the Unitree G1 demos that is a reduced-order command model; tracking error of the walking policy enters as a measured disturbance bound in the `robust` variant, not as a guarantee about the full robot.
- **Sampled implementation.** Constraints are enforced at the controller rate on a continuous-time condition. Inter-sample behaviour, solver tolerances and the `fallback` action on QP failure all matter in practice; the examples report minimum barrier values and fallback counts for this reason.
- **Monte Carlo results are empirical.** Rollout-based risk estimates in the stochastic and risk-aware examples are evaluations, not proofs.

The controller variants target the following model classes:

| Variant | Model |
|---------|-------|
| `vanilla` | $\dot{x} = f(x) + g(x)u$ |
| `robust` | $\dot{x} = f(x) + g(x)u + w$ with a bounded additive disturbance $\lVert w \rVert \le w_{\max}$ (2-norm or sup-norm, set by `disturbance_norm`) |
| `stochastic` | $dx = \big(f(x) + g(x)u\big)dt + \sigma(x)dw$ with $w$ a Wiener process |

## Examples and tutorials

**Examples** use pre-built systems from `cbfkit.systems` (unicycle, single and double
integrator, kinematic bicycle, quadrotor, fixed-wing UAV, Van der Pol, a nonlinear 2D system,
pedestrians) and need no code generation. [`examples/README.md`](examples/README.md) lists them in a recommended order:

```bash
python examples/unicycle/reach_goal/unicycle_reach_avoid_cbf.py
python examples/unicycle/reach_goal/mppi_cbf.py
cbfkit-bench list          # benchmark scenarios with sweep configs and comparisons
```

**Tutorials** show how to define a new system and generate its dynamics, controllers,
certificates and ROS2 node with `cbfkit.codegen`. See [`tutorials/README.md`](tutorials/README.md).

| Tutorial | Description |
|----------|-------------|
| [`code_generation_tutorial.ipynb`](tutorials/code_generation_tutorial.ipynb) | Generate dynamics, controllers, and certificates for a Van der Pol oscillator |
| [`multi_robot_coordination.ipynb`](tutorials/multi_robot_coordination.ipynb) | Multi-robot CBF coordination with code generation |
| [`rectify_relative_degree.ipynb`](tutorials/rectify_relative_degree.ipynb) | High-order CBFs for constraints with relative degree above one |
| [`mppi_cbf_reach_avoid.py`](tutorials/mppi_cbf_reach_avoid.py) | MPPI + CBF for unicycle reach-avoid |
| [`mppi_stl_reach_avoid.py`](tutorials/mppi_stl_reach_avoid.py) | MPPI with Signal Temporal Logic specifications |
| [`single_integrator_dynamic_obstacles.py`](tutorials/single_integrator_dynamic_obstacles.py) | Dynamic obstacle avoidance |
| [`multi_robot_3d_reachavoid.py`](tutorials/multi_robot_3d_reachavoid.py) | 3D multi-robot reach-avoid rendered with Manim |

There is no separate API reference yet. The docstrings, the examples above, and the
[project page](https://bardhh.github.io/cbfkit/) are the documentation.

## Integrations

### Gymnasium: safe RL without retraining

`SafetyFilterWrapper` filters every action from an RL policy through a CBF-QP before it
reaches the environment. It works with PPO, SAC, or any algorithm that emits continuous
actions, and needs no change to the policy. Requirements: a `Box` action space, a
control-affine model of the environment, barrier certificates for it, and the
observation-to-state and action-to-control mappings if they are not the identity.

<p align="center"><img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/safe_rl_gymnasium.gif" width="85%" alt="Safe RL: naive vs CBF-filtered policy"></p>

```bash
pip install "cbfkit[gymnasium] @ git+https://github.com/bardhh/cbfkit.git"
python examples/gymnasium/safe_single_integrator.py
```

### Isaac Lab and PyTorch: batched policy commands

`BatchedSafetyFilter` runs one CBFKit controller across many parallel environments with
independent histories, clocks, random streams and resets. The optional `torch` extra adds
`TorchSafetyFilter`, which copies PyTorch tensors through DLPack on their current device rather
than aliasing them; it is an inference bridge and does not carry gradients back to the policy.

The warehouse example filters planar velocity requests before a frozen ANYmal-C walking
policy in Isaac Lab, using simulator obstacle state. Across 24 trials per controller with
matching scenario settings and a crossing cart:

| Controller | Obstacle contacts | Falls | Clean deliveries |
|------------|------------------:|------:|-----------------:|
| Unfiltered policy | 23 | 19 | 0 |
| Distance-based stop rule | 0 | 3 | 21 |
| CBF-QP filter | 0 | 2 | 22 |

Two startup falls occurred in both the stop and CBF runs. Median filter latency for the
eight-robot batch was 2.8 ms including the bridge. The counts are descriptive: hidden simulator
and policy state was not held equal across modes, so they do not establish a causal advantage
over stopping. The [validation report](examples/isaac_lab/validation/REPORT.md) states what
the experiment does and does not establish.

[Integration guide](examples/isaac_lab/README.md) ·
[Reproduce the warehouse demo](examples/isaac_lab/WAREHOUSE.md) ·
[Measurements and limitations](examples/isaac_lab/validation/REPORT.md)

### MuJoCo/MJX: humanoid locomotion under reduced-order CBFs

<p align="center">
  <img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/g1_scramble.gif" width="48%" alt="Unitree G1 crossing a 40-pedestrian scramble under tracked-agent HOCBFs with a social MPPI planner">
  <img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/g1_corridor_sidestep.gif" width="48%" alt="Unitree G1 turning sideways to pass a gap certified by a rotating-ellipse footprint CBF">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/g1_plaza.gif" width="48%" alt="Unitree G1 crossing a plaza among pillars and reactive pedestrians under high-order CBFs">
  <img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/g1_navigate.gif" width="48%" alt="Unitree G1 walking to a goal around an obstacle under a robust CoM CBF">
</p>

A Unitree G1 runs in MJX through `sim.execute(plant=MujocoPlant(...))` with a pretrained
walking policy underneath and a CBF-QP on the centre-of-mass command above it: keep-out
barriers on obstacles and tracked pedestrians, a rotating-ellipse footprint for squeezing
through gaps, and an optional socially tuned MPPI planner in front of the filter. The
certificate covers the command-side reduced model; tracking error enters as a measured
disturbance bound.

```bash
pip install "cbfkit[mujoco] @ git+https://github.com/bardhh/cbfkit.git"
python examples/mujoco/g1_navigate.py
```

[Scramble crossing](examples/mujoco/g1_scramble.py) ·
[Certified sidestep](examples/mujoco/g1_corridor.py) ·
[Plaza among pedestrians](examples/mujoco/g1_plaza.py) ·
[Goal past an obstacle](examples/mujoco/g1_navigate.py) ·
[Sampling MPC over MJX rollouts](src/cbfkit/controllers/mjx_sampling_mpc/)

### ROS2

CBFKit is a Python toolbox; it does not depend on ROS. The code generation pipeline
(`cbfkit.codegen.create_new_system.generate_model`) emits a ROS2 controller node script
next to each generated model, under `<model>/ros2/controller.py`, which you adapt to your
message types. The [code generation tutorial](tutorials/code_generation_tutorial.ipynb) walks
through it.

## QP solver

CBF-CLF-QPs are tiny (a handful of decision variables, tens of constraints) and are solved at
the control rate. CBFKit ships a Mehrotra predictor-corrector primal-dual interior-point solver
written for this regime, selected with `solver=get_solver("fast")` on any CBF-QP controller;
the default remains `get_solver("jaxopt")` (OSQP), and CVXOPT and CasADi backends are available
behind extras. The fast solver's advantage is reliability rather than raw throughput: its
barrier-regularized Newton system stays well-conditioned on the slack-relaxed, ill-conditioned
QPs that stall OSQP (the G1 scramble example hit one such degenerate QP), it runs a fixed budget
of 16 Newton iterations (benign problems converge by iteration 8), and it cold-restarts if a
warm start stalls.

Wall time per solve with each solver wrapped in `jax.jit`, which is the path the simulator's
JIT mode and any jitted controller take, on one fixed random positive-definite QP per size,
200 repetitions after a warm-up call, CPU only (Apple M5 Max, JAX 0.6.2, float64):

| Size (n×m) | JAXopt OSQP | `fast` |
|------------|------------:|-------:|
| 2×5        | 30 µs       | 34 µs  |
| 4×10       | 38 µs       | 43 µs  |
| 8×20       | 83 µs       | 50 µs  |

Called eagerly from Python instead, OSQP takes about 45 ms per solve at every size because
each of its iterations is dispatched separately, while the fast solver stays near 50 µs; that
eager gap is what earlier versions of this README reported as a 700–880× speed-up. Both
solvers agree to about 1e-5 on these problems; the benchmark checks timing, not accuracy.
Reproduce either table with:

```bash
python benchmarks/qp_solver_comparison.py --jit --no-plot   # jitted, JAX solvers only
python benchmarks/qp_solver_comparison.py                   # eager, includes CVXOPT
```

## Gallery

### Neural CBF: learn a candidate barrier from data

When obstacles are hard to describe analytically (point clouds, occupancy maps, scanned
environments), a small network can learn `h(x)` from labelled safe/unsafe states and plug
straight into the CBF-QP controller. The result is a fitted candidate barrier; validity in
the sense of [Scope and assumptions](#scope-and-assumptions) has to be checked separately.

<p align="center"><img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/neural_cbf.gif" width="65%" alt="Neural CBF: agent avoiding a learned obstacle"></p>

```bash
python examples/neural_cbf/neural_cbf_obstacle_avoidance.py
```

### Multi-robot 3D coordination with Manim

Multi-robot reach-avoid in 3D rendered with CBFKit's Manim backend. The same backend renders
2D `CBFAnimator` scenes: pass `backend="manim"` (or `"manim-<low|medium|high|production>"`)
and `save("out.mp4")` writes the video (`.gif` also supported).

<p align="center"><img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/multi_robot_3d.gif" width="70%" alt="Manim 3D render of multi-robot reach-avoid"></p>

```bash
python tutorials/multi_robot_3d_reachavoid.py
```

<details>
<summary><strong>More examples (13 tiles)</strong></summary>

<table>
  <tr>
    <td align="center" width="33%">
      <img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/risk_aware_cvar.gif" width="100%" alt="Ellipsoidal-obstacle CBF"><br>
      <sub><b>Ellipsoidal-obstacle CBF</b> <a href="examples/unicycle/reach_goal/ellipsoidal_obstacle_cbf.py" title="Source: examples/unicycle/reach_goal/ellipsoidal_obstacle_cbf.py">🔗</a><br>Unicycle reach-goal with linear class-K</sub>
    </td>
    <td align="center" width="33%">
      <img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/stochastic_cbf.gif" width="100%" alt="Stochastic CBF"><br>
      <sub><b>Stochastic CBF (SDE)</b> <a href="examples/unicycle/reach_goal/stochastic_cbf.py" title="Source: examples/unicycle/reach_goal/stochastic_cbf.py">🔗</a><br>Safety under Brownian disturbance</sub>
    </td>
    <td align="center" width="33%">
      <img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/robust_cbf.gif" width="100%" alt="Robust CBF"><br>
      <sub><b>Robust CBF</b> <a href="examples/unicycle/reach_goal/robust_cbf.py" title="Source: examples/unicycle/reach_goal/robust_cbf.py">🔗</a><br>Worst-case bounded disturbance</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/mppi_rollouts.gif" width="100%" alt="MPPI rollouts"><br>
      <sub><b>MPPI rollout sampling</b> <a href="examples/unicycle/reach_goal/mppi_cbf.py" title="Source: examples/unicycle/reach_goal/mppi_cbf.py">🔗</a><br>Sampling-based planning</sub>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/mppi_stl.gif" width="100%" alt="MPPI reach-avoid"><br>
      <sub><b>MPPI reach-avoid</b> <a href="examples/single_integrator/mppi_reach_avoid.py" title="Source: examples/single_integrator/mppi_reach_avoid.py">🔗</a><br>Sampling-based planning with goal + obstacle cost</sub>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/ped_mppi.gif" width="100%" alt="MPPI navigation among pedestrians"><br>
      <sub><b>MPPI among pedestrians</b> <a href="examples/pedestrian/navigate_among_pedestrians/crowded.py" title="Source: examples/pedestrian/navigate_among_pedestrians/crowded.py">🔗</a><br>Sampling-based planning with moving agents</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/multi_robot_2d.gif" width="100%" alt="Multi-robot 2D coordination"><br>
      <sub><b>Multi-robot 2D</b> <a href="examples/single_integrator/multi_robot_coordination.py" title="Source: examples/single_integrator/multi_robot_coordination.py">🔗</a><br>Coordination via shared CBFs</sub>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/pedestrian_head_on.gif" width="100%" alt="Pedestrian head-on"><br>
      <sub><b>Pedestrian head-on</b> <a href="examples/pedestrian/navigate_among_pedestrians/head_on.py" title="Source: examples/pedestrian/navigate_among_pedestrians/head_on.py">🔗</a><br>Dynamic-agent avoidance</sub>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/fixed_wing_3d.gif" width="100%" alt="Fixed-wing aerial 3D"><br>
      <sub><b>Fixed-wing aerial 3D</b> <a href="examples/fixed_wing/reach_drop_point/ekf.py" title="Source: examples/fixed_wing/reach_drop_point/ekf.py">🔗</a><br>UAV reach-drop-point in 3D</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/ekf_estimation.gif" width="100%" alt="EKF state estimation"><br>
      <sub><b>EKF state estimation</b> <a href="examples/unicycle/reach_goal/ekf.py" title="Source: examples/unicycle/reach_goal/ekf.py">🔗</a><br>Unicycle reach-goal under measurement noise</sub>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/van_der_pol_clf.gif" width="100%" alt="Van der Pol CLF"><br>
      <sub><b>Van der Pol (CLF)</b> <a href="examples/van_der_pol/regulation/perfect_sensing.py" title="Source: examples/van_der_pol/regulation/perfect_sensing.py">🔗</a><br>Nonlinear regulation to the origin</sub>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/mpc_double_integrator.gif" width="100%" alt="Model Predictive Control"><br>
      <sub><b>Model Predictive Control</b> <a href="examples/double_integrator/mpc_tracking.py" title="Source: examples/double_integrator/mpc_tracking.py">🔗</a><br>Receding-horizon LTI tracking</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/quadrotor_6dof.gif" width="100%" alt="Quadrotor 6-DOF geometric tracking"><br>
      <sub><b>Quadrotor 6-DOF</b> <a href="examples/quadrotor_6dof/geometric_tracking.py" title="Source: examples/quadrotor_6dof/geometric_tracking.py">🔗</a><br>Geometric SE(3) tracking + altitude CBF</sub>
    </td>
    <td align="center" colspan="2">
      <img src="https://raw.githubusercontent.com/bardhh/cbfkit/main/media/showcase/monte_carlo_safety.gif" width="49%" alt="Monte Carlo safety evaluation"><br>
      <sub><b>Monte Carlo safety evaluation</b> <a href="examples/single_integrator/monte_carlo_safety.py" title="Source: examples/single_integrator/monte_carlo_safety.py">🔗</a><br>200 stochastic CBF rollouts (<code>jax.vmap</code>), live empirical risk</sub>
    </td>
  </tr>
</table>
</details>

*Also in the repository: estimate-feedback risk-aware CBFs, adaptive CVaR-CBF, barrier-activated
controllers, parameter sweeps, quadrotor attitude control, and a 2D Manim animator.*

## Simulation architecture

![cbfkit_architecture](https://github.com/user-attachments/assets/9ca32a8d-4fb5-420d-8742-cb6545a65889)

`cbfkit.simulation.simulator.execute` runs the loop
planner → nominal controller → safety controller → plant → integrator → sensor → estimator.
If the planner returns a **control trajectory**, the nominal controller is skipped and the
safety controller receives it directly. If it returns a **state trajectory**, the nominal
controller converts it to a control input first. Pass `use_jit=True` for the `lax.scan`
path, `plant=MujocoPlant(...)` in place of `dynamics` + `integrator` for MJX, and use
`conduct_monte_carlo` or the `jax.vmap` path for batches.

In the simulator each component is a pure function returning `(output, updated_data)`:

| Component | Signature | Returns |
|-----------|-----------|---------|
| Dynamics | `(x)` | `(f, g)` |
| Nominal controller | `(t, x, key, reference)` | `(u, ControllerData)` |
| Controller (safety filter) | `(t, x, u_nom, key, data)` | `(u, ControllerData)` |
| Planner | `(t, x, u_prev, key, data)` | `(u_traj \| None, PlannerData)` |
| Cost function | `(state, action)` | `cost` |

Legacy controller signatures such as `(t, x)` or `(t, x, u_nom)` are adapted automatically by
`cbfkit.controllers.setup_controller`. The `SafetyFilter` and Gymnasium wrappers hold this
per-step data for you when you run the controller outside the simulator.

## Contributing and project status

Bug reports and feature requests go to the
[issue tracker](https://github.com/bardhh/cbfkit/issues). [`CONTRIBUTING.md`](CONTRIBUTING.md)
describes the development setup, the pre-commit hooks, and what CI checks. Version history is
on the [releases page](https://github.com/bardhh/cbfkit/releases); the package version is read
from `src/cbfkit/VERSION`.

## Citing CBFKit

If you use CBFKit in your research, please cite the [paper](https://arxiv.org/abs/2404.07158):

```bibtex
@misc{black2024cbfkit,
  title={CBFKIT: A Control Barrier Function Toolbox for Robotics Applications},
  author={Mitchell Black and Georgios Fainekos and Bardh Hoxha and Hideki Okamoto and Danil Prokhorov},
  year={2024},
  eprint={2404.07158},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).
