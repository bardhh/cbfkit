# CBFKit: A Control Barrier Function Toolbox for Robotics Applications

CBFKit is a Python/ROS2 toolbox designed to facilitate safe planning and control for robotics applications, particularly in uncertain environments. The toolbox utilizes Control Barrier Functions (CBFs) to provide formal safety guarantees while offering flexibility and ease of use. We additionally provide efficient JAX implementation of Model Predictive Path Integral (MPPI) control with support for reach-avoid specifications.

## Table of Contents
- [Key Features](#key-features)
- [Supported Models](#supported-models)
- [Applications](#applications)
- [Installation](#installation)
- [Start with Tutorials](#tutorials)
- [ROS2](#ros2)
- [Examples](#examples)
- [Citing CBFKit](#citing-cbfkit)
- [License](#license)

## Key Features
- **Generalized Framework:** Supports the design of CBFs for various robotic systems operating in both deterministic and stochastic settings.
- **ROS Integration:** Seamlessly connects with ROS2, enabling multi-robot applications, environment and map encoding, and integration with motion planning algorithms.
- **Diverse CBF Options:** Provides a variety of CBF formulations and algorithms to address different control scenarios.
- **Model-based and Model-free Control:** Accommodates both model-based control strategies using system dynamics and model-free control approaches. Model-free algorithms to be added soon.
- **Safety Guarantee:** CBFs provide mathematically rigorous guarantees of safety by ensuring the system remains within a defined safe operating region.
- **Flexibility:** Allows users to specify custom safety constraints and barrier functions to tailor the control behavior to specific needs.
- **Multi-layer Architecture:** Allows seamless integration of planners, nominal controllers, and safety filter controllers.
- **Efficiency:** Leverages JAX for efficient automatic differentiation and jaxopt for fast quadratic program (QP) solving, enabling real-time control applications. Includes optional **JIT compilation** (`use_jit=True`) for high-performance simulation loops.
- **Code Generation:** Simplifies model creation with automatic code generation for dynamics, controllers, and certificate functions.
- **Usability:** Includes tutorials and examples for a smooth learning curve and rapid prototyping.
- **Functional Programming:** Built on functional programming principles, emphasizing data immutability and programmatic determinism.

## Supported Models
CBFKit accommodates a range of control-affine system models:
- **Deterministic Continuous-time ODEs:** $\dot{x} = f(x) + g(x)u$
- **ODEs with Bounded Disturbances:** $\dot{x} = f(x) + g(x)u + Mw$
- **Stochastic Differential Equations (SDEs):** $dx = (f(x) + g(x)u)dt + \sigma(x)dw$

## Applications
CBFKit can be applied to diverse robotics applications, including:
- **Autonomous Navigation:** Ensure collision avoidance with static and dynamic obstacles.
- **Human-Robot Interaction:** Guarantee safety during collaborative tasks and physical interaction.
- **Manipulation:** Control robot arms while avoiding obstacles and joint limits.
- **Multi-Robot Coordination:** Coordinate the movement of multiple robots while maintaining safe distances and avoiding collisions.

## Installation

### Prerequisites
- **Python 3.10 – 3.12** (3.13+ is not yet supported)

### Local Installation (pip)

1. Clone the repository:
   ```bash
   git clone https://github.com/bardhh/cbfkit.git
   cd cbfkit
   ```

2. Install the package:
   ```bash
   pip install .
   ```

3. **(Optional)** Install extras depending on your use case:
   ```bash
   # Code generation support (needed by most tutorials)
   pip install ".[codegen]"

   # Visualization (matplotlib)
   pip install ".[vis]"

   # CasADi support
   pip install ".[casadi]"

   # Everything needed for development (linting, testing, notebooks, etc.)
   pip install ".[dev]"

   # Or combine several extras at once
   pip install ".[codegen,vis,dev]"
   ```

4. **(Optional)** For an editable/development install so that local changes are reflected immediately:
   ```bash
   pip install -e ".[dev]"
   ```

> **Note:** On Apple Silicon (aarch64/arm64), `kvxopt` is installed in place of `cvxopt`. This is handled automatically.

### Docker

CBFKit is also deployable via Docker. After setting up Docker (refer to the [official Docker documentation](https://docs.docker.com/get-started/) for detailed instructions), proceed with one of the following methods:

#### VS Code Dev Container
1. Open the project in VS Code.
2. When prompted, reopen the folder in container and choose the **CBFKit CPU Dev Container** definition located at `.devcontainer/cbfkit-container`.
3. The container uses the standard `Dockerfile` through Docker Compose so macOS hosts always build the CPU image, while Linux users can optionally select the GPU profile (see below).

#### Docker Compose (command line)
The dev containers are backed by `.devcontainer/docker-compose.yml`, so you can use the same configuration outside of VS Code:

1. Build the CPU image (works on macOS via x86 emulation):
   ```bash
   docker compose -f .devcontainer/docker-compose.yml build cbfkit
   ```
2. Start an interactive development shell:
   ```bash
   docker compose -f .devcontainer/docker-compose.yml run --rm cbfkit bash
   ```
3. When you are done, clean up the container:
   ```bash
   docker compose -f .devcontainer/docker-compose.yml down
   ```

#### GPU Development (Linux hosts only)
The Compose file exposes a `cbfkit_gpu` service behind a `gpu` profile. Linux users with NVIDIA GPUs can enable it with:

```bash
docker compose -f .devcontainer/docker-compose.yml --profile gpu build cbfkit_gpu
docker compose -f .devcontainer/docker-compose.yml --profile gpu run --rm cbfkit_gpu bash
```

macOS builds always target the CPU image because GPU passthrough is not supported.

## Start with Tutorials
Explore the `tutorials` directory to help you get started with CBFKit.

**Note:** Many tutorials utilize the code generation feature of CBFKit. Please ensure you have installed the necessary dependencies by running `pip install "cbfkit[codegen]"` before running them.

For a quick start without additional dependencies, try `tutorials/unicycle_reach_avoid.py`:
```bash
python tutorials/unicycle_reach_avoid.py
```

Open the Python notebook in the `tutorials` directory to get started. The script `code_generation_tutorial.ipynb` automatically generates the controller, plant, and certificate function for a Van der Pol oscillator. It also generates ROS2 nodes for the plant, controller, sensor, and estimator. These serve as a starting point for developing your own CBF-based controller.

Generated files/folders:
```
van_der_pol_oscillator
 ┣ certificate_functions
 ┃ ┣ barrier_functions
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┣ barrier_1.py
 ┃ ┃ ┗ barrier_2.py
 ┃ ┣ lyapunov_functions
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┗ lyapunov_1.py
 ┃ ┗ __init__.py
 ┣ controllers
 ┃ ┣ __init__.py
 ┃ ┣ controller_1.py
 ┃ ┗ zero_controller.py
 ┣ ros2
 ┃ ┣ __init__.py
 ┃ ┣ config.py
 ┃ ┣ controller.py
 ┃ ┣ estimator.py
 ┃ ┣ plant_model.py
 ┃ ┗ sensor.py
 ┣ __init__.py
 ┣ constants.py
 ┣ plant.py
 ┗ run_ros2_nodes.sh
```

We recommend going through the tutorials in the following order to get familiar with the architecture of our library.
- `code_generation_tutorial.ipynb` (requires `cbfkit[codegen]`)
- `multi_robot_coordination.ipynb` (requires `cbfkit[codegen]`)
- `simulate_mppi_cbf.py` (requires `cbfkit[codegen]`)
- `simulate_mppi_stl.py` (requires `cbfkit[codegen]`)



## Simulation Architecture
- Every simulation must define a **planner**, **nominal controller**, and a **controller** where the output of planner is passed to nominal controller and the output of nominal controller is then passed to the controller. The **nominal controller** is expected to designed to generate control input that steers towards a state waypoint. The **controller** is designed to be a filter after nominal controller.
- The planner can return a state or control input trajectory. If the planner returns a control input trajectory, the nominal controller is skipped and the controller is directly employed. If the planner returns a state trajectory, then the nominal controller is called first to convert the desired state into corresponding control input command which is then passed to the controller.

The flowchart below summarizes the architecture

![cbfkit_architecture](https://github.com/user-attachments/assets/9ca32a8d-4fb5-420d-8742-cb6545a65889)

Each function (dynamics, cost, constraint, controller) must follow a specific structure.
- dynamics:
   * Input arguments:  x (state)
   * Return arguments:  f, g (dynamics matrix for affine dynamics x_dot = f(x) + g(x)u)
- nominal controller:
   * Input arguments: t (time), x (state), key (random key), reference (optional reference state)
   * Return arguments: u (control), data (dictionary with the key "u_nom" mapping to designed u )
   * Note: You can use `cbfkit.controllers.setup_nominal_controller` to adapt simpler functions like `f(t, x)` to this signature.
- cost function:
   * Input arguments: state (array), action (array)
   * Return arguments: cost (float)
- planner:
   * Input arguments: t (time), x (state), key (for random number generation), data (dictionary containing necessary information)
   * Return arguments: u (first control input in planned sequence (can be None if planner retutrns state trajectory instead of control trajectory)), data (dictionary containing extra information like found control or state trajectory)
- controller:
   * Input arguments: t (time), x (state), u_nom (nominal control input), key (for random number generation), data (dictionary containing necessary information)
   * Return arguments: u (control input), data (dictionary containing extra information)

The **data** *(python dictionary)* in planners and controllers is designed to cater to needs of different types of comntrollers and planners. For example, CBF-QP does not need to maintain internal state but planners/controllers like MPC or MPPI need to initialize their initial guess with solution from previous time step when implemented in receding horizon fashion. Since we focus on functional programming for computational efficiency, instead of maintaining this internal state, we pass it as input and output arguments. Controllers like CBF-QP need to maintain any internal state and can have empty dictionary whereas MPPI stores its solution trajectory in the dictionary (and received it back at next time step of the simulation). The **data** must therefore be populated appropriately. In case of planners, the control trajectory must be associated with the key *u_traj* and state trajectory must be associated with the key *x_traj*. See `cbf_clf_qp_generator.py` and `mppi_generator.py` files to understand in more detail.

## Examples
Several additional examples of how to use CBFkit to conduct full simulations of arbitrary dynamical systems are provided, including a unicycle robot, fixed-wing aerial vehicle, and more, all of which may be found in the `examples` folder. Executable examples are provided in the subdirectories of `examples`. These scripts demonstrate various control scenarios and can be run directly.

See below for a simplified script to simulate a unicycle robot navigating toward a goal set amidst obstacles using Model Predictive Path Integral (MPPI) control as a planner and a Control Barrier Function (CBF) as a safety filter. A full version of this example with visualization can be found at `examples/unicycle/reach_goal/mppi_cbf_control.py`:

```python
import jax.numpy as jnp
from jax import Array, jit
import cbfkit.controllers.mppi as mppi_planner
import cbfkit.simulation.simulator as sim
from cbfkit.systems.unicycle.models.accel_unicycle import plant
from cbfkit.certificates import concatenate_certificates, rectify_relative_degree
from cbfkit.certificates.barrier_functions import ellipsoidal_barrier_factory
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller as cbf_controller
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.estimators import naive as estimator

# Simulation parameters
tf, dt = 10.0, 0.05
init_state = jnp.array([0.0, 0.0, 0.0, jnp.pi / 4])
desired_state = jnp.array([2.0, 4.0, 0.0, 0.0])
actuation_constraints = jnp.array([100.0, 100.0])

unicycle_dynamics = plant(lam=1.0)

# MPPI Cost Functions
@jit
def stage_cost(state: Array, action: Array) -> Array:
    x, y, xd, yd = state[0], state[1], desired_state[0], desired_state[1]
    return 10.0 * ((x - xd) ** 2 + (y - yd) ** 2)

@jit
def terminal_cost(state: Array, action: Array) -> Array:
    x, y, xd, yd = state[0], state[1], desired_state[0], desired_state[1]
    return 100.0 * ((x - xd) ** 2 + (y - yd) ** 2)

# MPPI Configuration
mppi_args = {
    "robot_state_dim": 4,
    "robot_control_dim": 2,
    "prediction_horizon": 50,
    "num_samples": 500,
    "plot_samples": 30,
    "time_step": dt,
    "use_GPU": False,
    "costs_lambda": 0.1,
    "cost_perturbation": 0.5,
}

# Instantiate MPPI Planner
mppi_local_planner = mppi_planner.vanilla_mppi(
    control_limits=actuation_constraints,
    dynamics_func=unicycle_dynamics,
    trajectory_cost=None,
    stage_cost=stage_cost,
    terminal_cost=terminal_cost,
    mppi_args=mppi_args,
)

# Define Obstacles and Generate Barriers
obstacles, ellipsoids = [(1.0, 2.0, 0.0)], [(0.5, 1.5)]
cbf_factory, _, _ = ellipsoidal_barrier_factory(
    system_position_indices=(0, 1), obstacle_position_indices=(0, 1), ellipsoid_axis_indices=(0, 1)
)

barriers = [
    rectify_relative_degree(
        function=cbf_factory(obs, ell),
        system_dynamics=unicycle_dynamics,
        state_dim=len(init_state),
        form="exponential",
    )(certificate_conditions=zeroing_barriers.linear_class_k(5.0))
    for obs, ell in zip(obstacles, ellipsoids)
]

controller = cbf_controller(
    control_limits=actuation_constraints,
    dynamics_func=unicycle_dynamics,
    barriers=concatenate_certificates(*barriers),
)

# Simulation Execution
x, u, z, p, dkeys, dvals, planner_data, planner_data_keys = sim.execute(
    x0=init_state,
    dt=dt,
    num_steps=int(tf / dt),
    dynamics=unicycle_dynamics,
    integrator=integrator,
    planner=mppi_local_planner,
    nominal_controller=None,
    controller=controller,
    sensor=sensor,
    estimator=estimator,
    filepath="mppi_cbf_results",
    verbose=True,
    planner_data={
        "u_traj": jnp.zeros((mppi_args["prediction_horizon"], mppi_args["robot_control_dim"])),
        "x_traj": jnp.tile(desired_state.reshape(-1, 1), (1, int(tf / dt) + 1)),
        "prev_robustness": None,
    },
    use_jit=True,
)
```


## ROS2
The ROS2 nodes are generated in the `ros2` directory. The nodes are generated for the plant, controller, sensor, and estimator. This support is in the initial stage and will be improved soon.

To run the nodes, execute the following command in the `van_der_pol_oscillator` directory:
```bash
bash run_ros2_nodes.sh
```
The generated nodes interact as follows:
- The `plant_model` node simulates the physical system and publishes the system state.
- The `sensor` node receives the system state and adds noise to simulate real-world measurements.
- The `estimator` node receives the noisy measurements and estimates the true system state.
- The `controller` node receives the estimated state and computes the control input based on the CBF formulation.


## Citing CBFKit
If you use CBFKit in your research, please cite the following [PAPER](https://arxiv.org/abs/2404.07158):
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
CBFKit is distributed under the BSD 3-Clause License. Refer to the `LICENSE` file for detailed terms.
