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
CBFKit is readily deployable via a Docker image. After setting up Docker (refer to the [official Docker documentation](https://docs.docker.com/get-started/) for detailed instructions), proceed with one of the following methods:

### 1. VS Code Dev Container Launch
1. Open the project in VS Code.
2. When prompted, reopen the folder in container and choose the **CBFKit CPU Dev Container** definition located at `.devcontainer/cbfkit-container`.
3. The container uses the standard `Dockerfile` through Docker Compose so macOS hosts always build the CPU image, while Linux users can optionally select the GPU profile (see below).

### 2. Docker Compose (command line)
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

### 3. GPU Development (Linux hosts only)
The Compose file exposes a `cbfkit_gpu` service behind a `gpu` profile. Linux users with NVIDIA GPUs can enable it with:

```bash
docker compose -f .devcontainer/docker-compose.yml --profile gpu build cbfkit_gpu
docker compose -f .devcontainer/docker-compose.yml --profile gpu run --rm cbfkit_gpu bash
```

macOS builds always target the CPU image because GPU passthrough is not supported.

## Start with Tutorials
Explore the `tutorials` directory to help you get started with CBFKit. Open the Python notebook in the `tutorials` directory to get started. The script `simulate_new_control_system.ipynb` automatically generates the controller, plant, and certificate function for a Van der Pol oscillator. It also generates ROS2 nodes for the plant, controller, sensor, and estimator. These serve as a starting point for developing your own CBF-based controller.

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
- `simulate_new_control_system.ipynb`
- `multi_robot_example.ipynb`
- `simulate_mppi_cbf.py`
- `simulate_mppi_cbf_ellipsoidal_stochastic_cbf.py`
- `simulate_mppi_stl.py`



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
   * Input arguments: t (time), x (state)
   * Return arguments: u (control), data (dictionary with the key "u_nom" mapping to designed u )
- cost function:
   * Input arguments: state_and_time (concatenated state and time in 1D array)
   * Return arguments: cost (float)
- planner:
   * Input arguments: t (time), x (state), key (for random number generation), data (dictionary containing necessary information)
   * Return arguments: u (first control input in planned sequence (can be None if planner retutrns state trajectory instead of control trajectory)), data (dictionary containing extra information like found control or state trajectory)
- controller:
   * Input arguments: t (time), x (state), u_nom (nominal control input), key (for random number generation), data (dictionary containing necessary information)
   * Return arguments: u (control input), data (dictionary containing extra information)

The **data** *(python dictionary)* in planners and controllers is designed to cater to needs of different types of comntrollers and planners. For example, CBF-QP does not need to maintain internal state but planners/controllers like MPC or MPPI need to initialize their initial guess with solution from previous time step when implemented in receding horizon fashion. Since we focus on functional programming for computational efficiency, instead of maintaining this internal state, we pass it as input and output arguments. Controllers like CBF-QP need to maintain any internal state and can have empty dictionary whereas MPPI stores its solution trajectory in the dictionary (and received it back at next time step of the simulation). The **data** must therefore be populated appropriately. In case of planners, the control trajectory must be associated with the key *u_traj* and state trajectory must be associated with the key *x_traj*. See `cbf_clf_qp_generator.py` and `mppi_generator.py` files to understand in more detail.

## Examples
Several additional examples of how to use CBFkit to conduct full simulations of arbitrary dynamical systems are provided, including a unicycle robot, fixed-wing aerial vehicle, and more, all of which may be found in the `examples` folder. Any file contained within `examples` or any of its subdirectories whose name takes the form of `*main.py` is an executable example that may be referenced when a user is building their own application.

See below for the script used to simulate a unicycle robot navigating toward a goal set amidst three ellipsoidal obstacles using Model Predictive Path Integral (MPPI) control as a planner and a Control Barrier Function (CBF) as a safety filter. This file can be found at `examples/unicycle/start_to_goal/mppi_cbf_start_to_goal_main.py`:

```python
import jax.numpy as jnp
from jax import Array, jit

import cbfkit.controllers.mppi as mppi_planner
import cbfkit.simulation.simulator as sim
import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
from cbfkit.certificates import concatenate_certificates, rectify_relative_degree
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller as cbf_controller

# Simulation parameters
tf = 10.0
dt = 0.05
file_path = "examples/unicycle/start_to_goal/results/"

init_state = jnp.array([0.0, 0.0, 0.0, jnp.pi / 4])
desired_state = jnp.array([2.0, 4.0, 0.0, 0.0])
actuation_constraints = jnp.array([100.0, 100.0])  # Effectively, no control limits

unicycle_dynamics = unicycle.plant(l=1.0)
# ... (dynamics configuration) ...

# MPPI Cost Functions
@jit
def stage_cost(state_and_time: Array, action: Array) -> Array:
    x, y = state_and_time[0], state_and_time[1]
    xd, yd = desired_state[0], desired_state[1]
    dist_sq = (x - xd) ** 2 + (y - yd) ** 2
    return 10.0 * dist_sq

@jit
def terminal_cost(state_and_time: Array, action: Array) -> Array:
    x, y = state_and_time[0], state_and_time[1]
    xd, yd = desired_state[0], desired_state[1]
    dist_sq = (x - xd) ** 2 + (y - yd) ** 2
    return 100.0 * dist_sq

# MPPI Configuration
mppi_args = {
    "robot_state_dim": 4,
    "robot_control_dim": 2,
    "prediction_horizon": 50,
    "num_samples": 500,
    "plot_samples": 30,
    "time_step": dt,
    "use_GPU": True,
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

# ... (Barrier function setup) ...

controller = cbf_controller(
    control_limits=actuation_constraints,
    dynamics_func=unicycle_dynamics,
    barriers=barrier_packages,
)

# Simulation Execution (with JIT)
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
    filepath=file_path + "mppi_cbf_results",
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
