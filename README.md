# CBFKit: A Control Barrier Function Toolbox for Robotics Applications

CBFKit is a Python/ROS2 toolbox designed to facilitate safe planning and control for robotics applications, particularly in uncertain environments. The toolbox utilizes Control Barrier Functions (CBFs) to provide formal safety guarantees while offering flexibility and ease of use.

## Table of Contents
- [Key Features](#key-features)
- [Supported Models](#supported-models)
- [Applications](#applications)
- [Installation](#installation)
- [Tutorials](#tutorials)
- [ROS2](#ros2)
- [Citing CBFKit](#citing-cbfkit)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Key Features
- **Generalized Framework:** Supports the design of CBFs for various robotic systems operating in both deterministic and stochastic settings.
- **ROS Integration:** Seamlessly connects with ROS2, enabling multi-robot applications, environment and map encoding, and integration with motion planning algorithms.
- **Diverse CBF Options:** Provides a variety of CBF formulations and algorithms to address different control scenarios.
- **Model-based and Model-free Control:** Accommodates both model-based control strategies using system dynamics and model-free control approaches.
- **Safety Guarantee:** CBFs provide mathematically rigorous guarantees of safety by ensuring the system remains within a defined safe operating region.
- **Flexibility:** Allows users to specify custom safety constraints and barrier functions to tailor the control behavior to specific needs.
- **Efficiency:** Leverages JAX for efficient automatic differentiation and jaxopt for fast quadratic program (QP) solving, enabling real-time control applications.
- **Code Generation:** Simplifies model creation with automatic code generation for dynamics, controllers, and certificate functions.
- **Usability:** Includes tutorials and examples for a smooth learning curve and rapid prototyping.

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

### 1. VS Code DevContainer Launch
1. Open the project in VS Code.
2. Click the green button at the bottom right of the window to launch the DevContainer.
3. All necessary components are pre-installed for immediate use.

### 2. Docker Command Line
1. Build the image:
   ```
   docker build -t cbfkit:latest -f Dockerfile.$(uname -m) .
   ```
2. Run the container:
   ```
   docker run -it --name container-name -v .:/workspace cbfkit:latest
   ```

## Examples

We provide several examples of how to use CBFkit to conduct full simulations of arbitrary dynamical systems, including a unicycle robot, fixed-wing aerial vehicle, and more, all of which may be found in the ```examples``` folder. Any file contained within ```examples``` or any of its subdirectories whose name takes the form of ```*main.py``` is an executable example.

See below for the script used to simulate a unicycle robot navigating toward a goal set amidst three ellipsoidal obstacles, which may be found at ```examples/unicycle/vanilla_cbf_start_to_goal_main.py```:

```python
import jax.numpy as jnp

import cbfkit.systems.unicycle.models.olfatisaber2002approximate as unicycle
import cbfkit.simulation.simulator as sim
from cbfkit.controllers.model_based.cbf_clf_controllers import (
    vanilla_cbf_clf_qp_controller as cbf_controller,
)
from cbfkit.controllers.utils.certificate_packager import concatenate_certificates
from cbfkit.controllers.utils.barrier_conditions import zeroing_barriers

# Simulation parameters
tf = 10.0
dt = 0.01
file_path = "examples/unicycle/start_to_goal/results/"

approx_unicycle_dynamics = unicycle.plant(l=1.0)
init_state = jnp.array([0.0, 0.0, jnp.pi / 4])
desired_state = jnp.array([2.0, 4.0, 0])
actuation_constraints = jnp.array([100.0, 100.0])  # Effectively, no control limits

approx_uniycle_nom_controller = unicycle.controllers.proportional_controller(
    dynamics=approx_unicycle_dynamics,
    Kp_pos=1,
    Kp_theta=0.01,
    desired_state=desired_state,
)

obstacles = [
    (1.0, 2.0, 0.0),
    (-1.0, 1.0, 0.0),
    (0.5, -1.0, 0.0),
]
ellipsoids = [
    (0.5, 1.5),
    (1.0, 0.75),
    (0.75, 0.5),
]

barriers = [
    unicycle.certificate_functions.barrier_functions.obstacle_ca(
        certificate_conditions=zeroing_barriers.linear_class_k(2.0),
        obstacle=obs,
        ellipsoid=ell,
    )
    for obs, ell in zip(obstacles, ellipsoids)
]
barrier_packages = concatenate_certificates(*barriers)

controller = cbf_controller(
    control_limits=actuation_constraints,
    nominal_input=approx_uniycle_nom_controller,
    dynamics_func=approx_unicycle_dynamics,
    barriers=barrier_packages,
)

# Simulation imports
from cbfkit.integration import forward_euler as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.estimators import naive as estimator


x, u, z, p, data, data_keys = sim.execute(
    x0=init_state,
    dt=dt,
    num_steps=int(tf / dt),
    dynamics=approx_unicycle_dynamics,
    integrator=integrator,
    controller=controller,
    sensor=sensor,
    estimator=estimator,
    filepath=file_path + "vanilla_cbf_results",
)
```

## Tutorials
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

## ROS2
The ROS2 nodes are generated in the `ros2` directory. The nodes are generated for the plant, controller, sensor, and estimator.

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
If you use CBFKit in your research, please cite the following paper:
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
