"""Tutorial: Single integrator navigation among dynamic obstacles."""
#####################################################################
#####################################################################
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp

from cbfkit.codegen.create_new_system.generate_model import generate_model


def compute_theta_d(x, y, th):
    """Computes the desired heading angle relative to the current angle.

    Args:
        x: Current x position
        y: Current y position
        th: Current heading angle

    Returns:
        Desired heading angle expression
    """
    thd = f"arctan2(yg - {y}, xg - {x})"
    return f"{th} + arctan2(sin({thd} - {th}), cos({thd} - {th}))"


def norm(x, y):
    """Computes Euclidean distance to goal.

    Args:
        x: Current x position
        y: Current y position

    Returns:
        Distance expression
    """
    z = f"jnp.linalg.norm(jnp.array([{x} - xg, {y} - yg]))"
    return z


params = {}

# State indexing
x, y = "x[0]", "x[1]"  # Vehicle position
ox1, oy1 = "x[2]", "x[3]"  # Obstacle 1 position
ox2, oy2 = "x[4]", "x[5]"  # Obstacle 2 position
ox3, oy3 = "x[6]", "x[7]"  # Obstacle 3 position
ox4, oy4 = "x[8]", "x[9]"  # Obstacle 4 position
ox5, oy5 = "x[10]", "x[11]"  # Obstacle 5 position

# Drift dynamics: different behaviors for each obstacle
drift = [
    "0",  # vehicle x
    "0",  # vehicle y
    "-0.5",  # obstacle 1 x (constant velocity left)
    "0",  # obstacle 1 y
    "0.3",  # obstacle 2 x (constant velocity right)
    "0.3",  # obstacle 2 y (diagonal motion)
    "0.2",  # obstacle 3 x (constant velocity right)
    "0.2",  # obstacle 3 y (constant velocity up)
    "-0.4",  # obstacle 4 x (faster leftward motion)
    "-0.2",  # obstacle 4 y (constant downward motion)
    "0",  # obstacle 5 x (stationary)
    "0",  # obstacle 5 y
]

# Control matrix: extend for all obstacles
control_mat = [
    "[1, 0]",  # x control
    "[0, 1]",  # y control
] + [
    "[0, 0]"
] * 10  # no control for obstacles

# Barrier functions for all obstacles
barriers = [
    f"({x} - {ox1})**2 + ({y} - {oy1})**2 - r**2",
    f"({x} - {ox2})**2 + ({y} - {oy2})**2 - r**2",
    f"({x} - {ox3})**2 + ({y} - {oy3})**2 - r**2",
    f"({x} - {ox4})**2 + ({y} - {oy4})**2 - r**2",
    f"({x} - {ox5})**2 + ({y} - {oy5})**2 - r**2",
]
params["cbf"] = [{"r: float": 0.5}] * 5  # safety radius for each obstacle

# Nominal controller (proportional control to goal)
u_nom = f"-kp * ({x} - xg), -kp * ({y} - yg)"
params["controller"] = {"kp: float": 1.0, "xg: float": 1.0, "yg: float": 1.0}


# Generate the model from the provided dynamics and parameters:
generate_model(
    directory="./tutorials/models",
    model_name="single_integrator",
    drift_dynamics=drift,
    control_matrix=control_mat,
    barrier_funcs=barriers,
    nominal_controller=u_nom,
    params=params,
)

#####################################################################
#####################################################################

import models.single_integrator as unicycle
from models.single_integrator.certificate_functions.barrier_functions.barrier_1 import cbf

# Update initial state (reduced dimension)
# Update initial state for all obstacles
# Update initial state to include time
initial_state = jnp.array(
    [
        2.0,
        2.0,  # vehicle at (2,2)
        2.0,
        1.0,  # obstacle 1
        -1.0,
        -1.0,  # obstacle 2
        0.0,
        0.0,  # obstacle 3
        3.0,
        -1.0,  # obstacle 4
        1.0,
        0.0,  # obstacle 5
    ]
)

# Update actuation limits for 2D control
actuation_limits = jnp.array([1.0, 1.0])


# Load the plant dynamics and create a nominal controller aiming at goal (-2,-2)
dynamics = unicycle.plant()
nominal_controller = unicycle.controllers.controller_1(kp=1.0, xg=-2.0, yg=-2.0)

# After model generation, import the generated barrier functions
from models.single_integrator.certificate_functions.barrier_functions.barrier_1 import (
    cbf as barrier_1,
)
from models.single_integrator.certificate_functions.barrier_functions.barrier_2 import (
    cbf as barrier_2,
)
from models.single_integrator.certificate_functions.barrier_functions.barrier_3 import (
    cbf as barrier_3,
)
from models.single_integrator.certificate_functions.barrier_functions.barrier_4 import (
    cbf as barrier_4,
)
from models.single_integrator.certificate_functions.barrier_functions.barrier_5 import (
    cbf as barrier_5,
)

#####################################################################
#####################################################################
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import (
    cubic_class_k,
    linear_class_k,
)
from cbfkit.certificates import (
    concatenate_certificates,
)
from cbfkit.certificates import (
    rectify_relative_degree,
)

# Update the barrier certificates
cbf_packages = [
    rectify_relative_degree(
        [barrier_1, barrier_2, barrier_3, barrier_4, barrier_5][i](r=1.0),
        dynamics,
        len(initial_state),
        roots=-1.0 * jnp.ones((2,)),
    )
    for i in range(5)
]

# Combine all barrier certificates
barriers = concatenate_certificates(
    *[pkg(certificate_conditions=linear_class_k(5)) for pkg in cbf_packages]
)

# Construct the CBF-CLF-QP controller
# Update controller setup
controller = vanilla_cbf_clf_qp_controller(
    actuation_limits,
    dynamics,
    barriers,
    p_mat=jnp.eye(2),  # Simplified p_mat
)
#####################################################################
#####################################################################

from cbfkit.estimators import naive
from cbfkit.integration import runge_kutta_4
from cbfkit.sensors import perfect as sensor

# change the sensors for noisy data
# from cbfkit.sensors import unbiased_gaussian_noise as sensor
from cbfkit.simulation import simulator

# Simulate the closed-loop system
num_steps = 1000 if not os.getenv("CBFKIT_TEST_MODE") else 100
x, u, z, p, dkeys, dvals, _, _ = simulator.execute(
    x0=initial_state,
    dt=1e-2,
    num_steps=num_steps,
    dynamics=dynamics,
    integrator=runge_kutta_4,
    nominal_controller=nominal_controller,
    controller=controller,
    sensor=sensor,
    estimator=naive,
)

#####################################################################
#####################################################################

if os.getenv("CBFKIT_TEST_MODE"):
    sys.exit(0)

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from cbfkit.utils.animator import AnimationConfig, CBFAnimator

# Convert states to numpy for CBFAnimator
states_np = np.array(x)

# Compute axis limits from agent and obstacle trajectories
all_x = np.concatenate([states_np[:, 2 * i] for i in range(6)])
all_y = np.concatenate([states_np[:, 2 * i + 1] for i in range(6)])

animator = CBFAnimator(
    states_np,
    dt=1e-2,
    x_lim=(float(all_x.min()) - 1, float(all_x.max()) + 1),
    y_lim=(float(all_y.min()) - 1, float(all_y.max()) + 1),
    title="Single Integrator Trajectory with Moving Obstacle",
    config=AnimationConfig(blit=False, interval=20),
)
animator.add_goal((-2.0, -2.0), radius=0.25, color="g", label="Goal")
animator.add_trajectory(x_idx=0, y_idx=1, color="b", label="Trajectory", linewidth=2)

# Build to get axes for custom patches
fig, ax = animator.build()

# Agent point marker
(point,) = ax.plot([], [], "bo", markersize=6, label="Agent", zorder=3)

# Obstacle circles (dynamic — updated per frame)
circles = [
    patches.Circle(
        (states_np[0, 2 * i + 2], states_np[0, 2 * i + 3]),
        radius=0.5,
        edgecolor="r",
        facecolor=plt.cm.Set3(i / 5),
        alpha=0.5,
        label=f"Obstacle {i + 1}",
        zorder=1,
    )
    for i in range(5)
]
for circle in circles:
    ax.add_patch(circle)
ax.legend()


def custom_update(frame, _ax):
    """Update agent point and moving obstacle circles each frame."""
    point.set_data([states_np[frame, 0]], [states_np[frame, 1]])
    for i, circle in enumerate(circles):
        circle.center = (states_np[frame, 2 * i + 2], states_np[frame, 2 * i + 3])
    return [point, *circles]


animator.on_frame(custom_update)
animator.save("trajectory_animation")

# Show the final figure
plt.show()

# Plot the control inputs over time
fig2, ax2 = plt.subplots()
time_axis = jnp.linspace(0.0, 5.0, len(u[:, 0])).tolist()
ax2.plot(time_axis, u[:, 0], label="X Velocity")
ax2.plot(time_axis, u[:, 1], label="Y Velocity")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Control Input")
ax2.set_title("Control Inputs Over Time")
ax2.grid(True)
ax2.legend()
fig2.savefig("control_inputs.png")
plt.show()
