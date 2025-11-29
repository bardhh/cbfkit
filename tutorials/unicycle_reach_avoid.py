from typing import Any, Dict

#####################################################################
#####################################################################
import jax.numpy as jnp
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import tutorials.models.accel_unicycle as unicycle
from cbfkit.certificates import concatenate_certificates, rectify_relative_degree
from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
from cbfkit.codegen.create_new_system.generate_model import generate_model
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.estimators import naive
from cbfkit.integration import forward_euler
from cbfkit.sensors import perfect
from cbfkit.simulation import simulator
from cbfkit.utils.user_types import PlannerData
from tutorials.models.accel_unicycle.certificate_functions.barrier_functions.barrier_1 import (
    cbf,
)
from tutorials.models.accel_unicycle.certificate_functions.barrier_functions.barrier_2 import (
    cbf2_package,
)


def compute_theta_d(x, y, th):
    thd = f"arctan2(yg - {y}, xg - {x})"
    return f"{th} + arctan2(sin({thd} - {th}), cos({thd} - {th}))"


def norm(x, y):
    z = f"linalg.norm(array([{x} - xg, {y} - yg]))"
    return z


params: Dict[str, Any] = {}
x, y, v, th = "x[0]", "x[1]", "x[2]", "x[3]"
drift = [f"{v} * cos({th})", f"{v} * sin({th})", "0", "0"]
control_mat = ["[0, 0]", "[0, 0]", "[1, 0]", "[0, 1]"]
barriers = [f"({x} - xo)**2 + ({y} - yo)**2 - r**2", f"l**2 - {v}**2"]
params["cbf"] = [{"xo: float": 1.0, "yo: float": 1.0, "r: float": 1.0}, {"l: float": 1.0}]
u_nom = f"jnp.array([kp * ({norm(x ,y)} - {v}), kp * ({compute_theta_d(x, y, th)} - {th})])"
params["controller"] = {"kp: float": 1.0, "xg: float": 1.0, "yg: float": 1.0}

generate_model(
    directory="./tutorials/models",
    model_name="accel_unicycle",
    drift_dynamics=drift,
    control_matrix=control_mat,
    barrier_funcs=barriers,
    nominal_controller=u_nom,
    params=params,
)

#####################################################################
#####################################################################

initial_state = jnp.array([2.0, 2.0, 0.0, -3 * jnp.pi / 4])
actuation_limits = jnp.array([1.0, jnp.pi])
dynamics = unicycle.plant()
nominal_controller_func = unicycle.controllers.controller_1(kp=1.0, xg=-2.0, yg=-2.0)


#####################################################################
#####################################################################

cbf1_package = rectify_relative_degree(
    cbf(xo=0.9, yo=1.0, r=0.5), dynamics, len(initial_state), roots=-1.0 * jnp.ones((2,))
)

certificate_collection = concatenate_certificates(
    cbf1_package(certificate_conditions=linear_class_k(10.0)),
    cbf2_package(certificate_conditions=linear_class_k(1.0), l=1.0),
)
controller = vanilla_cbf_clf_qp_controller(
    control_limits=actuation_limits,
    dynamics_func=dynamics,
    barriers=certificate_collection,
    p_mat=jnp.diag(jnp.array([1.0, 0.1])),
)

#####################################################################
#####################################################################

x_sim, u_sim, z_sim, p_sim, dkeys, dvals, planner_data, planner_data_keys = simulator.execute(
    x0=initial_state,
    dt=1e-2,
    num_steps=1000,
    dynamics=dynamics,
    integrator=forward_euler,
    nominal_controller=nominal_controller_func,
    controller=controller,
    sensor=perfect,
    estimator=naive,
    planner_data=PlannerData(x_traj=jnp.zeros((4, 1001))),
    use_jit=True,
)

################################################0#####################
#####################################################################

# Create a figure and axis
fig, ax = plt.subplots()

# Add a circle patch to the plot
circle1 = patches.Circle((0.9, 1.0), radius=0.5, edgecolor="r", facecolor="k")

# Add the circle patch to the axis
ax.add_patch(circle1)

ax.plot(x_sim[:, 0], x_sim[:, 1])
# plt.show()

plt.savefig("trajectory_plot.png")

plt.plot(jnp.linspace(0.0, 5.0, len(u_sim[:, 0])), u_sim[:, 0])
plt.plot(jnp.linspace(0.0, 5.0, len(u_sim[:, 1])), u_sim[:, 1])

# plt.show()
