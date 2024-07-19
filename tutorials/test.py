from jax import Array, jit
import jax.numpy as jnp
import numpy as np

target_directory = "./tutorials"
model_name = "multi_augmented_single_integrators"
num_robots = 10
INITIAL_STATE = np.zeros(2 * num_robots)
goals = np.zeros(2 * num_robots)
DT = 0.1
TF = 10
N_STEPS = int(TF / DT) + 1
ACTUATION_LIMITS = 100 * jnp.ones(2 * num_robots)

# Set initial states and goals of all the robots
# The robot states are appended in a single vector
for i in range(num_robots):
    theta_disturbance = np.clip(np.random.normal(0, 1.0), -np.pi / 60, np.pi / 60)
    INITIAL_STATE[2 * i] = 2 * np.cos(2 * np.pi * i / num_robots + theta_disturbance)
    INITIAL_STATE[2 * i + 1] = 2 * np.sin(2 * np.pi * i / num_robots + theta_disturbance)
    goals[2 * i] = -2 * np.cos(2 * np.pi * i / num_robots) + 0.1
params = {}

state_constraint_funcs = []
for i in range(num_robots):
    for j in range(i + 1, num_robots):
        state_constraint_funcs.append(f"(x[{2*i}]-x[{2*j}])**2 + (x[{2*i+1}]-x[{2*j+1}])**2 - 1")

params["clf"] = []
lyapunov_functions = []
for i in range(num_robots):
    lyapunov_functions.append(f"(x[{2*i}]-goal[0])**2+(x[{2*i+1}]-goal[1])**2")
    params["clf"].append(
        {
            "goal: float": goals[2 * i : 2 * i + 2],
        }
    )

# Provides access to execute (sim.execute)
import cbfkit.simulation.simulator as sim

# Access to CBF-CLF-QP control law
import cbfkit.controllers_and_planners.model_based.cbf_clf_controllers as cbf_clf_controllers

# Necessary housekeeping for using multiple CBFs/CLFs
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.certificate_packager import (
    concatenate_certificates,
)

# Suite of zeroing barrier function derivative conditions (forms of Class K functions)
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.barrier_conditions import (
    zeroing_barriers,
)

# Exponentially stable derivative condition for CLF
from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.lyapunov_conditions.exponential_stability import (
    e_s,
)

# Assuming perfect, complete state information
from cbfkit.sensors import perfect as sensor

# With perfect sensing, we can use a naive estimate of the state
from cbfkit.estimators import naive as estimator

# Use forward-Euler numerical integration scheme
from cbfkit.utils.numerical_integration import forward_euler as integrator

from tutorials import multi_augmented_single_integrators

# Simulation Parameters
SAVE_FILE = f"tutorials/{model_name}/simulation_data"  # automatically uses .csv format

bs = []
for i in range(len(state_constraint_funcs)):
    func_name = (
        "multi_augmented_single_integrators.certificate_functions.barrier_functions."
        + "cbf"
        + str(i + 1)
        + "_package"
    )
    exec(f"func={func_name}")
    bs.append(func(certificate_conditions=zeroing_barriers.linear_class_k(alpha=1.0)))
barriers = concatenate_certificates(*bs)

ls = []
for i in range(len(lyapunov_functions)):
    func_name = (
        "multi_augmented_single_integrators.certificate_functions.lyapunov_functions."
        + "clf"
        + str(i + 1)
        + "_package"
    )
    exec(f"func={func_name}")
    # func = getattr(__main__, func_name)
    ls.append(
        func(
            certificate_conditions=e_s(c=2.0),
            goal=goals[2 * i : 2 * i + 2],
        )
    )
lyapunov = concatenate_certificates(*ls)

dynamics = multi_augmented_single_integrators.plant()

nominal_controller = multi_augmented_single_integrators.controllers.controller_1(goal=goals, k_p=1)

cbf_clf_controller = cbf_clf_controllers.vanilla_cbf_clf_qp_controller(
    control_limits=ACTUATION_LIMITS,
    nominal_input=nominal_controller,
    dynamics_func=dynamics,
    barriers=barriers,
    lyapunovs=lyapunov,
    relaxable_clf=True,
)

x, _u, _z, _p, dkeys, dvalues = sim.execute(
    x0=INITIAL_STATE,
    dt=DT,
    num_steps=N_STEPS,
    dynamics=dynamics,
    integrator=integrator,
    controller=cbf_clf_controller,
    sensor=sensor,
    estimator=estimator,  # perturbation=generate_stochastic_perturbation(sigma=sigma, dt=DT),
    filepath=SAVE_FILE,
)
