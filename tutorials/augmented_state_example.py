from jax import Array, jit
import jax.numpy as jnp

from cbfkit.codegen.create_new_system import generate_model

drift_dynamics = " [0, 0, 0, 0] "  #   "[x[1], -x[0] + epsilon * (1 - x[0]**2) * x[1]]"
control_matrix = "[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]"

target_directory = "./tutorials"
model_name = "two_augmented_single_integrators"

params = {}  # {"dynamics": {"epsilon: float": 0.5}}

generate_model.generate_model(
    directory=target_directory,
    model_name=model_name,
    drift_dynamics=drift_dynamics,
    control_matrix=control_matrix,
    params=params,
)

goal1_x = 1.0
goal1_y = 1.0
goal2_x = -1.0
goal2_y = -1.0
# nominal_control_law = " [ [-k_p * (x[0]-goal1_x)], [-k_p * (x[1]-goal1_y)], [-k_p * (x[2]-goal2_x)], [-k_p * (x[3]-goal2_y)] ] "  # "x[0] * (1 - k_p) - epsilon * (1 - x[0]**2) * x[1]"
nominal_control_law = " [ -k_p * (x[0]-goal1_x), -k_p * (x[1]-goal1_y), -k_p * (x[2]-goal2_x), -k_p * (x[3]-goal2_y) ] "  # "x[0] * (1 - k_p) - epsilon * (1 - x[0]**2) * x[1]"
params["controller"] = {
    "goal1_x: float": goal1_x,
    "goal1_y: float": goal1_y,
    "goal2_x: float": goal2_x,
    "goal2_y: float": goal2_y,
    "k_p: float": 1.0,
}
# {"k_p: float": 1.0, "epsilon: float": 0.5}

generate_model.generate_model(
    directory=target_directory,
    model_name=model_name,
    drift_dynamics=drift_dynamics,
    control_matrix=control_matrix,
    nominal_controller=nominal_control_law,
    params=params,
)

state_constraint_funcs = [
    "(x[0]-x[2])*(x[0]-x[2]) + (x[1]-x[3])*(x[1]-x[3]) - 1"
]  # ["5 - x[0]", "x[0] + 7"]

lyapunov_functions = " (x[0]-goal1_x)**2 + (x[1]-goal1_y)**2 + (x[2]-goal2_x) + (x[3]-goal2_y)**2"  # "x[0]**2 + x[1]**2 - radius"
# params["clf"] = [{"radius: float": 1.0}]
params["clf"] = [
    {
        "goal1_x: float": goal1_x,
        "goal1_y: float": goal1_y,
        "goal2_x: float": goal2_x,
        "goal2_y: float": goal2_y,
    }
]

generate_model.generate_model(
    directory=target_directory,
    model_name=model_name,
    drift_dynamics=drift_dynamics,
    control_matrix=control_matrix,
    barrier_funcs=state_constraint_funcs,
    lyapunov_funcs=lyapunov_functions,
    nominal_controller=nominal_control_law,
    params=params,
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

# To add stochastic perturbation to system dynamics
# from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation
# sigma = lambda x: jnp.array([[0, 0], [0, 0.05 * x[0]]])  # State-dependent diffusion term in SDE

from tutorials import two_augmented_single_integrators

# Simulation Parameters
SAVE_FILE = f"tutorials/{model_name}/simulation_data"  # automatically uses .csv format
DT = 0.1  # 1e-2
TF = 10  # 4  # 10.0
N_STEPS = int(TF / DT) + 1
INITIAL_STATE = jnp.array([-0.9, -0.9, 0.9, 0.9])
ACTUATION_LIMITS = jnp.array(
    [100.0, 100.0, 100.0, 100.0]
)  # Box control input constraint, i.e., -100 <= u <= 100

# Dynamics function with epsilon parameter: returns f(x), g(x), d(x)
eps = 0.5
dynamics = (
    two_augmented_single_integrators.plant()
)  # epsilon=eps, perturbation=generate_stochastic_perturbation(sigma, DT))

#! To do: separate box explaining
# Create barrier functions with linear class K function derivative conditions
b1 = two_augmented_single_integrators.certificate_functions.barrier_functions.cbf1_package(
    certificate_conditions=zeroing_barriers.linear_class_k(alpha=1.0),
)
# b2 = two_augmented_single_integrators.certificate_functions.barrier_functions.cbf2_package(
#     certificate_conditions=zeroing_barriers.linear_class_k(alpha=1.0),
# )
barriers = concatenate_certificates(b1)  # , b2)

#! To do: separate box explaining
# Create lyapunov function with exponential stability derivative condition
l1 = two_augmented_single_integrators.certificate_functions.lyapunov_functions.clf1_package(
    certificate_conditions=e_s(c=2.0),
    goal1_x=goal1_x,
    goal1_y=goal1_y,
    goal2_x=goal2_x,
    goal2_y=goal2_y,
)
lyapunov = concatenate_certificates(l1)

# Instantiate nominal controller
nominal_controller = two_augmented_single_integrators.controllers.controller_1(
    goal1_x=goal1_x, goal1_y=goal1_y, goal2_x=goal2_x, goal2_y=goal2_y, k_p=1
)  # (k_p=1.0, epsilon=eps)
# params["controller"] = {"goal1_x: float": goal1_x, "goal1_y: float": goal1_y, "goal2_x: float": goal2_x, "goal2_y: float": goal2_y, "k_p: float": 1.0}
# Instantiate CBF-CLF-QP control law
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

plot = True
if plot:
    from tutorials.plot_two_single_integrators import animate

    animate(
        states=x,
        estimates=_z,
        desired_state=jnp.array(
            [goal1_x, goal1_y, goal2_x, goal2_y]
        ),  # initial_conditions.desired_state,
        desired_state_radius=0.1,
        x_lim=(-5, 5),
        y_lim=(-5, 5),
        dt=DT,
        title="System Behavior",
        save_animation=True,
        animation_filename=target_directory + model_name + "_animation" + ".gif",
    )
