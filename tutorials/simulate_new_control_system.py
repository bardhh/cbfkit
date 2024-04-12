from jax import Array, jit
import jax.numpy as jnp
from cbfkit.codegen.create_new_system import generate_model

drift_dynamics = "[x[1], -x[0] + epsilon * (1 - x[0]**2) * x[1]]"
control_matrix = "[[0], [1]]"
target_directory = "./tutorials"
model_name = "van_der_pol_oscillator"
params = {"dynamics": {"epsilon: float": 0.5}}
generate_model.generate_model(
    directory=target_directory,
    model_name=model_name,
    drift_dynamics=drift_dynamics,
    control_matrix=control_matrix,
    params=params,
)
nominal_control_law = "x[0] * (1 - k_p) - epsilon * (1 - x[0]**2) * x[1]"
params["controller"] = {"k_p: float": 1.0, "epsilon: float": 0.5}
generate_model.generate_model(
    directory=target_directory,
    model_name=model_name,
    drift_dynamics=drift_dynamics,
    control_matrix=control_matrix,
    nominal_controller=nominal_control_law,
    params=params,
)
state_constraint_funcs = ["5 - x[0]", "x[0] + 7"]
lyapunov_functions = "x[0]**2 + x[1]**2 - radius"
params["clf"] = [{"radius: float": 1.0}]
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
from cbfkit.controllers.model_based.cbf_clf_controllers.vanilla_cbf_clf_qp_control_laws import (
    vanilla_cbf_clf_qp_controller,
)

# Necessary housekeeping for using multiple CBFs/CLFs
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.certificate_packager import (
    concatenate_certificates,
)

# Suite of zeroing barrier function derivative conditions (forms of Class K functions)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.barrier_conditions import (
    zeroing_barriers,
)

# Exponentially stable derivative condition for CLF
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.lyapunov_conditions.exponential_stability import (
    e_s,
)

# Assuming perfect, complete state information
from cbfkit.sensors import perfect as sensor

# With perfect sensing, we can use a naive estimate of the state
from cbfkit.estimators import naive as estimator

# Use forward-Euler numerical integration scheme
from cbfkit.utils.numerical_integration import forward_euler as integrator

# To add stochastic perturbation to system dynamics
from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation


@jit
def sigma(x):
    return jnp.array([[0, 0], [0, 0.05 * x[0]]])  # State-dependent diffusion term in SDE


from tutorials import van_der_pol_oscillator

# Simulation Parameters
SAVE_FILE = f"tutorials/{model_name}/simulation_data"
DT = 1e-2
TF = 10.0
N_STEPS = int(TF / DT) + 1
INITIAL_STATE = jnp.array([1.5, 0.25])
ACTUATION_LIMITS = jnp.array([100.0])  # Box control input constraint, i.e., -100 <= u <= 100
# Dynamics function with epsilon parameter: returns f(x), g(x), d(x)
eps = 0.5
dynamics = van_der_pol_oscillator.plant(
    epsilon=eps,
)

#! To do: separate box explaining
# Create barrier functions with linear class K function derivative conditions
b1 = van_der_pol_oscillator.certificate_functions.barrier_functions.cbf1_package(
    certificate_conditions=zeroing_barriers.linear_class_k(alpha=1.0),
)
b2 = van_der_pol_oscillator.certificate_functions.barrier_functions.cbf2_package(
    certificate_conditions=zeroing_barriers.linear_class_k(alpha=1.0),
)
barriers = concatenate_certificates(b1, b2)

#! To do: separate box explaining
# Create lyapunov function with exponential stability derivative condition
l1 = van_der_pol_oscillator.certificate_functions.lyapunov_functions.clf1_package(
    certificate_conditions=e_s(c=2.0),
    radius=1.0,
)
lyapunov = concatenate_certificates(l1)

# Instantiate nominal controller
nominal_controller = van_der_pol_oscillator.controllers.controller_1(k_p=1.0, epsilon=eps)

# Instantiate CBF-CLF-QP control law
cbf_clf_controller = vanilla_cbf_clf_qp_controller(
    control_limits=ACTUATION_LIMITS,
    nominal_input=nominal_controller,
    dynamics_func=dynamics,
    barriers=barriers,
    lyapunovs=lyapunov,
    relaxable_clf=True,
)
sim.execute(
    x0=INITIAL_STATE,
    dt=DT,
    num_steps=N_STEPS,
    dynamics=dynamics,
    perturbation=generate_stochastic_perturbation(sigma=sigma, dt=DT),
    integrator=integrator,
    controller=cbf_clf_controller,
    sensor=sensor,
    estimator=estimator,
    filepath=SAVE_FILE,
)
