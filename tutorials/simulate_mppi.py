from jax import Array, jit
import jax.numpy as jnp
from cbfkit.codegen.create_new_system import generate_mppi_model

# Simulation Parameters
target_directory = "./tutorials"
model_name = "mppi_si"
plot = True
SAVE_FILE = f"tutorials/{model_name}/simulation_data"
DT = 0.05  # 1e-2
TF = 10  # 10.0
N_STEPS = int(TF / DT) + 1
INITIAL_STATE = jnp.array([1.5, 0.25])
ACTUATION_LIMITS = jnp.array([5.0, 5.0])  # Box control input constraint, i.e., -100 <= u <= 100

# Initialize params dict for all dynamics and constraint function related parameters
params = {}

# Define  Dynamics
drift_dynamics = "[0, 0]"
control_matrix = "[[1, 0], [0, 1]]"

# Define stage and terminal cost functions
goal = jnp.array([7, 7])  # .reshape(-1, 1)
obstacle = jnp.array([5, 5])
obstacle_radius = 0.6

# MPPI stage and terminal cost functions: low cost when close to the goal and far away from the obstacles
stage_cost_function = "(x[0]-goal[0])**2 + (x[1]-goal[1])**2 + 10.0/(linalg.norm(x[0:2]-obstacle[0:2])-obstacle_radius)"
terminal_cost_function = "(x[0]-goal[0])**2 + (x[1]-goal[1])**2 + 10.0/max(array([(linalg.norm(x[0:2]-obstacle[0:2])-obstacle_radius),0.01]))"

params["stage_cost_function"] = {
    "goal: float": goal,
    "obstacle: float": obstacle,
    "obstacle_radius: float": obstacle_radius,
}
params["terminal_cost_function"] = {
    "goal: float": goal,
    "obstacle: float": obstacle,
    "obstacle_radius: float": obstacle_radius,
}

# Run script for automated genration of dynamics, cost functions python files
generate_mppi_model.generate_model(
    directory=target_directory,
    model_name=model_name,
    drift_dynamics=drift_dynamics,
    control_matrix=control_matrix,
    stage_cost_function=stage_cost_function,
    terminal_cost_function=terminal_cost_function,
    params=params,
)


# Provides access to execute (sim.execute)
import cbfkit.simulation.simulator as sim
# Access to MPPI control law
from cbfkit.controllers_and_planners.model_based.mppi.vanilla_mppi_laws import (
    vanilla_mppi_controller,
)
# Assuming perfect, complete state information
from cbfkit.sensors import perfect as sensor
# With perfect sensing, we can use a naive estimate of the state
from cbfkit.estimators import naive as estimator
# Use forward-Euler numerical integration scheme
from cbfkit.utils.numerical_integration import forward_euler as integrator
from tutorials import mppi_si



mppi_args = {
    "robot_state_dim": 2,
    "robot_control_dim": 2,
    "prediction_horizon": 80,
    "num_samples": 10000,
    "plot_samples": 30,
    "time_step": DT,
    "use_GPU": False,
    "costs_lambda": 0.03,
    "cost_perturbation": 0.1,
}

# Dynamics function with epsilon parameter: returns f(x), g(x), d(x)
# Load files generated with automated scripts before
eps = 0.5
dynamics = mppi_si.plant(
    epsilon=eps,
)
stage_cost = mppi_si.cost_functions.stage_cost_function.stage_cost(
    goal=goal, obstacle=obstacle, obstacle_radius=obstacle_radius
)
terminal_cost = mppi_si.cost_functions.terminal_cost_function.terminal_cost(
    goal=goal, obstacle=obstacle, obstacle_radius=obstacle_radius
)

# Instantiate MPPI control law
mppi_controller = vanilla_mppi(
    control_limits=ACTUATION_LIMITS,
    dynamics_func=dynamics,
    stage_cost=stage_cost,
    terminal_cost=terminal_cost,
    mppi_args=mppi_args,
)

# Run the simulation
x_, u_, z_, p_, data_keys_, data_items_ = sim.execute(
    x0=INITIAL_STATE,
    dt=DT,
    num_steps=N_STEPS,
    dynamics=dynamics,
    integrator=integrator,
    controller=mppi_controller,
    sensor=sensor,
    estimator=estimator,
    filepath=SAVE_FILE,
    controller_data={
        "action_trajectory": jnp.zeros(
            (mppi_args["prediction_horizon"], mppi_args["robot_control_dim"])
        )
    },
)

if plot:
    from tutorials.plot_helper.plot_mppi import animate

    animate(
        states=x_,
        estimates=z_,
        controller_data_keys=data_keys_,
        controller_data_items=data_items_,
        mppi_args=mppi_args,
        desired_state=goal,
        desired_state_radius=0.1,
        x_lim=(0, 10),
        y_lim=(0, 10),
        dt=DT,
        title="System Behavior",
        save_animation=True,
        animation_filename=target_directory + "/" + model_name + "/animation" + ".gif",
        obstacle=obstacle,
        obstacle_radius=obstacle_radius,
    )
