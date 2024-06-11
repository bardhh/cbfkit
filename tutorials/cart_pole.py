from jax import Array, jit
import jax.numpy as jnp

from cbfkit.codegen.create_new_system import generate_model

# parnameters

I = 0.099
M = 0.1
m = 0.1
l = 0.5
G = 9.81

k_1 = 1.0
k_2 = 1.0
k_3 = 1.0
k_4 = 1.0

Range = 10.0


# f + g * u  for cart pole
drift_dynamics = "[x[1], (m * l * ( m * l * G * sin(x[2]) * cos(x[2]) + (I + m * l ** 2) * x[3] ** 2 * sin(x[1]))) / ((I + m * l ** 2) * (M + m) - m ** 2 * l ** 2 * cos(x[2]) ** 2), x[3], (-m * l * ((m + M) * G * sin(x[2]) + m * l * x[3] ** 2 * sin(x[2]) * cos(x[2]))) / ((I + m * l ** 2) * (M + m) - m ** 2 * l ** 2 * cos(x[2]) ** 2)]"     # the f matrix
control_matrix = "[[0], [(I + m * l ** 2) / ((I + m * l ** 2) * (M + m) - m ** 2 * l ** 2 * cos(x[2]) ** 2 )], [0], [(-m * l * cos(x[2])) / ((I + m * l ** 2) * (M + m) - m ** 2 * l ** 2 * cos(x[2]) ** 2)]]"          # the control matrix g

target_directory = "./SGP_test"
madel_name = "cart_pole"
initial_state = [0,0,0,0]
params = {"dynamics": {"I : float": I,"M : float": M,"m : float": m,"G : float": G,"l : float": l,}}

nominal_control_law = "((( x[3] / l) * cos( x[2] ) * k_4 + k_3 * (x[1] + k_1 * x[3] * cos(x[2]) + k_2 * sin(x[2])) + k_1 * sin(x[2]) * ((G / l) * cos(x[2]) - x[3] ** 2) + k_2 * x[3] * cos(x[2]) ) ) / ((k_1 / l) * cos(x[2]) ** 2 - 1)"
params["controller"] = {"k_1 : float": k_1,"k_2 : float": k_2,"k_3 : float": k_3,"k_4 : float": k_4,"G : float": G,"l : float": l,}

lyapunov_funcs = "((k_1 * cos(x[2]) ** 2 / l) - 1) * x[3] ** 2 / 2  + (G / l) * (1 - cos(x[2]) ) + (k_3 / 2) * (x[1] + k_1 * x[3] * cos(x[2]) + k_2 * sin(x[2])) ** 2"
barrier_funcs = ["x[0] + range", "range - x[0]"]
params["clf"] = [{"k_1 : float": k_1,"k_2 : float": k_2,"k_3 : float": k_3,"G : float": G,"l : float": l,}]
params["cbf"] = [{"range: float": Range,},{"range: float": Range,}]

generate_model.generate_model(
    directory=target_directory,
    model_name=madel_name,
    drift_dynamics=drift_dynamics,
    initial_state=initial_state,
    control_matrix=control_matrix,
    nominal_controller=nominal_control_law,
    barrier_funcs=barrier_funcs,
    lyapunov_funcs=lyapunov_funcs,
    params=params,
)
