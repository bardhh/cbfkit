from cbfkit.codegen.create_new_system import generate_model

drift_dynamics = "[x[2], x[3], 0, 0, x[5], 0, 0, 0]"
control_matrix = "[[0, 0, 0, 0], [0, 0, 0, 0], [cos(x[4]) / m, cos(x[4]) / m, 0, 0], [sin(x[4]) / m, sin(x[4]) / m, 0, 0], [0, 0, 0, 0], [(x[1] - x[6]) / (1/12 * m * (a**2 + b**2)), (x[1] - x[7]) / (1/12 * m * (a**2 + b**2)), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]"

target_directory = "examples/rss_workshop"
model_name = "dynamics"
params = {"dynamics": {"m: float": 2.0, "a: float": 1.0, "b: float": 1.0}}


state_constraint_funcs = ["((x[0] - xo) / a)**10 + ((x[1] - yo) / b)**10 - 1 - r"]
params["cbf"] = [
    {"xo: float": 1.0, "yo: float": 1.0, "a: float": 1.0, "b: float": 1.0, "r: float": 1.0}
]


generate_model.generate_model(
    directory=target_directory,
    model_name=model_name,
    drift_dynamics=drift_dynamics,
    control_matrix=control_matrix,
    barrier_funcs=state_constraint_funcs,
    params=params,
)
