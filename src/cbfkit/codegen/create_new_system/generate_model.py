"""
Example
-------
model_name = "double_integrator_2d"
drift_dynamics = "[x[2], x[3], 0.0, 0.0]"
control_matrix = "[[0, 0], [0, 0], [1, 0], [0, 1]]"
barrier_functions = ["x[0]", "x[1] + 3"]
lyapunov_functions = "(x[0] - 2)**2 + (x[1] - 4)**2 - 1"
nominal_controller = "[kp * (x[0] - 1), kp * (x[1] - 2)]"
params = {"controller": {"kp: float": 1.0}}
generate_model(
    "./",
    model_name,
    drift_dynamics,
    control_matrix,
    barrier_functions,
    lyapunov_functions,
    nominal_controller,
    params,
)

"""

import logging
import os
import platform
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union

from jinja2 import Environment, FileSystemLoader

op_sys = platform.system()
DELIMITER = "/" if op_sys == "Darwin" or op_sys == "Linux" else "\\"
INIT_CONTENTS = ""
JAX_EXPRESSIONS = ["exp", "pi", "cos", "sin", "tan", "linalg.norm", "array", "max"]

# Jinja2 Setup
# Assuming templates are in ../templates relative to this file
TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates"
)
jinja_env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))


def create_folder(directory: str, folder_name: str) -> None:
    """Creates a new folder called 'folder_name' at the location specified by 'directory'."""
    folder_path = os.path.join(directory, folder_name)
    os.makedirs(folder_path, exist_ok=True)


def create_python_file(directory: str, file_name: str, file_contents: str):
    """Creates a python file called 'file_name' with contents 'file_contents' at 'directory'."""
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(file_contents)


LOGGER = logging.getLogger(__name__)


def run_black(file_path: str) -> None:
    """Format the given file with Black."""
    try:
        subprocess.run(
            ["black", file_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        LOGGER.error("Black executable not found while formatting %s", file_path)
        raise RuntimeError(
            "Black executable not found. Please ensure Black is installed and available."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode().strip() if exc.stderr else str(exc)
        LOGGER.error("Black failed to format %s: %s", file_path, stderr)
        raise RuntimeError(f"Black failed to format {file_path}: {stderr}") from exc


def extract_keys(input_string):
    pattern = r"\b(\w+)\s*:\s*(\w+)\b"
    matches = re.findall(pattern, input_string)
    keys = [match[0] for match in matches]
    return keys


def convert_string(input_string):
    keys = extract_keys(input_string)
    result_string = "".join([key + "," for key in keys])
    return result_string


def jaxify_expression(expr: str) -> str:
    """Helper to replace math functions with jnp versions."""
    for func in JAX_EXPRESSIONS:
        expr = expr.replace(func, "jnp." + func)
    expr = expr.replace("arcjnp.", "jnp.arc")  # Fix arcsin/arccos
    return expr


def generate_model(
    directory: str,
    model_name: str,
    drift_dynamics: Union[str, List[str]],
    control_matrix: Union[str, List[str]],
    stage_cost_function: Optional[Union[str, List, None]] = None,
    terminal_cost_function: Optional[Union[str, List, None]] = None,
    barrier_funcs: Optional[Union[str, List, None]] = None,
    lyapunov_funcs: Optional[Union[str, List, None]] = None,
    nominal_controller: Optional[Union[str, List, None]] = None,
    params: Optional[Union[Dict[str, Any], None]] = None,
) -> Tuple[int, int]:
    if params is None:
        params = {}

    # Process drift dynamics
    if type(drift_dynamics) == str:
        n_states = len(drift_dynamics.split(","))
        drift_dynamics_str = drift_dynamics
    elif type(drift_dynamics) == list:
        n_states = len(drift_dynamics)
        drift_dynamics_str = "[" + ",".join(drift_dynamics) + "]"
    else:
        raise TypeError("drift_dynamics is not type str or list!")

    # Process control matrix
    if type(control_matrix) == str:
        n_controls = len(control_matrix.split("],")[0].split(","))
        control_matrix_str = control_matrix
    elif type(control_matrix) == list:
        n_controls = len(control_matrix[0].split(","))
        control_matrix_str = "[" + ",".join(control_matrix) + "]"
    else:
        raise TypeError("drift_dynamics is not type str or list!")

    # JAXify strings
    drift_dynamics_str = jaxify_expression(drift_dynamics_str)
    control_matrix_str = jaxify_expression(control_matrix_str)

    # Create model root folder
    model_folder = directory + DELIMITER + model_name
    if not os.path.exists(model_folder):
        create_folder(directory, model_name)

    # Create model subfolders
    create_folder(model_folder, "cost_functions")
    create_folder(model_folder + DELIMITER + "cost_functions", "stage_cost_function")
    create_folder(model_folder + DELIMITER + "cost_functions", "terminal_cost_function")

    create_folder(model_folder, "certificate_functions")
    create_folder(model_folder + DELIMITER + "certificate_functions", "barrier_functions")
    create_folder(model_folder + DELIMITER + "certificate_functions", "lyapunov_functions")
    create_folder(directory + DELIMITER + model_name, "controllers")

    # Build init contents
    init_template = jinja_env.get_template("init.py.j2")

    model_init_imports = [
        "from .plant import plant",
        "from . import cost_functions",
        "from . import certificate_functions",
        "from . import controllers",
    ]
    model_init_contents = init_template.render(imports="\n".join(model_init_imports))

    if not os.path.isfile(model_folder + DELIMITER + "__init__.py"):
        create_python_file(model_folder, "__init__.py", model_init_contents)
    run_black(model_folder + DELIMITER + "__init__.py")

    if not os.path.isfile(model_folder + DELIMITER + "constants.py"):
        create_python_file(model_folder, "constants.py", "")

    # ... (Subfolder inits) ...
    create_python_file(model_folder + DELIMITER + "controllers", "__init__.py", "")

    cert_init_imports = ["from . import barrier_functions", "from . import lyapunov_functions"]
    cert_init_content = init_template.render(imports="\n".join(cert_init_imports))
    create_python_file(
        model_folder + DELIMITER + "certificate_functions", "__init__.py", cert_init_content
    )

    cost_init_imports = [
        "from . import stage_cost_function",
        "from . import terminal_cost_function",
    ]
    cost_init_content = init_template.render(imports="\n".join(cost_init_imports))
    create_python_file(
        model_folder + DELIMITER + "cost_functions", "__init__.py", cost_init_content
    )

    # Create plant.py using Jinja
    dynamics_args = (
        "".join([pp + ", " for pp in params["dynamics"].keys()])
        if "dynamics" in params.keys()
        else ""
    )

    plant_template = jinja_env.get_template("plant.py.j2")
    plant_contents = plant_template.render(
        dynamics_args=dynamics_args,
        drift_dynamics=drift_dynamics_str,
        control_matrix=control_matrix_str,
    )
    create_python_file(model_folder, "plant.py", plant_contents)
    run_black(model_folder + DELIMITER + "plant.py")

    # Cost functions
    cost_template = jinja_env.get_template("cost.py.j2")

    if stage_cost_function is not None:
        stage_cost_folder = (
            model_folder + DELIMITER + "cost_functions" + DELIMITER + "stage_cost_function"
        )

        stage_cost_str = jaxify_expression(stage_cost_function)
        stage_cost_args = (
            "".join([pp + ", " for pp in params["stage_cost_function"].keys()])
            if "stage_cost_function" in params.keys()
            else ""
        )

        stage_cost_contents = cost_template.render(
            n_states=n_states,
            cost_name="stage_cost",
            cost_args=stage_cost_args,
            cost_function=stage_cost_str,
        )

        stage_init_imports = ["from .stage_cost import stage_cost"]
        stage_init_contents = init_template.render(imports="\n".join(stage_init_imports))

        create_python_file(stage_cost_folder, "__init__.py", stage_init_contents)
        create_python_file(stage_cost_folder, "stage_cost.py", stage_cost_contents)
        run_black(stage_cost_folder + DELIMITER + "stage_cost.py")

    if terminal_cost_function is not None:
        terminal_cost_folder = (
            model_folder + DELIMITER + "cost_functions" + DELIMITER + "terminal_cost_function"
        )

        terminal_cost_str = jaxify_expression(terminal_cost_function)
        terminal_cost_args = (
            "".join([pp + ", " for pp in params["terminal_cost_function"].keys()])
            if "terminal_cost_function" in params.keys()
            else ""
        )

        terminal_cost_contents = cost_template.render(
            n_states=n_states,
            cost_name="terminal_cost",
            cost_args=terminal_cost_args,
            cost_function=terminal_cost_str,
        )

        terminal_init_imports = ["from .terminal_cost import terminal_cost"]
        terminal_init_contents = init_template.render(imports="\n".join(terminal_init_imports))

        create_python_file(terminal_cost_folder, "__init__.py", terminal_init_contents)
        create_python_file(terminal_cost_folder, "terminal_cost.py", terminal_cost_contents)
        run_black(terminal_cost_folder + DELIMITER + "terminal_cost.py")

    # Create barrier function contents
    if barrier_funcs is not None:
        barrier_folder = (
            model_folder + DELIMITER + "certificate_functions" + DELIMITER + "barrier_functions"
        )
        if type(barrier_funcs) is str:
            n_bars = 1
            barrier_funcs = [barrier_funcs]
        else:
            n_bars = len(barrier_funcs)

        barr_init_imports = [
            f"from .barrier_{bb+1} import cbf{bb+1}_package" for bb in range(n_bars)
        ]
        barr_init_contents = init_template.render(imports="\n".join(barr_init_imports))
        create_python_file(barrier_folder, "__init__.py", barr_init_contents)

        barrier_template = jinja_env.get_template("barrier.py.j2")

        for bb in range(n_bars):
            bf_str = jaxify_expression(barrier_funcs[bb])

            cbf_args = (
                "".join([pp + ", " for pp in params["cbf"][bb].keys()])
                if "cbf" in params.keys()
                else ""
            )
            cbf_args_call = convert_string(cbf_args)

            barrier_contents = barrier_template.render(
                n_states=n_states,
                cbf_args=cbf_args,
                cbf_args_call=cbf_args_call,
                barrier_func=bf_str,
                index=bb + 1,
            )
            create_python_file(barrier_folder, f"barrier_{bb+1}.py", barrier_contents)
            run_black(barrier_folder + DELIMITER + f"barrier_{bb+1}.py")

    # Create lyapunov function contents
    if lyapunov_funcs is not None:
        lyapunov_folder = (
            model_folder + DELIMITER + "certificate_functions" + DELIMITER + "lyapunov_functions"
        )

        if type(lyapunov_funcs) is str:
            n_lfs = 1
            lyapunov_funcs = [lyapunov_funcs]
        else:
            n_lfs = len(lyapunov_funcs)

        lyap_init_imports = [
            f"from .lyapunov_{ll+1} import clf{ll+1}_package" for ll in range(n_lfs)
        ]
        lyap_init_contents = init_template.render(imports="\n".join(lyap_init_imports))
        create_python_file(lyapunov_folder, "__init__.py", lyap_init_contents)

        lyapunov_template = jinja_env.get_template("lyapunov.py.j2")

        for ll in range(n_lfs):
            lf_str = jaxify_expression(lyapunov_funcs[ll])

            clf_args = (
                "".join([pp + ", " for pp in params["clf"][ll].keys()])
                if "clf" in params.keys()
                else ""
            )
            clf_args_call = convert_string(clf_args)

            lyapunov_contents = lyapunov_template.render(
                n_states=n_states,
                clf_args=clf_args,
                clf_args_call=clf_args_call,
                lyapunov_func=lf_str,
                index=ll + 1,
            )
            create_python_file(lyapunov_folder, f"lyapunov_{ll+1}.py", lyapunov_contents)
            run_black(lyapunov_folder + DELIMITER + f"lyapunov_{ll+1}.py")

    control_folder = model_folder + DELIMITER + "controllers"
    if nominal_controller is not None:
        if type(nominal_controller) is str:
            n_cons = 1
            nominal_controller = [nominal_controller]
        else:
            n_cons = len(nominal_controller)

        cont_init_imports = [
            f"from .controller_{cc+1} import controller_{cc+1}" for cc in range(n_cons)
        ]
        cont_init_contents = init_template.render(imports="\n".join(cont_init_imports))
        create_python_file(control_folder, "__init__.py", cont_init_contents)

        controller_template = jinja_env.get_template("controller.py.j2")

        for cc in range(n_cons):
            u_nom_str = jaxify_expression(nominal_controller[cc])

            control_args = (
                "".join([pp + ", " for pp in params["controller"].keys()])
                if "controller" in params.keys()
                else ""
            )

            controller_contents = controller_template.render(
                index=cc + 1, control_args=control_args, nominal_controller=u_nom_str
            )
            create_python_file(control_folder, f"controller_{cc+1}.py", controller_contents)
            run_black(control_folder + DELIMITER + f"controller_{cc+1}.py")

    # ROS2 Controller Node using Jinja
    ros2_folder = model_folder + "/ros2"
    os.makedirs(ros2_folder, exist_ok=True)
    if not os.path.isfile(ros2_folder + DELIMITER + "__init__.py"):
        create_python_file(ros2_folder, "__init__.py", "")

    controller_node_template = jinja_env.get_template("ros2_controller_node.py.j2")
    # Assuming class name convention
    class_name = "".join([word.capitalize() for word in model_name.split("_")]) + "Controller"

    node_contents = controller_node_template.render(class_name=class_name, model_name=model_name)
    controller_path = os.path.join(ros2_folder, "controller.py")
    with open(controller_path, "w") as f:
        f.write(node_contents)
    run_black(controller_path)

    print(f"Generated ROS2 controller node script at {controller_path}")

    return n_states, n_controls
