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

import os
import re
import textwrap
import platform
import subprocess
from typing import Any, Dict, List, Optional, Union, Tuple

op_sys = platform.system()
DELIMITER = "/" if op_sys == "Darwin" or op_sys == "Linux" else "\\"
INIT_CONTENTS = ""
JAX_EXPRESSIONS = ["exp", "pi", "cos", "sin", "tan", "linalg.norm", "array", "max"]


def create_folder(directory: str, folder_name: str) -> None:
    """Creates a new folder called 'folder_name' at the location specified by 'directory'.

    Args:
        directory (str): path to desired directory
        folder_name (str): name of folder to be created at 'directory'
    """
    # Create the folder
    folder_path = os.path.join(directory, folder_name)
    os.makedirs(folder_path, exist_ok=True)


def create_python_file(directory: str, file_name: str, file_contents: str):
    """Creates a python file called 'file_name' with contents 'file_contents' at 'directory'.

    Args:
        directory (str): location of filee
        file_name (str): name of file to be created
        file_contents (str): contents of file (including line spacing, etc.)
    """
    # Create the Python file with specified contents
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(file_contents)


def run_black(file_path):
    try:
        # Run the Black linter on the specified file
        subprocess.run(
            ["black", file_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        # print(f"Black successfully formatted {file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running Black: {e}")


def extract_keys(input_string):
    # Use regular expression to find patterns like "x: type"
    pattern = r"\b(\w+)\s*:\s*(\w+)\b"
    matches = re.findall(pattern, input_string)

    # Extract keys from matches
    keys = [match[0] for match in matches]

    return keys


def convert_string(input_string):
    keys = extract_keys(input_string)

    # Add a comma after each key
    result_string = "".join([key + "," for key in keys])

    return result_string


def generate_model(
    directory: str,
    model_name: str,
    drift_dynamics: Union[str, List[str]],
    control_matrix: Union[str, List[str]],
    stage_cost_function: Optional[
        Union[str, List, None]
    ] = None,  # cost function beside constraints
    terminal_cost_function: Optional[
        Union[str, List, None]
    ] = None,  # cost function beside constraints
    params: Optional[Union[Dict[str, Any], None]] = None,
) -> Tuple[int, int]:
    if params is None:
        params = {}

    # Process drift dynamics
    if type(drift_dynamics) == str:
        n_states = len(drift_dynamics.split(","))
        drift_dynamics = textwrap.dedent(drift_dynamics)
    elif type(drift_dynamics) == list:
        n_states = len(drift_dynamics)
        drift_dynamics = textwrap.dedent("[" + ",".join(drift_dynamics) + "]")
    else:
        raise TypeError("drift_dynamics is not type str or list!")

    # Process control matrix
    if type(control_matrix) == str:
        n_controls = len(control_matrix.split("],")[0].split(","))
        control_matrix = textwrap.dedent(control_matrix)
    elif type(control_matrix) == list:
        n_controls = len(control_matrix[0].split(","))
        control_matrix = textwrap.dedent("[" + ",".join(control_matrix) + "]")
    else:
        raise TypeError("drift_dynamics is not type str or list!")

    # Create model root folder
    model_folder = directory + DELIMITER + model_name
    if not os.path.exists(model_folder):
        create_folder(directory, model_name)

        # Create model subfolders
        create_folder(model_folder, "cost_functions")
        create_folder(model_folder + DELIMITER + "cost_functions", "stage_cost_function")
        create_folder(model_folder + DELIMITER + "cost_functions", "terminal_cost_function")

    # Build init contents
    model_init_contents = textwrap.dedent(
        """
        from .plant import plant
        from . import cost_functions
        """
    )

    cost_init_contents = textwrap.dedent(
        """
        from . import stage_cost_function
        from . import terminal_cost_function
        """
    )

    # Create proper init files
    if not os.path.isfile(model_folder + DELIMITER + "__init__.py"):
        create_python_file(model_folder, "__init__.py", model_init_contents)
    run_black(model_folder + DELIMITER + "__init__.py")
    if not os.path.isfile(model_folder + DELIMITER + "constants.py"):
        create_python_file(model_folder, "constants.py", "")

    if not os.path.isfile(model_folder + DELIMITER + "cost_functions" + DELIMITER + "__init__.py"):
        create_python_file(
            model_folder + DELIMITER + "cost_functions",
            "__init__.py",
            cost_init_contents,
        )
    run_black(model_folder + DELIMITER + "cost_functions" + DELIMITER + "__init__.py")

    # Create plant.py contents
    for expr in JAX_EXPRESSIONS:
        drift_dynamics = drift_dynamics.replace(expr, "jnp." + expr)
        control_matrix = control_matrix.replace(expr, "jnp." + expr)
    drift_dynamics = drift_dynamics.replace("arcjnp.", "arc")
    control_matrix = control_matrix.replace("arcjnp.", "arc")

    dynamics_args = (
        "".join([pp + ", " for pp in params["dynamics"].keys()])
        if "dynamics" in params.keys()
        else ""
    )
    plant_contents = textwrap.dedent(
        f'''
        import jax.numpy as jnp
        from jax import jit, Array, lax
        from typing import Optional, Union, Callable
        from cbfkit.utils.user_types import DynamicsCallable, DynamicsCallableReturns
        from .constants import *


        def plant({dynamics_args}**kwargs) -> DynamicsCallable:
            """
            Returns a function that represents the plant model,
            which computes the drift vector 'f' and control matrix 'g' based on the given state.

            States are the following:
                #! MANUALLY POPULATE

            Control inputs are the following:
                #! MANUALLY POPULATE

            Args:
                perturbation (Optional, Array): additive perturbation to the xdot dynamics
                kwargs: keyword arguments

            Returns:
                dynamics (Callable): takes state as input and returns dynamics components
                    f, g of the form dx/dt = f(x) + g(x)u

            """

            @jit
            def dynamics(x: Array) -> DynamicsCallableReturns:
                """
                Computes the drift vector 'f' and control matrix 'g' based on the given state x.

                Args:
                    x (Array): state vector

                Returns:
                    dynamics (DynamicsCallable): takes state as input and returns dynamics components f, g
                """
                f = jnp.array(
                    {drift_dynamics}
                )
                g = jnp.array(
                    {control_matrix}
                )

                return f, g

            return dynamics

    '''
    )
    create_python_file(model_folder, "plant.py", plant_contents)
    run_black(model_folder + DELIMITER + "plant.py")

    # Create barrier function contents
    # if cost_funcs is not None:

    if stage_cost_function is not None:
        stage_cost_folder = (
            model_folder + DELIMITER + "cost_functions" + DELIMITER + "stage_cost_function"
        )

        for expr in JAX_EXPRESSIONS:
            stage_cost_function = stage_cost_function.replace(expr, "jnp." + expr)

        stage_cost_init_contents = "\n".join([f"from .stage_cost import stage_cost"])

        stage_cost_args = (
            "".join([pp + ", " for pp in params["stage_cost_function"].keys()])
            if "stage_cost_function" in params.keys()
            else ""
        )

        stage_cost_contents = textwrap.dedent(
            f'''
            """
            #! MANUALLY POPULATE (docstring)
            """
            import jax.numpy as jnp
            from jax import jit, Array, lax
            from typing import List, Callable

            N = {n_states}


            ###############################################################################
            # Stage Cost Function
            ###############################################################################

            def stage_cost({stage_cost_args}**kwargs) -> Callable[[Array], Array]:
                """Super-level set convention.

                Args:
                    #! kwargs -- optional to manually populate

                Returns:
                    ret (float): value of constraint function evaluated at time and state

                """

                @jit
                def func(state_and_time: Array, action: Array) -> Array:
                    """Function to be evaluated.

                    Args:
                        state_and_time (Array): concatenated state vector and time

                    Returns:
                        Array: cbf value
                    """
                    x = state_and_time
                    return {stage_cost_function}

                return func

            '''
        )
        create_python_file(stage_cost_folder, "__init__.py", stage_cost_init_contents)
        create_python_file(stage_cost_folder, f"stage_cost.py", stage_cost_contents)
        run_black(stage_cost_folder + DELIMITER + f"stage_cost.py")

    if terminal_cost_function is not None:
        terminal_cost_folder = (
            model_folder + DELIMITER + "cost_functions" + DELIMITER + "terminal_cost_function"
        )

        for expr in JAX_EXPRESSIONS:
            terminal_cost_function = terminal_cost_function.replace(expr, "jnp." + expr)

        terminal_cost_init_contents = "\n".join([f"from .terminal_cost import terminal_cost"])

        terminal_cost_args = (
            "".join([pp + ", " for pp in params["terminal_cost_function"].keys()])
            if "terminal_cost_function" in params.keys()
            else ""
        )

        terminal_cost_contents = textwrap.dedent(
            f'''
            """
            #! MANUALLY POPULATE (docstring)
            """
            import jax.numpy as jnp
            from jax import jit, Array, lax
            from typing import List, Callable

            N = {n_states}


            ###############################################################################
            # Terminal Cost Function
            ###############################################################################

            def terminal_cost({terminal_cost_args}**kwargs) -> Callable[[Array], Array]:
                """Super-level set convention.

                Args:
                    #! kwargs -- optional to manually populate

                Returns:
                    ret (float): value of constraint function evaluated at time and state

                """

                @jit
                def func(state_and_time: Array, action: Array) -> Array:
                    """Function to be evaluated.

                    Args:
                        state_and_time (Array): concatenated state vector and time

                    Returns:
                        Array: cbf value
                    """
                    x = state_and_time
                    return {terminal_cost_function}

                return func

            '''
        )
        create_python_file(terminal_cost_folder, "__init__.py", terminal_cost_init_contents)
        create_python_file(terminal_cost_folder, f"terminal_cost.py", terminal_cost_contents)
        run_black(terminal_cost_folder + DELIMITER + f"terminal_cost.py")
