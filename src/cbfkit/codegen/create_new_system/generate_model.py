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
import platform
import re
import subprocess
import textwrap
from typing import Any, Dict, List, Optional, Tuple, Union

op_sys = platform.system()
DELIMITER = "/" if op_sys == "Darwin" or op_sys == "Linux" else "\\"
INIT_CONTENTS = ""
JAX_EXPRESSIONS = ["exp", "pi", "cos", "sin", "tan"]


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
        create_folder(model_folder, "certificate_functions")
        create_folder(model_folder + DELIMITER + "certificate_functions", "barrier_functions")
        create_folder(model_folder + DELIMITER + "certificate_functions", "lyapunov_functions")
        create_folder(directory + DELIMITER + model_name, "controllers")

    # Build init contents
    model_init_contents = textwrap.dedent(
        """
        from .plant import plant
        from . import certificate_functions
        from . import controllers
        """
    )

    certificate_init_contents = textwrap.dedent(
        ("from . import barrier_functions" if barrier_funcs is not None else "")
        + ("from . import lyapunov_functions" if lyapunov_funcs is not None else "")
    )

    # Create proper init files
    if not os.path.isfile(model_folder + DELIMITER + "__init__.py"):
        create_python_file(model_folder, "__init__.py", model_init_contents)
    run_black(model_folder + DELIMITER + "__init__.py")
    if not os.path.isfile(model_folder + DELIMITER + "constants.py"):
        create_python_file(model_folder, "constants.py", "")

    if not os.path.isfile(model_folder + DELIMITER + "controllers" + DELIMITER + "__init__.py"):
        create_python_file(model_folder + DELIMITER + "controllers", "__init__.py", INIT_CONTENTS)
    run_black(model_folder + DELIMITER + "controllers" + DELIMITER + "__init__.py")

    if not os.path.isfile(
        model_folder + DELIMITER + "certificate_functions" + DELIMITER + "__init__.py"
    ):
        create_python_file(
            model_folder + DELIMITER + "certificate_functions",
            "__init__.py",
            certificate_init_contents,
        )
    run_black(model_folder + DELIMITER + "certificate_functions" + DELIMITER + "__init__.py")

    # Create plant.py contents
    for expr in JAX_EXPRESSIONS:
        drift_dynamics = drift_dynamics.replace(expr, "jnp." + expr)
        control_matrix = control_matrix.replace(expr, "jnp." + expr)
    drift_dynamics = drift_dynamics.replace("arcjnp.", "jnp.arc")
    control_matrix = control_matrix.replace("arcjnp.", "jnp.arc")

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
    if barrier_funcs is not None:
        barrier_folder = (
            model_folder + DELIMITER + "certificate_functions" + DELIMITER + "barrier_functions"
        )
        if type(barrier_funcs) is str:
            n_bars = 1
            barrier_funcs = [barrier_funcs]
        else:
            n_bars = len(barrier_funcs)

        barr_init_contents = "\n".join(
            [f"from .barrier_{bb+1} import cbf{bb+1}_package" for bb in range(n_bars)]
        )

        for bb in range(n_bars):
            for expr in JAX_EXPRESSIONS:
                barrier_funcs[bb] = barrier_funcs[bb].replace(expr, "jnp." + expr)
            barrier_funcs[bb] = barrier_funcs[bb].replace("arcjnp.", "jnp.arc")

            cbf_args = (
                "".join([pp + ", " for pp in params["cbf"][bb].keys()])
                if "cbf" in params.keys()
                else ""
            )
            barrier_contents = textwrap.dedent(
                f'''
                """
                #! MANUALLY POPULATE (docstring)
                """
                import jax.numpy as jnp
                from jax import jit, jacfwd, jacrev, Array, lax
                from typing import List, Callable
                from cbfkit.controllers.model_based.cbf_clf_controllers.utils.certificate_packager import certificate_package

                N = {n_states}


                ###############################################################################
                # CBF
                ###############################################################################


                def cbf({cbf_args}**kwargs) -> Callable[[Array], Array]:
                    """Super-level set convention.

                    Args:
                        #! kwargs -- optional to manually populate

                    Returns:
                        ret (float): value of constraint function evaluated at time and state

                    """

                    @jit
                    def func(state_and_time: Array) -> Array:
                        """Function to be evaluated.

                        Args:
                            state_and_time (Array): concatenated state vector and time

                        Returns:
                            Array: cbf value
                        """
                        x = state_and_time
                        return {barrier_funcs[bb]}

                    return func


                def cbf_grad({cbf_args}**kwargs) -> Callable[[Array], Array]:
                    """Jacobian for the constraint function defined by cbf.

                    Args:
                        #! kwargs -- manually populate

                    Returns:
                        ret (float): value of constraint function evaluated at time and state

                    """
                    jacobian = jacfwd(cbf({convert_string(cbf_args)}**kwargs))

                    @jit
                    def func(state_and_time: Array) -> Array:
                        """_summary_

                        Args:
                            state_and_time (Array): concatenated state vector and time

                        Returns:
                            Array: cbf jacobian (gradient)
                        """

                        return jacobian(state_and_time)

                    return func


                def cbf_hess({cbf_args}**kwargs) -> Callable[[Array], Array]:
                    """Hessian for the constraint function defined by cbf.

                    Args:
                        #! kwargs -- manually populate

                    Returns:
                        ret (float): value of constraint function evaluated at time and state

                    """
                    hessian = jacrev(jacfwd(cbf({convert_string(cbf_args)}**kwargs)))

                    @jit
                    def func(state_and_time: Array) -> Array:
                        """_summary_

                        Args:
                            state_and_time (Array): concatenated state vector and time

                        Returns:
                            Array: cbf hessian
                        """

                        return hessian(state_and_time)

                    return func


                ###############################################################################
                # CBF{bb+1}
                ###############################################################################
                cbf{bb+1}_package = certificate_package(cbf, cbf_grad, cbf_hess, N)

                '''
            )
            create_python_file(barrier_folder, "__init__.py", barr_init_contents)
            create_python_file(barrier_folder, f"barrier_{bb+1}.py", barrier_contents)
            run_black(barrier_folder + DELIMITER + f"barrier_{bb+1}.py")

    # Create barrier function contents
    if lyapunov_funcs is not None:
        lyapunov_folder = (
            model_folder + DELIMITER + "certificate_functions" + DELIMITER + "lyapunov_functions"
        )
        if type(lyapunov_funcs) is str:
            n_lfs = 1
            lyapunov_funcs = [lyapunov_funcs]
        else:
            n_lfs = len(lyapunov_funcs)

        lyap_init_contents = "\n".join(
            [f"from .lyapunov_{ll+1} import clf{ll+1}_package" for ll in range(n_lfs)]
        )

        for ll in range(n_lfs):
            for expr in JAX_EXPRESSIONS:
                lyapunov_funcs[ll] = lyapunov_funcs[ll].replace(expr, "jnp." + expr)
            lyapunov_funcs[ll] = lyapunov_funcs[ll].replace("arcjnp.", "jnp.arc")

            clf_args = (
                "".join([pp + ", " for pp in params["clf"][ll].keys()])
                if "clf" in params.keys()
                else ""
            )
            lyapunov_contents = textwrap.dedent(
                f'''
                """
                #! MANUALLY POPULATE (docstring)
                """
                import jax.numpy as jnp
                from jax import jit, jacfwd, jacrev, Array, lax
                from typing import List, Callable
                from cbfkit.controllers.model_based.cbf_clf_controllers.utils.certificate_packager import certificate_package

                N = {n_states}


                ###############################################################################
                # CLF
                ###############################################################################


                def clf({clf_args}**kwargs) -> Callable[[Array], Array]:
                    """Super-level set convention.

                    Args:
                        #! kwargs -- optional to manually populate

                    Returns:
                        ret (float): value of goal function evaluated at time and state

                    """

                    @jit
                    def func(state_and_time: Array) -> Array:
                        """Function to be evaluated.

                        Args:
                            state_and_time (Array): concatenated state vector and time

                        Returns:
                            Array: clf value
                        """
                        x = state_and_time
                        return {lyapunov_funcs[ll]}

                    return func


                def clf_grad({clf_args}**kwargs) -> Callable[[Array], Array]:
                    """Jacobian for the goal function defined by clf.

                    Args:
                        #! kwargs -- manually populate

                    Returns:
                        ret (float): value of goal function evaluated at time and state

                    """
                    jacobian = jacfwd(clf({convert_string(clf_args)}**kwargs))

                    @jit
                    def func(state_and_time: Array) -> Array:
                        """_summary_

                        Args:
                            state_and_time (Array): concatenated state vector and time

                        Returns:
                            Array: clf jacobian (gradient)
                        """
                        
                        return jacobian(state_and_time)

                    return func


                def clf_hess({clf_args}**kwargs) -> Callable[[Array], Array]:
                    """Hessian for the goal function defined by clf.

                    Args:
                        #! kwargs -- manually populate

                    Returns:
                        ret (float): value of goal function evaluated at time and state

                    """
                    hessian = jacrev(jacfwd(clf({convert_string(clf_args)}**kwargs)))

                    @jit
                    def func(state_and_time: Array) -> Array:
                        """_summary_

                        Args:
                            state_and_time (Array): concatenated state vector and time

                        Returns:
                            Array: clf hessian
                        """

                        return hessian(state_and_time)

                    return func


                ###############################################################################
                # CLF{ll+1}
                ###############################################################################
                clf{ll+1}_package = certificate_package(clf, clf_grad, clf_hess, N)

                '''
            )
            create_python_file(lyapunov_folder, "__init__.py", lyap_init_contents)
            create_python_file(lyapunov_folder, f"lyapunov_{ll+1}.py", lyapunov_contents)
            run_black(lyapunov_folder + DELIMITER + f"lyapunov_{ll+1}.py")

    control_folder = model_folder + DELIMITER + "controllers"
    if nominal_controller is not None:
        if type(nominal_controller) is str:
            n_cons = 1
            nominal_controller = [nominal_controller]
        else:
            n_cons = len(nominal_controller)

        cont_init_contents = "\n".join(
            [f"from .controller_{cc+1} import controller_{cc+1}" for cc in range(n_cons)]
        )

        for cc in range(n_cons):
            for expr in JAX_EXPRESSIONS:
                nominal_controller[cc] = nominal_controller[cc].replace(expr, "jnp." + expr)
            nominal_controller[cc] = nominal_controller[cc].replace("arcjnp.", "jnp.arc")

            control_args = (
                "".join([pp + ", " for pp in params["controller"].keys()])
                if "controller" in params.keys()
                else ""
            )
            controller_contents = textwrap.dedent(
                f'''
            import jax.numpy as jnp
            from typing import *
            from jax import jit, Array, lax
            from cbfkit.utils.user_types import ControllerCallable, ControllerCallableReturns


            def controller_{cc+1}({control_args}) -> ControllerCallable:
                """
                Create a controller for the given dynamics.

                Args:
                    #! USER-POPULATE

                Returns:
                    controller (Callable): handle to function computing control

                """

                @jit
                def controller(t: float, x: Array) -> ControllerCallableReturns:
                    """Computes control input ({n_controls}x1).

                    Args:
                        t (float): time in sec
                        x (Array): state vector (or estimate if using observer/estimator)

                    Returns:
                        unom (Array): {n_controls}x1 vector
                        data: (dict): empty dictionary
                    """
                    # logging data
                    u_nom = {nominal_controller[cc]}
                    data = {{"u_nom": u_nom}}

                    return jnp.array(u_nom), data

                return controller
            '''
            )
            create_python_file(control_folder, "__init__.py", cont_init_contents)
            create_python_file(control_folder, f"controller_{cc+1}.py", controller_contents)
            run_black(control_folder + DELIMITER + f"controller_{cc+1}.py")
    else:
        controller_contents = textwrap.dedent(
            f'''
        import jax.numpy as jnp
        from jax import jit, Array, lax
        from cbfkit.utils.user_types import ControllerCallable, ControllerCallableReturns


        def zero_controller() -> ControllerCallable:
            """
            Create a zero controller for the given dynamics.

            Args:
                None

            Returns:
                controller (Callable): handle to function computing zero control

            """

            @jit
            def controller(_t: float, _state: Array) -> ControllerCallableReturns:
                """Computes zero control input ({n_controls}x1).

                Args:
                    _t (float): time in sec
                    _state (Array): state vector (or estimate if using observer/estimator)

                Returns:
                    zeros (Array): {n_controls}x1 zero vector
                    data: (dict): empty dictionary
                """
                # logging data
                data = {{}}

                return jnp.zeros(({n_controls},)), data

            return controller
        '''
        )
        create_python_file(control_folder, "zero_controller.py", controller_contents)
        run_black(control_folder + DELIMITER + "zero_controller.py")

    # ROS2 Generation
    ros2_folder = model_folder + "/ros2"
    os.makedirs(ros2_folder, exist_ok=True)

    ros2_init_contents = ""
    if not os.path.isfile(ros2_folder + DELIMITER + "__init__.py"):
        create_python_file(ros2_folder, "__init__.py", model_init_contents)

    # Controller Node

    controller_path = os.path.join(ros2_folder, "controller.py")
    node_contents = textwrap.dedent(
        f"""
    # Code-generated

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
    import jax.numpy as jnp
    import config

    class VanDerPolOscillatorController(Node):

        def __init__(self):
            super().__init__("{model_name}_controller")
            self.publisher = self.create_publisher(Float32MultiArray, "control_command", config.QUEUE_SIZE)
            self.subscription = self.create_subscription(Float32MultiArray, "state_estimate", self.listener_callback, config.QUEUE_SIZE)
            self.controller = self.get_controller_by_name(config.CONTROLLER_NAME, **config.CONTROLLER_PARAMS)
            self.reset_control_msg()
            self.timer = self.create_timer(config.TIMER_INTERVAL, self.publish_control)

        @staticmethod
        def get_controller_by_name(name, **kwargs):
            '''
            Dynamically import and return an instance of the specified controller.

            Parameters:
            - name: The name of the controller class to import and instantiate.
            - **kwargs: Keyword arguments to pass to the controller's constructor.

            Returns:
            An instance of the specified controller.
            '''
            # Assuming all controllers are in a module named 'controllers'
            module = __import__(config.MODULE_NAME + '.' + 'controllers', fromlist=[name])
            controller_class = getattr(module, name)
            return controller_class(**kwargs)

        def reset_control_msg(self):
            control_msg = Float32MultiArray()
            control_msg.data = [0.0]
            self.control_msg = control_msg

        def publish_control(self):
            self.publisher.publish(self.control_msg)

        def listener_callback(self, msg):
            estimate = jnp.array(msg.data)
            t = 0.0                                             #USER_DEFINED
            control_command, _ = self.controller(t, estimate)
            self.reset_control_msg()
            if control_command.ndim > 0:
                self.control_msg.data = [float(c) for c in control_command]
            else:
                self.control_msg.data = [float(control_command)]


    def main(args=None):
        rclpy.init(args=args)
        node = VanDerPolOscillatorController()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()


    if __name__ == "__main__":
        main()
    """
    )

    # Write the contents to the file
    with open(controller_path, "w") as f:
        f.write(node_contents)

    run_black(controller_path)

    print(f"Generated ROS2 controller node script at {controller_path}")

    # Sensor Node

    sensor_path = os.path.join(ros2_folder, "sensor.py")
    node_contents = textwrap.dedent(
        f"""
    # Code-generated

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
    import jax.numpy as jnp
    import config  # Import configurations

    class VanDerPolOscillatorSensor(Node):

        def __init__(self):
            super().__init__("{model_name}_sensor")
            self.publisher_ = self.create_publisher(Float32MultiArray, "sensor_measurement", config.QUEUE_SIZE)
            self.create_subscription(Float32MultiArray, "state", self.listener_callback, config.QUEUE_SIZE)
            self.sensor = self.get_sensor_by_name(config.SENSOR_NAME)

        @staticmethod
        def get_sensor_by_name(name):
            '''
            Dynamically load the sensor function.
            '''
            module_path = f"cbfkit.sensors"
            sensor_module = __import__(module_path, fromlist=[name])
            sensor_function = getattr(sensor_module, name)
            return sensor_function

        def listener_callback(self, msg):
            state = jnp.array(msg.data)
            measurement = self.sensor(0.0, state)
            sensor_msg = self.create_sensor_message(measurement)
            self.publisher_.publish(sensor_msg)

        def create_sensor_message(self, measurement):
            sensor_msg = Float32MultiArray()
            sensor_msg.data = [float(m) for m in measurement]
            return sensor_msg

    def main(args=None):
        rclpy.init(args=args)
        node = VanDerPolOscillatorSensor()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    if __name__ == "__main__":
        main()
    """
    )

    # Write the contents to the file
    with open(sensor_path, "w") as f:
        f.write(node_contents)

    print(f"Generated ROS2 sensor node script at {sensor_path}")

    run_black(sensor_path)

    # Estimator Node

    estimator_path = os.path.join(ros2_folder, "estimator.py")
    node_contents = textwrap.dedent(
        f"""
    # Code-generated

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
    import jax.numpy as jnp
    import config

    class VanDerPolOscillatorEstimator(Node):

        def __init__(self):
            super().__init__("{model_name}_estimator")
            self.publisher_ = self.create_publisher(Float32MultiArray, "state_estimate", config.QUEUE_SIZE)
            self.create_subscription(Float32MultiArray, "sensor_measurement", self.listener_callback, config.QUEUE_SIZE)
            self.create_subscription(Float32MultiArray, "control_command", self.listener_callback_controller, config.QUEUE_SIZE)
            self.estimate = jnp.zeros(1)
            self.covariance = jnp.zeros((1, 1))
            self.control = jnp.zeros(1)
            self.estimator = self.get_estimator_by_name(config.ESTIMATOR_NAME)

        @staticmethod
        def get_estimator_by_name(name):
            '''
            Dynamically return an instance of the specified estimator.

            Parameters:
            - name: The name of the estimator function to import and use.

            Returns:
            A reference to the specified estimator function.
            '''
            module_path = f'cbfkit.estimators.{{name}}'
            estimator_module = __import__(module_path, fromlist=[name])
            estimator_function = getattr(estimator_module, name)
            return estimator_function

        def listener_callback_controller(self, msg):
            self.control = jnp.array(msg.data)

        def listener_callback(self, msg):
            t = 0.0  # USER_DEFINED
            measurement = jnp.array(msg.data)
            self.estimate, self.covariance = self.estimator(t, measurement, self.estimate, self.control, self.covariance)
            estimate_msg = self.create_estimate_msg(self.estimate)
            self.publisher_.publish(estimate_msg)

        def create_estimate_msg(self, estimate):
            estimate_msg = Float32MultiArray()
            estimate_msg.data = [float(z) for z in estimate]
            return estimate_msg

    def main(args=None):
        rclpy.init(args=args)
        node = VanDerPolOscillatorEstimator()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    if __name__ == "__main__":
        main()
    """
    )

    # Write the contents to the file
    with open(estimator_path, "w") as f:
        f.write(node_contents)

    print(f"Generated ROS2 estimator node script at {estimator_path}")

    run_black(estimator_path)

    # plant_model node
    plant_path = os.path.join(ros2_folder, "plant_model.py")
    node_contents = textwrap.dedent(
        f"""
    # Code-generated

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
    import jax.numpy as jnp
    from jax import random
    import config  # Import configurations

    class VanDerPolOscillatorPlant(Node):

        def __init__(self):
            super().__init__("{model_name}_plant")
            self.publisher_ = self.create_publisher(Float32MultiArray, "state", config.QUEUE_SIZE)
            self.create_subscription(Float32MultiArray, "control_command", self.listener_callback, config.QUEUE_SIZE)
            self.dynamics = self.get_dynamics_by_name(config.PLANT_NAME, **config.PLANT_PARAMS)
            self.integrator = self.get_integrator_by_name(config.INTEGRATOR_NAME)
            self.state = jnp.array([1.5, 0.25])  # Initial state
            self.dt = config.DT
            self.key = random.PRNGKey(0)  # Initialize random key

        @staticmethod
        def get_dynamics_by_name(name, **kwargs):
            '''
            Dynamically load the plant dynamics function.
            '''
            dynamics_module = __import__(config.MODULE_NAME + "." + name, fromlist=[name])
            dynamics_function = getattr(dynamics_module, name)
            return dynamics_function(**kwargs)

        @staticmethod
        def get_integrator_by_name(name):
            '''
            Dynamically load the integrator function.
            '''
            module_path = f"cbfkit.integration.{{name}}"
            integrator_module = __import__(module_path, fromlist=[name])
            integrator_function = getattr(integrator_module, name)
            return integrator_function

        def listener_callback(self, msg):
            control = jnp.array(msg.data)
            f, g = self.dynamics(self.state)

            # Generate perturbation
            self.key, subkey = random.split(self.key)
            p = self.perturbation(self.state, control, f, g, subkey)

            # Continuous-time dynamics with perturbation
            xdot = f + jnp.matmul(g, control) + p(subkey)
            self.state = self.integrator(self.state, xdot, self.dt)

            state_msg = self.create_state_message(self.state)
            self.publisher_.publish(state_msg)

        def perturbation(self, x, _u, _f, _g, key):
            # Example of adding a simple noise perturbation
            noise = random.normal(key, x.shape) * 0.01  # USER_DEFINED
            return lambda _subkey: noise

        def create_state_message(self, state):
            state_msg = Float32MultiArray()
            state_msg.data = [float(x) for x in state]
            return state_msg

    def main(args=None):
        rclpy.init(args=args)
        node = VanDerPolOscillatorPlant()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    if __name__ == "__main__":
        main()

    """
    )

    # Write the contents to the file
    with open(plant_path, "w") as f:
        f.write(node_contents)

    run_black(plant_path)

    print(f"Generated ROS2 plant node script at {plant_path}")

    # generate config file

    config_path = os.path.join(ros2_folder, "config.py")
    config_contents = textwrap.dedent(
        f"""
    # Configuration settings for Ros2 application
    # Defaults for {model_name}

    LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARN, ERROR, FATAL
    QUEUE_SIZE = 10
    MODULE_NAME = "{model_name}"

    # Controller
    CONTROLLER_NAME = "controller_1"
    TIMER_INTERVAL = 0.1
    CONTROLLER_PARAMS = {{'k_p': 1.0, 'epsilon': 0.5}}

    # Estimator
    ESTIMATOR_NAME = "naive"

    # Plant
    PLANT_NAME = "plant"
    PLANT_PARAMS = {{'epsilon': 0.5}}
    INTEGRATOR_NAME = "forward_euler"
    DT = 0.01

    # Sensor
    SENSOR_NAME = "perfect"
    SENSOR_PARAMS = {{}}
        """
    )

    # Write the contents to config.py
    with open(config_path, "w") as f:
        f.write(config_contents)

    print(f"Generated configuration script at {config_path}")

    # generate bash script for nodes

    # Paths for the ROS2 node scripts relative to the bash script's execution
    node_scripts = [
        "ros2/plant_model.py",
        "ros2/sensor.py",
        "ros2/estimator.py",
        "ros2/controller.py",
    ]

    # The directory where the bash script will be saved
    bash_script_path = os.path.join(model_folder, "run_ros2_nodes.sh")

    # Prepare the bash script content
    bash_script_contents = """#!/bin/bash

    # Source ROS2 environment
    source /opt/ros/humble/setup.bash

    # Run ROS2 node scripts
    """
    # Retreat back one directory
    bash_script_contents += "cd ..\n"

    # Append commands for running the ROS2 node scripts
    for script in node_scripts[:-1]:  # Exclude the last script for now
        bash_script_contents += f"python3 {model_name}/{script} &\n"

    # Add the last script without the '&' to avoid running it in the background
    bash_script_contents += f"python3 {model_name}/{node_scripts[-1]}"

    # Write the bash script contents to the file
    with open(bash_script_path, "w") as bash_file:
        bash_file.write(bash_script_contents)

    os.chmod(bash_script_path, 0o755)

    print(f"Generated bash script to run ROS2 nodes at: {bash_script_path}")

    return n_states, n_controls
