# Code-generated

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import jax.numpy as jnp
from jax import random
import config  # Import configurations


class VanDerPolOscillatorPlant(Node):
    def __init__(self):
        super().__init__("dynamics_plant")
        self.publisher_ = self.create_publisher(Float32MultiArray, "state", config.QUEUE_SIZE)
        self.create_subscription(
            Float32MultiArray, "control_command", self.listener_callback, config.QUEUE_SIZE
        )
        self.dynamics = self.get_dynamics_by_name(config.PLANT_NAME, **config.PLANT_PARAMS)
        self.integrator = self.get_integrator_by_name(config.INTEGRATOR_NAME)
        self.state = jnp.array([1.5, 0.25])  # Initial state
        self.dt = config.DT
        self.key = random.PRNGKey(0)  # Initialize random key

    @staticmethod
    def get_dynamics_by_name(name, **kwargs):
        """
        Dynamically load the plant dynamics function.
        """
        dynamics_module = __import__(config.MODULE_NAME + "." + name, fromlist=[name])
        dynamics_function = getattr(dynamics_module, name)
        return dynamics_function(**kwargs)

    @staticmethod
    def get_integrator_by_name(name):
        """
        Dynamically load the integrator function.
        """
        module_path = f"cbfkit.integration.{name}"
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
