# Code-generated

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import jax.numpy as jnp
import config


class VanDerPolOscillatorController(Node):
    def __init__(self):
        super().__init__("dynamics_controller")
        self.publisher = self.create_publisher(
            Float32MultiArray, "control_command", config.QUEUE_SIZE
        )
        self.subscription = self.create_subscription(
            Float32MultiArray, "state_estimate", self.listener_callback, config.QUEUE_SIZE
        )
        self.controller = self.get_controller_by_name(
            config.CONTROLLER_NAME, **config.CONTROLLER_PARAMS
        )
        self.reset_control_msg()
        self.timer = self.create_timer(config.TIMER_INTERVAL, self.publish_control)

    @staticmethod
    def get_controller_by_name(name, **kwargs):
        """
        Dynamically import and return an instance of the specified controller.

        Parameters:
        - name: The name of the controller class to import and instantiate.
        - **kwargs: Keyword arguments to pass to the controller's constructor.

        Returns:
        An instance of the specified controller.
        """
        # Assuming all controllers are in a module named 'controllers'
        module = __import__(config.MODULE_NAME + "." + "controllers", fromlist=[name])
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
        t = 0.0  # USER_DEFINED
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
