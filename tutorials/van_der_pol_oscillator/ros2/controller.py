# Code-generated

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import jax.numpy as jnp
import config


class VanDerPolOscillatorController(Node):
    def __init__(self):
        super().__init__("van_der_pol_oscillator_controller")
        self.publisher = self.create_publisher(
            Float32MultiArray, "control_command", config.QUEUE_SIZE
        )
        self.subscription = self.create_subscription(
            Float32MultiArray,
            "state_estimate",
            self.listener_callback,
            config.QUEUE_SIZE,
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
        """
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
        # Dummy u_nom and key for ROS node usage
        u_nom = jnp.zeros(1)
        key = 0
        from cbfkit.utils.user_types import ControllerData

        data = ControllerData()

        control_command, _ = self.controller(t, estimate, u_nom, key, data)
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
