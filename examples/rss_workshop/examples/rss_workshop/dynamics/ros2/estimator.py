# Code-generated

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import jax.numpy as jnp
import config


class VanDerPolOscillatorEstimator(Node):
    def __init__(self):
        super().__init__("dynamics_estimator")
        self.publisher_ = self.create_publisher(
            Float32MultiArray, "state_estimate", config.QUEUE_SIZE
        )
        self.create_subscription(
            Float32MultiArray, "sensor_measurement", self.listener_callback, config.QUEUE_SIZE
        )
        self.create_subscription(
            Float32MultiArray,
            "control_command",
            self.listener_callback_controller,
            config.QUEUE_SIZE,
        )
        self.estimate = jnp.zeros(1)
        self.covariance = jnp.zeros((1, 1))
        self.control = jnp.zeros(1)
        self.estimator = self.get_estimator_by_name(config.ESTIMATOR_NAME)

    @staticmethod
    def get_estimator_by_name(name):
        """
        Dynamically return an instance of the specified estimator.

        Parameters:
        - name: The name of the estimator function to import and use.

        Returns:
        A reference to the specified estimator function.
        """
        module_path = f"cbfkit.estimators.{name}"
        estimator_module = __import__(module_path, fromlist=[name])
        estimator_function = getattr(estimator_module, name)
        return estimator_function

    def listener_callback_controller(self, msg):
        self.control = jnp.array(msg.data)

    def listener_callback(self, msg):
        t = 0.0  # USER_DEFINED
        measurement = jnp.array(msg.data)
        self.estimate, self.covariance = self.estimator(
            t, measurement, self.estimate, self.control, self.covariance
        )
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
