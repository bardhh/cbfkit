# Code-generated

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import jax.numpy as jnp
import config  # Import configurations


class VanDerPolOscillatorSensor(Node):
    def __init__(self):
        super().__init__("dynamics_sensor")
        self.publisher_ = self.create_publisher(
            Float32MultiArray, "sensor_measurement", config.QUEUE_SIZE
        )
        self.create_subscription(
            Float32MultiArray, "state", self.listener_callback, config.QUEUE_SIZE
        )
        self.sensor = self.get_sensor_by_name(config.SENSOR_NAME)

    @staticmethod
    def get_sensor_by_name(name):
        """
        Dynamically load the sensor function.
        """
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
