import rclpy
from typing import Callable, Tuple
from jax import Array


def sensor(compile_state: Callable[[], Array]) -> Callable[[], Tuple[float, Array]]:
    """Generates 'sense' function for ROS2 sensor object.

    Returns:
        Callable[[], Tuple[float, Array]]: _description_
    """
    rclpy.init()
    node = rclpy.create_node("sensor_node")

    def sense() -> Tuple[float, Array]:
        t = node.get_clock().now().to_msg().sec
        y = compile_state()

        rclpy.shutdown()

        return t, y

    return sense
