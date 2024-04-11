import rospy
from typing import Callable, Tuple
from jax import Array


def sensor(compile_state: Callable[[], Array]) -> Callable[[], Tuple[float, Array]]:
    """Generates 'sense' function for ROS sensor object.

    Returns:
        Callable[[], Tuple[float, Array]]: _description_
    """

    def sense() -> Tuple[float, Array]:
        t = rospy.get_rostime()
        y = compile_state()

        return t, y

    return sense
