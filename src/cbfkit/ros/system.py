"""module docstring here"""


from typing import Callable
import rospy
from cbfkit.utils.user_types import ControllerCallable, SensorCallable
from std_msgs.msg import Float64

NODE_NAME = rospy.get_name()
execution_time_publisher = rospy.Publisher(f"/{NODE_NAME}/execution_time", Float64, queue_size=1)


def spin(node_name: str, callback: Callable, frequency: float = 0.01):
    """_summary_

    Args:
        callback (Callable): _description_
        control_period (float, optional): _description_. Defaults to 0.01.
    """

    def spinner():
        rospy.init_node(node_name, anonymous=True)

        # Set up callback
        rospy.Timer(
            rospy.Duration(frequency),
            callback,
        )

        rospy.spin()

    return spinner


def stepper(
    sensor: SensorCallable,
    controller: ControllerCallable,
) -> Callable[[None], None]:
    """Step function to take the simulation forward one timestep. Designed
    to work generically with broad classes of dynamics, controllers, and
    estimators.

    Args:
        sensor (Callable): function handle to generate new state sensor
        controller (Callable): function handle to compute control input

    Returns:
        step (Callable): function handle for computing one step in ROS sim/experiment


    """

    def step(_something):
        """Callback for ROS spin. Essentially, grabs a sensor measurement
        and computes a control measurement

        Args:
            None

        Returns:
            None
        """
        start_time = rospy.Time.now().to_sec()

        # Get sensor measurement (subscribers inside) -- assume perfect measurements
        t, y = sensor()

        # Compute control input using ROS-wrapped controller (takes care of publishing)
        _u, _data = controller(t, y)

        # Compute execution time
        execution_time_ms = 1e3 * (rospy.Time.now().to_sec() - start_time)

        # Execution time message
        execution_time_msg = Float64()
        execution_time_msg.data = execution_time_ms
        execution_time_publisher.publish(execution_time_msg)

    return step
