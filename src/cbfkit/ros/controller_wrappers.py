import rospy
from jax import Array
from cbfkit.utils.user_types import ControllerCallable, ControllerCallableReturns
from typing import Tuple, Dict, Callable


def controller_wrapper(
    controller: ControllerCallable,
    backup_controller: ControllerCallable,
    publish: Callable[[Array, Array], None],
) -> ControllerCallable:
    """Wrapper for any generic controller function to publish to specified ROS topics.
    Requires the controller function handle, a list of publishers, and a handle to
    the publish function which processes the publish request for the specific
    configuration.

    Args:
        controller (ControllerCallable): handle to the controller function (computes input)
        publish (Callable[[Array, Array], None]): handle to the function which
            carries out the publish request. Must be configured for the specific environment.

    Returns:
        wrapped_controller (Callable[[float, Array], Tuple[Array, Dict]]): handle to the wrapped_controller function
    """

    def wrapped_controller(t: float, x: Array) -> ControllerCallableReturns:
        """_summary_

        Args:
            t (float): time in sec
            x (Array): estimated state vector

        Returns:
            Tuple: (computed input u, dict containing misc data)
        """
        if type(t) == rospy.Time:
            t = t.to_sec()

        try:
            u, data = controller(t, x)
            if "sub_data" in data.keys():
                if "violated" in data["sub_data"].keys():
                    if data["sub_data"]["violated"]:
                        raise ValueError("Violation of Safety Constraint!")
        except ValueError as e:
            rospy.loginfo(e)
            u, data = backup_controller(t, x)

        # Sends the computed input u and the estimated state x
        # since the published message may require x as well as u
        publish(u, x)

        return u, data

    return wrapped_controller


def ros_controller(extract_control: Callable[[], Tuple[Array, Dict]]) -> ControllerCallable:
    """_summary_

    Args:
        extract_control (Callable): _description_

    Returns:
        Tuple[Array, Dict]: contains computed input u and dictionary containing extra data
    """

    def controller(_t: float, _x: Array) -> ControllerCallableReturns:
        return extract_control(), {}

    return controller
