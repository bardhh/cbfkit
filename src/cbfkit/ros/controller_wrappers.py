from typing import Callable, Dict, Optional, Tuple, cast

import rospy
from jax import Array

from cbfkit.utils.user_types import (
    Control,
    ControllerCallable,
    ControllerCallableReturns,
    ControllerData,
    Key,
    State,
    Time,
)


def controller_wrapper(
    controller: ControllerCallable,
    backup_controller: ControllerCallable,
    publish: Callable[[Array, Array], None],
) -> ControllerCallable:
    """Wrapper for any generic controller function to publish to specified ROS topics. Requires the
    controller function handle, a list of publishers, and a handle to the publish function which
    processes the publish request for the specific configuration.

    Args:
        controller (ControllerCallable): handle to the controller function (computes input)
        publish (Callable[[Array, Array], None]): handle to the function which
            carries out the publish request. Must be configured for the specific environment.

    Returns
    -------
        wrapped_controller (Callable[[float, Array], Tuple[Array, Dict]]): handle to the
        wrapped_controller function
    """

    def wrapped_controller(
        t: Time,
        x: State,
        u_nom: Control,
        key: Key,
        data: ControllerData,
    ) -> ControllerCallableReturns:
        """Invoke the primary controller and fall back to backup on constraint violations."""
        _t_float: float
        if isinstance(t, rospy.Time):
            _t_float = t.to_sec()
        elif isinstance(t, Array):
            _t_float = float(t[0])  # Assuming it's a scalar array
        else:
            _t_float = t

        try:
            u, data = controller(_t_float, x, u_nom, key, data)
            sub_data = data.sub_data or {}
            if "violated" in sub_data and sub_data["violated"]:
                raise ValueError("Violation of Safety Constraint!")
        except ValueError as e:
            rospy.loginfo(e)
            u, data = backup_controller(_t_float, x, u_nom, key, data)

        # Sends the computed input u and the estimated state x
        # since the published message may require x as well as u
        publish(u, x)

        return u, data

    return wrapped_controller


def ros_controller(extract_control: Callable[[], Tuple[Array, Dict]]) -> ControllerCallable:
    """Adapt a ROS-side control extractor into the canonical controller callable."""

    def controller(
        _t: Time,
        _x: State,
        _u_nom: Optional[Control],
        _key: Key,
        _data: ControllerData,
    ) -> ControllerCallableReturns:
        u, d = extract_control()
        return u, ControllerData(sub_data=d)

    return controller
