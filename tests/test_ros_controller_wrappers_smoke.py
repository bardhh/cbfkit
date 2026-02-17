import importlib
import sys
import types

import jax.numpy as jnp

from cbfkit.utils.user_types import ControllerData


def _import_ros_controller_wrappers(monkeypatch):
    log_messages = []
    fake_rospy = types.ModuleType("rospy")

    class FakeTime:
        def __init__(self, sec):
            self._sec = sec

        def to_sec(self):
            return self._sec

    fake_rospy.Time = FakeTime
    fake_rospy.loginfo = log_messages.append
    monkeypatch.setitem(sys.modules, "rospy", fake_rospy)
    sys.modules.pop("cbfkit.ros.controller_wrappers", None)
    module = importlib.import_module("cbfkit.ros.controller_wrappers")
    return module, FakeTime, log_messages


def _import_ros2_controller_wrappers(monkeypatch):
    error_messages = []
    fake_rclpy = types.ModuleType("rclpy")
    fake_rclpy_time = types.ModuleType("rclpy.time")
    fake_rclpy_logging = types.ModuleType("rclpy.logging")

    class FakeTime:
        def __init__(self, sec):
            self._sec = sec

        def to_msg(self):
            return types.SimpleNamespace(sec=self._sec)

    class FakeLogger:
        def error(self, message):
            error_messages.append(message)

    fake_rclpy_time.Time = FakeTime
    fake_rclpy.time = fake_rclpy_time
    fake_rclpy_logging.get_logger = lambda _name: FakeLogger()

    monkeypatch.setitem(sys.modules, "rclpy", fake_rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.time", fake_rclpy_time)
    monkeypatch.setitem(sys.modules, "rclpy.logging", fake_rclpy_logging)
    sys.modules.pop("cbfkit.ros2.controller_wrappers", None)
    module = importlib.import_module("cbfkit.ros2.controller_wrappers")
    return module, FakeTime, error_messages


def test_ros_controller_wrapper_uses_backup_on_violation(monkeypatch):
    wrappers, fake_time_type, log_messages = _import_ros_controller_wrappers(monkeypatch)
    published = []
    backup_calls = []

    def publish(u, x):
        published.append((u, x))

    def primary_controller(_t, _x, _u_nom, _key, _data):
        return jnp.array([1.0]), ControllerData(sub_data={"violated": True})

    def backup_controller(t, x, u_nom, key, data):
        backup_calls.append((t, x, u_nom, key, data))
        return jnp.array([2.0]), ControllerData(sub_data={"source": "backup"})

    wrapped = wrappers.controller_wrapper(primary_controller, backup_controller, publish)
    u, data = wrapped(
        fake_time_type(3.5),
        jnp.array([0.0]),
        jnp.array([0.0]),
        jnp.array([0, 1], dtype=jnp.uint32),
        ControllerData(),
    )

    assert bool(jnp.allclose(u, jnp.array([2.0])))
    assert data.sub_data == {"source": "backup"}
    assert len(backup_calls) == 1
    assert len(log_messages) == 1
    assert "Violation of Safety Constraint!" in str(log_messages[0])
    assert len(published) == 1
    assert bool(jnp.allclose(published[0][0], jnp.array([2.0])))


def test_ros2_controller_wrapper_uses_backup_on_violation(monkeypatch):
    wrappers, fake_time_type, error_messages = _import_ros2_controller_wrappers(monkeypatch)
    published = []
    backup_calls = []

    def publish(u, x):
        published.append((u, x))

    def primary_controller(_t, _x, _u_nom, _key, _data):
        return jnp.array([1.0]), ControllerData(sub_data={"violated": True})

    def backup_controller(t, x, u_nom, key, data):
        backup_calls.append((t, x, u_nom, key, data))
        return jnp.array([4.0]), ControllerData(sub_data={"source": "backup"})

    wrapped = wrappers.controller_wrapper(primary_controller, backup_controller, publish)
    u, data = wrapped(
        fake_time_type(9),
        jnp.array([0.0]),
        jnp.array([0.0]),
        jnp.array([0, 1], dtype=jnp.uint32),
        ControllerData(),
    )

    assert bool(jnp.allclose(u, jnp.array([4.0])))
    assert data.sub_data == {"source": "backup"}
    assert len(backup_calls) == 1
    assert len(error_messages) == 1
    assert "Violation of Safety Constraint!" in str(error_messages[0])
    assert len(published) == 1
    assert bool(jnp.allclose(published[0][0], jnp.array([4.0])))


def test_ros_controller_helper_wraps_metadata(monkeypatch):
    wrappers, _, _ = _import_ros_controller_wrappers(monkeypatch)
    wrapped = wrappers.ros_controller(
        lambda: (jnp.array([7.0]), {"mode": "ros", "active": True})
    )

    u, data = wrapped(
        0.0,
        jnp.array([0.0]),
        jnp.array([0.0]),
        jnp.array([0, 1], dtype=jnp.uint32),
        ControllerData(),
    )

    assert bool(jnp.allclose(u, jnp.array([7.0])))
    assert data.sub_data == {"mode": "ros", "active": True}


def test_ros2_controller_helper_wraps_metadata(monkeypatch):
    wrappers, _, _ = _import_ros2_controller_wrappers(monkeypatch)
    wrapped = wrappers.ros_controller(
        lambda: (jnp.array([8.0]), {"mode": "ros2", "active": True})
    )

    u, data = wrapped(
        0.0,
        jnp.array([0.0]),
        jnp.array([0.0]),
        jnp.array([0, 1], dtype=jnp.uint32),
        ControllerData(),
    )

    assert bool(jnp.allclose(u, jnp.array([8.0])))
    assert data.sub_data == {"mode": "ros2", "active": True}
