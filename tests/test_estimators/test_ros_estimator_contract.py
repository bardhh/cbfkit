"""ROS stepping accepts both standard estimates and EKF gain telemetry."""

import jax.numpy as jnp
import pytest

from cbfkit.ros._experiment import stepper


@pytest.mark.parametrize("with_gain", [False, True])
def test_ros_step_preserves_estimate_and_covariance(with_gain):
    estimate = jnp.array([2.0])
    covariance = jnp.eye(1)

    def estimator(t, y, z, u, p):
        result = (estimate, covariance)
        return result + (jnp.eye(1),) if with_gain else result

    def controller(t, z):
        return -z, {"time": t}

    step = stepper(lambda: jnp.zeros(1), controller, estimator)
    u, z, p, data = step(0.5, None, jnp.zeros(1), None)
    assert z is estimate and p is covariance
    assert jnp.array_equal(u, -estimate)
    assert data == {"time": 0.5}
