"""Crowd adapters must accept the public default ControllerData."""

import jax.numpy as jnp

from cbfkit.systems.pedestrian.manager import CrowdManager
from cbfkit.utils.user_types import ControllerData


def test_crowd_controller_initializes_missing_sub_data():
    inner = ControllerData(complete=True)

    def robot(t, x, nominal, key, data):
        assert isinstance(data, ControllerData)
        return jnp.ones(2), inner

    controller = CrowdManager().get_nominal_controller(robot)
    control, data = controller(0.0, jnp.zeros(4), data=ControllerData(error=True))
    assert jnp.array_equal(control, jnp.ones(2))
    assert data.sub_data["inner_controller_data"] is inner
    assert data.error is True
