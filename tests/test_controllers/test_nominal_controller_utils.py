import jax.numpy as jnp
from jax import random
import pytest
from cbfkit.controllers.utils import setup_nominal_controller
from cbfkit.utils.user_types import ControllerData

def test_setup_nominal_controller_2_args():
    def simple_ctrl(t, x):
        return -x

    nominal_ctrl = setup_nominal_controller(simple_ctrl)

    t = 0.0
    x = jnp.array([1.0, 2.0])
    key = random.PRNGKey(0)

    u, data = nominal_ctrl(t, x, key, None)

    assert jnp.allclose(u, -x)
    assert jnp.allclose(data.u_nom, -x)

def test_setup_nominal_controller_3_args():
    def tracking_ctrl(t, x, ref):
        if ref is not None:
            return ref - x
        return -x

    nominal_ctrl = setup_nominal_controller(tracking_ctrl)

    t = 0.0
    x = jnp.array([1.0, 2.0])
    key = random.PRNGKey(0)
    ref = jnp.array([2.0, 4.0])

    u, data = nominal_ctrl(t, x, key, ref)

    assert jnp.allclose(u, ref - x)
    assert jnp.allclose(data.u_nom, ref - x)

    # Test with None ref
    # If the user function handles None, it should work
    u_none, _ = nominal_ctrl(t, x, key, None)
    assert jnp.allclose(u_none, -x)

def test_setup_nominal_controller_4_args():
    def complex_ctrl(t, x, key, ref):
        return x, ControllerData(u_nom=x)

    nominal_ctrl = setup_nominal_controller(complex_ctrl)

    t = 0.0
    x = jnp.array([1.0, 2.0])
    key = random.PRNGKey(0)

    u, data = nominal_ctrl(t, x, key, None)
    assert jnp.allclose(u, x)

    # Test 4 args returning only array
    def complex_ctrl_array(t, x, key, ref):
        return x

    nominal_ctrl_arr = setup_nominal_controller(complex_ctrl_array)
    u_arr, data_arr = nominal_ctrl_arr(t, x, key, None)
    assert jnp.allclose(u_arr, x)
    assert jnp.allclose(data_arr.u_nom, x)

def test_unsupported_signature():
    def bad_ctrl(t):
        return t

    with pytest.raises(ValueError):
        setup_nominal_controller(bad_ctrl)
