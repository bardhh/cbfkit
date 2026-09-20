import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cbfkit.systems.mujoco._policy_commands import normalize_torso_command


@pytest.mark.parametrize("size", [3, 4])
def test_constant_commands(size):
    x = jnp.zeros(2)
    np.testing.assert_array_equal(normalize_torso_command(None, size)(0.0, x, {}), np.zeros(size))
    command = tuple(range(size))
    np.testing.assert_array_equal(normalize_torso_command(command, size)(0.0, x, {}), command)


def test_scheduled_and_state_aware_commands_under_jit():
    scheduled = jax.jit(normalize_torso_command(lambda t: jnp.ones(3) * t, 3))
    aware = jax.jit(normalize_torso_command(lambda t, x, sub: x + sub["offset"] + t, 3))
    np.testing.assert_array_equal(scheduled(2.0, jnp.zeros(3), {}), [2.0, 2.0, 2.0])
    np.testing.assert_array_equal(aware(2.0, jnp.ones(3), {"offset": 3.0}), [6.0, 6.0, 6.0])


def test_invalid_signature_fails_at_setup():
    with pytest.raises(ValueError, match="torso command"):
        normalize_torso_command(lambda t, x: x, 3)
