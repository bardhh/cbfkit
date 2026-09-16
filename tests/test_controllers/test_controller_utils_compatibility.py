import jax.numpy as jnp
import pytest

import cbfkit.simulation.simulator as sim
from cbfkit.controllers.utils import setup_controller
from cbfkit.integration.forward_euler import forward_euler
from cbfkit.utils.user_types import ControllerData


def _dummy_dynamics(_x):
    return jnp.zeros((1,)), jnp.array([[1.0]])


def test_setup_controller_two_arg_signature():
    def legacy_controller(t, x):
        return jnp.array([t + x[0]])

    wrapped = setup_controller(legacy_controller)
    u_nom = jnp.array([0.25])
    u, data = wrapped(
        0.5,
        jnp.array([1.0]),
        u_nom,
        jnp.array([0, 0], dtype=jnp.uint32),
        ControllerData(),
    )

    assert jnp.allclose(u, jnp.array([1.5]))
    assert isinstance(data, ControllerData)
    assert data.u_nom is u_nom


def test_setup_controller_four_arg_key_data_signature():
    def legacy_controller(t, x, key, data):
        _ = (t, x, key, data)
        return jnp.array([2.0]), ControllerData(complete=True)

    wrapped = setup_controller(legacy_controller)
    u, data = wrapped(
        0.0,
        jnp.array([0.0]),
        jnp.array([0.0]),
        jnp.array([0, 0], dtype=jnp.uint32),
        ControllerData(),
    )

    assert jnp.allclose(u, jnp.array([2.0]))
    assert data.complete


@pytest.mark.parametrize("signature", ["key_data", "nominal_key"])
def test_explicit_four_arg_layout_ignores_parameter_names(signature):
    received = []

    def ambiguous(a, b, c, d):
        received.append((c, d))
        return jnp.array([1.0])

    nominal = jnp.array([2.0])
    key = jnp.array([0, 0], dtype=jnp.uint32)
    data = ControllerData()
    wrapped = setup_controller(ambiguous, signature=signature)
    wrapped(0.0, jnp.zeros(1), nominal, key, data)
    expected = (key, data) if signature == "key_data" else (nominal, key)
    assert received[0][0] is expected[0]
    assert received[0][1] is expected[1]


def test_explicit_layout_rejects_other_arities():
    with pytest.raises(ValueError, match="four-argument"):
        setup_controller(lambda t, x: x, signature="key_data")


def test_setup_controller_rejects_unsupported_signature():
    def bad_controller(_t, _x, _u_nom, _key, _data, _extra):
        return jnp.array([0.0])

    with pytest.raises(ValueError, match="Unsupported controller signature"):
        setup_controller(bad_controller)


def test_execute_accepts_legacy_two_arg_controller():
    def legacy_controller(_t, _x):
        return jnp.array([0.0])

    results = sim.execute(
        x0=jnp.array([0.0]),
        dt=0.1,
        num_steps=3,
        dynamics=_dummy_dynamics,
        integrator=forward_euler,
        controller=legacy_controller,  # type: ignore[arg-type]
        verbose=False,
        use_jit=False,
    )

    assert results.states.shape[0] == 3
    assert results.controls.shape == (3, 1)


def test_simulator_factory_accepts_legacy_two_arg_controller():
    def legacy_controller(_t, _x):
        return jnp.array([0.0])

    simulate_iter = sim.simulator(
        dt=0.1,
        num_steps=3,
        dynamics=_dummy_dynamics,
        integrator=forward_euler,
        planner=None,
        nominal_controller=None,
        controller=legacy_controller,  # type: ignore[arg-type]
        sensor=None,
        estimator=None,
        perturbation=None,
        sigma=None,
        key=jnp.array([0, 0], dtype=jnp.uint32),
    )

    steps = list(simulate_iter(jnp.array([0.0])))
    assert len(steps) == 3
    assert steps[-1].control is not None
    assert steps[-1].control.shape == (1,)
