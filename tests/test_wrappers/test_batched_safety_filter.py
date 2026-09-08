"""Behavioral coverage for independent, device-resident safety filtering."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cbfkit.utils.user_types import ControllerData
from cbfkit.wrappers.batched import BatchedSafetyFilter


def accumulating_controller(t, x, action, key, data):
    previous = 0.0 if data.sub_data is None else data.sub_data["total"]
    total = previous + action
    return total + t, ControllerData(sub_data={"total": total})


def test_history_and_selective_reset_are_independent():
    sf = BatchedSafetyFilter(accumulating_controller, num_envs=3, dt=0.1)
    x = jnp.zeros((3, 1))
    action = jnp.array([[1.0], [2.0], [3.0]])
    np.testing.assert_allclose(sf.filter(x, action)[0], action)
    np.testing.assert_allclose(sf.filter(x, action)[0], 2 * action + 0.1)
    sf.reset(jnp.array([False, True, False]))
    np.testing.assert_allclose(sf.filter(x, action)[0], [[3.2], [2], [9.2]])
    np.testing.assert_allclose(sf.time, [0.3, 0.1, 0.3])


def test_failure_is_isolated_and_nonfinite_output_detected():
    def controller(t, x, u, key, data):
        return jnp.where(x < 0, jnp.inf, u), ControllerData(error=x[0] == 0)

    sf = BatchedSafetyFilter(controller, num_envs=3, fallback="zero")
    u, info = sf.filter(jnp.array([[-1.0], [0.0], [1.0]]), jnp.ones((3, 1)))
    np.testing.assert_array_equal(u, [[0], [0], [1]])
    np.testing.assert_array_equal(info["fallback_used"], [True, True, False])
    assert isinstance(info["fallback_used"], jax.Array)


def test_random_streams_are_independent_and_reset_reproducible():
    def controller(t, x, u, key, data):
        return jax.random.uniform(key, u.shape), ControllerData()

    sf = BatchedSafetyFilter(controller, num_envs=3, seed=42)
    x = jnp.zeros((3, 1))
    first = sf.filter(x, x)[0]
    assert len(np.unique(first)) == 3
    sf.filter(x, x)
    sf.reset()
    np.testing.assert_array_equal(sf.filter(x, x)[0], first)


def test_invalid_shapes_and_configuration():
    with pytest.raises(ValueError):
        BatchedSafetyFilter(accumulating_controller, num_envs=0)
    sf = BatchedSafetyFilter(accumulating_controller, num_envs=2)
    with pytest.raises(ValueError):
        sf.filter(jnp.zeros((1, 2)), jnp.zeros((2, 1)))
    with pytest.raises(ValueError):
        sf.reset(jnp.array([0, 1]))


def test_failed_history_is_discarded_and_pending_reset_masks_accumulate():
    def controller(t, x, u, key, data):
        previous = 0.0 if data.sub_data is None else data.sub_data["previous"]
        out = u + previous
        return out, ControllerData(sub_data={"previous": x})

    sf = BatchedSafetyFilter(controller, num_envs=3)
    action = jnp.ones((3, 1))
    # Invalid input is flagged even when the controller returns a finite action.
    _, info = sf.filter(jnp.array([[jnp.nan], [2.0], [3.0]]), action)
    np.testing.assert_array_equal(info["fallback_used"], [True, False, False])
    out, _ = sf.filter(jnp.zeros((3, 1)), action)
    np.testing.assert_allclose(out, [[1.0], [3.0], [4.0]])
    sf.filter(jnp.ones((3, 1)), action)
    sf.reset(jnp.array([True, False, False]))
    sf.reset(jnp.array([False, False, True]))
    np.testing.assert_allclose(sf.filter(jnp.zeros((3, 1)), action)[0], [[1.0], [2.0], [1.0]])


def test_qp_matches_scalar_filters_over_steps_and_reset():
    from cbfkit.certificates import generate_certificate
    from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
    from cbfkit.optimization.quadratic_program import get_solver
    from cbfkit.systems.single_integrator.dynamics import two_dimensional_single_integrator
    from cbfkit.wrappers import SafetyFilter

    kwargs = dict(
        dynamics=two_dimensional_single_integrator(),
        barriers=generate_certificate(lambda x: x[0], linear_class_k(1.0), input_style="state"),
        control_limits=jnp.ones(2),
        solver=get_solver("fast"),
    )
    sf = BatchedSafetyFilter.from_cbf_qp(num_envs=2, **kwargs)
    single = [SafetyFilter.from_cbf_qp(**kwargs) for _ in range(2)]
    x = jnp.array([[0.1, 0.0], [2.0, 0.0]])
    action = jnp.array([[-1.0, 0.0], [-0.5, 0.0]])
    for step in range(4):
        if step == 2:
            sf.reset(jnp.array([True, False]))
            single[0].reset()
        u, info = sf.filter(x, action)
        expected = jnp.stack([f.filter(x[i], action[i])[0] for i, f in enumerate(single)])
        np.testing.assert_allclose(u, expected, atol=1e-5)
        assert np.all(np.asarray(u[:, 0] + x[:, 0]) >= -1e-5)
        assert not np.any(info["fallback_used"])


def test_float32_inputs_reach_controller_in_float64():
    def controller(t, x, u, key, data):
        assert x.dtype == u.dtype == jnp.float64
        # This increment vanishes if arithmetic is performed in float32.
        return u + 1e-10 * x, ControllerData(sub_data=(x, [u]))

    sf = BatchedSafetyFilter(controller, num_envs=2)
    x = jnp.ones((2, 1), dtype=jnp.float32)
    applied, info = sf.filter(x, x)
    assert applied.dtype == info["u_qp"].dtype == info["u_nom"].dtype == jnp.float64
    np.testing.assert_allclose(applied, 1.0 + 1e-10, rtol=0, atol=1e-14)
    # A custom non-dictionary history must remain usable after the cold call.
    sf.filter(x, x)
    sf.reset(jnp.array([True, False]))
    sf.filter(x, x)


def test_new_seed_persists_only_for_selected_environments():
    def controller(t, x, u, key, data):
        return jax.random.uniform(key, u.shape), ControllerData()

    sf = BatchedSafetyFilter(controller, num_envs=2, seed=1)
    x = jnp.zeros((2, 1))
    original = sf.filter(x, x)[0]
    sf.reset(jnp.array([True, False]), seed=9)
    reseeded = sf.filter(x, x)[0]
    assert not np.array_equal(reseeded[0], original[0])
    sf.reset()
    actual = sf.filter(x, x)[0]
    np.testing.assert_array_equal(actual[0], reseeded[0])
    np.testing.assert_array_equal(actual[1], original[1])
