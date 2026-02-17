
import jax.numpy as jnp
import pytest
from cbfkit.utils.user_types import SimulationResults

def test_simulation_results_api():
    """Test the dictionary-like access of SimulationResults."""

    states = jnp.array([[0.0, 0.0]])
    controls = jnp.array([[1.0]])
    estimates = jnp.array([])
    covariances = jnp.array([])

    c_keys = ["sol", "status"]
    c_vals = [jnp.array([0.5]), jnp.array([1])]

    p_keys = ["x_traj"]
    p_vals = [jnp.array([[0.1, 0.1]])]

    res = SimulationResults(
        states, controls, estimates, covariances,
        c_keys, c_vals, p_keys, p_vals
    )

    # 1. Test field access via string
    assert jnp.all(res["states"] == states)

    # 2. Test controller data access via string
    assert jnp.all(res["sol"] == c_vals[0])

    # 3. Test planner data access via string
    assert jnp.all(res["x_traj"] == p_vals[0])

    # 4. Test legacy unpacking
    x, u, z, p, ck, cv, pk, pv = res
    assert jnp.all(x == states)

    # 5. Test index access
    assert jnp.all(res[0] == states)

    # 6. Test KeyError
    with pytest.raises(KeyError, match="not found in SimulationResults"):
        _ = res["non_existent"]

def test_simulation_results_api_collisions():
    """Test priority order in case of key collisions."""
    # Priority: fields > controller > planner

    states = jnp.array([1])

    # Controller uses "states" key (collision with field)
    c_keys = ["states"]
    c_vals = [jnp.array([2])] # This should be masked

    # Planner uses "states" key (collision with field)
    p_keys = ["states"]
    p_vals = [jnp.array([3])] # This should be masked

    res = SimulationResults(
        states, jnp.array([]), jnp.array([]), jnp.array([]),
        c_keys, c_vals, p_keys, p_vals
    )

    # Should get field value
    assert jnp.all(res["states"] == states)

    # Controller vs Planner collision
    # Controller uses "unique_c"
    # Planner uses "unique_c"

    c_keys = ["shared"]
    c_vals = [jnp.array([10])]

    p_keys = ["shared"]
    p_vals = [jnp.array([20])]

    res = SimulationResults(
        states, jnp.array([]), jnp.array([]), jnp.array([]),
        c_keys, c_vals, p_keys, p_vals
    )

    # Should get controller value (priority 2)
    assert jnp.all(res["shared"] == c_vals[0])


def test_simulation_results_legacy_aliases_and_tuple_helper():
    states = jnp.array([[1.0, 2.0]])
    controls = jnp.array([[0.5]])
    estimates = jnp.array([[1.1, 2.1]])
    covariances = jnp.array([[[1.0, 0.0], [0.0, 1.0]]])

    c_keys = ["solver_status"]
    c_vals = [jnp.array([1])]
    p_keys = ["x_traj"]
    p_vals = [jnp.array([[1.0, 2.0]])]

    res = SimulationResults(
        states,
        controls,
        estimates,
        covariances,
        c_keys,
        c_vals,
        p_keys,
        p_vals,
    )

    assert res.controller_data_keys == c_keys
    assert res.controller_data_values == c_vals
    assert res.planner_data_keys == p_keys
    assert res.planner_data_values == p_vals

    legacy = res.as_tuple()
    assert len(legacy) == 8
    assert jnp.all(legacy[0] == states)
    assert jnp.all(legacy[1] == controls)
    assert legacy[4] == c_keys
    assert legacy[6] == p_keys

    tail = res[4:]
    assert isinstance(tail, tuple)
    assert tail[0] == c_keys
