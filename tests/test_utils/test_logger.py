import os
import pytest
import jax.numpy as jnp
from cbfkit.utils.logger import write_log
from cbfkit.simulation.callbacks import LoggingCallback
from cbfkit.simulation.utils import SimulationStepData

def test_write_log_list_of_dicts(tmp_path):
    filepath = tmp_path / "log_list.csv"
    data = [
        {"a": 1, "b": 2.0, "c": jnp.array([1, 2])},
        {"a": 3, "b": 4.0, "c": jnp.array([3, 4])}
    ]
    write_log(str(filepath), data)
    assert filepath.exists()
    content = filepath.read_text()
    assert "a,b,c" in content
    assert "1,2.0" in content
    assert "3,4.0" in content

def test_write_log_dict_of_lists(tmp_path):
    filepath = tmp_path / "log_dict.csv"
    data = {
        "a": [1, 3],
        "b": [2.0, 4.0],
        "c": [jnp.array([1, 2]), jnp.array([3, 4])]
    }
    write_log(str(filepath), data)
    assert filepath.exists()
    content = filepath.read_text()
    assert "a,b,c" in content
    assert "1,2.0" in content
    assert "3,4.0" in content

def test_write_log_dict_unequal_lengths(tmp_path):
    filepath = tmp_path / "log_unequal.csv"
    data = {
        "a": [1, 3],
        "b": [2.0], # Too short
    }
    with pytest.raises(ValueError, match="same length"):
        write_log(str(filepath), data)

def test_write_log_extension_handling(tmp_path):
    filepath = tmp_path / "log_no_ext"
    data = [{"x": 1}]
    write_log(str(filepath), data)
    expected_path = tmp_path / "log_no_ext.csv"
    assert expected_path.exists()

def test_logging_callback_integration(tmp_path):
    filepath = tmp_path / "callback_log"
    callback = LoggingCallback(str(filepath))
    callback.on_start(total_steps=2, dt=0.1)

    # Mock data
    data1 = SimulationStepData(
        state=jnp.array([1.0]),
        control=jnp.array([0.1]),
        estimate=jnp.array([1.0]),
        covariance=jnp.array([[1.0]]),
        controller_keys=["u_nom"],
        controller_values=[jnp.array([0.0])],
        planner_keys=[],
        planner_values=[]
    )
    callback.on_step(0, 0.0, data1)

    data2 = SimulationStepData(
        state=jnp.array([2.0]),
        control=jnp.array([0.2]),
        estimate=jnp.array([2.0]),
        covariance=jnp.array([[1.0]]),
        controller_keys=["u_nom"],
        controller_values=[jnp.array([0.0])],
        planner_keys=[],
        planner_values=[]
    )
    callback.on_step(1, 0.1, data2)

    callback.on_end(success=True)

    expected_path = tmp_path / "callback_log.csv"
    assert expected_path.exists()
    content = expected_path.read_text()

    # Check headers
    assert "state" in content
    assert "control" in content
    assert "controller_u_nom" in content

    # Check values
    # JAX arrays format as [1.] for jnp.array([1.0])
    assert "[1.]" in content
    assert "[2.]" in content
