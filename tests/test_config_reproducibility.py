import sys
import importlib
import jax.numpy as jnp
import numpy as np
import pytest

def test_ukf_config_reproducibility():
    """Test that UKFEstimationConfig in examples.van_der_pol.common.config is deterministic."""

    # First import
    if "examples.van_der_pol.common.config" in sys.modules:
        del sys.modules["examples.van_der_pol.common.config"]

    import examples.van_der_pol.common.config as config1
    val1 = config1.ukf_state_estimation.initial_state

    # Second import (reload)
    # We must delete the module from sys.modules to force re-execution of the module body
    del sys.modules["examples.van_der_pol.common.config"]
    del config1

    import examples.van_der_pol.common.config as config2
    val2 = config2.ukf_state_estimation.initial_state

    # Use explicit assertion failure message
    np.testing.assert_array_equal(val1, val2,
                                  err_msg="Config initial_state should be deterministic across reloads")
