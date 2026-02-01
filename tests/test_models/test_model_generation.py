"""
Test Module for generating arbitrary dynamics models with barrier and lyapunov functions.
=========================

This module contains tests for verifying functionality of the arbitrary model generation
feature.

Tests
-----
Code generation for models, nominal controllers, barrier functions, and lyapunov functions
specified for the:
- 2D Double Integrator Model
- Kinematic Bicycle Model
- Kinematic Unicycle Model

Setup
-----
- No set up required

Examples
--------
To run all tests in this module (from the root of the repository):
    $ python -m unittest tests.test_models.test_model_generation
"""

import unittest
from typing import Optional, List, Union, Dict, Any
from cbfkit.codegen.create_new_system.generate_model import generate_model


class TestModelGeneration(unittest.TestCase):
    """Test suite for Model Generation functionality.

    Args:
        unittest (_type_): _description_
    """

    def test_2d_double_integrator_generation(self):
        """Tests the model generation capabilities of a 2D integrator system."""
        drift_dynamics = "[x[2], x[3], 0.0, 0.0]"
        control_matrix = "[[0, 0], [0, 0], [1, 0], [0, 1]]"
        barrier_functions = ["x[0]", "x[1] + 3"]
        lyapunov_functions = "(x[0] - 2)**2 + (x[1] - 4)**2 - 1"
        nominal_controller = "[kp * (x[0] - 1), kp * (x[1] - 2)]"
        params = {"controller": {"kp: float": 1.0}}
        return self._test_model_generation(
            "tests/test_models/generated",
            "double_integrator_2d",
            (4, 2),
            drift_dynamics,
            control_matrix,
            barrier_functions,
            lyapunov_functions,
            nominal_controller,
            params,
        )

    def test_kinematic_bicycle_generation(self):
        """Tests the model generation capabilities of a kinematic bicycle system."""
        drift_dynamics = [
            "x[4] * (cos(x[2]) - sin(x[2]) * tan(x[3]))",
            "x[4] * (sin(x[2]) + cos(x[2]) * tan(x[3]))",
            "x[4] / lr * sin(x[3])",
            "0",
            "0",
        ]

        control_matrix = ["[0, 0]", "[0, 0]", "[0, 0]", "[1, 0]", "[0, 1]"]

        barrier_functions = "(x[0] - 2)**2 + (x[1] + 3)**2 - (2*2.0)**2"

        return self._test_model_generation(
            "tests/test_models/generated",
            "kinematic_bicycle",
            (5, 2),
            drift_dynamics,
            control_matrix,
            barrier_functions,
            params={"dynamics": {"lr: float": 1.0}},
        )

    def test_unicycle_generation(self):
        """Tests the model generation capabilities of a unicycle system."""
        drift_dynamics = "[0,0,0]"
        control_matrix = "[[cos(x[2]), 0], [sin(x[2]), 0], [0, 1]]"
        barrier_functions = "(x[0] - 2)**2 + (x[1] + 3)**2 - (2*2.0)**2"

        return self._test_model_generation(
            "tests/test_models/generated",
            "unicycle",
            (3, 2),
            drift_dynamics,
            control_matrix,
            barrier_functions,
        )

    def _test_model_generation(
        self,
        target_directory: str,
        model_name: str,
        dims: int,
        drift_dynamics: str,
        control_matrix: str,
        barrier_funcs: Optional[Union[str, List, None]] = None,
        lyapunov_funcs: Optional[Union[str, List, None]] = None,
        nominal_controller: Optional[Union[str, List, None]] = None,
        params: Optional[Union[Dict[str, Any], None]] = None,
    ):
        """Generic model generation test function for arbitrary new model.

        Args:
            target_directory (str): directory where new model should be placed
            model_name (str): name of new model
            drift_dynamics (str): defines drift (f) dynamics
            control_matrix (str): defines control matrix (g)
            barrier_funcs (Optional[Union[str, List, None]], optional): defines
                barrier functions. Defaults to None.
            lyapunov_funcs (Optional[Union[str, List, None]], optional): defines
                lyapunov functions. Defaults to None.
            nominal_controller (Optional[Union[str, List, None]], optional): defines
                nominal controllers. Defaults to None.
            params (Optional[Union[Dict[str, Any], None]], optional): dynamics, cbf,
                clf, nominal control params. Defaults to None.
        """
        if params is None:
            params = {}

        generated_states, generated_controls = generate_model(
            target_directory,
            model_name,
            drift_dynamics,
            control_matrix,
            barrier_funcs,
            lyapunov_funcs,
            nominal_controller,
            params,
        )

        self.assertTrue(
            (dims[0] == generated_states) and (dims[1] == generated_controls)
        )  # pylint: disable=W1503


if __name__ == "__main__":
    unittest.main()
