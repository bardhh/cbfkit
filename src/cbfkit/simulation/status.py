"""Simulation status checking, error formatting, and default callables."""

from typing import Any, List, Optional

import jax.numpy as jnp
import numpy as np
from jax import Array

from .ui import print_error, print_warning

SOLVER_STATUS_MAP = {
    -99: "NO_STATUS_AVAILABLE",
    -10: "INTEGRATION_NAN_ERROR",
    -2: "NAN_INPUT_DETECTED",
    -1: "NAN_DETECTED",
    0: "UNSOLVED (Likely Infeasible)",
    1: "SOLVED",
    2: "MAX_ITER_REACHED",
    3: "PRIMAL_INFEASIBLE",
    4: "DUAL_INFEASIBLE",
    5: "MAX_ITER_REACHED (UNSOLVED)",
}


def _format_error_status(status_code: Any) -> str:
    """Formats a solver status code into a human-readable string."""
    if isinstance(status_code, (int, float, jnp.ndarray, np.ndarray)):
        try:
            code = int(status_code)
            if code in SOLVER_STATUS_MAP:
                return f"{SOLVER_STATUS_MAP[code]} (Status: {code})"
            else:
                return f"Status: {code}"
        except (ValueError, TypeError):
            pass
    return str(status_code)


def _check_simulation_status(
    controller_data_keys: List[str],
    controller_data_values: List[Array],
    planner_data_keys: List[str],
    planner_data_values: List[Array],
    nan_detected: bool = False,
) -> None:
    """Checks for simulation errors and prints warnings if found."""
    # Explicit check for NaNs
    if nan_detected:
        print_error("Simulation failed due to NaNs in state trajectory.")

    # Check controller errors
    if "error" in controller_data_keys:
        idx = controller_data_keys.index("error")
        errors = controller_data_values[idx]
        if jnp.any(errors):
            first_error_idx = int(jnp.argmax(errors).item())

            status_msg = ""
            if "error_data" in controller_data_keys:
                idx_data = controller_data_keys.index("error_data")
                error_data = controller_data_values[idx_data]
                status = error_data[first_error_idx].item()
                status_msg = f" ({_format_error_status(status)})"

            print_warning(
                f"Simulation stopped early due to controller error at step {first_error_idx}{status_msg}."
            )

    # Check solver status telemetry from controller data.
    status_key = None
    if "sub_data_solver_status" in controller_data_keys:
        status_key = "sub_data_solver_status"
    elif "error_data" in controller_data_keys:
        status_key = "error_data"

    if status_key:
        idx_data = controller_data_keys.index(status_key)
        status_codes = controller_data_values[idx_data]
        max_iter_like_mask = (status_codes == 2) | (status_codes == 5)
        if "error" in controller_data_keys:
            idx_err = controller_data_keys.index("error")
            errors = controller_data_values[idx_err]
            max_iter_like_mask = max_iter_like_mask & (~errors)
        if jnp.any(max_iter_like_mask):
            count = int(jnp.sum(max_iter_like_mask).item())
            print_warning(
                f"Solver reached max iterations in {count} steps. "
                "These steps are treated as controller failures and may produce NaN controls."
            )

    # Check planner errors
    if "error" in planner_data_keys:
        idx = planner_data_keys.index("error")
        errors = planner_data_values[idx]
        if jnp.any(errors):
            first_error_idx = int(jnp.argmax(errors).item())
            print_warning(
                f"Simulation stopped early due to planner error at step {first_error_idx}."
            )


def _default_sensor(t, x, *, sigma=None, key=None, **kwargs):
    return x


def _default_estimator(t, y, z, u, c):
    return y, c if c is not None else jnp.zeros((len(y), len(y)))


def _default_perturbation(x, u, f, g):
    def p(key):
        return jnp.zeros_like(x)

    return p
