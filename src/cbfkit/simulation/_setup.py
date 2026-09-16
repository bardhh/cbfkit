"""Input validation before choosing a simulation execution backend."""

import warnings
from typing import Optional, Tuple

import jax.numpy as jnp
from jax import Array

from cbfkit.utils.user_types import (
    DiscretePlant,
    DynamicsCallable,
    IntegratorCallable,
    PerturbationCallable,
    StlTrajectoryCostCallable,
)


def _validate_dynamics_shapes(x0: Array, f_check: Array, g_check: Array) -> None:
    """Shape checks for the (f, g) returned by ``dynamics(x0)`` on the flat-state path."""
    if f_check.shape != x0.shape:
        msg = (
            f"Shape mismatch: Initial state 'x0' has shape {x0.shape}, "
            f"but dynamics drift 'f' has shape {f_check.shape}.\n"
            "The state vector must match the dynamics output shape."
        )
        if x0.ndim == 2 and x0.shape[1] == 1 and f_check.ndim == 1:
            msg += "\nTip: Pass a 1D array for 'x0' (e.g., use x0.ravel() or x0.flatten())."
        elif x0.shape[0] < f_check.shape[0]:
            msg += f"\nTip: System expects {f_check.shape[0]} states, but got {x0.shape[0]}."
        raise ValueError(msg)

    if f_check.ndim != 1:
        msg = (
            f"Dynamics function returned `f` with shape {f_check.shape}. "
            "Expected 1D array (shape (n,)).\n"
        )
        if f_check.ndim == 2 and f_check.shape[1] == 1:
            msg += (
                "It appears `f` is a column vector (n, 1). "
                "Please squeeze it to (n,) (e.g., using jnp.squeeze or .flatten())."
            )
        raise ValueError(msg)

    if g_check.ndim != 2:
        raise ValueError(
            f"Dynamics function returned `g` with shape {g_check.shape}. "
            "Expected 2D array (shape (n, m))."
        )


def validate_setup(
    x0: Array,
    dt: float,
    dynamics: Optional[DynamicsCallable],
    integrator: Optional[IntegratorCallable],
    plant: Optional[DiscretePlant],
    perturbation: Optional[PerturbationCallable],
    stl_trajectory_cost: Optional[StlTrajectoryCostCallable],
) -> Tuple[Array, Optional[Array]]:
    # Validate dynamics output — single call, reused for all checks
    x0 = jnp.atleast_1d(jnp.asarray(x0))
    if plant is not None:
        if dynamics is not None or integrator is not None:
            warnings.warn(
                "plant= given: 'dynamics' and 'integrator' are ignored.", UserWarning, stacklevel=3
            )
        if x0.shape != (plant.state_dim,):
            raise ValueError(
                f"x0 has shape {x0.shape} but plant.state_dim is {plant.state_dim} "
                f"(expected [qpos | qvel | com_xyz] for MujocoPlant)."
            )
        if abs(float(dt) - float(plant.dt)) > 1e-9:
            raise ValueError(
                f"dt={dt} must equal plant.dt={plant.dt} "
                "(use MujocoPlant(substeps=...) to change it)."
            )
        if perturbation is not None:
            raise NotImplementedError(
                "perturbation is not supported on the plant path (v1); "
                "use the plant's own domain randomisation."
            )
        if stl_trajectory_cost is not None:
            raise NotImplementedError(
                "stl_trajectory_cost requires the eager path, which is debug-only for plants."
            )
        g_check = None
    elif dynamics is None or integrator is None:
        raise ValueError("Either plant= or both dynamics= and integrator= must be given.")
    else:
        try:
            f_check, g_check = dynamics(x0)
        except Exception as e:
            raise ValueError(
                f"Dynamics evaluation failed for initial state 'x0' with shape {x0.shape}.\n"
                f"Ensure 'x0' has the correct dimensions for the system.\n"
                f"Original error: {e}"
            ) from e

        _validate_dynamics_shapes(x0, f_check, g_check)

    return x0, g_check
