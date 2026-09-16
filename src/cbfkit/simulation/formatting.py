"""Simulation data formatting utilities."""

import warnings
from typing import Tuple

import jax.numpy as jnp
import numpy as np

from cbfkit.utils.user_types import SimulationResults

from .utils import SimulationStepData


def format_return_data(
    data: Tuple[SimulationStepData, ...],
) -> SimulationResults:
    """Extracts simulation data into JAX arrays."""
    if not data:
        return SimulationResults(
            jnp.array([]),
            jnp.array([]),
            jnp.array([]),
            jnp.array([]),
            [],
            [],
            [],
            [],
        )

    # Optimization: Transpose tuple of NamedTuples to NamedTuple of tuples
    transposed = SimulationStepData(*zip(*data))

    states = jnp.stack(transposed.state)
    controls = jnp.stack(transposed.control)
    estimates = jnp.stack(transposed.estimate)
    covariances = jnp.stack(transposed.covariance)

    controller_data_keys = []
    controller_data_values = []
    planner_data_keys = []
    planner_data_values = []

    def process_keys_values(keys, values_tuple_of_lists):
        processed_keys = []
        processed_values = []

        if not values_tuple_of_lists:
            return processed_keys, processed_values

        vals_by_key = list(zip(*values_tuple_of_lists))

        if not vals_by_key:
            return processed_keys, processed_values

        for i, key in enumerate(keys):
            vals = vals_by_key[i]

            first_valid = next((v for v in vals if v is not None), None)
            if first_valid is None or isinstance(first_valid, (dict, str, list, tuple)):
                continue

            if any(v is None for v in vals):
                if isinstance(first_valid, (int, float, jnp.ndarray, np.ndarray)):
                    default_val = -99
                    is_float = False
                    if hasattr(first_valid, "dtype"):
                        is_float = jnp.issubdtype(first_valid.dtype, jnp.floating)
                    elif isinstance(first_valid, float):
                        is_float = True

                    if is_float:
                        default_val = jnp.nan

                    vals = [v if v is not None else default_val for v in vals]
                else:
                    continue

            try:
                arr = jnp.stack(vals)
                processed_keys.append(key)
                processed_values.append(arr)
            except ValueError as exc:
                # Field had inconsistent shapes across timesteps; drop it but warn
                # so users don't see a silent KeyError when they look it up later.
                warnings.warn(
                    f"Field {key!r} dropped from SimulationResults: "
                    f"inconsistent shapes across timesteps ({exc}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return processed_keys, processed_values

    if len(data) > 0:
        controller_data_keys, controller_data_values = process_keys_values(
            data[0].controller_keys, transposed.controller_values
        )
        planner_data_keys, planner_data_values = process_keys_values(
            data[0].planner_keys, transposed.planner_values
        )

    return SimulationResults(
        states,
        controls,
        estimates,
        covariances,
        controller_data_keys,
        controller_data_values,
        planner_data_keys,
        planner_data_values,
    )


def format_bulk_log(xs, us, zs, cs, c_datas, p_datas, num_steps):
    """Convert stacked JIT outputs to the column-oriented CSV logging contract."""
    # Optimization (Bolt): Use bulk logging instead of per-step loop
    c_keys = list(c_datas._fields)
    p_keys = list(p_datas._fields)

    log_dict = {
        "state": list(np.array(xs)),
        "control": list(np.array(us)),
        "estimate": list(np.array(zs)),
        "covariance": list(np.array(cs)),
    }

    def process_bulk_data(keys, data_obj, prefix):
        for k in keys:
            val = getattr(data_obj, k)
            # val could be Array(T, ...), Dict[str, Array(T, ...)], or None
            if val is None:
                log_dict[f"{prefix}_{k}"] = [None] * num_steps
            elif isinstance(val, dict):
                # Unstack dict of arrays -> list of dicts
                # First convert to numpy to speed up iteration
                val_np = {}
                for sk, sv in val.items():
                    try:
                        if isinstance(sv, tuple):
                            val_np[sk] = list(zip(*sv))
                        else:
                            val_np[sk] = list(np.array(sv))
                    except Exception as exc:
                        warnings.warn(
                            f"Could not convert sub-data field "
                            f"{prefix}.{k}.{sk!r} to numpy ({exc}); "
                            f"logging Nones for this field.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        val_np[sk] = [None] * num_steps
                # zip now works on lists of values
                vals = [dict(zip(val_np.keys(), t)) for t in zip(*val_np.values())]
                log_dict[f"{prefix}_{k}"] = vals
            else:
                log_dict[f"{prefix}_{k}"] = list(np.array(val))

    process_bulk_data(c_keys, c_datas, "controller")
    process_bulk_data(p_keys, p_datas, "planner")

    return log_dict
