"""Simulation data formatting utilities."""

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
            except ValueError:
                pass
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
