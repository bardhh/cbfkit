"""A stateless waypoint-sequencing planner.

``waypoint_route`` publishes the *current* waypoint as a one-column ``x_traj`` so the
simulator's ``resolve_nominal_control`` hands it to the nominal controller as ``ref``. No
extra planner state is needed: the current index is recovered as the waypoint nearest to
the previously published one, and advanced when the tracked position is within ``radius``.
Waypoints must therefore be distinct.
"""

from typing import Optional, Sequence, Tuple

import jax.numpy as jnp
from jax import Array, jit

from cbfkit.utils.user_types import (
    Control,
    Key,
    PlannerCallable,
    PlannerCallableReturns,
    PlannerData,
    State,
    Time,
)


def waypoint_route(
    waypoints: Sequence[Sequence[float]],
    radius: float,
    position_indices: Tuple[int, ...] = (0, 1),
) -> PlannerCallable:
    """Sequence through ``waypoints`` (N x d), switching within ``radius`` of the current one."""
    wps = jnp.asarray(waypoints, dtype=float)
    if wps.ndim != 2 or wps.shape[0] < 1:
        raise ValueError("waypoints must be an (N, d) array with N >= 1")
    idx_pos = jnp.asarray(position_indices)
    n = wps.shape[0]

    @jit
    def _step(x: Array, prev_wp: Array) -> Array:
        idx = jnp.argmin(jnp.linalg.norm(wps - prev_wp, axis=1))
        pos = jnp.take(x, idx_pos)
        reached = jnp.linalg.norm(pos - wps[idx]) < radius
        idx = jnp.minimum(idx + reached.astype(idx.dtype), n - 1)
        return wps[idx]

    def process(
        t: Time, x: State, u_nom: Optional[Control], key: Key, data: PlannerData
    ) -> PlannerCallableReturns:
        prev = wps[0] if data.x_traj is None else jnp.asarray(data.x_traj)[:, 0]
        wp = _step(jnp.asarray(x), prev)
        return jnp.zeros(jnp.shape(x)), data._replace(x_traj=wp.reshape(-1, 1), u_traj=None)

    return process


__all__ = ["waypoint_route"]
