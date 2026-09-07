"""Batched, JIT-compiled filtering with independent environment histories."""

import math
from collections.abc import Mapping
from typing import Callable

import jax
import jax.numpy as jnp

from cbfkit.utils.user_types import ControllerData

from .safety_filter import SafetyFilter


class BatchedSafetyFilter:
    """Apply one pure JAX controller to a fixed batch of environments.

    Inputs have shape ``(num_envs, state_dim/action_dim)``. Diagnostics remain
    JAX arrays; filtering does not convert device values to Python scalars.
    Controllers must support JIT/vmap and return a stable ControllerData tree
    after their first call. This stateful Python facade is not itself meant to
    be wrapped in jit or shared between threads.

    Like SafetyFilter, controller inputs are normalized to float64, including
    float32 policy inputs. The JAX interface returns controller-precision
    actions; the optional Torch adapter casts applied actions back for callers.
    A step with any reset may evaluate both cold and warm solves for the whole
    batch. With staggered episodic resets this cost can occur on every step.

    Failure defaults to NaN actions (``fallback='nan'``), making the failure
    visible. ``zero`` or a pure callable ``(state, nominal_action) -> action``
    can be selected explicitly. No fallback is automatically a safe control.
    """

    def __init__(
        self,
        controller,
        *,
        num_envs: int,
        dt: float = 0.01,
        seed: int = 0,
        fallback: str | Callable = "nan",
    ):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not callable(fallback) and fallback not in ("nan", "zero", "passthrough"):
            raise ValueError("fallback must be nan, zero, passthrough, or callable")
        self.num_envs = num_envs
        self._dt = dt
        self._seed = seed
        self._data = None
        self._signature = None
        self._time = jnp.zeros(num_envs)
        self._keys = self._initial_keys(seed)
        self._base_keys = self._keys
        self._reset_mask = jnp.ones(num_envs, dtype=bool)

        def cold(t, x, u, key):
            return controller(t, x, u, key, ControllerData())

        def warm(t, x, u, key, data, reset):
            return jax.lax.cond(
                reset, lambda: cold(t, x, u, key), lambda: controller(t, x, u, key, data)
            )

        self._cold = jax.jit(jax.vmap(cold))
        batch_warm = jax.vmap(controller)
        batch_reset = jax.vmap(warm)

        @jax.jit
        def advance(t, x, u, keys, data, reset):
            # The normal path executes only warm solves. A mixed-reset step
            # may evaluate both branches under vmap, trading extra work for
            # static batch shapes and no host synchronization.
            return jax.lax.cond(
                jnp.any(reset),
                lambda: batch_reset(t, x, u, keys, data, reset),
                lambda: batch_warm(t, x, u, keys, data),
            )

        self._advance = advance

        @jax.jit
        def finish(x, nominal, proposed, error):
            failed = (
                error
                | ~jnp.all(jnp.isfinite(proposed), axis=-1)
                | ~jnp.all(jnp.isfinite(x), axis=-1)
                | ~jnp.all(jnp.isfinite(nominal), axis=-1)
            )
            if callable(fallback):
                backup = jax.vmap(fallback)(x, nominal)
            elif fallback == "zero":
                backup = jnp.zeros_like(nominal)
            elif fallback == "passthrough":
                backup = nominal
            else:
                backup = jnp.full_like(nominal, jnp.nan)
            applied = jnp.where(failed[:, None], backup, proposed)
            intervened = ~jnp.all(jnp.isclose(applied, nominal, atol=1e-4), axis=-1)
            return applied, failed, intervened

        self._finish = finish
        self._split = jax.jit(jax.vmap(jax.random.split))

    def _initial_keys(self, seed):
        return jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(seed), i))(
            jnp.arange(self.num_envs)
        )

    @classmethod
    def from_cbf_qp(
        cls,
        *,
        num_envs: int,
        dt: float = 0.01,
        seed: int = 0,
        fallback: str | Callable = "nan",
        **kwargs,
    ):
        """Build from the same dynamics/barriers/options as SafetyFilter.

        Use a JIT-compatible solver, e.g. ``solver=get_solver('fast')``.
        Parameters and barrier definitions are shared; variable scene geometry
        should be encoded in each environment's state by the caller.
        """
        scalar = SafetyFilter.from_cbf_qp(dt=dt, seed=seed, **kwargs)
        return cls(scalar._controller, num_envs=num_envs, dt=dt, seed=seed, fallback=fallback)

    @property
    def time(self):
        """Per-environment elapsed times as a device array."""
        return self._time

    def filter(self, states, actions):
        """Return applied actions and batched device-array diagnostics."""
        states, actions = jnp.asarray(states), jnp.asarray(actions)
        for name, value in (("states", states), ("actions", actions)):
            if value.ndim != 2 or value.shape[0] != self.num_envs or value.shape[1] == 0:
                raise ValueError(f"{name} must have shape ({self.num_envs}, dimension > 0)")
            if not jnp.issubdtype(value.dtype, jnp.floating):
                raise ValueError(f"{name} must have a floating dtype")
        if len(states.devices()) != 1 or states.devices() != actions.devices():
            raise ValueError("states and actions must reside on the same single device")
        device = next(iter(states.devices()))
        signature = (states.shape, states.dtype, actions.shape, actions.dtype, device)
        if self._signature is not None and signature != self._signature:
            raise ValueError(
                "State/action shapes, dtypes and device must remain fixed for this filter"
            )
        if self._signature is None:
            self._time, self._keys, self._reset_mask = jax.device_put(
                (self._time, self._keys, self._reset_mask), device
            )
            self._base_keys = jax.device_put(self._base_keys, device)
        states = states.astype(jnp.float64)
        actions = actions.astype(jnp.float64)
        keys = self._split(self._keys)
        if self._data is None:
            proposed, data = self._cold(self._time, states, actions, keys[:, 1])
        else:
            proposed, data = self._advance(
                self._time, states, actions, keys[:, 1], self._data, self._reset_mask
            )
        if proposed.shape != actions.shape:
            raise ValueError("Controller output must match the action batch shape")
        applied, failed, intervened = self._finish(states, actions, proposed, data.error)
        sub = data.sub_data if isinstance(data.sub_data, Mapping) else {}
        info = dict(
            u_nom=actions,
            u_qp=proposed,
            u_applied=applied,
            intervened=intervened,
            fallback_used=failed,
            controller_error=data.error,
            solver_status=sub.get("solver_status", data.error_data),
            barrier_values=sub.get("bfs"),
        )
        self._signature = signature
        self._data = data
        self._keys = keys[:, 0]
        self._time = self._time + self._dt
        # A failed solve must not poison the next step's warm-start history.
        self._reset_mask = failed
        return applied, info

    def reset(self, mask=None, *, seed=None):
        """Reset all environments or a boolean mask before their next action.

        Reset environments restart their time, controller history, and random
        stream. The same seed reproduces the same stream per environment;
        pass a new seed to obtain different streams and replace the selected
        environments' base seeds for subsequent resets. Unselected histories
        and streams are untouched. Multiple pending masks accumulate.
        """
        mask = jnp.ones(self.num_envs, dtype=bool) if mask is None else jnp.asarray(mask)
        if mask.shape != (self.num_envs,) or mask.dtype != jnp.bool_:
            raise ValueError(f"reset mask must be boolean with shape ({self.num_envs},)")
        initial = self._base_keys if seed is None else self._initial_keys(seed)
        device = next(iter(self._keys.devices()))
        initial = jax.device_put(initial, device)
        mask = jax.device_put(mask, device)
        if seed is not None:
            self._base_keys = jnp.where(mask[:, None], initial, self._base_keys)
        self._keys = jnp.where(mask[:, None], initial, self._keys)
        self._time = jnp.where(mask, 0.0, self._time)
        self._reset_mask = self._reset_mask | mask
