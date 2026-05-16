# Gymnasium Safety Filter Wrapper — Design Spec

## Overview

Add a `SafetyFilter` class and an optional Gymnasium `Wrapper` to CBFKit, enabling
CBF-based safety filtering for reinforcement learning workflows. The safety filter
intercepts actions, solves a CBF-QP to produce a minimally-modified safe action, and
returns diagnostics (barrier values, solver status, intervention flag).

**Goals:**
- Let RL researchers add CBF safety to any continuous Gymnasium env in <20 lines
- Provide a standalone `SafetyFilter` usable outside Gymnasium (ROS, hardware loops, MPC)
- Keep Gymnasium as an optional dependency

**Non-goals for v1:**
- Stable Baselines3 training integration (future example)
- Learned dynamics models (interface ready, implementation deferred)
- Discrete action spaces
- Vectorized environments (`gymnasium.vector`)
- Colab notebook or benchmark tables

**Design note:** CBFKit follows functional composition patterns (factory functions returning
callables). `SafetyFilter` uses a class because it manages mutable per-step state (time,
PRNG key, solver warm-start) and the Gymnasium `Wrapper` API requires class inheritance.
The class is thin — it delegates all math to the existing controller callables.

## Architecture

```
SafetyFilter (standalone, no gym dep)
    |
    |-- from_cbf_qp(dynamics, barriers, control_limits, ...)  # convenience
    |-- from_controller(controller, ...)                       # full control
    |
    |-- filter(state, action) -> (safe_action, info)
    |-- reset(seed=None)
    |-- barrier_values(state, t=None)
    |
SafetyFilterWrapper (gymnasium.Wrapper subclass, optional dep)
    |
    |-- from_cbf_qp(env, dynamics, barriers, control_limits, ...) # one-liner
    |-- wraps env.step() to filter actions through SafetyFilter
    |-- wraps env.reset() to reset SafetyFilter state
```

## Component Details

### `SafetyFilter` (`src/cbfkit/wrappers/safety_filter.py`)

Core class that wraps a CBFKit `ControllerCallable` and manages per-step state.

**Construction — two paths:**

```python
# Path 1: From CBF-QP specification
sf = SafetyFilter.from_cbf_qp(
    dynamics,                    # DynamicsCallable: (x) -> (f, g)
    barriers,                    # CertificateInput (collection, list, or tuple)
    control_limits,              # Array: symmetric actuation limits
    variant="vanilla",           # "vanilla" | "robust" | "stochastic"
    lyapunovs=None,              # optional CertificateInput
    fallback="passthrough",      # "passthrough" | "zero" | callable(state, action) -> action
    dt=0.01,                     # timestep for internal time tracking
    seed=0,                      # PRNG seed
    **kwargs,                    # forwarded to the QP generator (relaxable_clf, etc.)
)

# Path 2: From pre-built controller
sf = SafetyFilter.from_controller(
    controller,                  # any ControllerCallable
    dt=0.01,
    seed=0,
    fallback="passthrough",
    barriers=None,               # optional CertificateCollection, for barrier_values() only
)
```

**Variant string mapping** (uses pre-built generators from `cbfkit.controllers.cbf_clf`):
- `"vanilla"` -> `vanilla_cbf_clf_qp_controller(control_limits, dynamics, barriers, lyapunovs, **kwargs)`
- `"robust"` -> `robust_cbf_clf_qp_controller(control_limits, dynamics, barriers, lyapunovs, **kwargs)`
- `"stochastic"` -> `stochastic_cbf_clf_qp_controller(control_limits, dynamics, barriers, lyapunovs, **kwargs)`

Risk-aware variants (`risk_aware`, `risk_aware_path_integral`) are excluded from v1 because
`risk_aware_cbf_clf_qp_controller` requires a `RiskAwareParams` object with additional
configuration (sigma, t_max, eta, p_bound) that doesn't fit the simple variant-string API.
Users who need risk-aware filtering should use `from_controller()` with a pre-built controller.

**`filter(state, action) -> (safe_action, info)`:**

1. Cast `state` to float64: `state = jnp.asarray(state, dtype=jnp.float64)`
2. Cast `action` to float64: `action = jnp.asarray(action, dtype=jnp.float64)`
3. Call `controller(self._t, state, action, key, self._data) -> (u_safe, updated_data)`
4. Check for NaN in `u_safe` (can occur even without `error=True` due to numerical issues):
   - If NaN detected, treat as solver failure (same as `error=True`)
5. If `updated_data.error` is True or NaN detected:
   - Apply fallback strategy:
     - `"passthrough"`: return the original `action` (not `u_safe`)
     - `"zero"`: return `jnp.zeros_like(action)`
     - callable: return `fallback(state, action)`
   - Set `info["fallback_used"] = True`
6. Compute `intervened = not jnp.allclose(action, u_safe, atol=1e-4)` (matches typical OSQP tolerance)
7. Increment `self._t += dt`, split `self._key`, store `self._data = updated_data`
8. Return `(u_safe, info)`

**`info` dict (always returned):**
```python
{
    "u_nom": Array,                # original action
    "u_safe": Array,               # filtered action
    "intervened": bool,            # was action modified (allclose with atol=1e-4)
    "barrier_values": Array | None,  # h(x) values, None if barriers unavailable
    "solver_status": int | None,     # from updated_data.error_data
    "controller_error": bool,        # bool(updated_data.error)
    "fallback_used": bool,           # was fallback triggered
}
```

**Info extraction mapping from `ControllerData`:**
- `info["solver_status"]` = `updated_data.error_data`
- `info["controller_error"]` = `bool(updated_data.error)`
- `info["barrier_values"]` = `updated_data.sub_data.get("bfs")` if `updated_data.sub_data` else recomputed from stored barriers, else `None`

**`barrier_values(state, t=None) -> Array | None`:**
- If barriers were provided, iterates `barriers.functions` (index 0 of `CertificateCollection`),
  calls each `f(t, state)`, returns `jnp.array([f(t, state) for f in barriers[0]])`.
- `t` defaults to `self._t` (internal time counter).
- Returns `None` if no barriers available.

**`reset(seed=None)`:**
- Resets `self._t = 0.0`
- Re-initializes `self._data = ControllerData()` (all defaults: `error=False`, `sub_data=None`, etc.
  The CBF-CLF-QP controller handles `None` sub_data gracefully via an `if is not None` guard.)
- If `seed` is provided, re-initializes `self._key = jax.random.PRNGKey(seed)`

**Internal state:**
- `self._t: float` — incremented by `dt` on each `filter()` call
- `self._key: PRNGKey` — split on each call
- `self._data: ControllerData` — carries solver warm-start via `sub_data["solver_params"]`

**Thread safety:** `SafetyFilter` is NOT safe for use with vectorized environments
(`gymnasium.vector.AsyncVectorEnv`) or concurrent calls. Each instance maintains mutable
state that is not thread-safe.

### `SafetyFilterWrapper` (`src/cbfkit/wrappers/gymnasium.py`)

Thin Gymnasium `Wrapper` subclass (~80 lines). Requires `gymnasium>=1.0`.
Does not support older gymnasium versions (pre-1.0 `reset()` returned only `obs`).

**Import hygiene:**
```python
# Always works (no gymnasium dep):
from cbfkit.wrappers import SafetyFilter

# Requires gymnasium:
from cbfkit.wrappers.gymnasium import SafetyFilterWrapper
```

Importing `cbfkit.wrappers.gymnasium` without gymnasium installed raises:
```
ImportError: SafetyFilterWrapper requires gymnasium. Install with: pip install cbfkit[gymnasium]
```

**Construction:**
```python
# From pre-built SafetyFilter:
safe_env = SafetyFilterWrapper(
    env,
    safety_filter=sf,
    obs_to_state=None,           # callable: obs -> state (identity if None)
    action_to_control=None,      # callable: gym action -> cbfkit control (identity if None)
    control_to_action=None,      # callable: cbfkit control -> gym action (identity if None)
)

# One-liner convenience:
safe_env = SafetyFilterWrapper.from_cbf_qp(
    env,
    dynamics=dynamics,
    barriers=barriers,
    control_limits=limits,
    obs_to_state=lambda obs: obs[:2],  # extract position from richer observation
)
```

**Construction validation:**
- `env.action_space` must be `gymnasium.spaces.Box`. Raises `ValueError` for
  `Discrete`, `MultiDiscrete`, etc.
- If `action_to_control` is None (identity), validates that `env.action_space.shape`
  is compatible with `control_limits` dimensionality (when available from the safety filter).

**`step(action)`:**
1. Guard: raise `RuntimeError("Call reset() before step()")` if `_last_obs` is None
2. Convert obs to state: `state = obs_to_state(last_obs)`
3. Convert action to control: `control = action_to_control(action)`
4. Filter: `safe_control, filter_info = safety_filter.filter(state, control)`
5. Convert back: `safe_action = control_to_action(safe_control)`
6. Cast to env dtype and clip: `safe_action = np.clip(np.asarray(safe_action, dtype=env.action_space.dtype), env.action_space.low, env.action_space.high)`
7. Forward: `obs, reward, terminated, truncated, info = env.step(safe_action)`
8. Store `_last_obs = obs`
9. Nest filter info: `info["safety_filter"] = filter_info`
10. Return

**`reset(**kwargs)`:**
1. `obs, info = env.reset(**kwargs)`
2. `self._last_obs = obs`
3. Forward seed if present: `self.safety_filter.reset(seed=kwargs.get("seed"))`
4. Return `obs, info`

**`render()`:** Passes through to `env.render()` (inherited from `gymnasium.Wrapper`).

### Custom Environment (`src/cbfkit/envs/gymnasium.py`)

**`CBFKit/SafeSingleIntegratorObstacles-v0`**

A simple 2D navigation env using the existing `two_dimensional_single_integrator()` dynamics
from `src/cbfkit/systems/single_integrator/dynamics.py`.

- **State:** `[x, y]` (2D position)
- **Observation:** `[x, y, goal_x, goal_y]` (Box, float32)
- **Action:** `[vx, vy]` (Box, float32, clipped to [-1, 1])
- **Dynamics:** `dx/dt = u` (single integrator, Euler integration)
- **Obstacles:** configurable list of `(cx, cy, radius)` tuples, default 3 obstacles
- **Reward:** `-distance_to_goal`, `-100` on collision, `+100` on goal reached
- **Terminated:** collision or goal reached (within 0.1 radius)
- **Truncated:** step count exceeds `max_steps` (default 500)
- **Render:** `"rgb_array"` and `"human"` modes via matplotlib

**Registration:** Explicit, not automatic (gymnasium is optional):
```python
from cbfkit.envs.gymnasium import register_envs
register_envs()
env = gymnasium.make("CBFKit/SafeSingleIntegratorObstacles-v0")
```

### Barrier Construction Helper (`src/cbfkit/envs/gymnasium.py`)

The example env module includes a helper to construct circular obstacle barriers
for the single integrator, since no pre-built barriers exist for this system:

```python
def circular_obstacle_barriers(obstacles, alpha=1.0):
    """Build CBF barriers for circular obstacles.

    Args:
        obstacles: list of (cx, cy, radius) tuples
        alpha: class-K gain for zeroing CBF condition

    Returns:
        CertificateCollection for use with SafetyFilter
    """
    # For each obstacle: h(x) = (x - cx)^2 + (y - cy)^2 - r^2
    # Uses generate_certificate with linear_class_k(alpha)
    # Returns concatenate_certificates(*all_barriers)
```

This uses existing CBFKit utilities: `generate_certificate` (auto-diffs jacobian/hessian),
`linear_class_k` (zeroing barrier condition), `concatenate_certificates`.

### Example Script (`examples/gymnasium/safe_single_integrator.py`)

Demonstrates the full story:
1. Register env, create it
2. Build barriers with `circular_obstacle_barriers(env.obstacles)`
3. Define a naive "drive straight to goal" policy
4. Run **without** safety filter — show collisions
5. Run **with** `SafetyFilterWrapper.from_cbf_qp(...)` — show safe trajectories
6. `obs_to_state=lambda obs: obs[:2]` shown explicitly
7. Save PNG: side-by-side unsafe vs safe trajectories with obstacles
8. Print terminal summary: collisions, goal reached, min barrier value, intervention rate

### Example README (`examples/gymnasium/README.md`)

~25 lines, suitable for embedding in the main project README. Shows complete
barrier construction — no undefined variables:

```python
import jax.numpy as jnp
import gymnasium
from cbfkit.envs.gymnasium import register_envs, circular_obstacle_barriers
from cbfkit.wrappers.gymnasium import SafetyFilterWrapper
from cbfkit.systems.single_integrator.dynamics import two_dimensional_single_integrator

register_envs()
env = gymnasium.make("CBFKit/SafeSingleIntegratorObstacles-v0")

# Build barrier functions from the environment's obstacle list
barriers = circular_obstacle_barriers(env.unwrapped.obstacles, alpha=1.0)

safe_env = SafetyFilterWrapper.from_cbf_qp(
    env,
    dynamics=two_dimensional_single_integrator(),
    barriers=barriers,
    control_limits=jnp.array([1.0, 1.0]),
    obs_to_state=lambda obs: obs[:2],
)
obs, info = safe_env.reset()
for _ in range(500):
    action = env.action_space.sample()  # random policy
    obs, reward, terminated, truncated, info = safe_env.step(action)
    if info["safety_filter"]["intervened"]:
        print("CBF safety filter intervened!")
    if terminated or truncated:
        break
```

## File Inventory

| File | Lines (est.) | Description |
|------|-------------|-------------|
| `src/cbfkit/wrappers/__init__.py` | 10 | Exports `SafetyFilter`, `__all__` |
| `src/cbfkit/wrappers/safety_filter.py` | 200 | Core `SafetyFilter` class |
| `src/cbfkit/wrappers/gymnasium.py` | 80 | `SafetyFilterWrapper` + import guard |
| `src/cbfkit/envs/__init__.py` | 5 | Package marker, `__all__` |
| `src/cbfkit/envs/gymnasium.py` | 180 | Custom env + `register_envs()` + `circular_obstacle_barriers()` |
| `examples/gymnasium/safe_single_integrator.py` | 120 | Demo script with plots |
| `examples/gymnasium/README.md` | 30 | Quick-start snippet |
| `tests/test_wrappers/test_safety_filter.py` | 200 | All filter + wrapper + fallback + optional dep tests |

**Total: ~825 lines of new code**

## Dependencies

`pyproject.toml` addition:
```toml
[project.optional-dependencies]
gymnasium = ["gymnasium>=1.0"]
```

## Testing Strategy

**SafetyFilter tests:**
- Construction: both `from_cbf_qp` and `from_controller` paths
- `filter()`: returns correct shapes, `u_safe` is JAX array, info dict has all keys
- State management: `reset()` clears time/data/key, time increments by dt, key splits
- `barrier_values(state, t)`: returns array when barriers available, None when not
- `intervened` detection: allclose with atol=1e-4
- NaN detection: NaN in u_safe triggers fallback even without error flag

**Fallback tests:**
- Solver failure (via crippled solver or mock) triggers each fallback strategy
- `"passthrough"` returns original action, `"zero"` returns zeros, callable invoked correctly
- `info` preserves `solver_status` and `controller_error` regardless of fallback

**Gymnasium wrapper tests** (skipped if gymnasium not installed):
- `step()`/`reset()` flow produces valid returns
- `step()` before `reset()` raises `RuntimeError`
- `info["safety_filter"]` populated with correct keys
- dtype casting: JAX float64 -> Gymnasium float32 output
- action clipping to `env.action_space` bounds
- `from_cbf_qp` convenience constructor works
- `reset(seed=42)` forwards seed to both env and safety filter

**Optional dependency test:**
- `from cbfkit.wrappers import SafetyFilter` works without gymnasium
- `from cbfkit.wrappers.gymnasium import SafetyFilterWrapper` without gymnasium raises
  `ImportError` with install instructions
