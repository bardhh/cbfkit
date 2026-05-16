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

**Non-goals for v1 code:**
- Stable Baselines3 training integration (future example)
- Learned dynamics models (interface ready, implementation deferred)
- Discrete action spaces
- Vectorized environments (`gymnasium.vector`)

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
    |-- filter(state, action) -> (u_applied, info)
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

**Barrier normalization:** Both construction paths normalize barriers into a
`CertificateCollection` (the 5-tuple NamedTuple) at construction time and store it as
`self._barriers`. `from_cbf_qp` passes the raw `CertificateInput` to the QP generator
(which handles all input forms), then normalizes and stores the result for
`barrier_values()`. `from_controller` stores the `barriers` argument directly (must be
`None` or an already-normalized `CertificateCollection`).

**Variant string mapping** (uses pre-built generators from `cbfkit.controllers.cbf_clf`):
- `"vanilla"` -> `vanilla_cbf_clf_qp_controller(control_limits, dynamics, barriers, lyapunovs, **kwargs)`
- `"robust"` -> `robust_cbf_clf_qp_controller(control_limits, dynamics, barriers, lyapunovs, **kwargs)`
- `"stochastic"` -> `stochastic_cbf_clf_qp_controller(control_limits, dynamics, barriers, lyapunovs, **kwargs)`

Risk-aware variants (`risk_aware`, `risk_aware_path_integral`) are excluded from v1 because
`risk_aware_cbf_clf_qp_controller` requires a `RiskAwareParams` object with additional
configuration (sigma, t_max, eta, p_bound) that doesn't fit the simple variant-string API.
Users who need risk-aware filtering should use `from_controller()` with a pre-built controller.

**`filter(state, action) -> (u_applied, info)`:**

Variable naming convention — three distinct action values tracked through the pipeline:
- `u_nom`: the original action passed by the caller
- `u_qp`: the raw output from the QP solver (may be NaN on failure)
- `u_applied`: the action actually returned (equals `u_qp` on success, fallback on failure)

Steps:
1. Cast inputs to float64: `state = jnp.asarray(state, dtype=jnp.float64)`,
   `u_nom = jnp.asarray(action, dtype=jnp.float64)`
2. Call `controller(self._t, state, u_nom, key, self._data) -> (u_qp, updated_data)`
3. Check for NaN in `u_qp` (can occur even without `error=True` due to numerical issues):
   - If NaN detected, treat as solver failure (same as `error=True`)
4. Determine `u_applied`:
   - If `updated_data.error` is True or NaN detected:
     - Apply fallback strategy:
       - `"passthrough"`: `u_applied = u_nom`
       - `"zero"`: `u_applied = jnp.zeros_like(u_nom)`
       - callable: `u_applied = fallback(state, u_nom)`
     - Set `fallback_used = True`
   - Otherwise: `u_applied = u_qp`, `fallback_used = False`
5. Compute `intervened = not jnp.allclose(u_nom, u_applied, atol=1e-4)` (matches typical OSQP tolerance)
6. Increment `self._t += dt`, split `self._key`, store `self._data = updated_data`
7. Return `(u_applied, info)`

**`info` dict (always returned):**

All scalar values are converted to Python native types (`bool()`, `int()`, `float()`) to
avoid JAX tracer leaks into downstream code that doesn't expect JAX arrays.

```python
{
    "u_nom": Array,                # original action passed in
    "u_qp": Array,                 # raw QP solver output (may be NaN on failure)
    "u_applied": Array,            # action actually returned (matches return value)
    "intervened": bool,            # bool(not allclose(u_nom, u_applied, atol=1e-4))
    "barrier_values": Array | None,  # h(x) values, None if barriers unavailable
    "solver_status": int | None,     # int(updated_data.error_data) or None
    "controller_error": bool,        # bool(updated_data.error)
    "fallback_used": bool,           # was fallback triggered
}
```

**Info extraction mapping from `ControllerData`:**
- `info["solver_status"]` = `int(updated_data.error_data)` if not None, else `None`
- `info["controller_error"]` = `bool(updated_data.error)`
- `info["barrier_values"]` = `updated_data.sub_data.get("bfs")` if `updated_data.sub_data`
  else recomputed from `self._barriers` if available, else `None`

**`barrier_values(state, t=None) -> Array | None`:**
- If `self._barriers` is not None, iterates `self._barriers[0]` (the functions tuple),
  calls each `f(t, state)`, returns `jnp.array([f(t, state) for f in self._barriers[0]])`.
- `t` defaults to `self._t` (internal time counter).
- Returns `None` if no barriers stored.

**`reset(seed=None)`:**
- Resets `self._t = 0.0`
- Re-initializes `self._data = ControllerData()` (all defaults: `error=False`, `sub_data=None`, etc.
  The CBF-CLF-QP controller handles `None` sub_data gracefully via an `if is not None` guard.)
- If `seed` is provided, re-initializes `self._key = jax.random.PRNGKey(seed)`

**Internal state:**
- `self._t: float` — incremented by `dt` on each `filter()` call
- `self._key: PRNGKey` — split on each call
- `self._data: ControllerData` — carries solver warm-start via `sub_data["solver_params"]`
- `self._barriers: CertificateCollection | None` — stored at construction for `barrier_values()`

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
- **Obstacles:** configurable list of `(cx, cy, radius)` tuples
- **Reward:** `-distance_to_goal`, `-100` on collision, `+100` on goal reached
- **Terminated:** collision or goal reached (within 0.1 radius)
- **Truncated:** step count exceeds `max_steps` (default 500)
- **Render:** `"rgb_array"` and `"human"` modes via matplotlib

**Deterministic defaults** (for reproducible demo contrast):
```python
DEFAULT_START = np.array([0.0, 0.0])
DEFAULT_GOAL = np.array([4.0, 0.0])
DEFAULT_OBSTACLES = [(2.0, 0.3, 0.5), (3.0, -0.2, 0.4), (2.5, -0.5, 0.3)]
DEFAULT_DT = 0.05
DEFAULT_MAX_STEPS = 200
```

When `seed` is passed to `reset()`, the start/goal positions are deterministic (via
`np.random.Generator`). The defaults above are chosen so a naive "drive straight to goal"
policy reliably collides with the first obstacle, producing a clear visual contrast with
the CBF-filtered version.

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

Demonstrates the full story with deterministic seeding for reproducibility:
1. Register env, create with `seed=42`
2. Build barriers with `circular_obstacle_barriers(env.unwrapped.obstacles)`
3. Define a naive "drive straight to goal" policy: `action = normalize(goal - pos)`
4. Run **without** safety filter — show collisions (uses same seed)
5. Run **with** `SafetyFilterWrapper.from_cbf_qp(...)` — show safe trajectories (same seed)
6. `obs_to_state=lambda obs: obs[:2]` shown explicitly
7. Save PNG: `gymnasium_safe_vs_unsafe.png` — side-by-side trajectories with obstacles
8. Print terminal summary:

```
=== CBFKit Gymnasium Safety Filter Demo ===
Unsafe run:  collision=True   goal_reached=False  steps=47
Safe run:    collision=False  goal_reached=True   steps=156
  Min barrier value:    0.0312
  Intervention rate:    34.6%
  Max action change:    0.847
```

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
obs, info = safe_env.reset(seed=42)
for _ in range(200):
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
| `src/cbfkit/wrappers/safety_filter.py` | 220 | Core `SafetyFilter` class |
| `src/cbfkit/wrappers/gymnasium.py` | 80 | `SafetyFilterWrapper` + import guard |
| `src/cbfkit/envs/__init__.py` | 5 | Package marker, `__all__` |
| `src/cbfkit/envs/gymnasium.py` | 200 | Custom env + `register_envs()` + `circular_obstacle_barriers()` |
| `examples/gymnasium/safe_single_integrator.py` | 130 | Demo script with plots + terminal summary |
| `examples/gymnasium/README.md` | 30 | Quick-start snippet |
| `tests/test_wrappers/test_safety_filter.py` | 200 | All filter + wrapper + fallback + optional dep tests |

**Total: ~875 lines of new code**

## Dependencies

`pyproject.toml` addition:
```toml
[project.optional-dependencies]
gymnasium = ["gymnasium>=1.0"]
```

## Testing Strategy

**SafetyFilter tests:**
- Construction: both `from_cbf_qp` and `from_controller` paths
- `filter()`: returns correct shapes, info dict has all keys with correct Python types
- State management: `reset()` clears time/data/key, time increments by dt, key splits
- `barrier_values(state, t)`: returns array when barriers available, None when not
- `intervened` detection: allclose with atol=1e-4
- NaN detection: NaN in u_qp triggers fallback even without error flag
- Scalar conversion: `intervened`, `controller_error`, `solver_status` are Python native types

**Fallback tests:**
- Solver failure (via crippled solver or mock) triggers each fallback strategy
- `"passthrough"` returns u_nom, `"zero"` returns zeros, callable invoked correctly
- `info["u_qp"]` preserves raw solver output (NaN on failure)
- `info["u_applied"]` matches the returned action
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

**CI integration:** `pyproject.toml` `[dev]` extras should include `gymnasium` so that
gymnasium wrapper tests run in the default CI path. Add a note in the test file:
`pytest.importorskip("gymnasium")` for graceful skip when not installed.

## Release & Visibility (post-v1)

The following artifacts amplify adoption but are out of scope for the initial code PR.
They should follow as separate efforts once the code lands:

- **Main README snippet:** Embed the 20-line example from `examples/gymnasium/README.md`
  under a "Safe RL" section in the project README
- **Demo media:** The example script saves `gymnasium_safe_vs_unsafe.png`. Convert to
  a GIF/short video for README and social media
- **Comparison table:** Unsafe policy vs clipped action vs CBF safety filter (collision
  rate, goal success, constraint violation). Requires SB3 integration first.
- **CITATION.cff / Zenodo DOI:** Project-level decision. A versioned release with DOI
  makes the toolbox citable in safe-RL papers
- **SB3 notebook:** Colab notebook showing CBFKit + Stable-Baselines3 PPO training with
  safety filter. High visibility, depends on v1 landing first.
- **Framing:** Position as "CBFKit for Safe Reinforcement Learning: composable Gymnasium
  safety filters backed by control barrier functions" — targets the safe-RL search
  terms, not just "CBF toolbox"
