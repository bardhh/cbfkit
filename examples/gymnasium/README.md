# CBFKit Gymnasium Safety Filter

Add a CBF safety layer to any continuous Gymnasium environment.

## Quick Start

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
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = safe_env.step(action)
    if info["safety_filter"]["intervened"]:
        print("CBF safety filter intervened!")
    if terminated or truncated:
        break
```

## Run the Demo

```bash
python examples/gymnasium/safe_single_integrator.py
```

This compares a naive "drive straight to goal" policy with and without CBF safety
filtering, saves a side-by-side plot, and prints a summary.
