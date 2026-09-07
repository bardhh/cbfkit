# Filter batched robot-policy commands

Use CBFKit's existing JAX controllers with parallel environments whose state
and commands are PyTorch tensors. `BatchedSafetyFilter` maintains separate
controller history, time, random streams, and resets for each environment.
`TorchSafetyFilter` is an optional tensor adapter used by the warehouse example.

The default CBFKit import does not import or require PyTorch, Gymnasium, or
Isaac Lab. Pure-JAX callers use `BatchedSafetyFilter` directly. The bridge
copies tensors through DLPack on their current device to protect against
simulator buffer reuse; it is not a zero-copy API and does not carry gradients
back to a PyTorch policy. CUDA tensors require CUDA-enabled JAX on the same GPU.

## Use the API

Install `pip install -e '.[torch]'` for the optional bridge (PyTorch ≥2.8 and
JAX ≥0.6.2). This raises the JAX version floor only when selecting the extra. Inside Isaac Lab,
use its supported PyTorch runtime and the [pinned Docker setup](docker/README.md).
The bridge is explicitly selected; it is not included in the `all` or `dev` extras.

Like the scalar filter, the batched filter normalizes controller inputs to
float64 and returns controller-precision JAX actions. Float32 policy tensors
are supported at the boundary; the Torch adapter casts applied actions back
to the policy's dtype. This is not a float32 QP implementation.

```python
import jax.numpy as jnp
import torch
from cbfkit.certificates import generate_certificate
from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
from cbfkit.optimization.quadratic_program import get_solver
from cbfkit.systems.single_integrator.dynamics import two_dimensional_single_integrator
from cbfkit.wrappers import BatchedSafetyFilter
from cbfkit.wrappers.torch import TorchSafetyFilter

# Planar velocity commands; the example model's safe set is x >= 0.
base = BatchedSafetyFilter.from_cbf_qp(
    num_envs=2, dt=0.02,
    dynamics=two_dimensional_single_integrator(),
    barriers=generate_certificate(lambda x: x[0], linear_class_k(1.0), input_style="state"),
    control_limits=jnp.ones(2), solver=get_solver("fast"),
)
bridge = TorchSafetyFilter(base)
state = torch.tensor([[0.1, 0.0], [1.0, 0.0]])
requested_velocity = torch.tensor([[-1.0, 0.0], [-1.0, 0.0]])
applied_velocity, info = bridge.filter(state, requested_velocity)
if info["fallback_used"].any():
    raise RuntimeError("Filter failed; choose a task-specific backup before actuation")
# Dispatch applied_velocity using the task's physical command mapping.
bridge.reset(torch.tensor([True, False]))  # Only environment zero restarted.
```

Call `reset(terminated | truncated)` after an environment performs in-step
resets, before filtering its next action. A full `reset(seed=...)` restarts all
histories and random streams. Calls that bypass this adapter must notify it
of resets explicitly. Batch size, shapes, dtypes, and device stay fixed for an
instance; the stateful Python interface is not thread-safe or itself JIT-wrapped.

The default fallback is NaN. A failed environment discards its solver history
on the next call; other histories remain intact. Explicit `zero`, `passthrough`,
or callable fallbacks are available, but none is automatically a safe backup.
Any reset may evaluate both cold and warm solves for the entire batch under
JAX `vmap`. With staggered episode endings this extra work can occur on every
step; budget for it when measuring episodic workloads. The generic controller
API does not assume that a zero solver history is equivalent to a cold call.
A reset with a new seed replaces the selected environments' base random
streams, including for subsequent resets that omit a seed.

## Real robot example

The [warehouse demo](WAREHOUSE.md) filters world-frame velocity requests before
an unchanged pretrained ANYmal-C walking policy. It explicitly converts those
velocities into the robot's body frame. It uses simulator obstacle state;
the cameras visualize the experiment and do not supply perception.

The library does not infer what arbitrary policy actions mean. Joint offsets,
normalized actions, world/body coordinates, and downstream clipping require
explicit task-specific mappings. A planar CBF does not certify articulated
robot tracking or contact dynamics.

## Measure the integration cost

Use identical dtype, device, solver settings, and precision for both paths:

```bash
python benchmarks/batched_safety_filter.py --dtype float32 --steps 100
python benchmarks/batched_safety_filter.py --dtype float32 --steps 100 --torch
```

For CUDA, set `JAX_PLATFORM_NAME=gpu` and `JAX_PLATFORMS=cuda,cpu` for both
commands. Repeat and alternate run order. Each row records input, filter, and
output dtype separately. These measurements include filtering,
returned diagnostics, and synchronization, excluding input preparation and
compilation. The bridged path additionally includes its ownership copies and
conversion. They measure the boundary's incremental end-to-end cost, not the
time of an individual copy operation. See [recorded validation](validation/REPORT.md).
