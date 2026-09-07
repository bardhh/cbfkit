"""Synchronized batched QP timings; CPU by default, CUDA when JAX is configured.

Run: python benchmarks/batched_safety_filter.py --batches 1 64 1024 4096
This is a filter microbenchmark, not an Isaac Lab robot benchmark.
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from cbfkit.certificates import generate_certificate
from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
from cbfkit.optimization.quadratic_program import get_solver
from cbfkit.systems.single_integrator.dynamics import two_dimensional_single_integrator
from cbfkit.wrappers import BatchedSafetyFilter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 64, 1024, 4096])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--torch", action="store_true", help="Include PyTorch/DLPack bridge overhead"
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
        help="Input dtype; the filter normalizes controller inputs to float64",
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    for batch in args.batches:
        sf = BatchedSafetyFilter.from_cbf_qp(
            num_envs=batch,
            dynamics=two_dimensional_single_integrator(),
            barriers=generate_certificate(lambda x: x[0], linear_class_k(1.0), input_style="state"),
            control_limits=jnp.ones(2),
            solver=get_solver("fast"),
        )
        dtype = getattr(jnp, args.dtype)
        x = (
            jnp.zeros((batch, 2), dtype=dtype)
            .at[:, 0]
            .set(jnp.linspace(0.05, 1.0, batch, dtype=dtype))
        )
        action = jnp.tile(jnp.array([-1.0, 0.0], dtype=dtype), (batch, 1))
        if args.torch:
            import torch

            from cbfkit.wrappers.torch import TorchSafetyFilter

            bridge = TorchSafetyFilter(sf)
            torch_action = torch.utils.dlpack.from_dlpack(action).clone()

            def prepare(state):
                return torch.utils.dlpack.from_dlpack(state).clone()

            def run(state):
                result = bridge.filter(state, torch_action)
                if torch_action.is_cuda:
                    torch.cuda.synchronize(torch_action.device)
                return result

        else:

            def prepare(state):
                return state

            def run(state):
                return jax.block_until_ready(sf.filter(state, action))

        initial = prepare(x)
        start = time.perf_counter()
        # Compile cold and steady-state signatures before measurement.
        for _ in range(2):
            run(initial)
        warmup = time.perf_counter() - start
        durations, violations, failures = [], 0, 0
        for step in range(args.steps):
            # Vary the active bound to exercise changing QPs, outside timing.
            state = x.at[:, 0].multiply(1.0 + 0.1 * jnp.sin(jnp.asarray(step * 0.1, dtype=dtype)))
            state.block_until_ready()
            input_state = prepare(state)
            if args.torch and torch_action.is_cuda:
                torch.cuda.synchronize(torch_action.device)
            start = time.perf_counter()
            u, info = run(input_state)
            durations.append(time.perf_counter() - start)
            if args.torch:
                failures += int(torch.sum(info["fallback_used"]))
                violations += int(torch.sum(u[:, 0] + input_state[:, 0] < -1e-5))
            else:
                failures += int(jnp.sum(info["fallback_used"]))
                violations += int(jnp.sum(u[:, 0] + state[:, 0] < -1e-5))
        print(
            json.dumps(
                dict(
                    batch=batch,
                    steps=args.steps,
                    device=str(jax.devices()[0]),
                    jax_version=jax.__version__,
                    torch_version=torch.__version__ if args.torch else None,
                    bridge=args.torch,
                    input_dtype=str(x.dtype),
                    filter_dtype=str(info["u_nom"].dtype).removeprefix("torch."),
                    output_dtype=str(u.dtype).removeprefix("torch."),
                    warmup_s=warmup,
                    mean_batch_ms=1000 * sum(durations) / len(durations),
                    p95_batch_ms=1000 * float(np.percentile(durations, 95)),
                    controls_per_second=batch * len(durations) / sum(durations),
                    constraint_violations=violations,
                    failed_controls=failures,
                )
            )
        )


if __name__ == "__main__":
    main()
