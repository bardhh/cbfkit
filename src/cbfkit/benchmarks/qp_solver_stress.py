"""QP solver stress benchmark scenario."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from cbfkit.benchmarks.registry import register_scenario
from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.controllers.cbf_clf.generate_constraints import (
    generate_compute_vanilla_clf_constraints,
    generate_compute_zeroing_cbf_constraints,
)
from cbfkit.integration import forward_euler
from cbfkit.simulation import simulator
from cbfkit.utils.jit_monitor import JitMonitor
from cbfkit.utils.user_types import CertificateCollection


def _dynamics(x):
    return jnp.zeros(2), jnp.eye(2)


def _make_cbf(c, r):
    return (
        lambda _t, x: jnp.sum((x - c) ** 2) - r**2,
        lambda _t, x: 2 * (x - c),
        lambda _t, _x: 2 * jnp.eye(2),
        lambda _t, _x: 0.0,
        lambda h: 1.0 * h,
    )


@register_scenario("qp_solver_stress", description="QP solver reliability and safety stress test.")
def qp_solver_stress(seed: int) -> dict[str, float | int]:
    """Run a dense-obstacle QP stress scenario and return summary metrics."""
    x0, dt, tf = jnp.array([0.0, 0.0]), 0.01, 2.0
    num_steps = int(tf / dt)

    key = jax.random.PRNGKey(seed)
    key_centers, key_radii = jax.random.split(key)
    centers = jax.random.uniform(key_centers, (10, 2), minval=2.0, maxval=8.0)
    radii = jax.random.uniform(key_radii, (10,), minval=0.5, maxval=1.0)

    barrier_tuples = [_make_cbf(centers[i], radii[i]) for i in range(10)]
    barriers = CertificateCollection(*[list(x) for x in zip(*barrier_tuples)])

    controller = cbf_clf_qp_generator(
        generate_compute_zeroing_cbf_constraints,
        generate_compute_vanilla_clf_constraints,
    )(
        control_limits=jnp.array([5.0, 5.0]),
        dynamics_func=_dynamics,
        barriers=barriers,
        relaxable_cbf=False,
        relaxable_clf=True,
    )

    def nominal_controller(t, x, _k, _r):
        return 2.0 * (jnp.array([10.0, 10.0]) - x), None

    JitMonitor.reset()
    start = time.time()
    results = simulator.execute(
        x0=x0,
        dt=dt,
        num_steps=num_steps,
        dynamics=_dynamics,
        integrator=forward_euler,
        controller=controller,
        nominal_controller=nominal_controller,
        use_jit=True,
        verbose=False,
    )
    total_time = time.time() - start

    c_data = results.controller_data
    iters = c_data.get("sub_data_solver_iter", c_data.get("solver_iter"))
    status = c_data.get("sub_data_solver_status", c_data.get("solver_status"))
    bfs = c_data.get("sub_data_bfs", c_data.get("bfs"))

    if iters is not None:
        iter_arr = np.asarray(iters)
        mean_solver_iter = float(np.mean(iter_arr))
        max_solver_iter = float(np.max(iter_arr))
    else:
        mean_solver_iter = 0.0
        max_solver_iter = 0.0

    if status is not None:
        status_arr = np.asarray(status)
        solver_failure_steps = int(np.sum(status_arr != 1))
    else:
        solver_failure_steps = 0

    if bfs is not None:
        bfs_arr = np.asarray(bfs)
        min_barrier_value = float(np.min(bfs_arr))
        safety_violation_steps = int(np.sum(np.any(bfs_arr < -1e-4, axis=1)))
    else:
        min_barrier_value = 0.0
        safety_violation_steps = 0

    return {
        "success": int(solver_failure_steps == 0 and safety_violation_steps == 0),
        "safety_violations": int(safety_violation_steps > 0),
        "solver_failures": int(solver_failure_steps > 0),
        "avg_step_ms": (total_time / num_steps) * 1000.0,
        "execution_time_s": total_time,
        "jit_compiles": int(JitMonitor.get_counts().get("simulator_jit", 0)),
        "mean_solver_iter": mean_solver_iter,
        "max_solver_iter": max_solver_iter,
        "solver_failure_steps": solver_failure_steps,
        "min_barrier_value": min_barrier_value,
        "safety_violation_steps": safety_violation_steps,
    }
