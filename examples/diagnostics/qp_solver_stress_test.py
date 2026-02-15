"""
Scout Diagnostic: QP Solver Stress Test.
Measures solver reliability, JIT overhead, and constraint violations.
Command: python examples/diagnostics/qp_solver_stress_test.py
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
from cbfkit.simulation import simulator
from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.controllers.cbf_clf.generate_constraints import (
    generate_compute_zeroing_cbf_constraints,
    generate_compute_vanilla_clf_constraints,
)
from cbfkit.utils.user_types import CertificateCollection
from cbfkit.utils.jit_monitor import JitMonitor
from cbfkit.integration import forward_euler

def dynamics(x): return jnp.zeros(2), jnp.eye(2)

def make_cbf(c, r):
    return (
        lambda t, x: jnp.sum((x - c)**2) - r**2,
        lambda t, x: 2 * (x - c),
        lambda t, x: 2 * jnp.eye(2),
        lambda t, x: 0.0,
        lambda h: 1.0 * h
    )

def run_benchmark():
    print("🚀 Scout: QP Solver Stress Test")
    x0, dt, tf = jnp.array([0.0, 0.0]), 0.01, 2.0
    num_steps = int(tf / dt)

    # Generate 10 random obstacles
    key = jax.random.PRNGKey(0)
    centers = jax.random.uniform(key, (10, 2), minval=2.0, maxval=8.0)
    radii = jax.random.uniform(key, (10,), minval=0.5, maxval=1.0)
    barriers = CertificateCollection(*[list(x) for x in zip(*[make_cbf(centers[i], radii[i]) for i in range(10)])])

    controller = cbf_clf_qp_generator(
        generate_compute_zeroing_cbf_constraints, generate_compute_vanilla_clf_constraints
    )(
        control_limits=jnp.array([5.0, 5.0]), dynamics_func=dynamics, barriers=barriers,
        relaxable_cbf=False, relaxable_clf=True,
    )

    JitMonitor.reset()
    start = time.time()
    res = simulator.execute(
        x0=x0, dt=dt, num_steps=num_steps, dynamics=dynamics, integrator=forward_euler,
        controller=controller, nominal_controller=lambda t, x, k, r: (2.0*(jnp.array([10.,10.])-x), None),
        use_jit=True, verbose=False
    )
    total_time = time.time() - start

    # Reporting
    c_data = res.controller_data
    # Flattened keys: sub_data_solver_iter, sub_data_solver_status, sub_data_bfs
    iters = c_data.get('sub_data_solver_iter', c_data.get('solver_iter'))
    status = c_data.get('sub_data_solver_status', c_data.get('solver_status'))
    bfs = c_data.get('sub_data_bfs', c_data.get('bfs'))

    print(f"⏱️  Time: {total_time:.4f}s | JIT Compiles: {JitMonitor.get_counts().get('simulator_jit', 0)}")

    if iters is not None and status is not None:
        fail = np.sum(status != 1)
        print(f"🧩 Solver: Mean Iter={np.mean(iters):.1f}, Max={np.max(iters)}, Failures={fail} ({fail/len(status):.1%})")

    if bfs is not None:
        min_h = np.min(bfs)
        print(f"🛡️  Constraints: Min h={min_h:.4f} (Violated steps: {np.sum(np.any(bfs < -1e-4, axis=1))})")

if __name__ == "__main__":
    run_benchmark()
