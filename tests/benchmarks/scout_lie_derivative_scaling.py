"""
Scout: Lie Derivative Scaling Benchmark
=======================================

Measures the magnitude of Lie derivatives (Lf*h, Lg*h) across the state space.
Vanishing Lg*h (Control Barrier Gradient) indicates loss of control authority,
often leading to "stuck" robots or solver failures.

Metrics:
- Distribution of |Lf*h| and ||Lg*h|| (Min, Max, Mean, Median)
- Vanishing Gradient Frequency (Points where ||Lg*h|| < Tolerance)

Run with:
    python tests/benchmarks/scout_lie_derivative_scaling.py
"""

import sys
import os
import jax
import jax.numpy as jnp
import numpy as np

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from cbfkit.systems.unicycle.models import accel_unicycle
from cbfkit.certificates import rectify_relative_degree
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers

def ellipsoid_cbf(obstacle, ellipsoid):
    P = jnp.diag(1.0 / (ellipsoid**2))
    def h(x_and_t):
        pos = x_and_t[:2]
        diff = pos - obstacle[:2]
        return jnp.dot(diff, jnp.dot(P, diff)) - 1.0
    return h

def run():
    print("🔎 Scout: Lie Derivative Scaling Benchmark")
    print("------------------------------------------")

    # 1. Setup System
    dyn = accel_unicycle.plant(lam=1.0)

    # Obstacle at (2,0), radius 0.5
    obs = jnp.array([2.0, 0.0, 0.0])
    ell = jnp.array([0.5, 0.5])

    # 2. Generate Barrier Certificate
    # Relative degree 2 -> exponential rectification
    # This creates the "effective" barrier function B(x) with relative degree 1
    barrier_factory = rectify_relative_degree(
        function=ellipsoid_cbf(obs, ell),
        system_dynamics=dyn,
        state_dim=4,
        form="exponential",
    )

    # Get the certificate collection
    certs = barrier_factory(zeroing_barriers.linear_class_k(1.0))

    # Extract h(x) and grad_h(x)
    # Note: certificate functions take (t, x)
    grad_h_fn = jax.jit(certs.jacobians[0])

    # 3. Grid Search
    # Grid x from 0 to 1.5 (approaching obstacle boundary at 1.5)
    # Grid y from -1 to 1
    # v = 1.0, theta = 0.0 (moving towards obstacle)

    xs = np.linspace(0.0, 1.5, 20)
    ys = np.linspace(-1.0, 1.0, 20)

    lf_mags = []
    lg_mags = []
    vanishing_count = 0
    total_points = 0

    print(f"  Scanning {len(xs)*len(ys)} states...")

    for x_pos in xs:
        for y_pos in ys:
            # State: x, y, v, theta
            state = jnp.array([x_pos, y_pos, 1.0, 0.0])
            t = 0.0

            # Compute Dynamics
            f, g = dyn(state)

            # Compute Gradient
            grad_h = grad_h_fn(t, state)

            # Compute Lie Derivatives
            lf = jnp.dot(grad_h, f)
            lg = jnp.dot(grad_h, g) # Result is vector of size (control_dim,)

            lf_mag = jnp.abs(lf)
            lg_mag = jnp.linalg.norm(lg)

            lf_mags.append(float(lf_mag))
            lg_mags.append(float(lg_mag))

            if lg_mag < 1e-6:
                vanishing_count += 1

            total_points += 1

    # 4. Report Stats
    lf_mags = np.array(lf_mags)
    lg_mags = np.array(lg_mags)

    print("\n📊 Lie Derivative Statistics:")
    print(f"  {'Metric':<10} {'|Lf*h|':<15} {'||Lg*h||':<15}")
    print(f"  {'Mean':<10} {np.mean(lf_mags):<15.4e} {np.mean(lg_mags):<15.4e}")
    print(f"  {'Median':<10} {np.median(lf_mags):<15.4e} {np.median(lg_mags):<15.4e}")
    print(f"  {'Min':<10} {np.min(lf_mags):<15.4e} {np.min(lg_mags):<15.4e}")
    print(f"  {'Max':<10} {np.max(lf_mags):<15.4e} {np.max(lg_mags):<15.4e}")

    print("\n⚠️  Vanishing Gradients:")
    print(f"  Points with ||Lg*h|| < 1e-6: {vanishing_count} / {total_points} ({vanishing_count/total_points*100:.1f}%)")

    if vanishing_count > 0:
        print("  -> Warning: Some states have no control authority over the barrier.")
        print("     Solver may fail or behave unpredictably in these regions.")
    else:
        print("  -> Good: Control authority exists across the scanned region.")

if __name__ == "__main__":
    run()
