
import time
import jax
import jax.numpy as jnp
import numpy as np
from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.controllers.cbf_clf.generate_constraints import (
    generate_compute_zeroing_cbf_constraints,
    generate_compute_vanilla_clf_constraints,
)
from cbfkit.utils.user_types import ControllerData

def benchmark_controller():
    # Setup
    n_controls = 2
    n_states = 4
    n_barriers = 10

    control_limits = jnp.array([1.0, 1.0])

    def dynamics(x):
        f = jnp.zeros((n_states,))
        g = jnp.zeros((n_states, n_controls))
        return f, g

    # Mock barriers (CertificateCollection)
    # functions, jacobians, hessians, partials, conditions
    barriers = (
        [lambda t, x: 1.0] * n_barriers,
        [lambda t, x: jnp.zeros((n_states,))] * n_barriers,
        [],
        [lambda t, x: 0.0] * n_barriers,
        [lambda h: h] * n_barriers
    )

    lyapunovs = ([], [], [], [], [])

    # Generate controller
    controller_gen = cbf_clf_qp_generator(
        generate_compute_zeroing_cbf_constraints,
        generate_compute_vanilla_clf_constraints
    )

    controller = controller_gen(
        control_limits=control_limits,
        dynamics_func=dynamics,
        barriers=barriers,
        lyapunovs=lyapunovs,
        relaxable_cbf=True # Adds slack variables
    )

    # Inputs
    t = 0.0
    x = jnp.zeros((n_states,))
    u_nom = jnp.zeros((n_controls,))
    key = jax.random.PRNGKey(0)
    data = ControllerData(
        error=0, error_data=0, complete=False,
        sol=jnp.zeros((n_controls + n_barriers,)),
        u=jnp.zeros((n_controls,)),
        u_nom=jnp.zeros((n_controls,)),
        sub_data={}
    )

    # Warmup
    print("Compiling...")
    t0 = time.time()
    u, d = controller(t, x, u_nom, key, data)
    u.block_until_ready()
    print(f"Compilation took {time.time() - t0:.4f}s")

    # Benchmark
    N = 1000
    print(f"Running {N} iterations...")
    t0 = time.time()
    for _ in range(N):
        u, d = controller(t, x, u_nom, key, d)
        u.block_until_ready()

    avg_time = (time.time() - t0) / N
    print(f"Average time per call: {avg_time*1000:.4f} ms")

if __name__ == "__main__":
    benchmark_controller()
