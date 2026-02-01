import time
import jax
import jax.numpy as jnp
from jaxopt import OSQP


def bench_warm_start():
    print(f"{'Mode':<10} {'Avg (us)':<10} {'Iter':<10} {'Factor':<10}")

    Q = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    G = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    h_base = jnp.array([2.0, 2.0, 2.0, 2.0])
    # Optimum at (10, 10), outside box [-2, 2]
    q_vec = jnp.array([-10.0, -10.0])

    solver = OSQP()

    @jax.jit
    def step(h, init_params):
        sol, state = solver.run(init_params=init_params, params_obj=(Q, q_vec), params_ineq=(G, h))
        return sol, state.iter_num

    h_vals = [h_base + i * 0.01 for i in range(100)]

    # Cold start
    step(h_vals[0], None)  # Compile for init_params=None
    t0, iters = time.time(), 0
    for h in h_vals:
        _, it = step(h, None)
        iters += int(it)
    t_cold = (time.time() - t0) * 1e6 / 100
    iter_cold = iters / 100

    # Warm start
    params, _ = step(h_vals[0], None)  # Get initial params
    step(h_vals[0], params)  # Compile for init_params=KKTSolution

    t0, iters = time.time(), 0
    for h in h_vals:
        params, it = step(h, params)
        iters += int(it)
    t_warm = (time.time() - t0) * 1e6 / 100
    iter_warm = iters / 100

    print(f"{'Cold':<10} {t_cold:<10.2f} {iter_cold:<10.1f} {'1.0x':<10}")
    print(f"{'Warm':<10} {t_warm:<10.2f} {iter_warm:<10.1f} {t_cold/t_warm:.1f}x")


if __name__ == "__main__":
    bench_warm_start()
