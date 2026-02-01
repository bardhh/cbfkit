import time
import jax.numpy as jnp
from jax import jit
from jaxopt import OSQP
from cbfkit.controllers.cbf_clf.generate_constraints import (
    generate_compute_cbf_clf_constraints,
    generate_compute_zeroing_cbf_constraints,
    generate_compute_vanilla_clf_constraints,
)
from cbfkit.utils.user_types import CertificateCollection

def bench_iterations():
    print(f"{'Constraints':<12} {'Iter (avg)':<12} {'Time (ms)':<10}")
    h, grad = lambda t, x: x[0] - 0.5, lambda t, x: jnp.array([1.0, 0.0])
    hess, dyn = lambda t, x: jnp.zeros((2, 2)), lambda x: (jnp.zeros(2), jnp.eye(2))
    dt, cond = lambda t, x: 0.0, lambda val: val

    solver, limits = OSQP(), jnp.array([10., 10.])

    for n in [1, 10, 50, 100]:
        barriers = CertificateCollection([h]*n, [grad]*n, [hess]*n, [dt]*n, [cond]*n)
        aug_limits = jnp.hstack([limits, 1e4 * jnp.ones(n)])

        compute = generate_compute_cbf_clf_constraints(
            generate_compute_zeroing_cbf_constraints, generate_compute_vanilla_clf_constraints,
            aug_limits, dyn, barriers, CertificateCollection([],[],[],[],[]), relaxable_cbf=True
        )

        p_mat, q_vec = jnp.eye(2 + n), jnp.zeros(2 + n)

        @jit
        def step(t, x):
            g, h_val, _ = compute(t, x)
            sol, state = solver.run(params_obj=(p_mat, q_vec), params_ineq=(g, h_val))
            return state.iter_num

        step(0.0, jnp.zeros(2)).block_until_ready()
        t0, steps, acc = time.time(), 100, 0
        for _ in range(steps): acc += int(step(0.0, jnp.zeros(2)))
        print(f"{n:<12} {acc/steps:<12.2f} {(time.time() - t0)/steps*1000:.4f}")

if __name__ == "__main__":
    bench_iterations()
