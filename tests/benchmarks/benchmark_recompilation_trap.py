import os
import sys
import time
import jax
import jax.numpy as jnp

sys.path.append(os.getcwd() + "/src")

from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller as make_ctl
from cbfkit.utils.user_types import CertificateCollection, ControllerData

def h(t, x): return x[0] - 0.5
def grad(t, x): return jnp.array([1.0, 0.0])
def hess(t, x): return jnp.zeros((2, 2))
def dt(t, x): return 0.0
def cond(v): return v
def dyn(x): return jnp.zeros(2), jnp.eye(2)

def run():
    print(f"{'Method':<10} {'Avg (ms)':<10} {'Factor':<10}")
    limits = jnp.array([1.0, 1.0])
    barriers = CertificateCollection([h], [grad], [hess], [dt], [cond])
    x, u, k, d = jnp.zeros(2), jnp.zeros(2), jax.random.PRNGKey(0), ControllerData()

    # 1. Reuse (Fast): Compiled once, reused many times
    ctl = make_ctl(limits, dyn, barriers)
    ctl(0.0, x, u, k, d)[0].block_until_ready()  # Warmup
    t0 = time.time()
    for _ in range(20):
        ctl(0.0, x, u, k, d)[0].block_until_ready()
    t_reuse = (time.time() - t0) / 20 * 1000

    # 2. Regenerate (Slow): Re-creates closure and triggers JIT every step
    t0 = time.time()
    for _ in range(5):
        # Simulating "dynamic barriers" by recreating controller each step
        make_ctl(limits, dyn, barriers)(0.0, x, u, k, d)[0].block_until_ready()
    t_regen = (time.time() - t0) / 5 * 1000

    print(f"{'Reuse':<10} {t_reuse:<10.2f} {'1.0x':<10}")
    print(f"{'Regen':<10} {t_regen:<10.2f} {t_regen / t_reuse:.1f}x (Trap!)")

if __name__ == "__main__":
    run()
