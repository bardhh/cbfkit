import time
import jax
import jax.numpy as jnp
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.utils.user_types import CertificateCollection, ControllerData

def bench():
    print(f"{'N':<5} {'Compile (s)':<12} {'Exec (ms)':<10}")

    def dyn(x):
        return jnp.zeros_like(x), jnp.eye(2)

    def h(t, x):
        return x[0]

    def grad(t, x):
        return jnp.array([1.0, 0.0])

    def hess(t, x):
        return jnp.zeros((2, 2))

    def dt(t, x):
        return 0.0

    def cond(val):
        return val

    for n in [1, 10, 50, 100]:
        barriers = CertificateCollection(
            [h]*n, [grad]*n, [hess]*n, [dt]*n, [cond]*n
        )

        ctl = vanilla_cbf_clf_qp_controller(
            control_limits=jnp.array([10., 10.]),
            dynamics_func=dyn,
            barriers=barriers
        )

        x = jnp.zeros(2)
        u_nom = jnp.zeros(2)
        key = jax.random.PRNGKey(0)
        data = ControllerData(error=False, error_data=None, complete=False, sol=None, u=None, u_nom=None, sub_data={})

        t0 = time.time()
        # First call triggers compile
        ctl(0.0, x, u_nom, key, data)[0].block_until_ready()
        t1 = time.time()

        # Second call (execution loop)
        steps = 100
        t_start_exec = time.time()
        for _ in range(steps):
            ctl(0.0, x, u_nom, key, data)[0].block_until_ready()
        t_end_exec = time.time()

        compile_time = t1 - t0
        exec_time_ms = (t_end_exec - t_start_exec) / steps * 1000

        # Approximate pure compile time by removing one execution time
        pure_compile = compile_time - (exec_time_ms / 1000)

        print(f"{n:<5} {pure_compile:<12.4f} {exec_time_ms:.4f}")

if __name__ == "__main__":
    bench()
