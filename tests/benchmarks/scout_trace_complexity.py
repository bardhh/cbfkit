"""
Scout: JAX Trace Complexity Benchmark
=====================================
Measures JAXPR trace size and compile time vs constraint count.
"""
import sys, os, time, jax
from jax import core
import jax.numpy as jnp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from cbfkit.systems.unicycle.models import accel_unicycle
from cbfkit.certificates import rectify_relative_degree, concatenate_certificates
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller as make_ctl
from cbfkit.utils.user_types import ControllerData

def run():
    print(f"🔎 Scout Trace Complexity\n{'N':<5} {'Eqns':<8} {'Consts':<8} {'T(ms)':<8}")
    dyn, x, t, u, key, data = accel_unicycle.plant(), jnp.zeros(4), 0.0, jnp.zeros(2), jax.random.PRNGKey(0), ControllerData()

    def count(j):
        c = len(j.eqns)
        for e in j.eqns:
            for v in e.params.values():
                if isinstance(v, core.Jaxpr): c += count(v)
                elif isinstance(v, core.ClosedJaxpr): c += count(v.jaxpr)
                elif isinstance(v, list) and v:
                    if isinstance(v[0], core.Jaxpr): c += sum(count(x) for x in v)
                    elif isinstance(v[0], core.ClosedJaxpr): c += sum(count(x.jaxpr) for x in v)
        return c

    for n in [1, 10, 50, 100]:
        h = lambda i: rectify_relative_degree(lambda x: x[0]-i, dyn, 4, form="exponential")(zeroing_barriers.linear_class_k(1.0))
        ctl = make_ctl(jnp.array([10., 10.]), dyn, concatenate_certificates(*[h(float(i)) for i in range(n)]))
        try:
            jaxpr = jax.make_jaxpr(ctl)(t, x, u, key, data)
            t0 = time.time()
            jax.jit(ctl)(t, x, u, key, data)[0].block_until_ready()
            print(f"{n:<5} {count(jaxpr.jaxpr):<8} {len(jaxpr.jaxpr.constvars):<8} {(time.time()-t0)*1000:<8.1f}")
        except Exception as e: print(f"{n:<5} ERROR: {e}")

if __name__ == "__main__": run()
