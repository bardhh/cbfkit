import unittest
import jax.numpy as jnp
import jax
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.utils.user_types import ControllerData

class TestJitStability(unittest.TestCase):
    """
    Regression test for JIT compilation stability.

    Ensures that the controller does not recompile unnecessarily when:
    1. Called repeatedly with different state values (but same shape/type).
    2. Called with updated sub_data from previous steps.

    Excessive recompilation causes severe performance degradation.
    """

    def test_controller_cache_stability(self):
        # 1. Setup simple system
        # 2D single integrator
        def dynamics(x):
            return jnp.zeros((2,)), jnp.eye(2)

        # Simple barrier h(x) = x[0] >= 0
        def h(t, x): return x[0]
        def grad(t, x): return jnp.array([1.0, 0.0])
        def hess(t, x): return jnp.zeros((2, 2))
        def partial_t(t, x): return 0.0
        def condition(val): return 1.0 * val

        barriers = ([h], [grad], [hess], [partial_t], [condition])
        control_limits = jnp.array([10.0, 10.0])

        # 2. Instantiate controller
        # This returns the JIT-compiled function
        controller = vanilla_cbf_clf_qp_controller(
            control_limits=control_limits,
            dynamics_func=dynamics,
            barriers=barriers,
            relaxable_cbf=False,
            relaxable_clf=True,
        )

        # 3. Run loop
        key = jax.random.PRNGKey(0)
        data = ControllerData(
            error=False,
            error_data=0,
            complete=False,
            sol=jnp.array([]),
            u=jnp.zeros(2),
            u_nom=jnp.zeros(2),
            sub_data={}
        )

        # Prime the cache (1st compilation)
        t = 0.0
        x = jnp.array([1.0, 1.0])
        u_nom = jnp.array([0.0, 0.0])

        u, data = controller(t, x, u_nom, key, data)

        cache_size_1 = controller._cache_size()
        self.assertEqual(cache_size_1, 1, f"Expected cache size 1 after first call, got {cache_size_1}")

        # Second call (warm start data now present in data.sub_data)
        # This typically triggers a 2nd compilation because sub_data structure changes (from empty dict to populated dict)
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, (2,))
        u_nom = jax.random.normal(key, (2,))

        u, data = controller(t + 0.1, x, u_nom, key, data)

        cache_size_2 = controller._cache_size()
        # We allow it to grow to 2, but warn if it's more.
        # Ideally it is 2.

        # Run 10 times with different states, reusing data
        for i in range(10):
            key, subkey = jax.random.split(key)
            x = jax.random.normal(subkey, (2,))
            u_nom = jax.random.normal(key, (2,))

            u, data = controller(t + 0.2 + i*0.1, x, u_nom, key, data)

        final_cache_size = controller._cache_size()

        # Should stay at cache_size_2 (which should be 2, or 1 if somehow structure matched)
        self.assertEqual(final_cache_size, cache_size_2,
                         f"JIT cache size grew from {cache_size_2} to {final_cache_size}. "
                         "Controller is recompiling on every step! "
                         "Check for non-static Python control flow or changing static arg structures.")

        self.assertLessEqual(final_cache_size, 2, "Cache size exceeds 2, implying instability beyond warm-start transition.")

if __name__ == '__main__':
    unittest.main()
