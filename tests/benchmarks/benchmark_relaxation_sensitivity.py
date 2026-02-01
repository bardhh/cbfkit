import os
import sys
import jax.numpy as jnp
from jax import random

# Ensure cbfkit is in path
sys.path.append(os.getcwd() + "/src")

from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.utils.user_types import ControllerData

def benchmark():
    """
    Benchmarks the sensitivity of the Relaxable CBF-QP controller to the slack variable penalty.

    Scenario:
        State: x = -2.0 (Unsafe)
        Dynamics: x_dot = u
        Barrier: h(x) = x >= 0 (Requires u >= 2.0 to satisfy alpha=1.0)
        Limit: u <= 0.5

        Gap: Required u (2.0) - Max u (0.5) = 1.5 violation.

    We sweep the slack penalty.
    - Low penalty: Slack is cheap, solver should find a solution easily.
    - High penalty: Slack is expensive, problem becomes ill-conditioned or hits limits.
    """
    print(f"Running Relaxation Sensitivity Benchmark...")
    print(f"Scenario: Infeasible constraints (u <= 0.5) vs Safety (requires u >= 2.0)")
    print("-" * 65)
    print(f"{'Penalty':<12} {'Status':<12} {'Error':<8} {'U[0]':<10} {'Slack est.':<10}")
    print("-" * 65)

    # Dynamics: x_dot = u
    def dynamics(x): return jnp.array([0.0]), jnp.array([[1.0]])

    # Barrier: h(x) = x >= 0
    # At x = -2.0, h = -2.0.
    # Class K: alpha * h = 1.0 * -2.0 = -2.0
    # Condition: Lfh + Lgh u >= -alpha * h
    #            0 + 1*u >= 2.0
    # Limit: u <= 0.5
    # Conflict: u >= 2.0 AND u <= 0.5. Gap = 1.5.

    def h(t, x): return x[0]
    def grad(t, x): return jnp.array([1.0])
    def hess(t, x): return jnp.array([[0.0]])
    def partial_t(t, x): return 0.0
    def condition(val): return 1.0 * val

    barriers = ([h], [grad], [hess], [partial_t], [condition])
    x = jnp.array([-2.0])
    u_nom = jnp.array([0.0])
    key = random.PRNGKey(0)

    # Sweep penalties
    penalties = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7]

    for p in penalties:
        # Create controller with specific penalty
        ctl = vanilla_cbf_clf_qp_controller(
            control_limits=jnp.array([0.5]),
            dynamics_func=dynamics,
            barriers=barriers,
            nominal_input=None,
            relaxable_cbf=True,
            slack_penalty_cbf=p,
        )

        # Run
        data = ControllerData()
        u, data = ctl(0.0, x, u_nom, key, data)

        # Interpret
        try:
            status_code = int(data.error_data)
        except:
            status_code = -1

        status_map = {1: "SOLVED", 0: "UNSOLVED", 2: "MAX_ITER"}
        status_str = status_map.get(status_code, f"CODE_{status_code}")

        u_val = f"{u[0]:.4f}"

        # Slack estimate: u + delta >= 2.0 => delta >= 2.0 - u
        # If u = 0.5, delta ~ 1.5
        slack_est = "N/A"
        if not jnp.isnan(u[0]):
            slack = 2.0 - u[0]
            slack_est = f"{slack:.4f}"

        print(f"{p:<12.1e} {status_str:<12} {str(data.error):<8} {u_val:<10} {slack_est:<10}")

if __name__ == "__main__":
    benchmark()
