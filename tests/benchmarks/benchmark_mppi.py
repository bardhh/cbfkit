
import time
import jax
import jax.numpy as jnp
from cbfkit.controllers.mppi.mppi_source import setup_mppi

# Dummy dynamics
def dynamics_func(state):
    # simple integrator
    # g needs to be (4, 2)
    return jnp.zeros_like(state), jnp.zeros((len(state), 2))

def stage_cost(state, action):
    return jnp.sum(state**2) + jnp.sum(action**2)

def terminal_cost(state):
    return jnp.sum(state**2)

def benchmark():
    robot_state_dim = 4
    robot_control_dim = 2
    horizon = 50
    samples = 1000 # Increased samples to make it measurable

    compute_rollout_costs = setup_mppi(
        dyn_func=dynamics_func,
        trajectory_cost_func=None,
        stage_cost_func=stage_cost,
        terminal_cost_func=terminal_cost,
        robot_state_dim=robot_state_dim,
        robot_control_dim=robot_control_dim,
        horizon=horizon,
        samples=samples,
        control_bound=100.0,
        dt=0.05,
        use_GPU=True,
    )

    key = jax.random.PRNGKey(0)
    U = jnp.zeros((horizon, robot_control_dim))
    init_state = jnp.zeros((robot_state_dim, 1))

    # Warmup
    print("Warming up...")
    compute_rollout_costs(key, U, init_state, 0.0, None, None)

    # Benchmark
    N = 500
    print(f"Running {N} iterations...")
    start = time.time()
    for _ in range(N):
        key, subkey = jax.random.split(key)
        # We need to block until ready to measure actual computation time on GPU
        res = compute_rollout_costs(subkey, U, init_state, 0.0, None, None)
        res[0].block_until_ready()

    end = time.time()
    avg_time = (end - start) / N
    print(f"Average time per iteration: {avg_time*1000:.4f} ms")

if __name__ == "__main__":
    benchmark()
