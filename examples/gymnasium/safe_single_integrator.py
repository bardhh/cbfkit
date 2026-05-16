"""CBFKit Gymnasium Safety Filter Demo.

Compares a naive "drive straight to goal" policy with and without
CBF safety filtering on the single-integrator obstacle avoidance env.

Usage:
    python examples/gymnasium/safe_single_integrator.py
    CBFKIT_TEST_MODE=1 python examples/gymnasium/safe_single_integrator.py  # skip plots
"""

import os

import gymnasium
import jax.numpy as jnp
import numpy as np

from cbfkit.envs.gymnasium import circular_obstacle_barriers, register_envs
from cbfkit.systems.single_integrator.dynamics import two_dimensional_single_integrator
from cbfkit.wrappers.gymnasium import SafetyFilterWrapper

TEST_MODE = os.environ.get("CBFKIT_TEST_MODE", "0") == "1"
SEED = 42
MAX_STEPS = 200 if not TEST_MODE else 50


def naive_policy(obs):
    """Drive straight toward goal."""
    pos = obs[:2]
    goal = obs[2:4]
    direction = goal - pos
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return np.zeros(2, dtype=np.float32)
    return (direction / norm).astype(np.float32)


def run_episode(env, seed, max_steps):
    """Run one episode and return trajectory + stats."""
    obs, _ = env.reset(seed=seed)
    trajectory = [obs[:2].copy()]
    collision = False
    goal_reached = False
    barrier_vals = []
    interventions = 0
    total_steps = 0

    for _ in range(max_steps):
        action = naive_policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append(obs[:2].copy())
        total_steps += 1

        if "safety_filter" in info:
            if info["safety_filter"]["intervened"]:
                interventions += 1
            bv = info["safety_filter"]["barrier_values"]
            if bv is not None:
                barrier_vals.append(float(np.min(np.asarray(bv))))

        if terminated:
            collision = info.get("collision", False)
            goal_reached = info.get("goal_reached", False)
            break
        if truncated:
            break

    return {
        "trajectory": np.array(trajectory),
        "collision": collision,
        "goal_reached": goal_reached,
        "steps": total_steps,
        "min_barrier": min(barrier_vals) if barrier_vals else None,
        "intervention_rate": interventions / max(total_steps, 1),
    }


def main():
    register_envs()

    # --- Unsafe run ---
    env_unsafe = gymnasium.make("CBFKit/SafeSingleIntegratorObstacles-v0")
    unsafe_result = run_episode(env_unsafe, seed=SEED, max_steps=MAX_STEPS)

    # --- Safe run ---
    env_safe = gymnasium.make("CBFKit/SafeSingleIntegratorObstacles-v0")
    barriers = circular_obstacle_barriers(env_safe.unwrapped.obstacles, alpha=1.0)
    safe_env = SafetyFilterWrapper.from_cbf_qp(
        env_safe,
        dynamics=two_dimensional_single_integrator(),
        barriers=barriers,
        control_limits=jnp.array([1.0, 1.0]),
        obs_to_state=lambda obs: obs[:2],
    )
    safe_result = run_episode(safe_env, seed=SEED, max_steps=MAX_STEPS)

    # --- Print summary ---
    print("\n=== CBFKit Gymnasium Safety Filter Demo ===")
    print(
        f"Unsafe run:  collision={unsafe_result['collision']:<7} "
        f"goal_reached={unsafe_result['goal_reached']:<7} "
        f"steps={unsafe_result['steps']}"
    )
    print(
        f"Safe run:    collision={safe_result['collision']:<7} "
        f"goal_reached={safe_result['goal_reached']:<7} "
        f"steps={safe_result['steps']}"
    )
    if safe_result["min_barrier"] is not None:
        print(f"  Min barrier value:    {safe_result['min_barrier']:.4f}")
    print(f"  Intervention rate:    {safe_result['intervention_rate']:.1%}")

    # --- Plot ---
    if not TEST_MODE:
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            obstacles = env_unsafe.unwrapped.obstacles
            goal = env_unsafe.unwrapped._default_goal

            for ax, result, title in [
                (ax1, unsafe_result, "Unsafe (no CBF)"),
                (ax2, safe_result, "Safe (CBF filter)"),
            ]:
                for cx, cy, r in obstacles:
                    circle = plt.Circle((cx, cy), r, color="red", alpha=0.3)
                    ax.add_patch(circle)
                ax.plot(*goal, "g*", markersize=15, label="Goal")
                traj = result["trajectory"]
                ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=2, label="Trajectory")
                ax.plot(traj[0, 0], traj[0, 1], "bs", markersize=8, label="Start")
                ax.plot(traj[-1, 0], traj[-1, 1], "bo", markersize=8, label="End")
                ax.set_xlim(-0.5, 5)
                ax.set_ylim(-1.5, 1.5)
                ax.set_aspect("equal")
                ax.set_title(title)
                ax.legend(loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = os.path.join(os.path.dirname(__file__), "gymnasium_safe_vs_unsafe.png")
            plt.savefig(save_path, dpi=150)
            print(f"\nPlot saved to: {save_path}")
            plt.close()
        except ImportError:
            print("matplotlib not available, skipping plot.")


if __name__ == "__main__":
    main()
