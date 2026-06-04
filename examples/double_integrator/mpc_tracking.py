"""Receding-horizon MPC tracking a goal with an LTI double integrator."""
import os
import sys

# Add the project root to the path so we can import cbfkit + examples.
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import jax.numpy as jnp
import numpy as np

from cbfkit.optimization.mpc.quadratic_cost_linear_dynamics import (
    generate_mpc_solver_quadratic_cost_linear_dynamics,
)

# In test mode we shorten the horizon loop and skip the (slow) GIF render.
TEST_MODE = bool(os.getenv("CBFKIT_TEST_MODE"))


def main() -> float:
    """Run the receding-horizon MPC loop and (optionally) save the animation.

    Returns the final Euclidean position error to the goal.
    """
    dt = 0.1
    # Discrete-time double integrator: state [px, py, vx, vy], control [ax, ay].
    A = jnp.array(
        [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    B = jnp.array([[0.0, 0.0], [0.0, 0.0], [dt, 0.0], [0.0, dt]])
    Q = jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0]))
    R = 0.1 * jnp.eye(2)
    Qn = 50.0 * Q
    N = 20
    solve = generate_mpc_solver_quadratic_cost_linear_dynamics(A, B, Q, R, Qn, N)

    goal = jnp.array([4.0, 4.0, 0.0, 0.0])
    ref_horizon = jnp.tile(goal, (N, 1))  # (N, 4) constant reference over the horizon
    x = jnp.array([0.0, 0.0, 0.0, 0.0])
    n_steps = 40 if not TEST_MODE else 5

    xs = [np.asarray(x)]
    preds = []
    for _ in range(n_steps):
        concatenated_x_xr = jnp.vstack([x.reshape(1, -1), ref_horizon])  # (N+1, 4)
        x_opt, u_opt = solve(concatenated_x_xr)  # x_opt (4, N+1), u_opt (2, N)
        u = u_opt[:, 0]
        x = A @ x + B @ u
        xs.append(np.asarray(x))
        preds.append(np.asarray(x_opt.T))  # (N+1, 4) predicted state horizon
    xs = np.array(xs)

    final_err = float(np.linalg.norm(xs[-1, :2] - np.asarray(goal)[:2]))
    print(f"Final position error to goal: {final_err:.4f}")

    if TEST_MODE:
        return final_err

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(float(goal[0]), float(goal[1]), "g*", markersize=18, label="Goal")
    (realized,) = ax.plot([], [], "b-", lw=2, label="Realized")
    (pred,) = ax.plot(
        [], [], color="orange", ls="--", lw=1.5, alpha=0.85, label="Predicted horizon"
    )
    dot = ax.scatter([], [], s=80, color="blue", zorder=5)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Model Predictive Control — receding-horizon LTI tracking", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    def update(i):
        realized.set_data(xs[: i + 1, 0], xs[: i + 1, 1])
        dot.set_offsets([[xs[i, 0], xs[i, 1]]])
        p = preds[min(i, len(preds) - 1)]
        pred.set_data(p[:, 0], p[:, 1])
        return realized, pred, dot

    anim = FuncAnimation(fig, update, frames=len(xs), interval=100, blit=True)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, "mpc_tracking.gif")
    anim.save(out, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"Saved animation to {out}")

    return final_err


if __name__ == "__main__":
    main()
