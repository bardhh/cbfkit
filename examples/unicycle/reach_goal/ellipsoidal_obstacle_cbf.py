"""Unicycle reach-goal with a vanilla CBF-CLF QP filter avoiding one ellipsoidal obstacle."""
import os
import sys

# Add the project root to the path so we can import examples
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import jax.numpy as jnp
import numpy as np

import cbfkit.simulation.simulator as sim
import cbfkit.systems.unicycle.models.olfatisaber2002approximate as unicycle
from cbfkit.certificates import concatenate_certificates, rectify_relative_degree
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.systems.unicycle import proportional_controller
from cbfkit.utils.user_types import PlannerData
from examples.unicycle.common.ellipsoidal_obstacle import cbf as ellipsoid_cbf

# Test mode: short horizon, skip the GIF render so tests stay fast.
TEST_MODE = bool(os.getenv("CBFKIT_TEST_MODE"))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
GIF_PATH = os.path.join(RESULTS_DIR, "ellipsoidal_obstacle_cbf.gif")


def main() -> str:
    dyn = unicycle.plant(lam=1.0)
    x0 = jnp.array([0.0, 0.0, jnp.pi / 2])
    xg = jnp.array([4.0, 4.0, 0.0])
    obs = jnp.array([2.0, 2.0, 0.0])
    ell = jnp.array([0.6, 0.6])
    barriers = concatenate_certificates(
        rectify_relative_degree(
            function=ellipsoid_cbf(obs, ell),
            system_dynamics=dyn,
            state_dim=3,
            form="exponential",
            roots=jnp.array([-1.0]),
        )(certificate_conditions=zeroing_barriers.linear_class_k(alpha=2.0))
    )
    nominal = proportional_controller(dynamics=dyn, Kp_pos=1, Kp_theta=0.01)
    controller = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([5.0, 5.0]),
        nominal_input=nominal,
        dynamics_func=dyn,
        barriers=barriers,
    )
    tf = 8.0 if not TEST_MODE else 1.0
    dt = 0.02
    n = int(tf / dt)
    res = sim.execute(
        x0=x0,
        dt=dt,
        num_steps=n,
        dynamics=dyn,
        integrator=integrator,
        nominal_controller=nominal,
        controller=controller,
        sensor=sensor,
        estimator=estimator,
        planner_data=PlannerData(
            u_traj=None,
            x_traj=jnp.tile(xg.reshape(-1, 1), (1, n + 1)),
            prev_robustness=None,
        ),
        use_jit=True,
    )
    states = np.asarray(res["states"])

    final_dist = float(np.linalg.norm(states[-1, :2] - np.asarray(xg[:2])))
    print(f"Final distance to goal: {final_dist:.4f}")

    if TEST_MODE:
        return GIF_PATH

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(
        plt.matplotlib.patches.Ellipse(
            (float(obs[0]), float(obs[1])),
            float(ell[0]) * 2,
            float(ell[1]) * 2,
            facecolor="red",
            alpha=0.35,
            edgecolor="red",
            lw=1.5,
        )
    )
    ax.plot(float(xg[0]), float(xg[1]), "g*", markersize=18, label="Goal")
    (line,) = ax.plot([], [], "b-", lw=2)
    dot = ax.scatter([], [], s=80, color="blue", zorder=5)
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.set_aspect("equal")
    ax.set_title("Ellipsoidal-obstacle CBF — unicycle reach-goal", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    def update(i):
        line.set_data(states[: i + 1, 0], states[: i + 1, 1])
        dot.set_offsets([[states[i, 0], states[i, 1]]])
        return line, dot

    stride = max(1, len(states) // 70)
    anim = FuncAnimation(fig, update, frames=range(0, len(states), stride), interval=100, blit=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    anim.save(GIF_PATH, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"Saved animation to {GIF_PATH}")
    return GIF_PATH


if __name__ == "__main__":
    main()
