"""6-DOF quadrotor geometric SE(3) tracking with a live altitude-CBF barrier value."""
import os
import sys

# Add the project root to the path so cbfkit + examples imports resolve.
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import jax.numpy as jnp
import numpy as np

import cbfkit.simulation.simulator as sim
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.systems.quadrotor_6dof.certificates.barrier_functions import h_alt
from cbfkit.systems.quadrotor_6dof.controllers.geometric import geometric_controller
from cbfkit.systems.quadrotor_6dof.models.quadrotor_6dof_dynamics import (
    quadrotor_6dof_dynamics,
)

TEST_MODE = bool(os.getenv("CBFKIT_TEST_MODE"))


def main() -> None:
    # Mass/inertia must be consistent between plant and controller: geometric_controller's
    # default gains are tuned for m≈4.34 kg, while quadrotor_6dof_dynamics defaults to
    # m=0.25 kg. Mismatch -> instant integration NaN. Use the heavier plant.
    m, jx, jy, jz = 4.34, 0.0820, 0.0845, 0.1377
    three_tuple = quadrotor_6dof_dynamics(m=m, jx=jx, jy=jy, jz=jz)

    def dyn(x):
        f, g, _s = three_tuple(x)
        return f, g

    desired = jnp.array([2.0, 1.5, 3.0])  # target (pn, pe, h)
    dt = 0.01
    tf = 6.0 if not TEST_MODE else 0.5
    n = int(tf / dt)

    # state layout: [pn, pe, h, u, v, w, phi, theta, psi, p, q, r]
    x0 = jnp.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    nominal = geometric_controller(
        dynamics=dyn, desired_state=desired, dt=dt, m=m, jx=jx, jy=jy, jz=jz
    )

    res = sim.execute(
        x0=x0,
        dt=dt,
        num_steps=n,
        dynamics=dyn,
        integrator=integrator,
        nominal_controller=nominal,
        sensor=sensor,
        estimator=estimator,
        use_jit=True,
    )
    states = np.asarray(res["states"])  # (n+1, 12)

    # Altitude-CBF barrier value h_alt(z, alt_limit). z = hstack([x, t]).
    # alt_limit must comfortably exceed our setpoint altitude (3 m) — pick 5 m.
    alt_limit = 5.0
    n_states_full = states.shape[0]
    ts = np.linspace(0.0, tf, n_states_full)
    h_vals = np.array(
        [
            float(h_alt(jnp.hstack([jnp.asarray(states[i]), jnp.asarray(ts[i])]), alt_limit))
            for i in range(n_states_full)
        ]
    )

    final_pos = states[-1, :3]
    dist = float(np.linalg.norm(final_pos - np.asarray(desired)))
    print(f"Final distance to goal: {dist:.4f} m")
    print(f"Minimum altitude-CBF value h_alt: {float(h_vals.min()):.4f} (positive => safe)")

    if TEST_MODE:
        return

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

    # Subsample frames for a compact GIF.
    stride = max(1, n_states_full // 80)
    idx = np.arange(0, n_states_full, stride)
    pn, pe, h_alt_traj = states[idx, 0], states[idx, 1], states[idx, 2]

    fig = plt.figure(figsize=(10, 5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_h = fig.add_subplot(1, 2, 2)

    ax3d.scatter(
        [float(desired[0])],
        [float(desired[1])],
        [float(desired[2])],
        color="green",
        s=120,
        marker="*",
        label="Goal",
        zorder=10,
    )
    (line3d,) = ax3d.plot([], [], [], "b-", lw=2, label="Quadrotor")
    dot3d = ax3d.scatter([], [], [], s=60, color="blue", zorder=11)
    pad = 0.5
    ax3d.set_xlim(min(pn.min(), float(desired[0])) - pad, max(pn.max(), float(desired[0])) + pad)
    ax3d.set_ylim(min(pe.min(), float(desired[1])) - pad, max(pe.max(), float(desired[1])) + pad)
    ax3d.set_zlim(0, alt_limit + 0.5)
    ax3d.set_xlabel("pn [m]")
    ax3d.set_ylabel("pe [m]")
    ax3d.set_zlabel("h [m]")
    ax3d.set_title("Quadrotor 6-DOF — geometric SE(3) tracking", fontsize=10)
    ax3d.legend(loc="upper right", fontsize=8)
    ax3d.view_init(elev=22, azim=-60)

    # h(z) trace: stays >0 ⇒ altitude envelope satisfied.
    ax_h.plot(ts, h_vals, color="purple", lw=1.5)
    (h_dot,) = ax_h.plot([], [], "o", color="purple", markersize=7)
    ax_h.axhline(0.0, color="red", ls="--", lw=1, alpha=0.7, label="Safety boundary h=0")
    ax_h.set_xlim(0, tf)
    ax_h.set_ylim(min(0.0, float(h_vals.min())) - 0.1, max(1.0, float(h_vals.max())) + 0.1)
    ax_h.set_xlabel("t [s]")
    ax_h.set_ylabel("$h_{\\rm alt}(z)$")
    ax_h.set_title("Altitude-CBF barrier value (positive ⇒ safe)", fontsize=10)
    ax_h.legend(loc="lower right", fontsize=8)
    ax_h.grid(True, alpha=0.3)

    def update(i):
        line3d.set_data(pn[: i + 1], pe[: i + 1])
        line3d.set_3d_properties(h_alt_traj[: i + 1])
        dot3d._offsets3d = ([pn[i]], [pe[i]], [h_alt_traj[i]])
        # Map subsampled index back to full-resolution h_vals index for the dot.
        full_i = idx[i]
        h_dot.set_data([ts[full_i]], [h_vals[full_i]])
        return line3d, dot3d, h_dot

    anim = FuncAnimation(fig, update, frames=len(idx), interval=100, blit=False)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, "geometric_tracking.gif")
    anim.save(out, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"Saved animation to {out}")


if __name__ == "__main__":
    main()
