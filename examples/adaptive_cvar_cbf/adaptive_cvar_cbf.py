"""Adaptive CVaR-CBF control for a double integrator with obstacle avoidance."""
import os

import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from cbfkit.controllers.adaptive_cvar_cbf import adaptive_cvar_cbf_controller
from cbfkit.simulation import simulator
from cbfkit.systems.double_integrator import models


class Obstacle:
    def __init__(self, x0, radius, dt):
        self.x_curr = np.array(x0).reshape(-1, 1)
        self.velocity_xy = np.zeros((2, 1))
        self.radius = radius
        self.dt = dt
        self.noise = [[0.01] * 4, [0.01] * 4]
        self.xlog = [self.x_curr]

    def step(self):
        # Simple static or slowly moving obstacle
        # For demo, let's make it static
        self.xlog.append(self.x_curr)


def nominal_controller(t, x, key, data=None):
    # PD controller to target
    target = jnp.array([4.0, 4.0])
    # Keep the nominal command moderate so the CVaR-CBF QP stays feasible near obstacles.
    kp = 0.3
    kd = 1.5

    pos_err = target - x[:2]
    vel_des = kp * pos_err
    # clamp vel
    v_norm = jnp.linalg.norm(vel_des)
    v_max = 1.5
    vel_des = jnp.where(v_norm > v_max, vel_des * v_max / v_norm, vel_des)

    acc_des = kd * (vel_des - x[2:4])

    # clamp acc
    a_norm = jnp.linalg.norm(acc_des)
    a_max = 1.5
    acc_des = jnp.where(a_norm > a_max, acc_des * a_max / a_norm, acc_des)

    return acc_des, data


def main():
    dt = 0.1
    tf = 20.0
    num_steps = int(tf / dt)

    # System
    dynamics = models.double_integrator_4d.plant()

    # Robot setup
    x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
    radius = 0.3

    # Obstacles
    obs1 = Obstacle([2.0, 2.5, 0.0, 0.0], 0.3, dt)
    obstacles = [obs1]

    # Controller Setup
    # Discrete dynamics approximation for the controller
    A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = np.array([[0.5 * dt**2, 0], [0, 0.5 * dt**2], [dt, 0], [0, dt]])

    dyn_model = {
        "A": A,
        "B": B,
        "dt": dt,
        "u_min": -2.0 * np.ones(2),
        "u_max": 2.0 * np.ones(2),
        "radius": radius,
    }

    params = {"htype": "dist", "S": 15, "beta": 0.99}
    noise_params = [[0.01] * 4, [0.01] * 4]

    controller = adaptive_cvar_cbf_controller(
        dynamics_model=dyn_model, obstacles=obstacles, params=params, noise_params=noise_params
    )

    # Integrator (Euler for simplicity matching the discrete logic)
    def integrator(x, xdot, dt):
        return x + xdot(x) * dt

    # Simulation
    print("Starting Simulation...")

    import jax.random as random

    # Seed numpy for consistent uncertainty generation
    np.random.seed(42)
    key = random.PRNGKey(0)

    sim_iter = simulator.simulator(
        dt=dt,
        num_steps=num_steps,
        dynamics=dynamics,
        integrator=integrator,
        planner=None,
        nominal_controller=nominal_controller,
        controller=controller,
        sensor=None,
        estimator=None,
        perturbation=None,
        sigma=None,
        key=key,
    )

    states = [x0]
    controls = []

    # Run loop
    for step_data in sim_iter(x0):
        # Update obstacles (if dynamic)
        for obs in obstacles:
            obs.step()

        states.append(step_data.state)
        controls.append(step_data.control)

        if step_data.controller_values:
            keys = step_data.controller_keys
            vals = step_data.controller_values
            if "sub_data" in keys:
                idx = keys.index("sub_data")
                sub_data = vals[idx]
                if isinstance(sub_data, dict) and "solver_status" in sub_data:
                    status = sub_data["solver_status"]
                    if status not in ["Solve_Succeeded", "Solved_To_Acceptable_Level"]:
                        print(f"Step {len(states)}: Solver failed with status: {status}")

        if jnp.linalg.norm(step_data.state[:2] - jnp.array([4.0, 4.0])) < 0.5:
            print("Goal Reached!")
            break

    print("Simulation Complete.")

    # Visualization
    states = np.array(states)
    controls = np.array(controls)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-2, 6)
    ax.set_ylim(-2, 6)
    ax.set_aspect("equal")
    ax.grid(True)

    robot_patch = Circle((x0[0], x0[1]), radius, color="b", alpha=0.8)
    ax.add_patch(robot_patch)
    target_patch = Circle((4, 4), 0.2, color="g")
    ax.add_patch(target_patch)

    obs_patches = []
    for obs in obstacles:
        p = Circle((obs.x_curr[0].item(), obs.x_curr[1].item()), obs.radius, color="r", alpha=0.6)
        ax.add_patch(p)
        obs_patches.append(p)

    (robot_line,) = ax.plot([], [], "b--")

    def update(frame):
        if frame >= len(states):
            return []
        pos = states[frame]
        robot_patch.center = (pos[0], pos[1])
        robot_line.set_data(states[: frame + 1, 0], states[: frame + 1, 1])
        return [robot_patch, robot_line]

    ani = animation.FuncAnimation(fig, update, frames=len(states), blit=True, interval=100)
    anim_path = os.path.join(results_dir, "cbfkit_demo.gif")

    from cbfkit.utils.animator import save_animation

    save_animation(ani, anim_path)


if __name__ == "__main__":
    main()
