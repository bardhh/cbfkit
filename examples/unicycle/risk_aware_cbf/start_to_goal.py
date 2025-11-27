import jax.numpy as jnp

import cbfkit.systems.unicycle.models.olfatisaber2002approximate as unicycle
import cbfkit.simulation.simulator as sim
from cbfkit.sensors import perfect as sensor
from cbfkit.estimators import naive as estimator
from cbfkit.integration import forward_euler as integrator

approx_unicycle_dynamics = unicycle.plant(l=1.0)
init_state = jnp.array([0.0, 0.0, jnp.pi / 4])
desired_state = jnp.array([4.0, 4.0, 0])

# For start-to-goal without obstacles, use proportional controller
controller = unicycle.controllers.proportional_controller(
    dynamics=approx_unicycle_dynamics, Kp_pos=1, Kp_theta=0.01
)

tf = 10.0
dt = 0.01

x, u, z, p, data_keys, data_values, planner_data, planner_data_keys = sim.execute(
    x0=init_state,
    dt=dt,
    num_steps=int(tf / dt),
    dynamics=approx_unicycle_dynamics,
    integrator=integrator,
    nominal_controller=controller,
    sensor=sensor,
    estimator=estimator,
    planner_data={
        "u_traj": None,
        "x_traj": jnp.tile(desired_state.reshape(-1, 1), (1, int(tf / dt) + 1)),
        "prev_robustness": None,
    },
    use_jit=True,
)

bicycle_states = jnp.asarray(x)

plot = 1
animate = 0  # Disabled - needs visualization refactoring
save = 1

if plot:
    import matplotlib.pyplot as plt
    from examples.unicycle.common.visualizations import plot_trajectory

    plot_trajectory(
        states=bicycle_states,
        desired_state=desired_state,
        desired_state_radius=0.1,
        x_lim=(-2, 6),
        y_lim=(-2, 6),
        title="System Behavior",
    )
    plt.show()

if animate:
    from examples.unicycle.common.visualizations import animate as animate_trajectory

    animate_trajectory(
        states=bicycle_states,
        desired_state=desired_state,
        desired_state_radius=0.1,
        x_lim=(-2, 6),
        y_lim=(-2, 6),
        dt=dt,
        title="System Behavior",
        save_animation=save,
        animation_filename="examples/unicycle/risk_aware_cbf/results/start_to_goal1.gif",
    )
