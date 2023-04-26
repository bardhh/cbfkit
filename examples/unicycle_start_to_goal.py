import jax.numpy as jnp

import cbfkit.models.unicycle as unicycle
import cbfkit.system as system

approx_unicycle_dynamics = unicycle.approx_unicycle_dynamics(l=1.0)
init_state = jnp.array([0.0, 0.0, jnp.pi])
desired_state = jnp.array([2, 4, 0])

approx_uniycle_nom_controller = unicycle.approx_unicycle_nominal_controller(
    dynamics=approx_unicycle_dynamics, Kp_pos=1, Kp_theta=0.01, desired_state=desired_state
)

# * uncomment to test step function
# new_state = system.step(
#     state=init_state, dynamics=bicycle_dynamics, controller=bicycle_nom_controller, dt=0.1
# )

dt = 0.05

bicycle_states = system.simulate(
    state=init_state,
    dynamics=approx_unicycle_dynamics,
    controller=approx_uniycle_nom_controller,
    dt=dt,
    num_steps=200,
)

bicycle_states = jnp.asarray(bicycle_states)

plot = 0
animate = 1

if plot:
    unicycle.plot_trajectory(
        states=bicycle_states,
        desired_state=desired_state,
        desired_state_radius=0.1,
        x_lim=(-4, 4),
        y_lim=(-4, 4),
        title="System Behavior",
    )

if animate:
    unicycle.animate(
        states=bicycle_states,
        desired_state=desired_state,
        desired_state_radius=0.1,
        x_lim=(-4, 4),
        y_lim=(-4, 4),
        dt=dt,
        title="System Behavior",
    )
