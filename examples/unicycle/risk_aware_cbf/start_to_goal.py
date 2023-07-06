import jax.numpy as jnp

import cbfkit.models.unicycle as unicycle
import cbfkit.system as system
from cbfkit.cbf_controllers import (
    risk_aware_cbf_controller as cbf_controller,
)

approx_unicycle_dynamics = unicycle.approx_unicycle_dynamics(l=1.0, stochastic=True)
init_state = jnp.array([0.0, 0.0, jnp.pi / 4])
desired_state = jnp.array([2.0, 4.0, 0])

approx_uniycle_nom_controller = unicycle.approx_unicycle_nominal_controller(
    dynamics=approx_unicycle_dynamics, Kp_pos=1, Kp_theta=0.01, desired_state=desired_state
)

controller = cbf_controller(
    nominal_input=approx_uniycle_nom_controller,
    dynamics_func=approx_unicycle_dynamics,
    barriers=unicycle.barrier_funcs,
    alpha=jnp.array([2.0] * 3),
)

tf = 10.0
dt = 0.01

bicycle_states, bicycle_data_keys, bicycle_data_values = system.simulate(
    state=init_state,
    dynamics=approx_unicycle_dynamics,
    controller=controller,
    dt=dt,
    num_steps=int(tf / dt),
)

bicycle_states = jnp.asarray(bicycle_states)

plot = 0
animate = 1
save = 1

if plot:
    unicycle.plot_trajectory(
        states=bicycle_states,
        desired_state=desired_state,
        desired_state_radius=0.1,
        x_lim=(-2, 6),
        y_lim=(-2, 6),
        title="System Behavior",
    )

if animate:
    unicycle.animate(
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
