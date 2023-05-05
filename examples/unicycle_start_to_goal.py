import jax.numpy as jnp

import cbfkit.models.unicycle as unicycle
import cbfkit.system as system
from cbfkit.cbf_controllers import (
    adaptive_control_barrier_function_controller,
    adaptive_risk_aware_cbf_controller,
    adaptive_stochastic_cbf_controller,
    control_barrier_function_controller,
    risk_aware_cbf_controller,
    stochastic_cbf_controller,
)

approx_unicycle_dynamics = unicycle.approx_unicycle_dynamics(l=1.0, stochastic=True)
init_state = jnp.array([0.0, 0.0, jnp.pi])
desired_state = jnp.array([2, 4, 0])

approx_uniycle_nom_controller = unicycle.approx_unicycle_nominal_controller(
    dynamics=approx_unicycle_dynamics, Kp_pos=1, Kp_theta=0.01, desired_state=desired_state
)

# * uncomment to test step function
# new_state = system.step(
#     state=init_state, dynamics=bicycle_dynamics, controller=bicycle_nom_controller, dt=0.1
# )


cbf_controller = adaptive_risk_aware_cbf_controller(
    nominal_input=approx_uniycle_nom_controller,
    dynamics_func=approx_unicycle_dynamics,
    barrier_funcs=unicycle.barrier_functions,
    barrier_jacobians=unicycle.barrier_jacobians,
    barrier_hessians=unicycle.barrier_hessians,
)

dt = 0.05

bicycle_states = system.simulate(
    state=init_state,
    dynamics=approx_unicycle_dynamics,
    # controller=approx_uniycle_nom_controller,
    controller=cbf_controller,
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
