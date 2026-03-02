import os
import sys

# Add the project root to the path so we can import examples
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, root_path)

import jax.numpy as jnp

import cbfkit.simulation.simulator as sim
import cbfkit.systems.unicycle.models.olfatisaber2002approximate as unicycle
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.systems.unicycle import proportional_controller
from cbfkit.utils.user_types import PlannerData

approx_unicycle_dynamics = unicycle.plant(lam=1.0)
init_state = jnp.array([0.0, 0.0, jnp.pi / 2])
desired_state = jnp.array([4.0, 4.0, 0])

# For start-to-goal without obstacles, use proportional controller
controller = proportional_controller(dynamics=approx_unicycle_dynamics, Kp_pos=1, Kp_theta=0.01)

tf = 10.0 if not os.environ.get("CBFKIT_TEST_MODE") else 1.0
dt = 0.01

x, u, z, p, controller_keys, controller_values, planner_keys, planner_values = sim.execute(
    x0=init_state,
    dt=dt,
    num_steps=int(tf / dt),
    dynamics=approx_unicycle_dynamics,
    integrator=integrator,
    nominal_controller=controller,
    sensor=sensor,
    estimator=estimator,
    planner_data=PlannerData(
        u_traj=None,
        x_traj=jnp.tile(desired_state.reshape(-1, 1), (1, int(tf / dt) + 1)),
        prev_robustness=None,
    ),
    use_jit=True,
)

bicycle_states = jnp.asarray(x)

plot = 1 if not os.environ.get("CBFKIT_TEST_MODE") else 0
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
        estimates=jnp.zeros_like(bicycle_states),
        desired_state=desired_state,
        desired_state_radius=0.1,
        x_lim=(-2, 6),
        y_lim=(-2, 6),
        dt=dt,
        title="System Behavior",
        save_animation=save,
        animation_filename="examples/unicycle/reach_goal/results/risk_aware_cbf.gif",
    )
