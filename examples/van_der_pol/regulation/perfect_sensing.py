"""Van der Pol regulation with perfect state measurements and CBF-CLF control."""
import os
import sys
from typing import List, Optional, Tuple

# Add the project root to the path so we can import examples
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, root_path)

import matplotlib

# Prevent backend override in imported visualization modules.
matplotlib.use = lambda *args, **kwargs: None

import jax.numpy as jnp
from jax import Array, jit

import cbfkit.simulation.simulator as sim
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.systems import van_der_pol
from cbfkit.utils.user_types import ControllerData, Key, State
from examples.van_der_pol.common.config import perfect_state_measurements as setup
from examples.van_der_pol.visualizations.path import animate

# Simulation/plot settings
tf = 5.0 if not os.getenv("CBFKIT_TEST_MODE") else 0.5
n_steps = int(tf / setup.dt)
plot = 1 if not os.getenv("CBFKIT_TEST_MODE") else 0
save = 0

# Controlled reverse-time Van der Pol dynamics
dynamics = van_der_pol.reverse_van_der_pol_oscillator(epsilon=setup.epsilon, sigma=setup.Q)


def regulation_controller(epsilon: float, k1: float = 4.0, k2: float = 4.0):
    """Nominal controller for reverse-time Van der Pol regulation.

    The input is chosen so the closed-loop error dynamics are damped and avoid the
    1/x2 singular amplification in the plant input channel near the goal.
    """

    @jit
    def controller(
        _t: float, x: State, _key: Key, _xd: Optional[State] = None
    ) -> Tuple[Array, ControllerData]:
        x1, x2 = x
        u = x2 * ((k1 - 1.0) * x1 - k2 * x2 + epsilon * (1.0 - x1**2) * x2)

        # logging data
        data = ControllerData()
        return jnp.array([u]), data

    return controller


nominal_controller = regulation_controller(setup.epsilon, k1=4.0, k2=4.0)


def execute() -> Tuple[Array, Array, Array, Array, List[str], List[Array], List[str], List[Array]]:
    x, u, z, p, c_keys, c_values, p_keys, p_values = sim.execute(
        x0=setup.initial_state,
        dynamics=dynamics,
        sensor=sensor,
        nominal_controller=nominal_controller,
        estimator=estimator,
        integrator=integrator,
        dt=setup.dt,
        sigma=setup.R,
        num_steps=n_steps,
        use_jit=True,
    )

    return x, u, z, p, c_keys, c_values, p_keys, p_values


(
    states,
    controls,
    estimates,
    covariances,
    data_keys,
    data_values,
    planner_keys,
    planner_values,
) = execute()

final_distance = jnp.linalg.norm(states[-1] - setup.desired_state)
print(f"Final Distance to Goal: {float(final_distance):.6f}")
print(f"Goal Reached (radius={setup.goal_radius}): {bool(final_distance <= setup.goal_radius)}")

if plot:
    bound = float(jnp.maximum(jnp.max(jnp.abs(states[:, 0])), jnp.max(jnp.abs(states[:, 1])))) + 0.5

    animate(
        states=states,
        estimates=estimates,
        desired_state=setup.desired_state,
        desired_state_radius=setup.goal_radius,
        x_lim=(-bound, bound),
        y_lim=(-bound, bound),
        dt=setup.dt,
        title="System Behavior",
        save_animation=bool(save),
        animation_filename="examples/van_der_pol/regulation/results/perfect_measurements.gif",
    )
