
import time
import jax.numpy as jnp
import pytest
from jax import random

import cbfkit.systems.unicycle.models.olfatisaber2002approximate as unicycle
from cbfkit.certificates import concatenate_certificates
from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
from cbfkit.controllers.cbf_clf.vanilla_cbf_clf_qp_control_laws import vanilla_cbf_clf_qp_controller
from cbfkit.integration import runge_kutta_4 as integrator_rk4
from cbfkit.simulation import simulator as sim
from cbfkit.utils.user_types import PlannerData

# Simulation setup
N = 3  # n_states
M = 2  # n_controls
TF = 1.0 # Shorten for speed
DT = 1e-2
N_STEPS = int(TF / DT)

# Initial conditions
x_max = 5.0
y_max = 5.0
ACTUATION_LIMITS = jnp.array([1e3, 1e3])

# Barrier function parameters
ALPHA = 0.5
GOAL = jnp.array([0.0, 0.0, 0])
OBSTACLES = [jnp.array([1.0, 2.0])]
ELLIPSOIDS = [jnp.array([0.5, 0.5])]

# Lyapunov function
bars = [
    unicycle.certificates.barrier_functions.obstacle_ca(
        certificate_conditions=linear_class_k(ALPHA),
        obstacle=jnp.array([obs[0], obs[1], 0.0]),
        ellipsoid=jnp.array([ell[0], ell[1]]),
    )
    for obs, ell in zip(OBSTACLES, ELLIPSOIDS)
]
BARRIERS = concatenate_certificates(*bars)

DYNAMICS = unicycle.plant(lam=1.0)
NOMINAL_CONTROLLER = unicycle.controllers.proportional_controller(
    dynamics=DYNAMICS,
    Kp_pos=1.0,
    Kp_theta=0.01,
)

CONTROLLER = vanilla_cbf_clf_qp_controller(
    control_limits=ACTUATION_LIMITS,
    dynamics_func=DYNAMICS,
    barriers=BARRIERS,
)

@pytest.fixture
def initial_state():
    return jnp.array([-2.0, -2.0, 0.0])

def test_simulator_jit_rk4_correctness(initial_state):
    """Tests that JIT-compiled execution with RK4 optimization runs without error."""
    planner_data = PlannerData(
        u_traj=None,
        x_traj=jnp.tile(GOAL.reshape(-1, 1), (1, N_STEPS + 1)),
        prev_robustness=None,
    )

    # Execute simulation
    x, u, z, c, _, _, _, _ = sim.execute(
        x0=initial_state,
        dt=DT,
        num_steps=N_STEPS,
        dynamics=DYNAMICS,
        integrator=integrator_rk4,
        planner=None,
        nominal_controller=NOMINAL_CONTROLLER,
        controller=CONTROLLER,
        sensor=None, # Use default
        estimator=None, # Use default
        key=random.PRNGKey(0),
        planner_data=planner_data,
        use_jit=True,
        verbose=False,
    )

    # Check for NaNs
    assert not jnp.any(jnp.isnan(x))
    assert x.shape == (N_STEPS, N)

    # Check that we moved somewhat towards goal or stayed safe
    # This is a loose check just to ensure dynamics were applied
    assert not jnp.allclose(x[0], x[-1])
