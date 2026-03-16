"""Tutorial: Unicycle reach-avoid with ellipsoidal barriers and speed limits."""
import os
import jax.numpy as jnp
from jax import jit
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("Matplotlib not found. Plotting disabled. Install 'cbfkit[vis]' to enable plotting.")

from cbfkit.certificates import concatenate_certificates, rectify_relative_degree
from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.estimators import naive
from cbfkit.integration import runge_kutta_4
from cbfkit.sensors import perfect
from cbfkit.simulation import simulator
from cbfkit.systems.unicycle.models.accel_unicycle.dynamics import accel_unicycle_dynamics
from cbfkit.certificates.barrier_functions import ellipsoidal_barrier_factory

# Simulation Parameters
initial_state = jnp.array([2.0, 2.0, 0.0, -3 * jnp.pi / 4])
actuation_limits = jnp.array([1.0, jnp.pi])
dt = 1e-2
num_steps = 1000 if not os.getenv("CBFKIT_TEST_MODE") else 50

# Dynamics
dynamics = accel_unicycle_dynamics()


# Nominal Controller
@jit
def nominal_controller_func(t, state, key, data):
    # Unpack state
    x, y, v, th = state

    # Target
    xg, yg = -2.0, -2.0
    kp = 1.0

    # Distance error
    dist = jnp.linalg.norm(jnp.array([x - xg, y - yg]))

    # Heading error
    thd = jnp.arctan2(yg - y, xg - x)
    theta_err = jnp.arctan2(jnp.sin(thd - th), jnp.cos(thd - th))

    # Control laws from tutorial
    accel = kp * (dist - v)
    omega = kp * theta_err

    return jnp.array([accel, omega]), {}


# Barriers
# 1. Obstacle Avoidance: (x-xo)^2 + (y-yo)^2 - r^2 >= 0
# The ellipsoidal_barrier_factory creates a factory for generating barrier functions.
# It expects the center position and the semi-axes lengths of the ellipsoid.
cbf_factory, _, _ = ellipsoidal_barrier_factory(
    system_position_indices=(0, 1),
    obstacle_position_indices=(0, 1),
    ellipsoid_axis_indices=(0, 1),
)

# Obstacle parameters: xo=0.9, yo=1.0, semi-axes=[0.5, 0.5] (circle with r=0.5)
obs_pos = jnp.array([0.9, 1.0])
obs_axes = jnp.array([0.5, 0.5])

# Create the barrier function h(x)
h_obs = cbf_factory(obs_pos, obs_axes)

# Rectify relative degree:
# The barrier function h(x) depends on position, but the control input (acceleration)
# appears in the second derivative of position. Thus, the relative degree is 2.
# rectify_relative_degree automatically computes the necessary derivatives and
# returns a function that generates the high-order barrier certificate.
obstacle_cbf_package = rectify_relative_degree(
    function=h_obs,
    system_dynamics=dynamics,
    state_dim=len(initial_state),
    form="exponential",
    roots=jnp.array([-1.0, -1.0]),  # Roots for the pole placement in the linear system
)

# 2. Speed Limit: l^2 - v^2 >= 0 => l=1.0
@jit
def h_speed(state_and_time):
    v = state_and_time[2]
    l = 1.0
    return l**2 - v**2


# The speed limit barrier function depends on velocity. The control input (acceleration)
# appears in the first derivative of velocity. Thus, the relative degree is 1.
speed_limit_cbf_package = rectify_relative_degree(
    function=h_speed,
    system_dynamics=dynamics,
    state_dim=len(initial_state),
    form="exponential",
    roots=jnp.array([-1.0]),
)

certificate_collection = concatenate_certificates(
    obstacle_cbf_package(certificate_conditions=linear_class_k(10.0)),
    speed_limit_cbf_package(certificate_conditions=linear_class_k(1.0)),
)

# Controller
controller = vanilla_cbf_clf_qp_controller(
    control_limits=actuation_limits,
    dynamics_func=dynamics,
    barriers=certificate_collection,
    p_mat=jnp.diag(jnp.array([1.0, 0.1])),
)

# Simulation
x_sim, u_sim, z_sim, p_sim, controller_keys, controller_values, planner_keys, planner_values = simulator.execute(
    x0=initial_state,
    dt=dt,
    num_steps=num_steps,
    dynamics=dynamics,
    integrator=runge_kutta_4,
    nominal_controller=nominal_controller_func,
    controller=controller,
    sensor=perfect,
    estimator=naive,
    use_jit=True,
)

# Plotting
if PLOT_AVAILABLE:
    fig, ax = plt.subplots()
    circle1 = patches.Circle((0.9, 1.0), radius=0.5, edgecolor="r", facecolor="k")
    ax.add_patch(circle1)
    ax.plot(x_sim[:, 0], x_sim[:, 1])
    ax.set_aspect("equal")
    plt.title("Unicycle Reach-Avoid with CBF")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.savefig("trajectory_plot.png")

    if not os.getenv("CBFKIT_TEST_MODE"):
        plt.show()

    plt.figure()
    plt.plot(
        jnp.linspace(0.0, dt * len(u_sim[:, 0]), len(u_sim[:, 0])), u_sim[:, 0], label="Accel"
    )
    plt.plot(
        jnp.linspace(0.0, dt * len(u_sim[:, 1]), len(u_sim[:, 1])), u_sim[:, 1], label="Omega"
    )
    plt.legend()
    plt.title("Control Inputs")
    plt.xlabel("Time (s)")
    plt.ylabel("Control")

    if not os.getenv("CBFKIT_TEST_MODE"):
        plt.show()
