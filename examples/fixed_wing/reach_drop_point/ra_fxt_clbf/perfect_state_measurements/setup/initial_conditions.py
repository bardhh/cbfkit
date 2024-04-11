import jax.numpy as jnp

# Time settings
tf = 5.0
dt = 1e-2

# px, py, pz, v, psi, gamma
initial_state = jnp.array([500.0, 250.0, 250.0, 100.0, -jnp.pi / 2, 0.0])
desired_state = jnp.array([0.0, 0.0, 200.0])
desired_velpos = jnp.array([-100.0, 0.0, 200.0])
goal_radius = 1.0

# Fixed-Wing UAV physical parameters
actuation_limits = jnp.array(
    [1e9, 1e9, jnp.tan(jnp.pi / 2.5)]
)  # effectively, no actuation limits (except roll)

# Obstacles
obstacle_locations = [
    jnp.array([0.0, 100.0, desired_state[2]]),
    jnp.array([0.0, -100.0, desired_state[2]]),
    # jnp.array([1000.0, 250.0, desired_state[2]]),
]
ellipsoid_radii = [
    [200, 50, 1000],
    [1000, 50, 1000],
    # [50, 100, 1000],
]


# FxTS parameters
Tg = 5.0
lookahead_time = 2.0
alpha = 100.0
e1 = 0.9
e2 = 1.1
c1 = 4.0
c2 = 1 / ((e2 - 1) * (Tg - 1 / (c1 * (1 - e1))))

# Control Parameters
QP_J = jnp.diag(jnp.array([100.0, 1.0, 1.0]))

# Stochasticity Parameters
Q = 0.5 * jnp.eye(len(initial_state))  # process noise
R = 0.05 * jnp.eye(len(initial_state))  # measurement noise
