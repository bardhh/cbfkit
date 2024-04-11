import jax.numpy as jnp

# Time settings
tf = 10.0
dt = 1e-2

# pn, pe, h, u, v, w, phi, theta, psi, p, q, r
initial_state = jnp.array([10.0, 5.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.1, -0.1, -0.01, 0.01, 0.0])
desired_state = jnp.array([0.0, 0.0, 1.0])

# Quadrotor physical parameters
m = 4.34
jx = 0.0820
jy = 0.0845
jz = 0.1377
# m = 1.5  # kg
# jx = jy = 0.05  # kg m^2
# jz = 0.057  # kg m^2
actuation_limits = jnp.array([2.0 * 9.81 * m, 1.0 / jx, 1.0 / jy, 1.0 / jz])

# FxTS parameters
Tg = tf
e1 = 0.5
e2 = 1.5
c1 = 0.5
c2 = 1 / ((e2 - 1) * (Tg - 1 / (c1 * (1 - e1))))

# Stochasticity Parameters
Q = 0.5 * jnp.eye(len(initial_state))  # process noise
R = 0.05 * jnp.eye(len(initial_state))  # measurement noise
