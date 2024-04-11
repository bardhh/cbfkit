import jax.numpy as jnp

# Time settings
tf = 1.0
dt = 1e-3

# x1, x2
initial_state = jnp.array([1.0, 1.0])
desired_state = jnp.array([0.0, 0.0])
goal_radius = 0.05

# Van der Pol oscillator physical parameters
actuation_limits = jnp.array([1e3])  # effectively, no actuation limits (except roll)
epsilon = 0.2

# FxTS parameters
Tg = tf
e1 = 0.5
e2 = 1.5
c1 = 4.0
c2 = 1 / ((e2 - 1) * (Tg - 1 / (c1 * (1 - e1))))
if c2 < 0:
    raise ValueError(f"Parameter c2 < 0: c2 = {c2:.2f}")

# Stochasticity Parameters
Q = 0.05 * jnp.array([[0.0, 0.0], [0.0, 1.0]])  # process noise
R = 0.05 * jnp.eye(len(initial_state))  # measurement noise

# RA-CLF Parameters
pg = 0.75
gamma_v = 11.0
eta_v = float(jnp.linalg.norm(jnp.dot(jnp.array([1.0, 1.0]), Q)))
