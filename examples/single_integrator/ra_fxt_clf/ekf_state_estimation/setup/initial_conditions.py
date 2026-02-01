import jax.numpy as jnp

# plot animations if true
VISUALIZE = False

# Time settings
tf = 1.0
dt = 1e-3

# Define goal set
desired_state = jnp.array([0.0, 0.0])
goal_radius = 0.05

# Physical parameters
actuation_limits = 1e6 * jnp.array([1.0, 1.0])  # effectively, no actuation limits (except roll)

# FxTS parameters
Tg = tf
e1 = 0.5
e2 = 1.5
c1 = 4.0
c2 = 1 / ((e2 - 1) * (Tg - 1 / (c1 * (1 - e1))))
if c2 < 0:
    raise ValueError(f"Parameter c2 < 0: c2 = {c2:.2f}")

# Stochasticity Parameters
Q = 2.0 * jnp.eye(len(desired_state))  # process noise
R = 0.25 * jnp.eye(len(desired_state))  # measurement noise

# RA-CLF Parameters
pg = 0.95
gamma_v = 1.0 - 0.5 * goal_radius**2
eta_v = float(jnp.linalg.norm(jnp.dot(jnp.array([1.0, 1.0]), Q)))

# save file
n_trials = 100
pkl_file = f"examples/single_integrator/ra_fxt_clf/ekf_state_estimation/results/ra_pi_clf_n{n_trials}_pg{int(pg * 100)}.pkl"
