"""Configuration classes for Van der Pol oscillator examples."""
import jax.numpy as jnp
import numpy as np


class BaseConfig:
    # Time settings
    tf = 1.0
    dt = 1e-3

    # Define goal set
    desired_state = jnp.array([0.0, 0.0])
    goal_radius = 0.05
    initial_state = jnp.array([2.0, 2.0])

    # Physical parameters
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

    # RA-CLF Parameters
    pg = 0.75
    gamma_v = 11.0
    eta_v = float(jnp.linalg.norm(jnp.dot(jnp.array([1.0, 1.0]), Q)))
    lambda_h = 1.0
    lambda_generator = None


class PerfectMeasurementsConfig(BaseConfig):
    R = 0.05 * jnp.eye(len(BaseConfig.desired_state))  # measurement noise


# For UKF, the existing code was using Unicycle parameters.
# To break dependency but preserve behavior (even if potentially wrong for VDP),
# we replicate the Unicycle config values here but adapted for VDP structure if needed.
# However, since the UKF example imports Unicycle dynamics, it expects Unicycle config structure (3 states).
# This is a mess. For now, I will define a UKFConfig that mimics the Unicycle one used previously.
class UKFEstimationConfig:
    x_max = 5.0
    y_max = 5.0
    # Note: These random values are generated once at import time
    rng = np.random.default_rng(0)
    x_rand = rng.uniform(low=-x_max, high=x_max)
    y_rand = rng.uniform(low=-y_max, high=y_max)
    a_rand = rng.uniform(low=-jnp.pi, high=jnp.pi)
    initial_state = jnp.array([x_rand, y_rand, a_rand])
    desired_state = jnp.array([0.0, 0.0, 0])
    Q = 0.5 * jnp.eye(len(initial_state))
    R = 0.05 * jnp.eye(len(initial_state))


# Aliases
perfect_state_measurements = PerfectMeasurementsConfig
ukf_state_estimation = UKFEstimationConfig
