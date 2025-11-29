import jax.numpy as jnp
import numpy as np


class BaseConfig:
    x_max = 5.0
    y_max = 5.0
    # Note: These random values are generated once at import time
    x_rand = np.random.uniform(low=-x_max, high=x_max)
    y_rand = np.random.uniform(low=-y_max, high=y_max)
    a_rand = np.random.uniform(low=-jnp.pi, high=jnp.pi)
    initial_state = jnp.array([x_rand, y_rand, a_rand])
    desired_state = jnp.array([0.0, 0.0, 0])
    Q = 0.5 * jnp.eye(len(initial_state))


class PerfectMeasurementsConfig(BaseConfig):
    R = None


class EKFEstimationConfig(BaseConfig):
    R = 0.05 * jnp.eye(len(BaseConfig.initial_state))


class UKFEstimationConfig(BaseConfig):
    R = 0.05 * jnp.eye(len(BaseConfig.initial_state))


class HybridEstimationConfig(BaseConfig):
    R = 0.05 * jnp.eye(len(BaseConfig.initial_state))


# Aliases matching original module names for compatibility
perfect_state_measurements = PerfectMeasurementsConfig
ekf_state_estimation = EKFEstimationConfig
ukf_state_estimation = UKFEstimationConfig
ekf_ukf_hybrid_state_estimation = HybridEstimationConfig
