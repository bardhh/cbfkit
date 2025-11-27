import jax.numpy as jnp

class BaseConfig:
    # Time settings
    tf = 1.0
    dt = 1e-3
    
    # Define goal set
    desired_state = jnp.array([0.0, 0.0])
    goal_radius = 0.05
    
    # Physical parameters
    actuation_limits = jnp.array([1e3, 1e3])
    
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
    
    # RA-CLF Parameters
    eta_v = float(jnp.linalg.norm(jnp.dot(jnp.array([1.0, 1.0]), Q)))

class PerfectMeasurementsConfig(BaseConfig):
    VISUALIZE = False
    R = 0.01 * jnp.eye(len(BaseConfig.desired_state))
    pg = 0.50
    gamma_v = 1.0 - 0.5 * BaseConfig.goal_radius**2
    n_trials = 10
    pkl_file = f"examples/single_integrator/ra_fxt_clf/results/perfect_measurements_n{n_trials}_pg{int(pg * 100)}.pkl"

class EKFEstimationConfig(BaseConfig):
    VISUALIZE = False
    actuation_limits = 1e6 * jnp.array([1.0, 1.0])
    R = 0.25 * jnp.eye(len(BaseConfig.desired_state))
    pg = 0.95
    gamma_v = 1.0 - 0.5 * BaseConfig.goal_radius**2
    n_trials = 10
    pkl_file = f"examples/single_integrator/ra_fxt_clf/results/ekf_estimation_n{n_trials}_pg{int(pg * 100)}.pkl"

class UKFEstimationConfig(BaseConfig):
    VISUALIZE = True
    R = 0.25 * jnp.eye(len(BaseConfig.desired_state))
    pg = 0.50
    gamma_v = 1.0 - 0.5 * BaseConfig.goal_radius**2
    n_trials = 3
    pkl_file = f"examples/single_integrator/ra_fxt_clf/results/ukf_estimation_n{n_trials}_pg{int(pg * 100)}.pkl"

# Aliases
perfect_state_measurements = PerfectMeasurementsConfig
ekf_state_estimation = EKFEstimationConfig
ukf_state_estimation = UKFEstimationConfig
