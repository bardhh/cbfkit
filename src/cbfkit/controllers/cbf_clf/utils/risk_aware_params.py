"""
risk_aware_params.py
================

Provides classes to compile parameters for Risk-Aware (RA-) or Risk-Aware Path Integral (RA-PI-) CBFs and CLFs.

Classes
---------
-Params: compiles parameters for RA/RA-PI CBF or CLF
-RiskAwareParams: compiles parameters for both RA/RA-PI CBF and CLF

Notes
-----
The RiskAwareParams class is necessary for instantiating risk_aware- and risk_aware_path_integral_cbf_clf_qp controllers.

Examples
--------
>>> import jax.numpy as jnp
>>> ra_params = RiskAwareParams(
>>>     t_max=10.0,
>>>     p_bound_b=0.05,
>>>     gamma_b=0.5,
>>>     eta_b=2.0,
>>>     p_bound_v=0.9,
>>>     gamma_v=10.0,
>>>     eta_v=2.0,
>>>     sigma=lambda x: 0.25 * jnp.eye(4),
>>>     varsigma=lambda x: 0.1 * jnp.eye(4),
>>> )

"""

from typing import Callable, Optional, Union, Tuple

from jax import Array


class RiskAwareParams:
    """Object to compile various parameters relevant to risk-aware control with CBFs and CLFs.

    Attributes
    ----------
        ra_cbf (Params): object containing parameters for RA/RA-PI CBFs
        ra_clf (Params): object containing parameters for RA/RA-PI CLFs
        sigma (Optional[Union[Callable[[Array], Array], None]] = None): diffusion function in stochastic plant dynamics
        varsigma (Optional[Union[Callable[[Array], Array], None]] = None): diffusion function in stochastic measurement dynamics
    """

    def __init__(
        self,
        t_max: Optional[Union[float, None]] = None,
        p_bound: Optional[Union[float, None]] = None,
        gamma: Optional[Union[float, Array, None]] = None,
        eta: Optional[Union[float, None]] = None,
        epsilon: Optional[Union[float, None]] = None,
        lambda_h: Optional[Union[float, None]] = None,
        lambda_generator: Optional[Union[float, None]] = None,
        sigma: Optional[Union[Callable[[Array], Array], None]] = None,
        varsigma: Optional[Union[Callable[[Array], Array], None]] = None,
        integrator_states: Optional[Union[Array, None]] = None,
        # Atlas: Added explicit CBF/CLF params to match docstring and enable unification
        p_bound_b: Optional[Union[float, None]] = None,
        gamma_b: Optional[Union[float, Array, None]] = None,
        eta_b: Optional[Union[float, None]] = None,
        p_bound_v: Optional[Union[float, None]] = None,
        gamma_v: Optional[Union[float, Array, None]] = None,
        eta_v: Optional[Union[float, None]] = None,
    ):
        """Constructor method for RiskAwareParams.

        Args:
            t_max (Optional[Union[float, None]] = None): maximum system operation time (sec)
            p_bound (Optional[Union[float, None]] = None): maximum tolerable risk of CBF constraint violation
            gamma (Optional[Union[float, Array, None]] = None): max value of CBF in set of initial conditions
            eta (Optional[Union[float, None]] = None): maximum dB/dx * sigma(x) term within constraint set
            epsilon (Optional[Union[float, None]] = None): maximum dB/dx * sigma(x) term within constraint set
            lambda_h (Optional[Union[float, None]] = None): maximum dB/dx * sigma(x) term within constraint set
            lambda_generator (Optional[Union[float, None]] = None): maximum dB/dx * sigma(x) term within constraint set
            sigma (Optional[Union[Callable[[Array], Array], None]] = None): diffusion function in stochastic plant dynamics
            varsigma (Optional[Union[Callable[[Array], Array], None]] = None): diffusion function in stochastic measurement dynamics
            integrator_states (Optional[Union[Array, None]] = None): states for path integral control
            p_bound_b (Optional[Union[float, None]] = None): CBF-specific p_bound
            gamma_b (Optional[Union[float, Array, None]] = None): CBF-specific gamma
            eta_b (Optional[Union[float, None]] = None): CBF-specific eta
            p_bound_v (Optional[Union[float, None]] = None): CLF-specific p_bound
            gamma_v (Optional[Union[float, Array, None]] = None): CLF-specific gamma
            eta_v (Optional[Union[float, None]] = None): CLF-specific eta
        """
        self.t_max = t_max
        self.p_bound = p_bound
        self.gamma = gamma
        self.eta = eta
        self.epsilon = epsilon
        self.lambda_h = lambda_h
        self.lambda_generator = lambda_generator
        self.sigma = sigma
        self.varsigma = varsigma
        self.integrator_states = integrator_states

        # Store specific params
        self.p_bound_b = p_bound_b
        self.gamma_b = gamma_b
        self.eta_b = eta_b
        self.p_bound_v = p_bound_v
        self.gamma_v = gamma_v
        self.eta_v = eta_v

    def decompose(self) -> Tuple["RiskAwareParams", "RiskAwareParams"]:
        """Decomposes the params into separate CBF and CLF parameter objects.

        Returns:
            Tuple[RiskAwareParams, RiskAwareParams]: (cbf_params, clf_params)
        """
        cbf_params = RiskAwareParams(
            t_max=self.t_max,
            p_bound=self.p_bound_b if self.p_bound_b is not None else self.p_bound,
            gamma=self.gamma_b if self.gamma_b is not None else self.gamma,
            eta=self.eta_b if self.eta_b is not None else self.eta,
            epsilon=self.epsilon,
            lambda_h=self.lambda_h,
            lambda_generator=self.lambda_generator,
            sigma=self.sigma,
            varsigma=self.varsigma,
            integrator_states=self.integrator_states,
        )

        clf_params = RiskAwareParams(
            t_max=self.t_max,
            p_bound=self.p_bound_v if self.p_bound_v is not None else self.p_bound,
            gamma=self.gamma_v if self.gamma_v is not None else self.gamma,
            eta=self.eta_v if self.eta_v is not None else self.eta,
            epsilon=self.epsilon,
            lambda_h=self.lambda_h,
            lambda_generator=self.lambda_generator,
            sigma=self.sigma,
            varsigma=self.varsigma,
            integrator_states=self.integrator_states,
        )

        return cbf_params, clf_params
