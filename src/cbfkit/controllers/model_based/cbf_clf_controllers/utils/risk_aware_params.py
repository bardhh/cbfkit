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

from typing import Union, Optional, Callable
from jax import Array


class Params:
    """
    Object to compile various parameters relevant to risk-aware control with CBFs/CLFs.

    Attributes:
        integrator_states (Union[Array, None]): vector of integrator states for RA-PI control
        t_max (Optional[Union[float, None]] = None): maximum system operation time (sec)
        p_bound (Optional[Union[float, None]] = None): maximum tolerable risk of certificate function violation
        gamma (Optional[Union[float, None]] = None): min/max value of certificate in set of initial conditions
        eta (Optional[Union[float, None]] = None): maximum dF/dx * sigma(x) term within constraint set / outside goal set
        epsilon (Optional[Union[float, None]] = None): probability of measurement bound holding
        lambda_h (Optional[Union[float, None]] = None): Lipschitz constant of h function in measurement model

    """

    integrator_states: Union[Array, None] = None

    def __init__(
        self,
        t_max: Optional[Union[float, None]] = None,
        p_bound: Optional[Union[float, None]] = None,
        gamma: Optional[Union[float, None]] = None,
        eta: Optional[Union[float, None]] = None,
        epsilon: Optional[Union[float, None]] = None,
        lambda_h: Optional[Union[float, None]] = None,
    ):
        """
        Constructor method for Params.

        Args:
            t_max (Optional[Union[float, None]] = None): maximum system operation time (sec)
            p_bound (Optional[Union[float, None]] = None): maximum tolerable risk of certificate function violation
            gamma (Optional[Union[float, None]] = None): min/max value of certificate in set of initial conditions
            eta (Optional[Union[float, None]] = None): maximum dF/dx * sigma(x) term within constraint set / outside goal set
            epsilon (Optional[Union[float, None]] = None): probability of measurement bound holding
            lambda_h (Optional[Union[float, None]] = None): Lipschitz constant of h function in measurement model
        """
        self.t_max = t_max
        self.p_bound = p_bound
        self.gamma = gamma
        self.eta = eta
        self.epsilon = epsilon
        self.lambda_h = lambda_h


class RiskAwareParams:
    """
    Object to compile various parameters relevant to risk-aware control with CBFs and CLFs.

    Attributes:
        ra_cbf (Params): object containing parameters for RA/RA-PI CBFs
        ra_clf (Params): object containing parameters for RA/RA-PI CLFs
        sigma (Optional[Union[Callable[[Array], Array], None]] = None): diffusion function in stochastic plant dynamics
        varsigma (Optional[Union[Callable[[Array], Array], None]] = None): diffusion function in stochastic measurement dynamics

    """

    def __init__(
        self,
        t_max: Optional[Union[float, None]] = None,
        p_bound_b: Optional[Union[float, None]] = None,
        gamma_b: Optional[Union[float, None]] = None,
        eta_b: Optional[Union[float, None]] = None,
        p_bound_v: Optional[Union[float, None]] = None,
        gamma_v: Optional[Union[float, None]] = None,
        eta_v: Optional[Union[float, None]] = None,
        sigma: Optional[Union[Callable[[Array], Array], None]] = None,
        varsigma: Optional[Union[Callable[[Array], Array], None]] = None,
    ):
        """Constructor method for RiskAwareParams.

        Args:
            t_max (Optional[Union[float, None]] = None): maximum system operation time (sec)
            p_bound_b (Optional[Union[float, None]] = None): maximum tolerable risk of CBF constraint violation
            gamma_b (Optional[Union[float, None]] = None): max value of CBF in set of initial conditions
            eta_b (Optional[Union[float, None]] = None): maximum dB/dx * sigma(x) term within constraint set
            p_bound_v (Optional[Union[float, None]] = None): minimum tolerable probability of reaching CLF goal set
            gamma_v (Optional[Union[float, None]] = None): max value of CLF in set of initial conditions
            eta_v (Optional[Union[float, None]] = None): maximum dV/dx * sigma(x) term outside goal set
            sigma (Optional[Union[Callable[[Array], Array], None]] = None): diffusion function in stochastic plant dynamics
            varsigma (Optional[Union[Callable[[Array], Array], None]] = None): diffusion function in stochastic measurement dynamics
        """
        self.ra_cbf = Params(t_max, p_bound_b, gamma_b, eta_b)
        self.ra_clf = Params(t_max, p_bound_v, gamma_v, eta_v)

        self.sigma = sigma
        self.varsigma = varsigma
