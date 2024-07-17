"""
lyapunov_conditions.py
================

This module contains functions to compute Lyapunov conditions for various degrees of stability.

Functions
---------
-a_s(): asymptotically stable
-e_s(c): exponentially stable
-ft_s(c, e): finite-time stable
-fxt_s(c1, c2, e1, e2): fixed-time stable

Notes
-----
The functions in this module are used for continuous-time, deterministic, nonlinear systems.

Examples
--------
>>> from jax import jacfwd, jacrev
>>> from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.certificate_packager import certificate_package, concatenate_certificates
>>> from cbfkit.controllers_and_planners.utils.lyapunov_conditions import e_s, fxt_s
>>> 
>>> def clf(goal):
>>>     def func(x):
>>>         return x[0] - goal
>>>     return func
>>>
>>> def clf_grad(goal):
>>>     jacobian = jacfwd(clf(goal))
>>>     def func(x):
>>>         return jacobian(x)
>>>     return func
>>>
>>> def clf_hess(goal):
>>>     hessian = jacrev(jacfwd(clf(goal)))
>>>     def func(x):
>>>         return hessian(x)
>>>     return func
>>> 
>>> package1 = certificate_package(clf, clf_grad, clf_hess, n=1)
>>>
>>> goal = 1.0
>>> c1, c2, e1, e2 = 1.0, 1.0, 0.75, 1.25
>>> as_lyapunov = concatenate_certificates(
>>>     package1(certificate_conditions=e_s(c1), goal=goal), 
>>> )
>>> fxts_lyapunov = concatenate_certificates(
>>>     package1(certificate_conditions=fxt_s(c1, c2, e1, e2), goal=goal), 
>>> )

"""

from typing import Callable
from jax import lax, Array


def e_s(c: float) -> Callable[[Array], Array]:
    """Generates function for computing RHS of Lyapunov conditions for Exponential stability:

    Vdot <= -c1*V

    Args:
        c1 (float): convergence constant

    Returns:
        Callable[[Array], Array]: ES Lyapunov conditions
    """
    return lambda V: lax.cond(V > 0, lambda _fake: -c * V, lambda _fake: 0.0, 0)
