"""
risk_aware_path_integral_cbf_clf_qp_controller.py
================

.. deprecated::
    **NON-FUNCTIONAL STUB — the path integral never accumulates.**

    The underlying constraint builder
    ``generate_constraints/risk_aware_path_integral_cbfs.py`` resets
    ``integrator_states`` to zero on every call (line 52: ``ra_params.integrator_states =
    jnp.zeros((n_bfs,))``).  The ``lax.cond`` on line 62 that was intended to *accumulate*
    the drift only runs once and then the attribute is overwritten, so ``I_L`` is always 0.
    This is an architectural limitation: the constraint builder receives only ``(t, x)``
    and cannot thread mutable carry through a JIT/scan loop.

    **Use** ``accumulating_risk_aware_cbf_controller`` from
    ``cbfkit.controllers.cbf_clf.accumulating_risk_aware_cbf`` instead.  It carries
    ``I_L`` correctly in ``ControllerData.sub_data["I_L"]`` and is JIT/scan-safe.

    This module is kept for backward compatibility only.  Its runtime behaviour is
    unchanged so that existing tests continue to pass.

Defines the controller function for a risk-aware path integral (RA-PI) CBF-CLF-QP control law
for stochastic, continuous-time, control-affine, nonlinear dynamical systems.

Functions
---------
-risk_aware_path_integral_cbf_clf_qp_controller: generates the function to compute the control
solution to the CBF-CLF-QP using RA-PI CBFs and RA-PI CLFs

Notes
-----
Relies on the cbf_clf_qp_generator function defined in cbf_clf_qp_generator.py
in the containing folder.

Examples
--------
>>> import jax.numpy as jnp
>>> import cbfkit.systems.unicycle.models.olfatisaber2002approximate as unicycle
>>> from cbfkit.certificates import concatenate_certificates
>>> from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
>>> from cbfkit.controllers.cbf_clf import risk_aware_path_integral_cbf_clf_qp_controller
>>>
>>> nominal_controller = unicycle.controllers.proportional_controller(k=1.0)
>>> actuation_limits = jnp.array([2.0, jnp.pi / 2])
>>> dynamics = unicycle.plant(l=1.0)
>>> obstacle = jnp.array([0.0, 0.0, 0.0])
>>> ellipsoid = jnp.array([1.0, 0.5])
>>> barrier = unicycle.certificate_function.barrier_functions.ellipsoidal_obstacle.obstacle_ca(
>>>     certificate_conditions=linear_class_k(1.0),
>>>     obstacle=obstacle,
>>>     ellipsoid=ellipsoid
>>> )
>>>
>>> controller = risk_aware_path_integral_cbf_clf_qp_controller(
>>>     nominal_input=nominal_controller,
>>>     dynamics_func=dynamics,
>>>     barriers=concatenate_certificates(barrier),
>>>     control_limits=actuation_limits,
>>> )
"""

from cbfkit.utils.user_types import CbfClfQpGenerator

from .cbf_clf_qp_generator import cbf_clf_qp_generator
from .generate_constraints import (
    generate_compute_ra_pi_cbf_constraints,
    generate_compute_ra_pi_clf_constraints,
)

risk_aware_path_integral_cbf_clf_qp_controller: CbfClfQpGenerator = cbf_clf_qp_generator(
    generate_compute_ra_pi_cbf_constraints,
    generate_compute_ra_pi_clf_constraints,
)
