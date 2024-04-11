"""
risk_aware_path_integral_cbf_clf_qp_controller.py
================

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
>>> from cbfkit.controllers.utils import concatenate_certificates
>>> from cbfkit.controllers.utils.barrier_conditions.zeroing_barriers import linear_class_k
>>> from cbfkit.controllers.model_based.cbf_clf_controllers import risk_aware_path_integral_cbf_clf_qp_controller
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

from cbfkit.utils.user_types import ControllerCallable
from .cbf_clf_qp_generator import cbf_clf_qp_generator
from .generate_constraints import (
    generate_compute_ra_pi_cbf_constraints,
    generate_compute_ra_pi_clf_constraints,
)

risk_aware_path_integral_cbf_clf_qp_controller: ControllerCallable = cbf_clf_qp_generator(
    generate_compute_ra_pi_cbf_constraints,
    generate_compute_ra_pi_clf_constraints,
)
