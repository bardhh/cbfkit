"""
stochastic_cbf_clf_qp_controller.py
================

Defines the controller function for a CBF-CLF-QP control law for deterministic,
continuous-time, control-affine, nonlinear dynamical systems subject to an additive,
bounded perturbation.

Functions
---------
-robust_cbf_clf_qp_controller: generates the function to compute the control solution
 to the CBF-CLF-QP using Robust CBFs and Robust CLFs

Notes
-----
Relies on the cbf_clf_qp_generator function defined in cbf_clf_qp_generator.py
in the containing folder.

Examples
--------
>>> import jax.numpy as jnp
>>> import cbfkit.systems.unicycle.models.olfatisaber2002approximate as unicycle
>>> from cbfkit.controllers_and_planners.utils import concatenate_certificates
>>> from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers.utils.barrier_conditions.zeroing_barriers import linear_class_k
>>> from cbfkit.controllers_and_planners.model_based.cbf_clf_controllers import robust_cbf_clf_qp_controller
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
>>> controller = robust_cbf_clf_qp_controller(
>>>     nominal_input=nominal_controller,
>>>     dynamics_func=dynamics,
>>>     barriers=concatenate_certificates(barrier),
>>>     control_limits=actuation_limits,
>>> )
"""

from cbfkit.utils.user_types import ControllerCallable
from .cbf_clf_qp_generator import cbf_clf_qp_generator
from .generate_constraints import (
    generate_compute_robust_cbf_constraints,
    generate_compute_robust_clf_constraints,
)

robust_cbf_clf_qp_controller: ControllerCallable = cbf_clf_qp_generator(
    generate_compute_robust_cbf_constraints,
    generate_compute_robust_clf_constraints,
)
