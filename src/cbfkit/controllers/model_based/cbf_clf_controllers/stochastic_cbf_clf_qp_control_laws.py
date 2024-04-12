"""
stochastic_cbf_clf_qp_controller.py
================

Defines the controller function for a CBF-CLF-QP control law for stochastic,
continuous-time, control-affine, nonlinear dynamical systems.

Functions
---------
-stochastic_cbf_clf_qp_controller: generates the function to compute the control solution
 to the CBF-CLF-QP using Stochastic CBFs and Stochastic CLFs

Notes
-----
Relies on the cbf_clf_qp_generator function defined in cbf_clf_qp_generator.py
in the containing folder.

Examples
--------
>>> import jax.numpy as jnp
>>> import cbfkit.systems.unicycle.models.olfatisaber2002approximate as unicycle
>>> from cbfkit.controllers.utils import concatenate_certificates
>>> from cbfkit.controllers.model_based.cbf_clf_controllers.utils.barrier_conditions.stochastic_barriers import linear_class_k
>>> from cbfkit.controllers.model_based.cbf_clf_controllers import robust_cbf_clf_qp_controller
>>> 
>>> nominal_controller = unicycle.controllers.proportional_controller(k=1.0)
>>> actuation_limits = jnp.array([2.0, jnp.pi / 2])
>>> dynamics = unicycle.plant(l=1.0)
>>> obstacle = jnp.array([0.0, 0.0, 0.0])
>>> ellipsoid = jnp.array([1.0, 0.5])
>>> barrier = unicycle.certificate_function.barrier_functions.ellipsoidal_obstacle.obstacle_ca(
>>>     obstacle=obstacle, ellipsoid=ellipsoid
>>> )
>>> 
>>> controller = stochastic_cbf_clf_qp_controller(
>>>     nominal_input=nominal_controller,
>>>     dynamics_func=dynamics,
>>>     barriers=concatenate_certificates(barrier),
>>>     control_limits=actuation_limits,
>>> )
"""

from cbfkit.utils.user_types import CbfClfQpGenerator
from .cbf_clf_qp_generator import cbf_clf_qp_generator
from .generate_constraints import (
    generate_compute_stochastic_cbf_constraints,
    generate_compute_stochastic_clf_constraints,
)

stochastic_cbf_clf_qp_controller: CbfClfQpGenerator = cbf_clf_qp_generator(
    generate_compute_stochastic_cbf_constraints,
    generate_compute_stochastic_clf_constraints,
)
