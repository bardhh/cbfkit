"""
risk_aware_cbf_clf_qp_controller.py
================

Defines the controller function for a risk-aware (RA) CBF-CLF-QP control law
for stochastic, continuous-time, control-affine, nonlinear dynamical systems.

Functions
---------
-risk_aware_cbf_clf_qp_controller: generates the function to compute the control
solution to the CBF-CLF-QP using RA CBFs and RA CLFs

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
>>> from cbfkit.controllers.cbf_clf import risk_aware_cbf_clf_qp_controller
>>> from cbfkit.controllers.cbf_clf.utils.risk_aware_params import RiskAwareParams
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
>>> ra_params = RiskAwareParams(p_bound=0.05, sigma=lambda x: jnp.zeros((3, 1)))
>>>
>>> controller = risk_aware_cbf_clf_qp_controller(
>>>     nominal_input=nominal_controller,
>>>     dynamics_func=dynamics,
>>>     barriers=concatenate_certificates(barrier),
>>>     control_limits=actuation_limits,
>>>     ra_params=ra_params,
>>> )
"""

from typing import Any, Optional

from jax import Array

from cbfkit.controllers.cbf_clf.utils.risk_aware_params import RiskAwareParams
from cbfkit.utils.user_types import (
    CbfClfQpGenerator,
    CertificateInput,
    ControllerCallable,
    DynamicsCallable,
    EMPTY_CERTIFICATE_COLLECTION,
)

from .cbf_clf_qp_generator import cbf_clf_qp_generator
from .generate_constraints import (
    generate_compute_estimate_feedback_ra_cbf_constraints,
    generate_compute_estimate_feedback_ra_clf_constraints,
    generate_compute_ra_cbf_constraints,
    generate_compute_ra_clf_constraints,
)


def _make_ra_controller(generator: CbfClfQpGenerator):
    """Create a risk-aware controller wrapper that decomposes ra_params for the generator."""

    def controller(
        control_limits: Array,
        dynamics_func: DynamicsCallable,
        barriers: Optional[CertificateInput] = EMPTY_CERTIFICATE_COLLECTION,
        lyapunovs: Optional[CertificateInput] = EMPTY_CERTIFICATE_COLLECTION,
        p_mat: Optional[Array] = None,
        ra_params: Optional[RiskAwareParams] = None,
        **kwargs: Any,
    ) -> ControllerCallable:
        if ra_params is not None:
            cbf_params, clf_params = ra_params.decompose()
            kwargs.setdefault("ra_cbf_params", cbf_params)
            kwargs.setdefault("ra_clf_params", clf_params)

        return generator(
            control_limits,
            dynamics_func,
            barriers,
            lyapunovs,
            p_mat,
            **kwargs,
        )

    return controller


risk_aware_cbf_clf_qp_controller = _make_ra_controller(
    cbf_clf_qp_generator(
        generate_compute_ra_cbf_constraints,
        generate_compute_ra_clf_constraints,
    )
)
risk_aware_cbf_clf_qp_controller.__doc__ = """Generates the control solution to the CBF-CLF-QP using RA CBFs/CLFs.

Automatically decomposes a unified ``ra_params`` into separate
``ra_cbf_params`` and ``ra_clf_params`` for the underlying generator.
"""

estimate_feedback_risk_aware_cbf_clf_qp_controller = _make_ra_controller(
    cbf_clf_qp_generator(
        generate_compute_estimate_feedback_ra_cbf_constraints,
        generate_compute_estimate_feedback_ra_clf_constraints,
    )
)
estimate_feedback_risk_aware_cbf_clf_qp_controller.__doc__ = """Generates the control solution to the CBF-CLF-QP using RA CBFs/CLFs with estimate feedback.

Automatically decomposes a unified ``ra_params`` into separate
``ra_cbf_params`` and ``ra_clf_params`` for the underlying generator.
"""
