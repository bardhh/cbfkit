"""
barrier_activated_cbf_clf_qp_control_laws.py
================

Defines the controller function for a Barrier-Activated CBF-CLF-QP control law.
This controller scales barrier constraints based on proximity to obstacles,
allowing for smoother navigation in cluttered environments by deactivating
distant constraints.

Functions
---------
-barrier_activated_cbf_clf_qp_controller: generates the function to compute the control solution.

Notes
-----
Relies on the cbf_clf_qp_generator function defined in cbf_clf_qp_generator.py.
"""

from cbfkit.utils.user_types import CbfClfQpGenerator

from .cbf_clf_qp_generator import cbf_clf_qp_generator
from .generate_constraints import (
    generate_compute_activated_cbf_constraints,
    generate_compute_vanilla_clf_constraints,
)

barrier_activated_cbf_clf_qp_controller: CbfClfQpGenerator = cbf_clf_qp_generator(
    generate_compute_activated_cbf_constraints,
    generate_compute_vanilla_clf_constraints,
)
