"""
CBFKit: A Python/JAX library for Control Barrier Functions.

This module enables 64-bit float precision by default for numerical stability
in QP solvers (OSQP/JAXopt) and MPPI exponential cost weighting.
"""

from jax import config

from ._version import __version__

# Enable 64-bit float precision for numerical stability
# This MUST happen before any other JAX imports in user code
config.update("jax_enable_x64", True)
