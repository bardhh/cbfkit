"""Neural Control Barrier Functions for CBFKit.

Learn barrier functions from data using JAX neural networks, producing
``CertificateCollection`` objects that integrate directly with the
CBF-CLF-QP controller pipeline.

Requires the ``neural`` extra: ``pip install cbfkit[neural]``
"""

from .model import CBFNetwork, create_neural_cbf, make_cbf_callable
from .training import cbf_loss, train_neural_cbf

__all__ = [
    "CBFNetwork",
    "cbf_loss",
    "create_neural_cbf",
    "make_cbf_callable",
    "train_neural_cbf",
]
