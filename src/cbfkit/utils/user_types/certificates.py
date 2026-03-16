"""Certificate (Barrier, Lyapunov) function types for CBFKit."""

from enum import Enum
from typing import Callable, List, NamedTuple, Tuple, Union

from jax import Array

from .data import State, Time


# Certificate Function Callables
class CertificateInputStyle(str, Enum):
    """Enumeration for certificate function input styles."""

    CONCATENATED = "concatenated"
    SEPARATED = "separated"
    STATE = "state"


CertificateCallable = Callable[[Time, State], Array]
CertificateJacobianCallable = Callable[[Time, State], Array]
CertificateHessianCallable = Callable[[Time, State], Array]
CertificatePartialCallable = Callable[[Time, State], Array]
CertificateConditionsCallable = Callable[[Array], Array]


class CertificateCollection(NamedTuple):
    """Collection of certificate functions and their derivatives."""

    functions: List[CertificateCallable]
    jacobians: List[CertificateJacobianCallable]
    hessians: List[CertificateHessianCallable]
    partials: List[CertificatePartialCallable]
    conditions: List[CertificateConditionsCallable]

    def __add__(self, other: "CertificateCollection") -> "CertificateCollection":
        if not isinstance(other, CertificateCollection):
            return NotImplemented
        return CertificateCollection(
            self.functions + other.functions,
            self.jacobians + other.jacobians,
            self.hessians + other.hessians,
            self.partials + other.partials,
            self.conditions + other.conditions,
        )


# Default empty collection
EMPTY_CERTIFICATE_COLLECTION = CertificateCollection([], [], [], [], [])


CertificateTuple = Tuple[
    CertificateCallable,
    CertificateJacobianCallable,
    CertificateHessianCallable,
    CertificatePartialCallable,
    CertificateConditionsCallable,
]
BarrierTuple = CertificateTuple
LyapunovTuple = CertificateTuple


# Legacy Certificate Tuple (matching CertificateCollection components)
CertificateLegacyTuple = Tuple[
    List[CertificateCallable],
    List[CertificateJacobianCallable],
    List[CertificateHessianCallable],
    List[CertificatePartialCallable],
    List[CertificateConditionsCallable],
]

# Valid input types for certificate collection arguments (barriers, lyapunovs)
CertificateInput = Union[
    CertificateCollection,
    List[CertificateCollection],
    Tuple[CertificateCollection, ...],
    CertificateLegacyTuple,
]


# Predictive Barrier Function Callable
PredictiveBarrierCollectionCallable = Callable[
    [],
    Tuple[
        List[CertificateCallable],
        List[CertificateJacobianCallable],
        List[CertificateHessianCallable],
        List[CertificatePartialCallable],
    ],
]
