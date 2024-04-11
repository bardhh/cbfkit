"""
user_types
================

This module contains a collection of user-defined Python types, for
the purpose of type-hinting other modules in this repository.

Types
---------
Time
State
Control
Estimate
Covariance
CertificateCallable
CertificateJacobianCallable
CertificateHessianCallable
CertificatePartialCallable
CertificateConditionsCallable
CertificateCollection
CertificateTuple
DynamicsCallableReturns
DynamicsCallable
ControllerCallableReturns
ControllerCallable
EstimatorCallable
SensorCallable
IntegratorCallable
QpSolverCallable
ComputeCertificateConstraintFunctionGenerator

Notes
-----
This file was initially written for a Python 3.11 distribution, for which the
native typing module contains a TypeAlias object. We had to revert to Python 3.8
for ROS-noetic, and TypeAlias does not exist in this distribution. Therefore, the
convention MyType = Type was originally written as MyType: TypeAlias = Type, 
but this has been removed by necessity.

Examples
--------
>>> from cbfkit.utils.user_types import *
>>> import jax.numpy as jnp
>>> x: State = jnp.array([1, 2, 2.4])

"""

from typing import Callable, Tuple, Dict, Any, List, Union, Optional
from jax import Array, random

# Define types for readability
State = Array
Control = Array
Estimate = Array
Covariance = Array

# Certificate (Barrier, Lyapunov, Barrier-Lyapunov, etc.) Function Callables
CertificateCallable = Callable[[float, State], Array]
CertificateJacobianCallable = Callable[[float, State], Array]
CertificateHessianCallable = Callable[[float, State], Array]
CertificatePartialCallable = Callable[[float, State], Array]
CertificateConditionsCallable = Callable[[Array], Array]
CertificateCollection = Tuple[
    List[CertificateCallable],
    List[CertificateJacobianCallable],
    List[CertificateHessianCallable],
    List[CertificatePartialCallable],
    List[CertificateConditionsCallable],
]
CertificateTuple = Tuple[
    CertificateCallable,
    CertificateJacobianCallable,
    CertificateHessianCallable,
    CertificatePartialCallable,
    CertificateConditionsCallable,
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


# Dynamics Callables
DynamicsCallableReturns = Tuple[Array, Array]
DynamicsCallable = Callable[[State], DynamicsCallableReturns]

# Perturbation Callables
PerturbationCallableReturns = Callable[[random.PRNGKey], Array]
PerturbationCallable = Callable[[State, Control, Array, Array], PerturbationCallableReturns]

# Controller Callables
ControllerCallableReturns = Tuple[Array, Dict[str, Any]]
ControllerCallable = Callable[[float, State], ControllerCallableReturns]

# Estimator Callables
EstimatorCallable = Callable[
    [float, Array, Array, Optional[Union[Array, None]], Optional[Union[Array, None]]],
    Tuple[Array, Array],
]

# Sensor Callables
SensorCallable = Callable[[float, Array], Array]

# Integrator Callable
IntegratorCallable = Callable[[State, Array, float], State]

# QP Solver Callables
QpSolverCallable = Callable[
    [Array, Array, Union[Array, None], Union[Array, None], Union[Array, None], Union[Array, None]],
    Tuple[Array, Dict[str, Any]],
]

# CBF-CLF-QP-Generators
GenerateComputeCertificateConstraintCallable = Callable[
    [Array, DynamicsCallable, CertificateCollection, CertificateCollection, Dict[str, Any]],
    Callable[[float, State], Tuple[Array, Array]],
]
CbfClfQpGenerator = Callable[
    [
        Array,
        ControllerCallable,
        DynamicsCallable,
        CertificateCollection,
        CertificateCollection,
        Union[Array, None],
        Dict[str, Any],
    ],
    ControllerCallable,
]

# ComputeCertificateConstraintFunctionGenerator
ComputeCertificateConstraintFunctionGenerator = Callable[
    [DynamicsCallable, CertificateCollection, Array, Dict[str, Any]],
    Callable[[float, Array], Tuple[Array, Array, Dict[str, Any]]],
]

# Miscellaneous
Time = float
NumSteps = int
