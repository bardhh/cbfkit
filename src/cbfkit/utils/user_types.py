from typing import TypeAlias, Callable, Tuple, Dict, Any, List, Union
from jax import Array

# Define types for readability
State: TypeAlias = Array

# Barrier Function Callables
BarrierCallable: TypeAlias = Callable[[float, State], Array]
BarrierJacobianCallable: TypeAlias = Callable[[float, State], Array]
BarrierHessianCallable: TypeAlias = Callable[[float, State], Array]
BarrierPartialCallable: TypeAlias = Callable[[float, State], Array]
BarrierCollectionCallable: TypeAlias = Callable[
    [],
    Tuple[
        List[BarrierCallable],
        List[BarrierJacobianCallable],
        List[BarrierHessianCallable],
        List[BarrierPartialCallable],
    ],
]
BarrierTuple: TypeAlias = Tuple[
    BarrierCallable, BarrierJacobianCallable, BarrierHessianCallable, BarrierPartialCallable
]


# Dynamics Callables
DynamicsCallableReturns: TypeAlias = Tuple[Array, Array, Array]
DynamicsCallable: TypeAlias = Callable[[State], DynamicsCallableReturns]

# Controller Callables
ControllerCallableReturns: TypeAlias = Tuple[Array, Dict[str, Any]]
ControllerCallable: TypeAlias = Callable[[float, State], ControllerCallableReturns]

# Lyapunov Function Callables
LyapunovCallable: TypeAlias = Callable[[float, State], Array]
LyapunovJacobianCallable: TypeAlias = Callable[[float, State], Array]
LyapunovHessianCallable: TypeAlias = Callable[[float, State], Array]
LyapunovPartialCallable: TypeAlias = Callable[[float, State], Array]
LyapunovConditionCallable: TypeAlias = Callable[[Array], Array]
LyapunovCollectionCallable: TypeAlias = Callable[
    [],
    Tuple[
        List[LyapunovCallable],
        List[LyapunovJacobianCallable],
        List[LyapunovHessianCallable],
        List[LyapunovPartialCallable],
        List[LyapunovConditionCallable],
    ],
]
LyapunovTuple: TypeAlias = Tuple[
    LyapunovCallable,
    LyapunovJacobianCallable,
    LyapunovHessianCallable,
    LyapunovPartialCallable,
    LyapunovConditionCallable,
]

# Miscellaneous
Time: TypeAlias = float
NumSteps: TypeAlias = int
