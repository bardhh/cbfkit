"""User types module.

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

from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    TypeAlias,
    TypedDict,
    Union,
)

from jax import Array, random

# Define types for readability
Time: TypeAlias = Union[float, Array]
State: TypeAlias = Array
Control: TypeAlias = Array
Estimate: TypeAlias = Array
Covariance: TypeAlias = Array
Key: TypeAlias = Array
NumSteps: TypeAlias = int


# Data Schemas
class ControllerData(NamedTuple):
    """Data structure for controller output."""

    error: bool = False
    error_data: Optional[Union[int, Array]] = None
    complete: bool = False
    sol: Optional[Array] = None
    u: Optional[Array] = None
    u_nom: Optional[Array] = None
    sub_data: Optional[Dict[str, Any]] = None


class PlannerData(NamedTuple):
    """Data structure for planner output."""

    u_traj: Optional[Array] = None
    x_traj: Optional[Array] = None
    prev_robustness: Optional[Union[float, Array]] = None
    error: bool = False
    xs: Optional[Array] = None
    sampled_x_traj: Optional[Array] = None


class SimulationResults(NamedTuple):
    """Results from a simulation execution."""

    states: State
    controls: Control
    estimates: Estimate
    covariances: Covariance
    controller_keys: List[str]
    controller_values: List[Array]
    planner_keys: List[str]
    planner_values: List[Array]

    @property
    def controller_data(self) -> Dict[str, Array]:
        """Returns controller data as a dictionary."""
        return dict(zip(self.controller_keys, self.controller_values))

    @property
    def planner_data(self) -> Dict[str, Array]:
        """Returns planner data as a dictionary."""
        return dict(zip(self.planner_keys, self.planner_values))

    def __getitem__(self, key: Union[int, str]) -> Any:
        """Allows dictionary-like access to simulation results.

        Args:
            key (Union[int, str]): Integer index (tuple access) or string key.

        Returns:
            Any: The requested data.

        Raises:
            KeyError: If the key is not found in fields, controller_data, or planner_data.

        Examples:
            >>> results["states"]  # Access field
            >>> results["sol"]     # Access controller data
            >>> results[0]         # Access by index (legacy)
        """
        if isinstance(key, int):
            return tuple.__getitem__(self, key)
        if isinstance(key, str):
            if key in self._fields:
                return getattr(self, key)
            if key in self.controller_keys:
                idx = self.controller_keys.index(key)
                return self.controller_values[idx]
            if key in self.planner_keys:
                idx = self.planner_keys.index(key)
                return self.planner_values[idx]
            raise KeyError(
                f"Key '{key}' not found in SimulationResults fields, controller_data, or planner_data."
            )
        return tuple.__getitem__(self, key)


# Certificate (Barrier, Lyapunov, Barrier-Lyapunov, etc.) Function Callables
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
# Supports single collection, list of collections, tuple of collections, or legacy component tuple
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


# Dynamics Callables
DynamicsCallableReturns = Tuple[Array, Array]
DynamicsCallable = Callable[[State], DynamicsCallableReturns]

# Perturbation Callables
PerturbationCallableReturns = Callable[[Key], Array]
PerturbationCallable = Callable[[State, Control, Array, Array], PerturbationCallableReturns]

# Controller Callables

ControllerCallableReturns = Tuple[Array, ControllerData]

ControllerCallable = Callable[
    [Time, State, Control, Key, ControllerData], ControllerCallableReturns
]

NominalControllerCallable = Callable[[Time, State, Key, Optional[State]], ControllerCallableReturns]
"""Callable for nominal controllers.

Args:
    t: Current time.
    x: Current state.
    key: PRNG key for randomization.
    reference: Optional reference state (from planner).

Returns:
    A tuple containing the control input and controller data.

See Also:
    cbfkit.controllers.setup_nominal_controller: Helper to adapt simple functions.
"""


# Planner Callables
PlannerCallableReturns = Tuple[Array, PlannerData]
PlannerCallable = Callable[
    [Time, State, Optional[Control], Key, PlannerData], PlannerCallableReturns
]

# Estimator Callables
EstimatorCallable = Callable[
    [Time, Array, Array, Optional[Array], Optional[Array]],
    Tuple[State, Covariance],
]


# Sensor Callables
class SensorCallable(Protocol):
    """Protocol for sensor callables."""

    def __call__(
        self,
        t: Time,
        x: Array,
        *,
        sigma: Optional[Array] = None,
        key: Optional[Array] = None,
        **kwargs: Any,
    ) -> Array:
        """Call method."""
        ...


# Integrator Callable
VectorFieldCallable = Callable[[State], Array]
IntegratorCallable = Callable[[State, VectorFieldCallable, float], State]


# QP Solver Callables
QpSolverCallable = Callable[
    [Array, Array, Union[Array, None], Union[Array, None], Union[Array, None], Union[Array, None]],
    Tuple[Array, Dict[str, Any]],
]


# Solver Params Type Alias
SolverParams: TypeAlias = Tuple[Any, Any]
"""Solver parameters type (usually (KKTSolution, OSQPState) from jaxopt)."""


class CbfClfQpConfig(TypedDict, total=False):
    """Configuration for CBF-CLF-QP controllers.

    Attributes:
        relaxable_clf (bool): Whether to treat CLF as a soft constraint (default: True).
        relaxable_cbf (bool): Whether to treat CBF as a soft constraint (default: False).
        tunable_class_k (bool): Whether to tune the Class K function parameter (default: False).
        slack_bound_cbf (Optional[float]): Maximum slack for CBF constraints (default: 100.0 or 1e4).
        slack_bound_clf (float): Maximum slack for CLF constraints (default: 1e9).
        slack_penalty_cbf (float): Penalty weight for CBF slack variables (default: 2e3).
        slack_penalty_clf (float): Penalty weight for CLF slack variables (default: 2e3).
        scale_cbf (float): Scaling factor for CBF slack normalization (computed if not provided).
        scale_clf (float): Scaling factor for CLF slack normalization (computed if not provided).
        clf_complete_tol (float): Tolerance for considering CLF task complete (default: 1e-3).
    """

    relaxable_clf: bool
    relaxable_cbf: bool
    tunable_class_k: bool
    slack_bound_cbf: Optional[float]
    slack_bound_clf: float
    slack_penalty_cbf: float
    slack_penalty_clf: float
    scale_cbf: float
    scale_clf: float
    clf_complete_tol: float


class CbfClfQpData(TypedDict, total=False):
    """Data returned by CBF-CLF-QP controllers in sub_data.

    Attributes:
        solver_params (SolverParams): Tuple of (KKTSolution, OSQPState) from jaxopt.
        solver_iter (Union[int, Array]): Number of iterations taken by the solver.
        solver_status (Union[int, Array]): Exit status of the solver.
        complete (bool): Whether the CLF task is complete.
        bfs (Array): Values of barrier functions.
        lfs (Array): Values of lyapunov functions.
        violated (Union[bool, Array]): Whether any barrier function is violated.
    """

    solver_params: SolverParams
    solver_iter: Union[int, Array]
    solver_status: Union[int, Array]
    complete: bool
    bfs: Array
    lfs: Array
    lfs_nom: Array
    violated: Union[bool, Array]
    activation_weights: Array


# CBF-CLF-QP-Generators
class GenerateComputeCertificateConstraintCallable(Protocol):
    """Protocol for generating compute certificate constraint function."""

    def __call__(
        self,
        control_limits: Array,
        dyn_func: DynamicsCallable,
        barriers: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
        lyapunovs: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
        **kwargs: Any,
    ) -> Callable[[Time, State], Tuple[Array, Array, CbfClfQpData]]:
        """Call method."""
        ...


class CbfClfQpGenerator(Protocol):
    """Protocol for CBF-CLF-QP generator."""

    def __call__(
        self,
        control_limits: Array,
        dynamics_func: DynamicsCallable,
        barriers: Optional[CertificateInput] = EMPTY_CERTIFICATE_COLLECTION,
        lyapunovs: Optional[CertificateInput] = EMPTY_CERTIFICATE_COLLECTION,
        p_mat: Optional[Array] = None,
        *,
        relaxable_clf: bool = True,
        relaxable_cbf: bool = False,
        tunable_class_k: bool = False,
        slack_bound_cbf: Optional[float] = None,
        slack_bound_clf: float = 1e9,
        slack_penalty_cbf: float = 2e3,
        slack_penalty_clf: float = 2e3,
        **kwargs: Any,
    ) -> ControllerCallable:
        """Call method."""
        ...


# ComputeCertificateConstraintFunctionGenerator
class ComputeCertificateConstraintFunctionGenerator(Protocol):
    """Protocol for computing certificate constraint function generator."""

    def __call__(
        self,
        control_limits: Array,
        dyn_func: DynamicsCallable,
        barriers: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
        lyapunovs: CertificateCollection = EMPTY_CERTIFICATE_COLLECTION,
        **kwargs: Any,
    ) -> Callable[[Time, Array], Tuple[Array, Array, CbfClfQpData]]:
        """Call method."""
        ...


# Miscellaneous

# Cost function Callables
StlTrajectoryCostCallable = Callable[[float, Array], Union[float, Array]]
TrajectoryCostCallableReturns = Tuple[Array]
TrajectoryCostCallable = Callable[[State, Control], TrajectoryCostCallableReturns]
StageCostCallableReturns = Tuple[Array]
StageCostCallable = Callable[[State, Control], StageCostCallableReturns]
TerminalCostCallableReturns = Tuple[Array]
TerminalCostCallable = Callable[[State, Control], TerminalCostCallableReturns]

# CBF-CLF-QP-Generators
GenerateComputeStageCostCallable = Callable[
    [Array, DynamicsCallable, StageCostCallable, TerminalCostCallable, Dict[str, Any]],
    Callable[[Time, State], Tuple[Array, Array]],
]
GenerateComputeTerminalCostCallable = Callable[
    [Array, DynamicsCallable, StageCostCallable, TerminalCostCallable, Dict[str, Any]],
    Callable[[Time, State], Tuple[Array, Array]],
]


class MppiGenerator(Protocol):
    """Protocol for MPPI generator."""

    def __call__(
        self,
        control_limits: Array,
        dynamics_func: DynamicsCallable,
        stage_cost: Optional[StageCostCallable] = None,
        terminal_cost: Optional[TerminalCostCallable] = None,
        trajectory_cost: Optional[TrajectoryCostCallable] = None,
        mppi_args: Any = None,
        **kwargs: Any,
    ) -> PlannerCallable:
        """Call method."""
        ...
