"""Callable type definitions and configuration types for CBFKit."""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeAlias,
    TypedDict,
    Union,
    TYPE_CHECKING,
)

from jax import Array

from .certificates import (
    CertificateCollection,
    CertificateInput,
    EMPTY_CERTIFICATE_COLLECTION,
)
from .data import Control, Covariance, Key, State, Time, ControllerData, PlannerData


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
#
# The unified solver signature accepts an optional ``init_params`` for
# warm-starting and returns a ``QpSolution`` (which supports tuple
# unpacking as ``(primal, status, params)``).  Backends that do not
# support warm-starting accept and ignore the argument.
QpSolverCallable = Callable[
    ...,
    Any,  # QpSolution — typed as Any to avoid circular import
]


if TYPE_CHECKING:
    from jaxopt.base import KKTSolution
    from jaxopt._src.osqp import OSQPState
else:
    KKTSolution = Any
    OSQPState = Any


# Solver Params Type Alias
SolverParams: TypeAlias = Tuple[KKTSolution, OSQPState]
"""Solver parameters type (specifically (KKTSolution, OSQPState) for QP solvers)."""


class CbfClfQpConfig(TypedDict, total=False):
    """Configuration for CBF-CLF-QP controllers."""

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
    """Data returned by CBF-CLF-QP controllers in sub_data."""

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

GenerateComputeStageCostCallable = Callable[
    [Array, DynamicsCallable, StageCostCallable, TerminalCostCallable, Dict[str, Any]],
    Callable[[Time, State], Tuple[Array, Array]],
]
GenerateComputeTerminalCostCallable = Callable[
    [Array, DynamicsCallable, StageCostCallable, TerminalCostCallable, Dict[str, Any]],
    Callable[[Time, State], Tuple[Array, Array]],
]


class MppiParameters(TypedDict, total=False):
    """Parameters for MPPI controller configuration."""

    robot_state_dim: int
    robot_control_dim: int
    prediction_horizon: int
    num_samples: int
    time_step: float
    use_GPU: bool
    costs_lambda: float
    cost_perturbation: float
    plot_samples: int


class MppiGenerator(Protocol):
    """Protocol for MPPI generator."""

    def __call__(
        self,
        control_limits: Array,
        dynamics_func: DynamicsCallable,
        stage_cost: Optional[StageCostCallable] = None,
        terminal_cost: Optional[TerminalCostCallable] = None,
        trajectory_cost: Optional[TrajectoryCostCallable] = None,
        mppi_args: Optional[MppiParameters] = None,
        **kwargs: Any,
    ) -> PlannerCallable:
        """Call method."""
        ...
