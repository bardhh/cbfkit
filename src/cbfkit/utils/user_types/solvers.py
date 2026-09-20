"""Solver contracts shared by controller types and backend implementations."""

from typing import Any, Optional, Protocol, Union, cast

from jax import Array


class QpSolution:
    """Return type for all QP solvers.

    Attributes:
        primal: Solution vector.
        status: Integer status code (1 = solved).
        params: Solver-specific state for warm-starting.  ``None`` for
            backends that do not support warm-starting.
    """

    __slots__ = ("primal", "status", "params")

    def __init__(self, primal: Array, status: Union[int, Array], params: Any = None):
        self.primal = primal
        self.status = status
        self.params = params

    # Support tuple unpacking: primal, status, params = solution
    def __iter__(self):
        return iter((self.primal, self.status, self.params))

    def __getitem__(self, idx):
        return (self.primal, self.status, self.params)[idx]

    def __repr__(self):
        return f"QpSolution(primal={self.primal}, status={self.status})"


class QpSolverFunction(Protocol):
    """Common solver call signature; warm-start state belongs to the backend."""

    def __call__(
        self,
        h_mat: Array,
        f_vec: Array,
        g_mat: Optional[Array] = None,
        h_vec: Optional[Array] = None,
        a_mat: Optional[Array] = None,
        b_vec: Optional[Array] = None,
        init_params: Any = None,
    ) -> QpSolution: ...


class QpSolverCallable(QpSolverFunction, Protocol):
    """Solver callable with explicit execution capabilities."""

    jit_compatible: bool
    solver_name: str


def with_solver_metadata(
    solver: QpSolverFunction, *, name: str, jit_compatible: bool
) -> QpSolverCallable:
    """Attach metadata without wrapping the function or changing JAX tracing."""
    decorated = cast(QpSolverCallable, solver)
    decorated.jit_compatible = jit_compatible
    decorated.solver_name = name
    return decorated
