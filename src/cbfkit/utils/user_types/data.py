"""Core data types and structures for CBFKit simulations."""

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, TypeAlias, Union

from jax import Array
import jax.numpy as jnp

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

    @classmethod
    def from_constant(cls, state: State) -> "PlannerData":
        """Creates a PlannerData object with a constant reference state.

        This helper creates a single-column trajectory, which the simulator
        will broadcast across all time steps, effectively treating it as a
        fixed setpoint/goal.

        Args:
            state (State): The constant reference state (e.g., goal).
                           Can be a 1D array or a list.

        Returns:
            PlannerData: A new instance with `x_traj` set to the provided state
                         (reshaped to a column vector).
        """
        state = jnp.atleast_1d(jnp.array(state))
        if state.ndim == 1:
            state = state.reshape(-1, 1)
        return cls(x_traj=state)


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

    @property
    def controller_data_keys(self) -> List[str]:
        """Legacy alias for ``controller_keys``."""
        return self.controller_keys

    @property
    def controller_data_values(self) -> List[Array]:
        """Legacy alias for ``controller_values``."""
        return self.controller_values

    @property
    def planner_data_keys(self) -> List[str]:
        """Legacy alias for ``planner_keys``."""
        return self.planner_keys

    @property
    def planner_data_values(self) -> List[Array]:
        """Legacy alias for ``planner_values``."""
        return self.planner_values

    def as_tuple(self) -> Tuple[
        State,
        Control,
        Estimate,
        Covariance,
        List[str],
        List[Array],
        List[str],
        List[Array],
    ]:
        """Returns the canonical legacy 8-tuple representation."""
        return tuple(self)  # type: ignore[return-value]

    def __getitem__(self, key: Union[int, str]) -> Any:
        """Allows dictionary-like access to simulation results.

        Args:
            key (Union[int, str]): Integer index (tuple access) or string key.

        Returns:
            Any: The requested data.

        Raises:
            KeyError: If the key is not found in fields, controller_data, or planner_data.
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
