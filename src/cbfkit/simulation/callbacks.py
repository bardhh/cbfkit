from typing import Any, Dict, List, Optional, Protocol, Union

from rich.progress import Progress, TaskID

from cbfkit.simulation.utils import SimulationStepData
from cbfkit.utils.logger import write_log

from .ui import create_progress, print_simulation_end


class SimulationCallback(Protocol):
    def on_start(self, total_steps: int, dt: float) -> None: ...

    def on_step(self, step_idx: int, time: float, data: SimulationStepData) -> None: ...

    def on_end(self, success: bool, message: str = "") -> None: ...


class ProgressCallback:
    def __init__(self) -> None:
        self._progress: Optional[Progress] = None
        self._task_id: Optional[TaskID] = None

    def on_start(self, total_steps: int, dt: float) -> None:
        self._progress = create_progress(total=total_steps)
        self._progress.start()
        self._task_id = self._progress.add_task("Simulating", total=total_steps)

    def on_step(self, step_idx: int, time: float, data: SimulationStepData) -> None:
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, advance=1)

    def on_end(self, success: bool, message: str = "") -> None:
        if self._progress is not None:
            self._progress.stop()
        print_simulation_end(success, message)


class LoggingCallback:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.log_data: Union[List[Dict[str, Any]], Dict[str, Any]] = []

    def on_start(self, total_steps: int, dt: float) -> None:
        self.log_data = []

    def log_bulk(self, data: Dict[str, Any]) -> None:
        """Sets the log data directly from a bulk dictionary (e.g. from JIT arrays)."""
        self.log_data = data

    def on_step(self, step_idx: int, time: float, data: SimulationStepData) -> None:
        step_log_entry = {
            "state": data.state,
            "control": data.control,
            "estimate": data.estimate,
            "covariance": data.covariance,
        }
        for i, key in enumerate(data.controller_keys):
            step_log_entry[f"controller_{key}"] = data.controller_values[i]
        for i, key in enumerate(data.planner_keys):
            step_log_entry[f"planner_{key}"] = data.planner_values[i]

        self.log_data.append(step_log_entry)

    def on_end(self, success: bool, message: str = "") -> None:
        if self.log_data:
            write_log(self.filepath, self.log_data)
