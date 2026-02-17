import numpy as np

import cbfkit.simulation.simulator as sim


def _patch_numpy_reductions(monkeypatch):
    monkeypatch.setattr(sim.jnp, "any", lambda arr: bool(np.any(arr)))
    monkeypatch.setattr(sim.jnp, "argmax", lambda arr: np.argmax(arr))
    monkeypatch.setattr(sim.jnp, "sum", lambda arr: np.sum(arr))


def test_check_simulation_status_treats_status_2_as_failure_warning(monkeypatch):
    warnings = []

    _patch_numpy_reductions(monkeypatch)
    monkeypatch.setattr(sim, "print_warning", lambda message: warnings.append(message))
    monkeypatch.setattr(sim, "print_error", lambda _message: None)

    sim._check_simulation_status(
        controller_data_keys=["error", "sub_data_solver_status"],
        controller_data_values=[np.array([False, False]), np.array([1, 2])],
        planner_data_keys=[],
        planner_data_values=[],
        nan_detected=False,
    )

    assert any("treated as controller failures" in message for message in warnings)


def test_check_simulation_status_treats_status_5_as_failure_warning(monkeypatch):
    warnings = []

    _patch_numpy_reductions(monkeypatch)
    monkeypatch.setattr(sim, "print_warning", lambda message: warnings.append(message))
    monkeypatch.setattr(sim, "print_error", lambda _message: None)

    sim._check_simulation_status(
        controller_data_keys=["error", "sub_data_solver_status"],
        controller_data_values=[np.array([False, False]), np.array([1, 5])],
        planner_data_keys=[],
        planner_data_values=[],
        nan_detected=False,
    )

    assert any("treated as controller failures" in message for message in warnings)
