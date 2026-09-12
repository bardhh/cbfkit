"""Corridor tuning options reach the showcase builder without running MJX."""

import pytest

from examples.mujoco import g1_showcase as showcase


@pytest.mark.parametrize("smoothing", ["-0.1", "1", "nan"])
def test_showcase_rejects_invalid_plan_smoothing(smoothing, monkeypatch):
    monkeypatch.setattr(showcase, "OVERRIDES", {})
    with pytest.raises(SystemExit) as exc:
        showcase.main(["simulate", "corridor", "--plan-smoothing", smoothing])
    assert exc.value.code == 2


def test_showcase_forwards_corridor_nominal_options(monkeypatch):
    monkeypatch.setattr(showcase, "OVERRIDES", {})
    captured = {}

    def simulate(*args):
        captured.update(showcase.OVERRIDES)

    monkeypatch.setattr(showcase, "simulate", simulate)
    showcase.main(
        [
            "simulate",
            "corridor",
            "--gap",
            "1.5",
            "--sidestep-commitment",
            "--plan-smoothing",
            "0.25",
        ]
    )
    assert captured["sidestep_commitment"] is True
    assert captured["plan_smoothing"] == 0.25
    assert captured["gap"] == 1.5


@pytest.mark.parametrize("args, expected", [([], True), (["--no-sidestep-commitment"], False)])
def test_showcase_commitment_default_and_baseline_switch(args, expected, monkeypatch):
    monkeypatch.setattr(showcase, "OVERRIDES", {})
    captured = {}
    monkeypatch.setattr(showcase, "simulate", lambda *args: captured.update(showcase.OVERRIDES))
    showcase.main(["simulate", "corridor", *args])
    assert captured["sidestep_commitment"] is expected


@pytest.mark.parametrize("args", [["--sidestep-commitment"], ["--plan-smoothing", "0.2"]])
def test_showcase_rejects_corridor_options_on_other_scenarios(args, monkeypatch):
    monkeypatch.setattr(showcase, "OVERRIDES", {})
    with pytest.raises(SystemExit) as exc:
        showcase.main(["simulate", "scramble", *args])
    assert exc.value.code == 2
