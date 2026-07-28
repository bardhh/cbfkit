"""Tests for the Manim backend in cbfkit.utils.visualization."""

import warnings

import numpy as np
import pytest

from cbfkit.utils.animators import deps
from cbfkit.utils.visualization import _parse_manim_backend, visualize_3d_multi_robot


def _make_synthetic_data(num_robots=2, sdim=3, n_steps=20):
    """Create minimal synthetic trajectory data for testing."""
    states = np.zeros((n_steps, sdim * num_robots))
    goals = np.zeros(sdim * num_robots)
    for i in range(num_robots):
        idx = sdim * i
        t = np.linspace(0, 2 * np.pi, n_steps)
        states[:, idx] = np.cos(t + i)
        states[:, idx + 1] = np.sin(t + i)
        states[:, idx + 2] = t / (2 * np.pi)
        goals[idx] = -1.0 * (i + 1)
        goals[idx + 1] = 1.0 * (i + 1)
        goals[idx + 2] = 1.0
    return states, goals


class TestManimQualityParsing:
    def test_bare_manim_defaults_to_low(self):
        assert _parse_manim_backend("manim") == "low_quality"

    @pytest.mark.parametrize(
        "suffix,expected",
        [
            ("low", "low_quality"),
            ("medium", "medium_quality"),
            ("high", "high_quality"),
            ("production", "production_quality"),
        ],
    )
    def test_quality_suffixes(self, suffix, expected):
        assert _parse_manim_backend(f"manim-{suffix}") == expected

    def test_invalid_suffix_raises(self):
        with pytest.raises(ValueError, match="Unknown Manim backend"):
            _parse_manim_backend("manim-ultra")

    def test_quality_passed_to_render(self, monkeypatch):
        """Ensure the quality kwarg reaches render_multi_robot_3d."""
        # Nothing here needs manim itself: the renderer is mocked and only the
        # dispatch path is under test.  Satisfy the gate rather than skipping --
        # manim is excluded from the [dev] extra, so a skip means "never in CI".
        monkeypatch.setattr(deps, "_HAS_MANIM", True)

        states, goals = _make_synthetic_data()
        captured = {}

        def mock_render(**kwargs):
            captured.update(kwargs)
            return "/tmp/mock.mp4"

        monkeypatch.setattr(
            "cbfkit.utils.visualizations.manim_3d_multi_robot.render_multi_robot_3d",
            mock_render,
        )
        visualize_3d_multi_robot(
            states=states,
            desired_states=goals,
            desired_state_radius=0.3,
            num_robots=2,
            backend="manim-high",
        )
        assert captured["quality"] == "high_quality"


class TestManimBackendDispatch:
    def test_manim_backend_raises_import_error_when_missing(self, monkeypatch):
        """Without manim installed, backend='manim' should raise ImportError."""
        # deps._HAS_MANIM is what _require_manim() consults; patching the
        # re-export on cbfkit.utils.animator does not affect the gate.
        monkeypatch.setattr(deps, "_HAS_MANIM", False)
        states, goals = _make_synthetic_data()
        with pytest.raises(ImportError, match=r"cbfkit\[manim\]"):
            visualize_3d_multi_robot(
                states=states,
                desired_states=goals,
                desired_state_radius=0.3,
                num_robots=2,
                backend="manim",
            )

    def test_manim_backend_forwards_subplot_data(self, monkeypatch):
        """The manim backend renders the distance panels rather than dropping them.

        It previously warned that ``include_min_distance_plot`` "will be
        ignored" while forwarding the data and drawing the panel anyway, so the
        warning told users the opposite of what happened.
        """
        # Only the dispatch path is under test and the renderer is mocked, so
        # satisfy the gate rather than skipping (manim is not in the [dev]
        # extra, so skipping here would mean this never runs in CI).
        monkeypatch.setattr(deps, "_HAS_MANIM", True)

        states, goals = _make_synthetic_data()

        # We can't actually render without a display, so mock render_multi_robot_3d
        recorded = {}

        def mock_render(**kwargs):
            recorded.update(kwargs)
            return "/tmp/mock_output.mp4"

        monkeypatch.setattr(
            "cbfkit.utils.visualizations.manim_3d_multi_robot.render_multi_robot_3d",
            mock_render,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            visualize_3d_multi_robot(
                states=states,
                desired_states=goals,
                desired_state_radius=0.3,
                num_robots=2,
                backend="manim",
                include_min_distance_plot=True,
                include_min_distance_to_obstacles_plot=True,
                threshold=0.5,
                safety_radius=0.25,
                ellipse_centers=[np.array([0.0, 0.0, 0.0])],
                ellipse_radii=[np.array([1.0, 1.0, 1.0])],
                ellipse_rotations=[np.eye(3)],
            )
            # Stronger than matching the old text: the path should be silent.
            assert [str(x.message) for x in w] == []

        # The panels are actually drawn, so the data must reach the renderer.
        assert recorded["min_dists"] is not None
        assert recorded["obs_dists"] is not None
        # ...along with the reference line and bubble size that make them readable.
        assert recorded["threshold"] == 0.5
        assert recorded["safety_radius"] == 0.25
