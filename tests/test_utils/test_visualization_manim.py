"""Tests for the Manim backend in cbfkit.utils.visualization."""

import warnings

import numpy as np
import pytest

import cbfkit.utils.animator as animator_module
from cbfkit.utils.visualization import visualize_3d_multi_robot


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


class TestManimBackendDispatch:
    def test_manim_backend_raises_import_error_when_missing(self, monkeypatch):
        """Without manim installed, backend='manim' should raise ImportError."""
        monkeypatch.setattr(animator_module, "_HAS_MANIM", False)
        states, goals = _make_synthetic_data()
        with pytest.raises(ImportError, match=r"cbfkit\[manim\]"):
            visualize_3d_multi_robot(
                states=states,
                desired_states=goals,
                desired_state_radius=0.3,
                num_robots=2,
                backend="manim",
            )

    def test_manim_backend_warns_on_subplot_features(self, monkeypatch):
        """Subplot features should emit warnings when using manim backend."""
        # Skip if manim is not installed
        if not animator_module._HAS_MANIM:
            pytest.skip("manim not installed")

        states, goals = _make_synthetic_data()

        # We can't actually render without a display, so mock render_multi_robot_3d
        import cbfkit.utils.visualization as vis_module

        def mock_render(**kwargs):
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
                ellipse_centers=[np.array([0.0, 0.0, 0.0])],
                ellipse_radii=[np.array([1.0, 1.0, 1.0])],
                ellipse_rotations=[np.eye(3)],
            )
            manim_warnings = [x for x in w if "Manim backend" in str(x.message)]
            assert len(manim_warnings) == 2
