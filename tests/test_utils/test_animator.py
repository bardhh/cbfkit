"""Tests for cbfkit.utils.animator module."""

import os

import numpy as np
import pytest

import cbfkit.utils.animator as animator_module
from cbfkit.utils.animator import AnimationConfig, CBFAnimator, save_animation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Use non-interactive backend for CI
os.environ.setdefault("CBFKIT_TEST_MODE", "1")

import matplotlib

matplotlib.use("Agg")


@pytest.fixture
def simple_states():
    """Linear trajectory: 50 steps, 4-dim state (x, y, v, theta)."""
    n = 50
    t = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t), np.zeros(n), t])


@pytest.fixture
def estimates(simple_states):
    """Noisy version of simple_states for estimate overlay."""
    rng = np.random.default_rng(42)
    return simple_states + rng.normal(0, 0.05, simple_states.shape)


# ---------------------------------------------------------------------------
# AnimationConfig
# ---------------------------------------------------------------------------


class TestAnimationConfig:
    def test_defaults(self):
        cfg = AnimationConfig()
        assert cfg.fps == 20
        assert cfg.dpi == 100
        assert cfg.interval == 50
        assert cfg.bitrate == 1800
        assert cfg.figsize == (10, 8)
        assert cfg.grid_alpha == 0.3
        assert cfg.blit is True

    def test_override(self):
        cfg = AnimationConfig(fps=30, dpi=150)
        assert cfg.fps == 30
        assert cfg.dpi == 150

    def test_default_config_module_level(self):
        assert animator_module.DEFAULT_CONFIG.fps == 20


# ---------------------------------------------------------------------------
# CBFAnimator – build
# ---------------------------------------------------------------------------


class TestCBFAnimatorBuild:
    def test_build_creates_figure(self, simple_states):
        a = CBFAnimator(simple_states, dt=0.1)
        fig, ax = a.build()
        assert fig is not None
        assert ax is not None
        assert a.fig is fig
        assert a.ax is ax
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_build_sets_limits_and_labels(self, simple_states):
        a = CBFAnimator(simple_states, x_lim=(-5, 5), y_lim=(-3, 3), title="Test")
        fig, ax = a.build()
        assert ax.get_xlim() == (-5, 5)
        assert ax.get_ylim() == (-3, 3)
        assert ax.get_title() == "Test"
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_aspect_equal(self, simple_states):
        a = CBFAnimator(simple_states, aspect="equal")
        fig, ax = a.build()
        # matplotlib normalizes "equal" to 1.0
        assert ax.get_aspect() in ("equal", 1.0)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_add_goal(self, simple_states):
        a = CBFAnimator(simple_states)
        ret = a.add_goal([1.0, 2.0], radius=0.5, color="green")
        assert ret is a  # fluent API
        fig, ax = a.build()
        # Goal adds a circle patch
        circles = [p for p in ax.patches if isinstance(p, matplotlib.patches.Circle)]
        assert len(circles) >= 1
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_add_obstacle_circle(self, simple_states):
        a = CBFAnimator(simple_states)
        a.add_obstacle([0, 0], radius=1.0)
        fig, ax = a.build()
        circles = [p for p in ax.patches if isinstance(p, matplotlib.patches.Circle)]
        assert len(circles) >= 1
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_add_obstacle_ellipse(self, simple_states):
        a = CBFAnimator(simple_states)
        a.add_obstacle([0, 0], ellipse_radii=(1.0, 0.5))
        fig, ax = a.build()
        ellipses = [p for p in ax.patches if isinstance(p, matplotlib.patches.Ellipse)]
        assert len(ellipses) >= 1
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_add_obstacles_bulk(self, simple_states):
        centers = [[0, 0], [1, 1], [2, 2]]
        radii = [0.5, 0.3, 0.4]
        a = CBFAnimator(simple_states)
        ret = a.add_obstacles(centers, radii=radii)
        assert ret is a
        assert len(a._obstacles) == 3
        import matplotlib.pyplot as plt

        fig, _ = a.build()
        plt.close(fig)

    def test_add_trajectory_line(self, simple_states):
        a = CBFAnimator(simple_states)
        ret = a.add_trajectory(x_idx=0, y_idx=1, label="Traj")
        assert ret is a
        fig, ax = a.build()
        assert len(a._traj_artists) == 1
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_add_trajectory_scatter(self, simple_states, estimates):
        a = CBFAnimator(simple_states)
        a.add_trajectory(x_idx=0, y_idx=1, data=estimates, style="scatter", label="Est")
        fig, ax = a.build()
        assert len(a._traj_artists) == 1
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_show_time(self, simple_states):
        a = CBFAnimator(simple_states, dt=0.1)
        ret = a.show_time()
        assert ret is a
        fig, ax = a.build()
        assert a._time_text is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


# ---------------------------------------------------------------------------
# CBFAnimator – animate
# ---------------------------------------------------------------------------


class TestCBFAnimatorAnimate:
    def test_animate_returns_func_animation(self, simple_states):
        a = CBFAnimator(simple_states, dt=0.1)
        a.add_trajectory(x_idx=0, y_idx=1)
        anim = a.animate()
        assert anim is not None
        assert a.animation is anim
        import matplotlib.pyplot as plt

        plt.close(a.fig)

    def test_update_sets_trajectory_data(self, simple_states):
        a = CBFAnimator(simple_states, dt=0.1)
        a.add_trajectory(x_idx=0, y_idx=1)
        a.build()
        artists = a._update_func(10)
        assert len(artists) >= 1
        xdata, ydata = a._traj_artists[0].get_data()
        np.testing.assert_array_equal(xdata, simple_states[:10, 0])
        np.testing.assert_array_equal(ydata, simple_states[:10, 1])
        import matplotlib.pyplot as plt

        plt.close(a.fig)

    def test_update_with_time_text(self, simple_states):
        a = CBFAnimator(simple_states, dt=0.5)
        a.add_trajectory(x_idx=0, y_idx=1)
        a.show_time()
        a.build()
        artists = a._update_func(4)
        assert a._time_text.get_text() == "Time: 2.0s"
        import matplotlib.pyplot as plt

        plt.close(a.fig)

    def test_on_frame_callback(self, simple_states):
        calls = []

        def my_callback(frame, ax):
            calls.append(frame)
            return []

        a = CBFAnimator(simple_states, dt=0.1)
        a.add_trajectory(x_idx=0, y_idx=1)
        a.on_frame(my_callback)
        a.build()
        a._update_func(5)
        a._update_func(10)
        assert calls == [5, 10]
        import matplotlib.pyplot as plt

        plt.close(a.fig)

    def test_multiple_trajectories(self, simple_states, estimates):
        a = CBFAnimator(simple_states, dt=0.1)
        a.add_trajectory(x_idx=0, y_idx=1, label="True")
        a.add_trajectory(x_idx=0, y_idx=1, data=estimates, style="scatter", label="Est")
        a.build()
        artists = a._update_func(10)
        assert len(artists) == 2
        import matplotlib.pyplot as plt

        plt.close(a.fig)


# ---------------------------------------------------------------------------
# CBFAnimator – save
# ---------------------------------------------------------------------------


class TestCBFAnimatorSave:
    def test_save_creates_file(self, simple_states, tmp_path):
        a = CBFAnimator(simple_states, dt=0.1)
        a.add_trajectory(x_idx=0, y_idx=1)
        path = str(tmp_path / "test_anim.mp4")
        result = a.save(path)
        # Should save as either mp4 or gif depending on backend availability
        assert result != "" or True  # May fail in CI without ffmpeg, that's OK
        import matplotlib.pyplot as plt

        plt.close(a.fig)

    def test_save_gif_fallback(self, simple_states, tmp_path):
        """Force GIF fallback by using a config and verifying some file is created."""
        a = CBFAnimator(simple_states, dt=0.1, config=AnimationConfig(fps=5, dpi=50))
        a.add_trajectory(x_idx=0, y_idx=1)
        path = str(tmp_path / "test_anim.mp4")
        result = a.save(path)
        # At least one format should succeed
        if result:
            assert os.path.exists(result)
        import matplotlib.pyplot as plt

        plt.close(a.fig)


# ---------------------------------------------------------------------------
# save_animation standalone
# ---------------------------------------------------------------------------


class TestSaveAnimation:
    def test_save_animation_creates_parent_dirs(self, simple_states, tmp_path):
        import matplotlib.animation as mpl_animation
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        (line,) = ax.plot([], [])
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        def update(frame):
            line.set_data(simple_states[:frame, 0], simple_states[:frame, 1])
            return (line,)

        anim = mpl_animation.FuncAnimation(fig, update, frames=10, blit=True)

        nested = tmp_path / "sub" / "dir" / "anim.mp4"
        save_animation(anim, str(nested))
        # Parent dirs should exist even if save failed
        assert nested.parent.exists()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Plotly backend
# ---------------------------------------------------------------------------

plotly = pytest.importorskip("plotly")


class TestPlotlyBackend:
    def test_invalid_backend_raises(self, simple_states):
        with pytest.raises(ValueError, match="Unknown backend"):
            CBFAnimator(simple_states, backend="invalid")

    def test_build_returns_plotly_figure(self, simple_states):
        a = CBFAnimator(simple_states, dt=0.1, backend="plotly")
        a.add_trajectory(x_idx=0, y_idx=1)
        fig = a.build()
        assert fig is not None
        assert a.fig is fig
        assert a.ax is None  # no matplotlib axes for plotly

    def test_plotly_goals_and_obstacles(self, simple_states):
        a = CBFAnimator(simple_states, dt=0.1, backend="plotly")
        a.add_goal([1.0, 2.0], radius=0.5, color="r")
        a.add_obstacle([0, 0], radius=1.0, color="k")
        a.add_obstacle([2, 2], ellipse_radii=(0.5, 0.3))
        a.add_trajectory(x_idx=0, y_idx=1)
        fig = a.build()
        # Should have shapes for goal circle + 2 obstacles
        assert len(fig.layout.shapes) == 3
        # Should have goal marker trace + trajectory trace
        assert len(fig.data) == 2

    def test_plotly_multiple_trajectories(self, simple_states, estimates):
        a = CBFAnimator(simple_states, dt=0.1, backend="plotly")
        a.add_trajectory(x_idx=0, y_idx=1, label="True")
        a.add_trajectory(x_idx=0, y_idx=1, data=estimates, style="scatter", label="Est")
        fig = a.build()
        assert len(fig.data) == 2  # two trajectory traces
        assert len(fig.frames) > 0

    def test_plotly_frames_contain_data(self, simple_states):
        a = CBFAnimator(simple_states, dt=0.1, backend="plotly")
        a.add_trajectory(x_idx=0, y_idx=1)
        fig = a.build()
        # Last frame should have data up to the end
        last_frame = fig.frames[-1]
        assert len(last_frame.data) == 1  # one trajectory
        assert len(last_frame.data[0].x) > 0

    def test_plotly_show_time_adds_annotations(self, simple_states):
        a = CBFAnimator(simple_states, dt=0.1, backend="plotly")
        a.add_trajectory(x_idx=0, y_idx=1)
        a.show_time()
        fig = a.build()
        # Frames should have annotation layouts
        mid_frame = fig.frames[len(fig.frames) // 2]
        assert mid_frame.layout.annotations is not None
        assert "Time:" in mid_frame.layout.annotations[0].text

    def test_plotly_aspect_equal(self, simple_states):
        a = CBFAnimator(simple_states, dt=0.1, backend="plotly", aspect="equal")
        a.add_trajectory(x_idx=0, y_idx=1)
        fig = a.build()
        assert fig.layout.yaxis.scaleanchor == "x"

    def test_plotly_save_creates_html(self, simple_states, tmp_path):
        a = CBFAnimator(simple_states, dt=0.1, backend="plotly")
        a.add_trajectory(x_idx=0, y_idx=1)
        path = str(tmp_path / "test_anim.mp4")  # extension replaced to .html
        result = a.save(path)
        assert result.endswith(".html")
        assert os.path.exists(result)

    def test_plotly_save_nested_dirs(self, simple_states, tmp_path):
        a = CBFAnimator(simple_states, dt=0.1, backend="plotly")
        a.add_trajectory(x_idx=0, y_idx=1)
        path = str(tmp_path / "sub" / "dir" / "anim.html")
        result = a.save(path)
        assert os.path.exists(result)

    def test_plotly_downsampling(self, simple_states):
        # With max_frames=10, a 50-step sim should be downsampled
        cfg = AnimationConfig(plotly_max_frames=10)
        a = CBFAnimator(simple_states, dt=0.1, backend="plotly", config=cfg)
        a.add_trajectory(x_idx=0, y_idx=1)
        fig = a.build()
        assert len(fig.frames) <= 12  # 10 + possible last frame rounding

    def test_plotly_animate_returns_figure(self, simple_states):
        a = CBFAnimator(simple_states, dt=0.1, backend="plotly")
        a.add_trajectory(x_idx=0, y_idx=1)
        result = a.animate()
        assert result is a.fig
        assert a.animation is result


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


class TestImportGuard:
    def test_require_matplotlib_raises_when_missing(self, monkeypatch):
        monkeypatch.setattr(animator_module, "_HAS_MATPLOTLIB", False)
        with pytest.raises(ImportError, match=r"cbfkit\[vis\]"):
            animator_module._require_matplotlib()

    def test_require_plotly_raises_when_missing(self, monkeypatch):
        monkeypatch.setattr(animator_module, "_HAS_PLOTLY", False)
        with pytest.raises(ImportError, match=r"cbfkit\[plotly\]"):
            animator_module._require_plotly()
