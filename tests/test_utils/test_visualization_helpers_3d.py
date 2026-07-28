"""Tests for the backend-shared 3D distance metrics.

``_compute_distance_metrics`` feeds the safety panels of every 3D backend
(plotly, matplotlib, manim), so it runs in CI without any optional extra.
"""

import numpy as np

from cbfkit.utils.visualizations.helpers_3d import _compute_distance_metrics


def _min_dists(states, num_robots, sdim):
    """Run the helper and return the per-step minimum over all robots."""
    goals = np.zeros(sdim * num_robots)
    _, min_dists, _ = _compute_distance_metrics(
        np.asarray(states, dtype=float),
        goals,
        num_robots,
        sdim,
        None,
        None,
        None,
        True,
        False,
    )
    return min_dists.min(axis=1)


class TestMinInterRobotDistance:
    """Separation must be measured over positions only.

    Regression: the helper used to reshape to the full per-robot state width and
    take the norm across every column, reporting sqrt(|dp|^2 + |dv|^2).  That
    overstates separation precisely when robots converge at speed -- the only
    regime the safety panel exists to monitor.
    """

    def test_coincident_robots_report_zero_despite_velocity(self):
        # Both robots at the origin -- a collision -- closing at 3 m/s each.
        # The velocity columns must not rescue the reported distance.
        states = [[0, 0, 0, 3, 0, 0, 0, 0, 0, -3, 0, 0]]
        assert _min_dists(states, num_robots=2, sdim=6)[0] == 0.0

    def test_velocity_columns_do_not_inflate_separation(self):
        # Robots 1.0 apart in x, with deliberately large opposing velocities.
        states = [[0, 0, 0, 5, -4, 2, 1, 0, 0, -5, 4, -2]]
        assert _min_dists(states, num_robots=2, sdim=6)[0] == 1.0

    def test_matches_position_only_norm_over_a_trajectory(self):
        rng = np.random.default_rng(0)
        n_steps, num_robots, sdim = 12, 3, 6
        states = rng.normal(0, 5, size=(n_steps, num_robots * sdim))

        got = _min_dists(states, num_robots, sdim)

        pos = states.reshape(n_steps, num_robots, sdim)[:, :, :3]
        expected = np.array(
            [
                min(
                    np.linalg.norm(pos[t, i] - pos[t, j])
                    for i in range(num_robots)
                    for j in range(num_robots)
                    if i != j
                )
                for t in range(n_steps)
            ]
        )
        assert np.allclose(got, expected)

    def test_position_only_state_layout_still_works(self):
        # sdim == 3 has no velocity columns; behaviour must be unchanged.
        states = [[0, 0, 0, 2, 0, 0]]
        assert _min_dists(states, num_robots=2, sdim=3)[0] == 2.0


class TestGoalAndObstacleDistances:
    """These already sliced positions correctly -- pin that they still do."""

    def test_goal_distance_ignores_velocity_columns(self):
        states = np.array([[0.0, 0.0, 0.0, 9.0, 9.0, 9.0]])
        goals = np.array([3.0, 4.0, 0.0, 0.0, 0.0, 0.0])
        goal_dists, _, _ = _compute_distance_metrics(
            states, goals, 1, 6, None, None, None, False, False
        )
        assert goal_dists[0, 0] == 5.0

    def test_obstacle_distance_ignores_velocity_columns(self):
        states = np.array([[10.0, 0.0, 0.0, 7.0, 7.0, 7.0]])
        goals = np.zeros(6)
        _, _, obs_dists = _compute_distance_metrics(
            states,
            goals,
            1,
            6,
            [np.array([0.0, 0.0, 0.0])],
            [np.array([2.0, 2.0, 2.0])],
            [np.eye(3)],
            False,
            True,
        )
        # Point 10 from centre, sphere radius 2 -> 8 to the surface.
        assert obs_dists[0, 0] == 8.0
