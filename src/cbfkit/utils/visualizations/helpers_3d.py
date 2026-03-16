"""Shared 3D geometry helpers for multi-robot visualizations."""

import numpy as np


def _plot_ellipse_3d(ax, center, radii, rotation, color="blue", alpha=0.2):
    """Plot a 3D ellipsoid on *ax* (matplotlib)."""
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)  # (3, n*n)
    pts = (rotation @ pts).T + center  # (n*n, 3)
    n = len(u)
    x = pts[:, 0].reshape(n, n)
    y = pts[:, 1].reshape(n, n)
    z = pts[:, 2].reshape(n, n)

    ax.plot_surface(x, y, z, color=color, rstride=4, cstride=4, alpha=alpha)


def _point_to_ellipsoid_distance(p, c, r, R):
    """Distance from point *p* to an ellipsoid (negative if inside)."""
    u = p - c
    norm_u = np.linalg.norm(u)
    if norm_u == 0:
        return -np.min(r)

    u_normalized = u / norm_u
    inv_r_squared = np.diag(1.0 / (r ** 2))
    K = u_normalized.T @ R @ inv_r_squared @ R.T @ u_normalized
    s = 1.0 / np.sqrt(K)
    x = c + s * u_normalized
    d = np.linalg.norm(p - x)
    return d if s >= 1 else -d


def _ellipsoid_mesh(center, radii, rotation, n=12):
    """Generate triangulated mesh vertices for a 3D ellipsoid (for Plotly)."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)  # (3, n*n)
    pts = (rotation @ pts).T + center  # (n*n, 3)

    # Build triangle indices for the grid (vectorized)
    rows, cols = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    p0 = (rows * n + cols).ravel()
    p1 = p0 + 1
    p2 = p0 + n
    p3 = p2 + 1
    ii = np.concatenate([p0, p0]).tolist()
    jj = np.concatenate([p1, p2]).tolist()
    kk = np.concatenate([p2, p3]).tolist()
    return pts[:, 0], pts[:, 1], pts[:, 2], ii, jj, kk


def _compute_distance_metrics(states, desired_states, num_robots, sdim,
                              ellipse_centers, ellipse_radii, ellipse_rotations,
                              include_min_dist, include_obs_dist):
    """Pre-compute distance arrays used by both backends."""
    N = len(states)

    # Goal distances (N, num_robots)
    goal_dists = np.zeros((N, num_robots))
    for i in range(num_robots):
        idx = sdim * i
        goal_dists[:, i] = np.linalg.norm(
            states[:, idx:idx + 3] - desired_states[idx:idx + 3], axis=1,
        )

    # Min inter-robot distances (vectorized over robots)
    min_dists = None
    if include_min_dist:
        min_dists = np.zeros((N, num_robots))
        for t in range(N):
            positions = states[t, :].reshape(num_robots, sdim)
            diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_dists[t, :] = np.min(dists, axis=1)

    # Min obstacle distances
    obs_dists = None
    if include_obs_dist and ellipse_centers is not None:
        num_obs = len(ellipse_centers)
        obs_dists = np.zeros((N, num_robots))
        for t in range(N):
            for i in range(num_robots):
                idx = sdim * i
                p = states[t, idx:idx + 3]
                dmin = np.inf
                for j in range(num_obs):
                    d = _point_to_ellipsoid_distance(
                        p, ellipse_centers[j], ellipse_radii[j], ellipse_rotations[j],
                    )
                    dmin = min(dmin, d)
                obs_dists[t, i] = dmin

    return goal_dists, min_dists, obs_dists
