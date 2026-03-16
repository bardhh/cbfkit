"""Plotly 3D multi-robot visualization backend."""

from pathlib import Path

import numpy as np

from .helpers_3d import _ellipsoid_mesh

# Plotly tab10-equivalent colours
_PLOTLY_TAB10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _visualize_3d_plotly(
    states, desired_states, desired_state_radius, num_robots,
    ellipse_centers, ellipse_radii, ellipse_rotations,
    x_lim, y_lim, z_lim, dt, sdim, title, save_animation,
    animation_filename, include_min_distance_plot,
    include_min_distance_to_obstacles_plot, threshold,
    goal_dists, min_dists, obs_dists,
):
    from cbfkit.utils.animators.helpers import (
        _compute_plotly_frame_step,
        _plotly_animation_controls,
    )
    from cbfkit.utils.animators.deps import _require_plotly
    _require_plotly()

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    N = len(states)
    time = np.arange(N) * dt
    colors = [_PLOTLY_TAB10[i % len(_PLOTLY_TAB10)] for i in range(num_robots)]

    # 3D scenes are heavier to render; use a higher floor than 2D
    frame_indices, frame_duration_ms = _compute_plotly_frame_step(
        dt, N, max_frames=200, min_frame_ms=80,
    )

    # --- subplot layout ---
    num_cols = 2
    specs_row = [{"type": "scene"}, {"type": "xy"}]
    titles = [title, "Distance to Goal"]
    if include_min_distance_plot:
        num_cols += 1
        specs_row.append({"type": "xy"})
        titles.append("Min Dist Between Robots")
    if include_min_distance_to_obstacles_plot:
        num_cols += 1
        specs_row.append({"type": "xy"})
        titles.append("Min Dist to Obstacles")

    widths = [2] + [1] * (num_cols - 1)
    fig = make_subplots(
        rows=1, cols=num_cols,
        specs=[specs_row],
        column_widths=widths,
        subplot_titles=titles,
    )

    # --- static 3D elements ---
    for i in range(num_robots):
        idx = sdim * i
        gx, gy, gz = float(desired_states[idx]), float(desired_states[idx + 1]), float(desired_states[idx + 2])
        fig.add_trace(
            go.Scatter3d(
                x=[gx], y=[gy], z=[gz],
                mode="markers",
                marker=dict(size=8, color=colors[i], symbol="diamond"),
                name=f"Goal {i + 1}",
                showlegend=True,
            ),
            row=1, col=1,
        )
        sx, sy, sz, si, sj, sk = _ellipsoid_mesh(
            np.array([gx, gy, gz]),
            np.array([desired_state_radius] * 3),
            np.eye(3), n=15,
        )
        fig.add_trace(
            go.Mesh3d(
                x=sx.tolist(), y=sy.tolist(), z=sz.tolist(),
                i=si, j=sj, k=sk,
                color=colors[i], opacity=0.15,
                showlegend=False,
            ),
            row=1, col=1,
        )

    # Ellipsoid obstacles
    if ellipse_centers is not None and ellipse_radii is not None and ellipse_rotations is not None:
        for ec, er, erot in zip(ellipse_centers, ellipse_radii, ellipse_rotations):
            mx, my, mz, ii, jj, kk = _ellipsoid_mesh(
                np.asarray(ec), np.asarray(er), np.asarray(erot),
            )
            fig.add_trace(
                go.Mesh3d(
                    x=mx.tolist(), y=my.tolist(), z=mz.tolist(),
                    i=ii, j=jj, k=kk,
                    color="black", opacity=0.2,
                    showlegend=False,
                ),
                row=1, col=1,
            )

    n_static = len(fig.data)

    # --- animated traces (initial = empty) ---
    for i in range(num_robots):
        fig.add_trace(
            go.Scatter3d(
                x=[], y=[], z=[],
                mode="lines",
                line=dict(color=colors[i], width=3),
                name=f"Robot {i + 1}",
                showlegend=True,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[], y=[],
                mode="lines",
                line=dict(color=colors[i], width=2),
                name=f"Robot {i + 1}",
                showlegend=False,
            ),
            row=1, col=2,
        )

    if include_min_distance_plot:
        col_md = 3
        for i in range(num_robots):
            fig.add_trace(
                go.Scatter(
                    x=[], y=[],
                    mode="lines",
                    line=dict(color=colors[i], width=2),
                    name=f"Robot {i + 1}",
                    showlegend=False,
                ),
                row=1, col=col_md,
            )
        if threshold is not None:
            fig.add_hline(
                y=threshold, line_dash="dash", line_color="red",
                row=1, col=col_md,
            )

    if include_min_distance_to_obstacles_plot:
        col_od = 3 if not include_min_distance_plot else 4
        for i in range(num_robots):
            fig.add_trace(
                go.Scatter(
                    x=[], y=[],
                    mode="lines",
                    line=dict(color=colors[i], width=2),
                    name=f"Robot {i + 1}",
                    showlegend=False,
                ),
                row=1, col=col_od,
            )

    n_animated = len(fig.data) - n_static
    animated_indices = list(range(n_static, n_static + n_animated))

    # --- animation frames ---
    frames = []
    for fi in frame_indices:
        t = fi * dt
        trace_data = []

        for i in range(num_robots):
            idx = sdim * i
            trace_data.append(
                go.Scatter3d(
                    x=states[:fi, idx].tolist(),
                    y=states[:fi, idx + 1].tolist(),
                    z=states[:fi, idx + 2].tolist(),
                )
            )
            trace_data.append(
                go.Scatter(
                    x=time[:fi].tolist(),
                    y=goal_dists[:fi, i].tolist(),
                )
            )

        if include_min_distance_plot:
            for i in range(num_robots):
                trace_data.append(
                    go.Scatter(
                        x=time[:fi].tolist(),
                        y=min_dists[:fi, i].tolist(),
                    )
                )

        if include_min_distance_to_obstacles_plot:
            for i in range(num_robots):
                trace_data.append(
                    go.Scatter(
                        x=time[:fi].tolist(),
                        y=obs_dists[:fi, i].tolist(),
                    )
                )

        frames.append(
            go.Frame(
                data=trace_data,
                traces=animated_indices,
                name=f"{t:.2f}s",
            )
        )

    fig.frames = frames

    # --- scene + axis layout ---
    menus, sliders = _plotly_animation_controls(
        frames, frame_duration_ms, button_y=-0.25,
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=list(x_lim), title="X [m]"),
            yaxis=dict(range=list(y_lim), title="Y [m]"),
            zaxis=dict(range=list(z_lim), title="Z [m]"),
            aspectmode="cube",
        ),
        width=350 * num_cols,
        height=500,
        margin=dict(l=30, r=30, t=60, b=100),
        template="plotly_white",
        updatemenus=menus,
        sliders=sliders,
    )

    fig.update_xaxes(title_text="Time [s]", row=1, col=2)
    fig.update_yaxes(title_text="Distance [m]", row=1, col=2)
    fig.update_yaxes(range=[0, float(np.max(goal_dists)) * 1.1], row=1, col=2)

    if include_min_distance_plot:
        fig.update_xaxes(title_text="Time [s]", row=1, col=3)
        fig.update_yaxes(title_text="Distance [m]", row=1, col=3)
        fig.update_yaxes(range=[0, float(np.max(min_dists)) * 1.1], row=1, col=3)

    if include_min_distance_to_obstacles_plot:
        col_od = 3 if not include_min_distance_plot else 4
        fig.update_xaxes(title_text="Time [s]", row=1, col=col_od)
        fig.update_yaxes(title_text="Distance [m]", row=1, col=col_od)
        fig.update_yaxes(range=[0, float(np.max(obs_dists)) * 1.1], row=1, col=col_od)

    # --- save / show ---
    if save_animation:
        output_path = Path(animation_filename).with_suffix(".html")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path), auto_open=False)
        abs_path = str(output_path.resolve())
        print(f"\nInteractive animation saved to: file://{abs_path}")
    else:
        fig.show()

    return fig
