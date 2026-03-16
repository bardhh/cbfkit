"""Plotly backend mixin for CBFAnimator."""

from pathlib import Path

import plotly.graph_objects as go

from .helpers import (
    _PLOTLY_FADE_BUCKETS,
    _SCATTER_ALPHA_FACTOR,
    _circle_shape,
    _compute_plotly_frame_step,
    _ellipse_svg_path,
    _plotly_animation_controls,
    _to_css_color,
)


class _PlotlyMixin:
    """Plotly-specific build / save / show methods.

    Mixed into :class:`~cbfkit.utils.animators.animator.CBFAnimator`.
    Expects the host class to provide ``_states``, ``_dt``, ``_config``,
    ``_goals``, ``_obstacles``, ``_trajectories``, ``_agents``,
    ``_predictions``, ``_show_time``, ``_x_lim``, ``_y_lim``, ``_title``,
    ``_aspect``, and ``_compute_prediction``.
    """

    def _build_plotly(self):
        """Build an animated Plotly figure from the declared elements."""
        cfg = self._config
        n_total = len(self._states)
        frame_indices, frame_duration_ms = _compute_plotly_frame_step(
            self._dt, n_total, max_frames=cfg.plotly_max_frames,
        )

        # --- static layout shapes (goals + obstacles) ---
        static_shapes: list = []
        for g in self._goals:
            pos = g["position"]
            css = _to_css_color(g["color"])
            static_shapes.append(
                dict(
                    type="circle",
                    xref="x", yref="y",
                    x0=float(pos[0]) - g["radius"],
                    y0=float(pos[1]) - g["radius"],
                    x1=float(pos[0]) + g["radius"],
                    y1=float(pos[1]) + g["radius"],
                    line=dict(color=css, dash="dash", width=1),
                    fillcolor="rgba(0,0,0,0)",
                )
            )

        for obs in self._obstacles:
            c = obs["center"]
            css = _to_css_color(obs["color"])
            opacity = obs["alpha"]
            if obs["ellipse_radii"] is not None:
                rx, ry = obs["ellipse_radii"]
            elif obs["radius"] is not None:
                rx = ry = obs["radius"]
            else:
                continue
            path_str = _ellipse_svg_path(float(c[0]), float(c[1]), float(rx), float(ry))
            static_shapes.append(
                dict(
                    type="path",
                    path=path_str,
                    xref="x", yref="y",
                    fillcolor=css,
                    opacity=opacity,
                    line=dict(color=css, width=1),
                )
            )

        # --- base traces ---
        base_traces: list = []

        # 1) Goal markers
        for g in self._goals:
            pos = g["position"]
            css = _to_css_color(g["color"])
            base_traces.append(
                go.Scatter(
                    x=[float(pos[0])],
                    y=[float(pos[1])],
                    mode="markers",
                    marker=dict(size=8, color=css, symbol="circle"),
                    name=g["label"],
                    showlegend=True,
                )
            )

        # 2) Trajectory traces
        for spec in self._trajectories:
            css = _to_css_color(spec["color"])
            mode = "markers" if spec["style"] == "scatter" else "lines"
            marker_opts = (
                dict(size=3, color=css, opacity=spec["alpha"] * _SCATTER_ALPHA_FACTOR)
                if spec["style"] == "scatter"
                else dict()
            )
            line_opts = (
                dict(color=css, width=spec["linewidth"])
                if spec["style"] != "scatter"
                else dict()
            )
            base_traces.append(
                go.Scatter(
                    x=[], y=[],
                    mode=mode,
                    name=spec["label"],
                    line=line_opts if line_opts else None,
                    marker=marker_opts if marker_opts else None,
                    opacity=spec["alpha"] if spec["style"] != "scatter" else 1.0,
                )
            )

        # 3) Agent traces: body marker + trail line per agent
        for spec in self._agents:
            css = _to_css_color(spec["body_color"])
            # Body marker
            base_traces.append(
                go.Scatter(
                    x=[], y=[],
                    mode="markers",
                    marker=dict(size=10, color=css, opacity=spec["body_alpha"]),
                    name=spec["label"],
                    showlegend=True,
                )
            )
            # Trail line
            trail_css = _to_css_color(spec["trail_color"])
            dash_map = {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}
            dash = dash_map.get(spec["trail_style"], "solid")
            base_traces.append(
                go.Scatter(
                    x=[], y=[],
                    mode="lines",
                    line=dict(color=trail_css, width=1.5, dash=dash),
                    opacity=spec["trail_alpha"],
                    name=spec["label"] + " trail",
                    showlegend=False,
                )
            )

        # 4) Prediction traces: _PLOTLY_FADE_BUCKETS sub-traces per prediction
        for spec in self._predictions:
            css = _to_css_color(spec["color"])
            dash_map = {"dotted": "dot", "dashed": "dash", "solid": "solid"}
            dash = dash_map.get(spec["linestyle"], "solid")
            for bucket in range(_PLOTLY_FADE_BUCKETS):
                opacity = spec["alpha"] * (1.0 - bucket / _PLOTLY_FADE_BUCKETS)
                base_traces.append(
                    go.Scatter(
                        x=[], y=[],
                        mode="lines",
                        line=dict(color=css, width=spec["linewidth"], dash=dash),
                        opacity=opacity,
                        name=spec["label"] if (bucket == 0 and spec["label"]) else "",
                        showlegend=(bucket == 0 and bool(spec["label"])),
                    )
                )

        n_traces = len(base_traces)

        # --- animation frames ---
        has_agents = len(self._agents) > 0
        frames: list = []
        for fi in frame_indices:
            t = fi * self._dt
            trace_updates: list = []

            # 1) Goal markers (unchanged)
            for g in self._goals:
                pos = g["position"]
                trace_updates.append(
                    go.Scatter(x=[float(pos[0])], y=[float(pos[1])])
                )

            # 2) Trajectory traces
            for spec in self._trajectories:
                src = spec["data"] if spec["data"] is not None else self._states
                trace_updates.append(
                    go.Scatter(
                        x=src[:fi, spec["x_idx"]].tolist(),
                        y=src[:fi, spec["y_idx"]].tolist(),
                    )
                )

            # 3) Agent traces (body marker + trail)
            for spec in self._agents:
                src = spec["data"] if spec["data"] is not None else self._states
                px = float(src[fi, spec["x_idx"]])
                py = float(src[fi, spec["y_idx"]])
                # Body
                trace_updates.append(go.Scatter(x=[px], y=[py]))
                # Trail
                trace_updates.append(
                    go.Scatter(
                        x=src[:fi, spec["x_idx"]].tolist(),
                        y=src[:fi, spec["y_idx"]].tolist(),
                    )
                )

            # 4) Prediction traces (bucketed fading)
            for spec in self._predictions:
                px, py = self._compute_prediction(spec, fi)
                n_pts = len(px)
                for bucket in range(_PLOTLY_FADE_BUCKETS):
                    if n_pts >= 2:
                        start = bucket * n_pts // _PLOTLY_FADE_BUCKETS
                        end = (bucket + 1) * n_pts // _PLOTLY_FADE_BUCKETS + 1
                        end = min(end, n_pts)
                        trace_updates.append(
                            go.Scatter(
                                x=list(px[start:end]),
                                y=list(py[start:end]),
                            )
                        )
                    else:
                        trace_updates.append(go.Scatter(x=[], y=[]))

            # --- per-frame layout (time annotation + agent shapes) ---
            layout_update = {}

            if self._show_time:
                layout_update["annotations"] = [
                    dict(
                        x=0.02, y=0.98,
                        xref="paper", yref="paper",
                        text=f"Time: {t:.1f}s",
                        showarrow=False,
                        font=dict(size=13),
                        bgcolor="white",
                        opacity=0.8,
                        bordercolor="gray",
                        borderwidth=1,
                    )
                ]

            # Agent body/zone as moving shapes
            if has_agents:
                frame_shapes = list(static_shapes)  # copy static shapes
                for spec in self._agents:
                    src = spec["data"] if spec["data"] is not None else self._states
                    ax_pos = float(src[fi, spec["x_idx"]])
                    ay_pos = float(src[fi, spec["y_idx"]])
                    # Body circle
                    frame_shapes.append(
                        _circle_shape(
                            ax_pos, ay_pos, spec["body_radius"],
                            spec["body_color"], alpha=spec["body_alpha"],
                        )
                    )
                    # Safety zone
                    if spec["zone_radius"] is not None:
                        frame_shapes.append(
                            _circle_shape(
                                ax_pos, ay_pos, spec["zone_radius"],
                                spec["zone_color"], alpha=spec["zone_alpha"],
                            )
                        )
                layout_update["shapes"] = frame_shapes

            frames.append(
                go.Frame(
                    data=trace_updates,
                    traces=list(range(n_traces)),
                    name=f"{t:.2f}s",
                    layout=layout_update if layout_update else None,
                )
            )

        # --- assemble figure ---
        plot_size = 600
        menus, sliders = _plotly_animation_controls(frames, frame_duration_ms)

        fig = go.Figure(
            data=base_traces,
            frames=frames,
            layout=go.Layout(
                title=dict(
                    text=self._title,
                    x=0.5, xanchor="center",
                    font=dict(size=16),
                ),
                xaxis=dict(
                    title="x [m]",
                    range=list(self._x_lim),
                    showgrid=True,
                    gridcolor="rgba(0,0,0,0.1)",
                    constrain="domain",
                ),
                yaxis=dict(
                    title="y [m]",
                    range=list(self._y_lim),
                    showgrid=True,
                    gridcolor="rgba(0,0,0,0.1)",
                    **(dict(scaleanchor="x", scaleratio=1) if self._aspect == "equal" else {}),
                    constrain="domain",
                ),
                shapes=static_shapes,
                width=plot_size + 90,
                height=plot_size + 160,
                margin=dict(l=60, r=30, t=60, b=100),
                updatemenus=menus,
                sliders=sliders,
                legend=dict(
                    x=1.0, y=1.0,
                    xanchor="right", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                ),
                template="plotly_white",
            ),
        )

        self._fig = fig
        return fig

    def _save_plotly(self, path: str) -> str:
        if self._fig is None:
            self._build_plotly()

        output_path = Path(path).with_suffix(".html")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fig.write_html(str(output_path), auto_open=False)
        abs_path = str(output_path.resolve())
        print(f"\nInteractive animation saved to: file://{abs_path}")
        return abs_path

    def _show_plotly(self):
        if self._fig is None:
            self._build_plotly()
        self._fig.show()
