"""Internal hud helpers; public API is cbfkit.systems.mujoco.showcase."""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

Vec = Union[Sequence[float], np.ndarray]

from ._showcase_geometry import _rgb, h_rgba

# --------------------------------------------------------------------------- HUD
# The band is rgb(24, 26, 30) at alpha 0.85 over mid-gray, pre-composited because the HUD
# is concatenated below the 3-D view rather than blended into it.
_BAND_RGB = tuple(0.85 * np.array([24.0, 26.0, 30.0]) / 255.0 + 0.15 * 0.5)
_FONT = "DejaVu Sans"
_TEXT = "#e8eaee"
_DIM = "#9aa0a8"
_GRID = "#4a4f57"
_PILL_BG = "#2d3138"
_HUD_WINDOW_S = 6.0
_INTERVENTION_FULL_SCALE = 0.5  # m/s at the right end of the bar
_MAX_PILLS = 5
_PILL_X0, _PILL_X1 = 0.705, 0.995  # the right block's horizontal extent
_PILL_Y_SINGLE = 0.17  # one row of pills sits at the bottom of the band
_PILL_Y_ROWS = (0.40, 0.13)  # two rows: below the intervention bar
_MIN_PILL_SCALE = 0.5  # pills shrink to this fraction of the base size before giving up
_HUD_CACHE_MAX = 2  # figures are expensive; keep the working set, close the rest


class HudRenderer:
    """Cached matplotlib HUD strip: figure and artists are built once and updated per frame.

    ``draw`` returns an ``(height, width, 3)`` uint8 array. Layout: a left text block
    (clock, mode, ``h_min``), a centre 6 s scrolling ``h_min`` plot with the zero line and
    a playhead, and a right block with the intervention bar and status pills.
    """

    def __init__(self, width: int, height: int) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.patches import Rectangle

        self.width = int(width)
        self.height = int(height)
        scale = self.width / 1280.0
        self._big = 22.0 * scale  # >= 30 px em at width 1280
        self._mid = 16.0 * scale  # >= 22 px em at width 1280
        self._small = 13.0 * scale

        self._fig = plt.figure(figsize=(self.width / 100.0, self.height / 100.0), dpi=100)
        self._fig.patch.set_facecolor(_BAND_RGB)

        self._t_text = self._fig.text(
            0.016, 0.70, "", color=_TEXT, fontsize=self._big, va="center", fontname=_FONT
        )
        self._mode_text = self._fig.text(
            0.016, 0.44, "", color=_DIM, fontsize=self._mid, va="center", fontname=_FONT
        )
        self._h_text = self._fig.text(
            0.016, 0.17, "", color=_TEXT, fontsize=self._big, va="center", fontname=_FONT
        )

        self._ax = self._fig.add_axes((0.27, 0.20, 0.40, 0.60))
        self._ax.set_facecolor("#1b1e23")
        for side in ("top", "right"):
            self._ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            self._ax.spines[side].set_color(_GRID)
        self._ax.tick_params(colors=_DIM, labelsize=self._small, length=3, pad=2)
        self._ax.set_xticks([])
        self._zero = self._ax.axhline(0.0, color="#d8dde4", lw=1.2, ls="--", alpha=0.8)
        # The second trace is the comparison run (unfiltered) in the side-by-side renders;
        # it stays empty, and out of the y-limits, unless ``draw`` is given ``h_hist2``.
        (self._trace2,) = self._ax.plot([], [], color="#ff6b6b", lw=1.6, ls="--", alpha=0.95)
        (self._trace,) = self._ax.plot([], [], color="#7fd4ff", lw=2.0)
        (self._playhead,) = self._ax.plot([], [], color=_TEXT, lw=1.4, alpha=0.9)
        (self._dot,) = self._ax.plot([], [], "o", ms=6.0, color=_TEXT)
        # A title rather than an in-axes label: the trace would otherwise run through it.
        self._title = f"h_min, last {int(_HUD_WINDOW_S)} s"
        self._ax.set_title(
            self._title,
            color=_DIM,
            fontsize=self._small,
            loc="left",
            pad=3.0,
            fontname=_FONT,
        )
        self._fill: Optional[Any] = None

        self._bar_caption = self._fig.text(
            _PILL_X0, 0.85, "", color=_DIM, fontsize=self._mid, va="center", fontname=_FONT
        )
        self._bar_ax = self._fig.add_axes((_PILL_X0, 0.58, 0.28, 0.15))
        self._bar_ax.set_xlim(0.0, _INTERVENTION_FULL_SCALE)
        self._bar_ax.set_ylim(0.0, 1.0)
        self._bar_ax.axis("off")
        self._bar_full_scale = _INTERVENTION_FULL_SCALE
        self._bar_bg = Rectangle((0.0, 0.0), _INTERVENTION_FULL_SCALE, 1.0, facecolor=_PILL_BG)
        self._bar_ax.add_patch(self._bar_bg)
        self._bar = Rectangle((0.0, 0.0), 0.0, 1.0, facecolor="#7fd4ff")
        self._bar_ax.add_patch(self._bar)

        self._pills = [
            self._fig.text(
                _PILL_X0,
                _PILL_Y_SINGLE,
                "",
                color=_TEXT,
                fontsize=self._mid,
                va="center",
                ha="left",
                bbox={"boxstyle": "round,pad=0.32", "facecolor": _PILL_BG, "edgecolor": "none"},
                fontname=_FONT,
            )
            for _ in range(_MAX_PILLS)
        ]
        for pill in self._pills:
            pill.set_visible(False)

    # -- layout helpers ----------------------------------------------------
    def _pill_width_frac(self, text: str, size: float) -> float:
        """Estimated pill width as a fraction of the figure (DejaVu Sans caps, + padding)."""
        em_px = size * 100.0 / 72.0
        return (0.62 * em_px * len(text) + 0.9 * em_px) / self.width

    def _pack_pills(self, entries: List[Tuple[str, str]], size: float) -> List[List[float]]:
        """Greedy row packing: x offsets per row for ``entries`` at font ``size``."""
        rows: List[List[float]] = [[]]
        x = _PILL_X0
        for text, _ in entries:
            w = self._pill_width_frac(text, size)
            if x + w > _PILL_X1 and rows[-1]:
                rows.append([])
                x = _PILL_X0
            rows[-1].append(x)
            x += w + 0.008
        return rows

    def _set_pills(self, entries: List[Tuple[str, str]]) -> None:
        """Lay the pills out in one or two rows, shrinking the font until they all fit.

        A pill is a status flag -- "QP not converged", a slack magnitude -- so dropping one
        silently would hide exactly the frames a viewer is looking for. This shrinks instead,
        and raises if even the smallest size cannot hold them.
        """
        if len(entries) > _MAX_PILLS:
            raise ValueError(
                f"{len(entries)} HUD pills requested but only {_MAX_PILLS} slots exist: "
                f"{[text for text, _ in entries]}"
            )
        size = self._mid
        rows = self._pack_pills(entries, size)
        while len(rows) > 2 and size > _MIN_PILL_SCALE * self._mid:
            size *= 0.9
            rows = self._pack_pills(entries, size)
        if len(rows) > 2:
            raise ValueError(
                f"HUD pills {[text for text, _ in entries]} do not fit in two rows even at "
                f"{_MIN_PILL_SCALE:.0%} of the base font size"
            )
        y_rows = [_PILL_Y_SINGLE] if len(rows) < 2 else _PILL_Y_ROWS
        offsets = [(x, y_rows[r]) for r, row in enumerate(rows) for x in row]
        for pill, (text, colour), (x, y) in zip(self._pills, entries, offsets):
            pill.set_position((x, y))
            pill.set_text(text)
            pill.set_fontsize(size)
            pill.set_color("#14161a")
            bbox = pill.get_bbox_patch()
            if bbox is not None:
                bbox.set_facecolor(colour)
            pill.set_visible(True)
        for pill in self._pills[len(offsets) :]:
            pill.set_visible(False)

    # -- drawing -----------------------------------------------------------
    def draw(
        self,
        t: float,
        h_min: float,
        h_hist: Optional[np.ndarray],
        t_hist: Optional[np.ndarray],
        intervention: float,
        active: bool,
        flags: Dict[str, bool],
        slack: Optional[float] = None,
        mode: str = "CBF-QP",
        h_hist2: Optional[np.ndarray] = None,
        intervention_caption: Optional[str] = None,
        ylim: Optional[Tuple[float, float]] = None,
        h_label: str = "h_min",
        full_scale: Optional[float] = None,
    ) -> np.ndarray:
        """Render one HUD frame; see :func:`hud_strip` for the argument meanings."""
        colour = h_rgba(h_min, 1.0)[:3]
        self._t_text.set_text(f"t = {float(t):5.1f} s")
        self._mode_text.set_text(str(mode))
        self._h_text.set_text(f"h min = {float(h_min):+.2f}")
        self._h_text.set_color(_rgb(colour))

        if self._fill is not None:
            self._fill.remove()
            self._fill = None
        t_lo, t_hi = float(t) - _HUD_WINDOW_S, float(t)
        if h_hist is not None and t_hist is not None and len(np.asarray(t_hist)) > 0:
            t_all = np.asarray(t_hist, dtype=float).reshape(-1)
            h_all = np.asarray(h_hist, dtype=float).reshape(-1)
            keep = (t_all >= t_lo) & (t_all <= t_hi + 1e-9)
            th, hh = t_all[keep], h_all[keep]
            self._trace.set_data(th, hh)
            if th.size:
                below = [bool(v) for v in hh < 0.0]
                self._fill = self._ax.fill_between(
                    th, hh, 0.0, where=below, color="#d94b56", alpha=0.35, interpolate=True
                )
                lo = min(float(hh.min()) - 0.05, -0.1)
                hi = max(float(hh.max()) + 0.05, 0.4)
            else:
                lo, hi = -0.1, 0.4
        else:
            self._trace.set_data([], [])
            lo, hi = -0.1, 0.4

        # The comparison trace shares ``t_hist``: both runs are logged on the same clock, so a
        # shorter ``h_hist2`` (a comparison run that ended earlier) is simply truncated to fit.
        if h_hist2 is not None and t_hist is not None:
            h_all2 = np.asarray(h_hist2, dtype=float).reshape(-1)
            t_all2 = np.asarray(t_hist, dtype=float).reshape(-1)
            n2 = min(h_all2.size, t_all2.size)
            t2, h2 = t_all2[:n2], h_all2[:n2]
            keep2 = (t2 >= t_lo) & (t2 <= t_hi + 1e-9)
            self._trace2.set_data(t2[keep2], h2[keep2])
            if keep2.any():
                lo = min(lo, float(h2[keep2].min()) - 0.05)
                hi = max(hi, float(h2[keep2].max()) + 0.05)
        else:
            self._trace2.set_data([], [])
        title = f"{h_label}, last {int(_HUD_WINDOW_S)} s" + (
            "   (red dashed: no filter)" if h_hist2 is not None else ""
        )
        if title != self._title:
            self._title = title
            self._ax.set_title(
                title, color=_DIM, fontsize=self._small, loc="left", pad=3.0, fontname=_FONT
            )

        # A caller that knows the whole clip pins the range once: autoscaling per frame makes
        # the trace jump every time the 6 s window slides past a peak.
        if ylim is not None:
            lo, hi = float(ylim[0]), float(ylim[1])
        self._ax.set_xlim(t_lo, t_hi)
        self._ax.set_ylim(lo, hi)
        # the lower tick is dropped when it would crowd the zero line's label
        ticks = [0.0, round(hi, 2)]
        if (0.0 - lo) / max(hi - lo, 1e-9) > 0.18:
            ticks.insert(0, round(lo, 2))
        self._ax.set_yticks(ticks)
        self._playhead.set_data([t_hi, t_hi], [lo, hi])
        self._dot.set_data([t_hi], [float(h_min)])
        self._dot.set_color(_rgb(colour))

        iv = float(intervention)
        self._bar_caption.set_text(
            f"intervention  |dv| = {iv:.2f} m/s"
            if intervention_caption is None
            else str(intervention_caption)
        )
        scale = _INTERVENTION_FULL_SCALE if full_scale is None else max(float(full_scale), 1e-9)
        if scale != self._bar_full_scale:
            self._bar_full_scale = scale
            self._bar_ax.set_xlim(0.0, scale)
            self._bar_bg.set_width(scale)
        self._bar.set_width(min(max(iv, 0.0), scale))
        self._bar.set_facecolor("#7fd4ff" if active else _GRID)

        entries: List[Tuple[str, str]] = []
        if active:
            entries.append(("CBF ACTIVE", "#3fc46a"))
        for name, on in (flags or {}).items():
            if on:
                entries.append((str(name).upper(), "#f0a830"))
        if slack is not None:
            hot = float(slack) > 1e-3
            entries.append((f"SLACK {float(slack):.3f}", "#d94b56" if hot else _DIM))
        self._set_pills(entries)

        self._fig.canvas.draw()
        buf = np.asarray(self._fig.canvas.buffer_rgba())  # type: ignore[attr-defined]
        return np.ascontiguousarray(buf[:, :, :3]).copy()

    def close(self) -> None:
        from matplotlib import pyplot as plt

        plt.close(self._fig)


_HUD_CACHE: "OrderedDict[Tuple[int, int], HudRenderer]" = OrderedDict()


def hud_strip(
    width: int,
    height: int,
    t: float,
    h_min: float,
    h_hist: Optional[np.ndarray],
    t_hist: Optional[np.ndarray],
    intervention: float,
    active: bool,
    flags: Dict[str, bool],
    slack: Optional[float] = None,
    mode: str = "CBF-QP",
    h_hist2: Optional[np.ndarray] = None,
    intervention_caption: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    h_label: str = "h_min",
    full_scale: Optional[float] = None,
) -> np.ndarray:
    """One HUD strip as an ``(height, width, 3)`` uint8 array.

    ``t`` is the current time and ``h_min`` the current minimum barrier value (it colours
    the numeric readout and the plot's playhead dot). ``h_hist``/``t_hist`` are the
    history arrays, windowed here to the last 6 s. ``intervention`` is
    ``|v_safe - v_nom|`` in m/s, drawn as a bar of full scale ``full_scale`` (0.5 by default;
    pass the control bound when the filtered variable is an acceleration rather than a
    velocity, or the bar pegs through every braking manoeuvre). ``active`` lights the "CBF ACTIVE"
    pill, ``flags`` adds one amber pill per true entry (short upper-case keys fit best,
    e.g. ``{"mppi fallback": False, "qp fail": True}``), and ``slack`` adds a slack pill
    for the relaxed-QP runs. ``mode`` is the small label under the clock.

    ``h_hist2`` overlays a second ``h_min`` history (the unfiltered comparison run) as a red
    dashed trace on the same clock as ``t_hist``, and ``intervention_caption`` replaces the
    bar's caption for runs where the filtered variable is not a velocity. ``ylim`` pins the
    plot's vertical range instead of autoscaling it to the visible 6 s, and ``h_label`` names
    the quantity plotted (for a caller that plots a reparametrised barrier).

    Figures are cached per ``(width, height)`` because this runs per frame; the cache holds
    the most recent few and closes the rest.
    """
    key = (int(width), int(height))
    renderer = _HUD_CACHE.pop(key, None)
    if renderer is None:
        renderer = HudRenderer(*key)
    _HUD_CACHE[key] = renderer  # re-inserted last: the dict doubles as an LRU order
    while len(_HUD_CACHE) > _HUD_CACHE_MAX:
        _, evicted = _HUD_CACHE.popitem(last=False)
        evicted.close()
    return renderer.draw(
        t,
        h_min,
        h_hist,
        t_hist,
        intervention,
        active,
        flags,
        slack,
        mode,
        h_hist2,
        intervention_caption,
        ylim,
        h_label,
    )
