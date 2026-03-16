"""Animation framework for CBFKit simulations.

Provides :class:`CBFAnimator` for building 2D trajectory animations with a
declarative API, :class:`AnimationConfig` for centralised defaults, and
:func:`save_animation` for robust file output with format fallback.

Supports three backends:

* ``"matplotlib"`` — generates MP4 / GIF animations via ``FuncAnimation``.
* ``"plotly"`` (default) — generates interactive HTML files with play/pause
  controls and a timeline slider.  Requires ``pip install cbfkit[plotly]``.
* ``"manim"`` — high-quality 3D animations (MP4) via Manim.  Currently only
  available for 3D multi-robot scenes via :func:`visualize_3d_multi_robot`.
  Requires ``pip install cbfkit[manim]``.
"""

from .animator import CBFAnimator
from .config import AnimationConfig, DEFAULT_CONFIG
from .deps import (
    _HAS_MANIM,
    _HAS_MATPLOTLIB,
    _HAS_PLOTLY,
    _require_manim,
    _require_matplotlib,
    _require_plotly,
)
from .helpers import save_animation

__all__ = [
    "AnimationConfig",
    "CBFAnimator",
    "DEFAULT_CONFIG",
    "save_animation",
    "_HAS_MATPLOTLIB",
    "_HAS_PLOTLY",
    "_HAS_MANIM",
    "_require_matplotlib",
    "_require_plotly",
    "_require_manim",
]
