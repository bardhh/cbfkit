"""Backward-compatible shim — all functionality lives in :mod:`cbfkit.utils.animators`."""

from cbfkit.utils.animators import (  # noqa: F401
    _HAS_MANIM,
    _HAS_MATPLOTLIB,
    _HAS_PLOTLY,
    DEFAULT_CONFIG,
    AnimationConfig,
    CBFAnimator,
    save_animation,
)


# Define _require_* locally so monkeypatching this module's _HAS_* flags
# (as existing tests do) continues to work correctly.
def _require_matplotlib():
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "Optional dependency 'matplotlib' not found. "
            "Please install cbfkit[vis] to use visualization features."
        )


def _require_plotly():
    if not _HAS_PLOTLY:
        raise ImportError(
            "Optional dependency 'plotly' not found. "
            "Please install cbfkit[plotly] to use the Plotly backend."
        )


def _require_manim():
    if not _HAS_MANIM:
        raise ImportError(
            "Optional dependency 'manim' not found. "
            "Please install cbfkit[manim] to use the Manim backend."
        )
