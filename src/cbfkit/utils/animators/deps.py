"""Optional dependency checks for animation backends."""

try:
    import matplotlib.animation as mpl_animation  # noqa: F401
    import matplotlib.pyplot as plt  # noqa: F401
    from matplotlib.collections import LineCollection  # noqa: F401
    from matplotlib.patches import Circle, Ellipse  # noqa: F401

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go  # noqa: F401

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

try:
    import manim  # noqa: F401

    _HAS_MANIM = True
except ImportError:
    _HAS_MANIM = False


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
