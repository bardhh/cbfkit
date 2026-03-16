"""Animation configuration defaults."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class AnimationConfig:
    """Centralised animation parameter defaults.

    Attributes
    ----------
    fps : int
        Frames per second for saved files.
    dpi : int
        Resolution (dots per inch) for saved files.
    interval : int
        Milliseconds between frames for interactive display.
    bitrate : int
        Bitrate for MP4 output (kbps).
    figsize : tuple of float
        Default figure size ``(width, height)`` in inches.
    grid_alpha : float
        Transparency of the background grid.
    blit : bool
        Whether to use blitting for faster rendering.
    plotly_max_frames : int
        Maximum number of frames for the Plotly backend.  Long simulations
        are downsampled to this count to keep the HTML file responsive.
    """

    fps: int = 20
    dpi: int = 100
    interval: int = 50
    bitrate: int = 1800
    figsize: Tuple[float, float] = (10, 8)
    grid_alpha: float = 0.3
    blit: bool = True
    plotly_max_frames: int = 200


DEFAULT_CONFIG = AnimationConfig()
"""Module-level default :class:`AnimationConfig` instance."""
