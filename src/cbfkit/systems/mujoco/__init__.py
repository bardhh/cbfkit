"""MuJoCo/MJX plant backend for CBFKit.

Optional extra: ``pip install cbfkit[mujoco]``. Not imported by
``cbfkit.systems`` so the core package stays importable without MuJoCo.
"""

from pathlib import Path

try:
    import mujoco
    from mujoco import mjx  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "cbfkit.systems.mujoco requires MuJoCo and MJX. " "Install with: pip install cbfkit[mujoco]"
    ) from exc

from .plant import MujocoPlant  # noqa: E402

MODELS_DIR = Path(__file__).parent / "models"


def load_model(name: str) -> mujoco.MjModel:
    """Load a vendored MJCF scene by directory name, e.g. ``load_model("cart_pole")``."""
    path = MODELS_DIR / name / "scene.xml"
    if not path.exists():
        raise FileNotFoundError(f"No vendored MuJoCo model named {name!r} at {path}")
    return mujoco.MjModel.from_xml_path(str(path))


__all__ = ["MODELS_DIR", "MujocoPlant", "load_model", "mujoco"]
