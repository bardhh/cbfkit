"""MJX sampling-based MPC (MPPI over spline knots) for ``MujocoPlant``.

Optional extra: ``pip install cbfkit[mujoco]``. Independent of
``cbfkit.controllers.mppi`` -- that module rolls out control-affine ODEs;
this one rolls out ``mjx.step``.
"""

try:
    from mujoco import mjx  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "cbfkit.controllers.mjx_sampling_mpc requires MuJoCo and MJX. "
        "Install with: pip install cbfkit[mujoco]"
    ) from exc

from .sampling_mpc import MpcState, SamplingMpc  # noqa: E402
from .spline import get_interp_func, interp_linear, interp_zero  # noqa: E402

__all__ = ["MpcState", "SamplingMpc", "get_interp_func", "interp_linear", "interp_zero"]
