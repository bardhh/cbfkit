"""Internal camera helpers; public API is cbfkit.systems.mujoco.showcase."""

from typing import Optional, Sequence, Union

import mujoco
import numpy as np

Vec = Union[Sequence[float], np.ndarray]


# --------------------------------------------------------------------------- camera
class CameraSchedule:
    """A never-cutting free camera: lagged look-at, fixed framing, slowly drifting azimuth.

    ``update(k, target_xy)`` returns an ``mjCAMERA_FREE`` camera whose ``lookat`` chases
    ``target_xy`` (at z = 0.8) through a first-order lag of time constant ``lag_s``, which
    filters the gait jitter out of the shot. The azimuth drifts at ``drift_deg_per_s``; when
    a ``heading`` is supplied the camera is additionally biased to sit behind-left of the
    walking direction, blended 50/50 with the drift so it never swings fast.

    ``k`` is the *logged step index* (as the markers callback receives it), so the lag is
    integrated over the actual elapsed ``dt * (k - k_prev)`` even when frames are skipped.
    The same ``MjvCamera`` instance is returned every call -- use it before the next update.

    The azimuth is a filtered state like the look-at, rate-limited to
    :data:`MAX_AZIMUTH_RATE_DEG_S`. Without that limit a heading that flips sign (a robot
    sidestepping, or one whose velocity passes through zero) moves the *target* azimuth by
    up to 180 deg and the shot whips around in a single frame.
    """

    LOOKAT_Z = 0.8
    BEHIND_LEFT_DEG = 180.0 - 35.0
    MAX_AZIMUTH_RATE_DEG_S = 30.0

    def __init__(
        self,
        distance: float,
        elevation: float,
        azimuth0: float = 135.0,
        drift_deg_per_s: float = 0.6,
        lag_s: float = 0.4,
        dt: float = 0.02,
    ) -> None:
        self.distance = float(distance)
        self.elevation = float(elevation)
        self.azimuth0 = float(azimuth0)
        self.drift_deg_per_s = float(drift_deg_per_s)
        self.lag_s = float(lag_s)
        self.dt = float(dt)
        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._cam.trackbodyid = -1
        self._lookat = np.array([0.0, 0.0, self.LOOKAT_Z])
        self._azimuth = self.azimuth0
        self._last_k: Optional[int] = None

    @staticmethod
    def _wrap180(deg: float) -> float:
        return float((deg + 180.0) % 360.0 - 180.0)

    def update(self, k: int, target_xy: Vec, heading: Optional[float] = None) -> mujoco.MjvCamera:
        """Camera for logged step ``k`` looking at ``target_xy`` (optionally heading-biased)."""
        target = np.asarray(target_xy, dtype=float).reshape(-1)[:2]
        goal = np.array([target[0], target[1], self.LOOKAT_Z])
        last_k = self._last_k
        first = last_k is None or k <= last_k
        elapsed = 0.0 if last_k is None else self.dt * (k - last_k)
        if first:
            self._lookat = goal  # first frame (or a reset): snap, do not ease in from the origin
        else:
            alpha = 1.0 - float(np.exp(-elapsed / max(self.lag_s, 1e-6)))
            self._lookat = self._lookat + alpha * (goal - self._lookat)

        azimuth = self.azimuth0 + self.drift_deg_per_s * self.dt * float(k)
        if heading is not None:
            behind_left = float(np.degrees(float(heading))) + self.BEHIND_LEFT_DEG
            azimuth = azimuth + 0.5 * self._wrap180(behind_left - azimuth)
        if first:
            self._azimuth = self._wrap180(azimuth)
        else:
            step = self._wrap180(azimuth - self._azimuth)
            limit = self.MAX_AZIMUTH_RATE_DEG_S * elapsed
            self._azimuth = self._wrap180(self._azimuth + max(-limit, min(limit, step)))
        self._last_k = int(k)

        self._cam.lookat[:] = self._lookat
        self._cam.distance = self.distance
        self._cam.elevation = self.elevation
        self._cam.azimuth = self._azimuth
        return self._cam
