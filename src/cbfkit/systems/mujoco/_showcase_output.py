"""Internal output helpers; public API is cbfkit.systems.mujoco.showcase."""

import math
import os
import shutil
import subprocess
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import BinaryIO, List, Optional, Sequence, Tuple, Union

import numpy as np

Vec = Union[Sequence[float], np.ndarray]


# --------------------------------------------------------------------------- compositing
def _dejavu_candidates(name: str) -> List[Path]:
    """Where a DejaVu ``.ttf`` might live: matplotlib's bundled copy first, then the system.

    ``find_spec`` locates matplotlib's data directory *without importing* matplotlib, so
    this module keeps matplotlib out of its import path (see the module docstring).
    """
    candidates: List[Path] = []
    try:
        spec = find_spec("matplotlib")
    except (ImportError, ValueError):  # pragma: no cover - defensive
        spec = None
    if spec is not None and spec.origin:
        candidates.append(Path(spec.origin).parent / "mpl-data" / "fonts" / "ttf" / name)
    return candidates + [
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
        Path("/opt/homebrew/share/fonts") / name,
        Path("/Library/Fonts") / name,
    ]


def _label_font(size: int):
    """DejaVu Sans Bold at ``size`` px for PIL, falling back to PIL's built-in bitmap font."""
    from PIL import ImageFont

    for cand in _dejavu_candidates("DejaVuSans-Bold.ttf"):
        if cand.is_file():
            return ImageFont.truetype(str(cand), size)
    return ImageFont.load_default()


def compose_panels(
    panels: List[np.ndarray],
    labels: List[str],
    hud: Optional[np.ndarray],
    gap: int = 8,
    bg: Tuple[int, int, int] = (18, 18, 20),
    captions: Optional[Sequence[Optional[str]]] = None,
) -> np.ndarray:
    """Lay panels out side by side with a label pill each, and the HUD strip underneath.

    Panels are concatenated horizontally with ``gap`` pixels of ``bg`` between them; the
    HUD (any width) goes below, and both rows are centred on the widest of the two.
    ``captions`` optionally adds a second, smaller pill under a panel's label (``None``
    for a panel that has none) -- for example to say that one panel is frozen on its last
    frame while the other runs on. Returns an ``(H, W, 3)`` uint8 array.
    """
    from PIL import Image, ImageDraw

    if not panels:
        raise ValueError("compose_panels needs at least one panel")
    if len(labels) != len(panels):
        raise ValueError(f"got {len(panels)} panels but {len(labels)} labels")
    frames = [np.asarray(p, dtype=np.uint8) for p in panels]
    row_w = sum(f.shape[1] for f in frames) + gap * (len(frames) - 1)
    row_h = max(f.shape[0] for f in frames)
    hud_arr = None if hud is None else np.asarray(hud, dtype=np.uint8)
    hud_w = 0 if hud_arr is None else hud_arr.shape[1]
    hud_h = 0 if hud_arr is None else hud_arr.shape[0]
    out_w = max(row_w, hud_w)
    out_h = row_h + (gap + hud_h if hud_arr is not None else 0)

    canvas = Image.new("RGB", (out_w, out_h), tuple(int(c) for c in bg))
    x = (out_w - row_w) // 2
    boxes = []
    for frame in frames:
        canvas.paste(Image.fromarray(frame), (x, 0))
        boxes.append((x, frame.shape[1]))
        x += frame.shape[1] + gap
    if hud_arr is not None:
        canvas.paste(Image.fromarray(hud_arr), ((out_w - hud_w) // 2, row_h + gap))

    size = max(14, int(round(row_h / 26.0)))
    font = _label_font(size)
    small_size = max(11, int(round(size * 0.72)))
    small_font = _label_font(small_size)
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    pad = max(6, size // 2)
    caption_list = list(captions) if captions is not None else [None] * len(boxes)
    if len(caption_list) != len(boxes):
        raise ValueError(f"got {len(boxes)} panels but {len(caption_list)} captions")

    def pill(x0: int, y0: int, text: str, use_font, fill) -> int:
        """Draw one rounded label at ``(x0, y0)``; returns the bottom edge."""
        try:
            left, top, right, bottom = draw.textbbox((x0, y0), text, font=use_font)
        except AttributeError:  # pragma: no cover - very old Pillow
            right, bottom = x0 + size * len(text) // 2, y0 + size
            left, top = x0, y0
        draw.rounded_rectangle(
            (left - pad, top - pad // 2, right + pad, bottom + pad // 2),
            radius=pad,
            fill=(18, 20, 24, 200),
        )
        draw.text((x0, y0), text, font=use_font, fill=fill)
        return int(bottom)

    for (px, _), text, caption in zip(boxes, labels, caption_list):
        x0, y0 = px + pad + 2, pad + 2
        bottom = pill(x0, y0, text, font, (232, 234, 238, 255))
        if caption:
            pill(x0, bottom + pad + pad // 2, caption, small_font, (200, 205, 212, 255))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    return np.asarray(canvas, dtype=np.uint8)


# --------------------------------------------------------------------------- writers
def ffmpeg_exe() -> str:
    """Path to an ffmpeg binary: ``imageio_ffmpeg``'s bundled one, else one on ``PATH``."""
    try:
        import imageio_ffmpeg  # type: ignore[import-untyped]

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # noqa: BLE001 - any import/lookup failure falls back to PATH
        found = shutil.which("ffmpeg")
        if found:
            return found
        raise RuntimeError(
            "no ffmpeg available: install imageio-ffmpeg (pip install imageio-ffmpeg) "
            "or put ffmpeg on PATH"
        ) from None


class FrameWriter:
    """Stream RGB frames to an MP4 or a palette-optimised GIF via ffmpeg.

    ``fps`` is the rate the added frames represent, so for a 0.02 s simulation rendered
    every other step you pass ``fps=25`` for a real-time MP4. ``speed`` only applies to
    GIFs: the frames are time-compressed by ``speed`` and the GIF is written at
    ``fps * speed``, i.e. ``FrameWriter(p, 10, kind="gif", speed=2.0)`` gives a 20 fps GIF
    playing 2x faster than real time at 10 frames per simulated second. ``gif_fps`` overrides
    that output rate on its own (the playback speed is unchanged), so the same added frames
    can be resampled to a smaller file.

    Frames must all be the same size; an odd width or height is padded by one pixel
    because ``yuv420p`` needs even dimensions. ``timeout`` bounds encoder shutdown
    and GIF conversion (seconds); it does not limit the duration of a streaming session.
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        fps: float,
        *,
        kind: str = "mp4",
        speed: float = 1.0,
        gif_width: int = 480,
        gif_colors: int = 64,
        gif_fps: Optional[float] = None,
        timeout: float = 30.0,
    ) -> None:
        if kind not in ("mp4", "gif"):
            raise ValueError(f"kind must be 'mp4' or 'gif', got {kind!r}")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.timeout = timeout
        self.path = Path(path)
        self.fps = float(fps)
        self.kind = kind
        self.speed = float(speed)
        self.gif_width = int(gif_width)
        self.gif_colors = int(gif_colors)
        self.gif_fps = None if gif_fps is None else float(gif_fps)
        self.n_frames = 0
        self._size: Optional[Tuple[int, int]] = None
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._stderr: Optional[BinaryIO] = None
        self._tmp: Optional[Path] = None
        self._closed = False

    # -- ffmpeg plumbing ---------------------------------------------------
    def _target(self) -> Path:
        if self.kind == "mp4":
            return self.path
        if self._tmp is None:
            fd, name = tempfile.mkstemp(prefix="cbfkit_showcase_", suffix=".mp4")
            os.close(fd)
            self._tmp = Path(name)
        return self._tmp

    def _start(self, width: int, height: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        crf = "18" if self.kind == "mp4" else "12"  # the GIF intermediate stays near-lossless
        cmd = [
            ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-framerate",
            f"{self.fps:g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            crf,
            "-pix_fmt",
            "yuv420p",
            "-r",
            f"{self.fps:g}",
            str(self._target()),
        ]
        # A file cannot fill a pipe and deadlock the encoder while we write frames.
        self._stderr = tempfile.TemporaryFile()
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stderr=self._stderr, bufsize=0
            )
        except OSError:
            self._stderr.close()
            self._stderr = None
            self._remove_partial()
            self._closed = True
            raise

    def _gif_pass(self, src: Path) -> None:
        # The time-compressed stream runs at ``fps * speed``; ``gif_fps`` resamples that to a
        # different output rate (dropping or duplicating frames) without changing the speed.
        out_fps = self.fps * self.speed if self.gif_fps is None else self.gif_fps
        vf = (
            f"setpts=(PTS-STARTPTS)/{self.speed:g},fps={out_fps:g},"
            f"scale={self.gif_width}:-1:flags=lanczos,split[a][b];"
            f"[a]palettegen=max_colors={self.gif_colors}:stats_mode=diff[p];"
            f"[b][p]paletteuse=dither=none:diff_mode=rectangle"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [ffmpeg_exe(), "-y", "-v", "error", "-i", str(src), "-vf", vf, str(self.path)],
                check=True,
                capture_output=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"ffmpeg GIF pass timed out for {self.path}") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or b"").decode(errors="replace")[:2000]
            raise RuntimeError(
                f"ffmpeg's GIF palette pass failed ({exc.returncode}) for {self.path}: {detail}"
            ) from exc

    # -- public API --------------------------------------------------------
    def add(self, frame_rgb_uint8: np.ndarray) -> None:
        """Append one ``(H, W, 3)`` uint8 RGB frame."""
        if self._closed:
            raise RuntimeError("FrameWriter is closed")
        frame = np.ascontiguousarray(np.asarray(frame_rgb_uint8, dtype=np.uint8))
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) RGB frame, got shape {frame.shape}")
        if frame.shape[0] % 2 or frame.shape[1] % 2:
            pad_h, pad_w = frame.shape[0] % 2, frame.shape[1] % 2
            frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)))
        h, w = frame.shape[:2]
        if self._size is None:
            self._size = (w, h)
            try:
                self._start(w, h)
            except BaseException:
                self._abort()
                raise
        elif self._size != (w, h):
            raise ValueError(f"frame size changed from {self._size} to {(w, h)}")
        assert self._proc is not None and self._proc.stdin is not None
        try:
            # Unbuffered writes can be short; send the complete frame.
            remaining = memoryview(frame.tobytes())
            while remaining:
                written = self._proc.stdin.write(remaining)
                if not written:
                    raise BrokenPipeError("encoder stopped accepting frames")
                remaining = remaining[written:]
        except BrokenPipeError as exc:
            # ffmpeg died mid-stream; its stderr says why, and without it the traceback is
            # just "broken pipe" on frame N.
            self._abort()
            raise RuntimeError(
                f"ffmpeg exited while writing {self.path}: {self._error_detail}"
            ) from exc
        self.n_frames += 1

    def _diagnostics(self) -> str:
        if self._stderr is None:
            return ""
        self._stderr.seek(0)
        return self._stderr.read(2000).decode(errors="replace")

    def _stop(self) -> None:
        """Reap the child, escalating from terminate to kill with bounded waits."""
        if self._proc is None or self._proc.poll() is not None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=self.timeout)

    def _release(self) -> None:
        if self._proc is not None and self._proc.stdin is not None:
            self._proc.stdin.close()
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
        if self._tmp is not None:
            self._tmp.unlink(missing_ok=True)

    def _remove_partial(self) -> None:
        if self._tmp is not None:
            self._tmp.unlink(missing_ok=True)
        # Do not remove a pre-existing target if no encoder was ever started.
        if self._proc is not None:
            self.path.unlink(missing_ok=True)

    def _abort(self) -> None:
        self._closed = True
        try:
            self._stop()
        finally:
            self._error_detail = self._diagnostics()
            try:
                self._remove_partial()
            finally:
                self._release()

    def close(self) -> str:
        """Finish encoding; ``timeout`` bounds shutdown and the GIF palette pass."""
        if self._closed:
            return str(self.path)
        self._closed = True
        try:
            if self._proc is None:
                raise RuntimeError(f"no frames were added, nothing to write to {self.path}")
            assert self._proc.stdin is not None
            self._proc.stdin.close()
            try:
                code = self._proc.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"ffmpeg timed out while closing {self.path}") from exc
            if code != 0:
                raise RuntimeError(f"ffmpeg failed ({code}): {self._diagnostics()}")
            if self.kind == "gif":
                assert self._tmp is not None
                self._gif_pass(self._tmp)
        except BaseException:
            self._abort()
            raise
        finally:
            self._release()
        return str(self.path)

    def __enter__(self) -> "FrameWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            # Preserve the user's original error even if process cleanup also fails.
            try:
                self._abort()
            except (OSError, subprocess.TimeoutExpired):
                pass
        else:
            self.close()
