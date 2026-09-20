"""Encoder failures use a controlled child process, independent of installed FFmpeg."""

import os
import signal
import sys
import time

import numpy as np
import pytest

from cbfkit.systems.mujoco import _showcase_output as output


@pytest.fixture
def encoder(monkeypatch, tmp_path):
    real_popen = output.subprocess.Popen

    def install(body):
        script = tmp_path / "encoder.py"
        script.write_text(body)
        monkeypatch.setattr(output, "ffmpeg_exe", lambda: str(script))
        monkeypatch.setattr(
            output.subprocess,
            "Popen",
            lambda args, **kwargs: real_popen([sys.executable, *args], **kwargs),
        )

    return install


def frame():
    return np.zeros((2, 2, 3), dtype=np.uint8)


def test_stalled_encoder_is_killed_and_partial_output_removed(encoder, tmp_path):
    encoder(
        """import pathlib, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path(sys.argv[-1]).write_bytes(b"partial")
sys.stdin.buffer.read()
time.sleep(60)
"""
    )
    writer = output.FrameWriter(tmp_path / "clip.mp4", 25, timeout=0.3)
    writer.add(frame())
    # Wait for the child to install its signal handler before exercising kill escalation.
    ready_deadline = time.monotonic() + 5
    while not writer.path.exists() and time.monotonic() < ready_deadline:
        time.sleep(0.01)
    assert writer.path.exists()
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        writer.close()
    assert time.monotonic() - start < 5
    assert writer._proc.poll() is not None
    if os.name == "posix":
        assert writer._proc.returncode == -signal.SIGKILL
    assert writer._proc.stdin.closed
    assert writer._stderr is None
    assert not writer.path.exists()


def test_large_stderr_does_not_deadlock_and_failure_cleans_gif_temp(encoder, tmp_path):
    encoder(
        """import pathlib, sys
pathlib.Path(sys.argv[-1]).write_bytes(b"partial")
sys.stderr.write("encoder error " * 100000)
sys.stderr.flush()
sys.stdin.buffer.read()
sys.exit(7)
"""
    )
    writer = output.FrameWriter(tmp_path / "clip.gif", 25, kind="gif", timeout=3)
    writer.add(frame())
    with pytest.raises(RuntimeError, match=r"ffmpeg failed \(7\): encoder error"):
        writer.close()
    assert not writer._tmp.exists()
    assert not writer.path.exists()
    assert writer._proc.poll() == 7


def test_user_exception_survives_abort(encoder, tmp_path):
    encoder(
        """import pathlib, sys, time
pathlib.Path(sys.argv[-1]).write_bytes(b"partial")
sys.stdin.buffer.read()
time.sleep(60)
"""
    )
    writer = output.FrameWriter(tmp_path / "clip.mp4", 25, timeout=0.3)
    with pytest.raises(ValueError, match="original error"):
        with writer:
            writer.add(frame())
            raise ValueError("original error")
    assert writer._proc.poll() is not None
    assert not writer.path.exists()


def test_gif_pass_timeout_removes_both_outputs(encoder, tmp_path):
    encoder(
        """import pathlib, sys, time
pathlib.Path(sys.argv[-1]).write_bytes(b"partial")
if "-vf" in sys.argv:
    time.sleep(60)
else:
    sys.stdin.buffer.read()
"""
    )
    writer = output.FrameWriter(tmp_path / "clip.gif", 25, kind="gif", timeout=0.5)
    writer.add(frame())
    with pytest.raises(RuntimeError, match="GIF pass timed out"):
        writer.close()
    assert not writer.path.exists()
    assert not writer._tmp.exists()


def test_missing_encoder_cleans_temporary_file(monkeypatch, tmp_path):
    monkeypatch.setattr(output, "ffmpeg_exe", lambda: str(tmp_path / "missing"))
    writer = output.FrameWriter(tmp_path / "clip.gif", 25, kind="gif")
    with pytest.raises(FileNotFoundError):
        writer.add(frame())
    assert not writer._tmp.exists()
    assert writer._stderr is None


def test_abort_before_start_preserves_existing_output(tmp_path):
    target = tmp_path / "existing.mp4"
    target.write_bytes(b"valid old output")
    with pytest.raises(ValueError):
        with output.FrameWriter(target, 25):
            raise ValueError("before encoding")
    assert target.read_bytes() == b"valid old output"


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_timeout_must_be_positive_and_finite(tmp_path, timeout):
    with pytest.raises(ValueError, match="timeout"):
        output.FrameWriter(tmp_path / "clip.mp4", 25, timeout=timeout)


def test_broken_pipe_reaps_encoder_and_reports_stderr(encoder, tmp_path):
    encoder(
        """import pathlib, sys
pathlib.Path(sys.argv[-1]).write_bytes(b"partial")
sys.stderr.write("encoder crashed")
sys.exit(9)
"""
    )
    writer = output.FrameWriter(tmp_path / "clip.mp4", 25, timeout=3)
    writer.add(frame())
    writer._proc.wait(timeout=3)
    with pytest.raises(RuntimeError, match="encoder crashed"):
        writer.add(frame())
    assert writer._proc.returncode == 9
    assert writer._stderr is None
    assert not writer.path.exists()


def test_locator_failure_closes_writer_without_removing_old_output(monkeypatch, tmp_path):
    def missing():
        raise RuntimeError("no ffmpeg")

    monkeypatch.setattr(output, "ffmpeg_exe", missing)
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"old output")
    writer = output.FrameWriter(target, 25)
    with pytest.raises(RuntimeError, match="no ffmpeg"):
        writer.add(frame())
    with pytest.raises(RuntimeError, match="closed"):
        writer.add(frame())
    assert target.read_bytes() == b"old output"
