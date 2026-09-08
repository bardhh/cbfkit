"""Prevent an Isaac Lab version change from reversing physical CBF mappings."""

import math
import runpy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
quaternion = runpy.run_path(
    str(Path(__file__).parents[2] / "examples/isaac_lab/isaac_quaternion.py")
)
rotation = quaternion["quaternion_yaw_cos_sin"]


@pytest.mark.parametrize("order", ["wxyz", "xyzw"])
@pytest.mark.parametrize("angle", [0.0, math.pi / 2, -math.pi / 2])
def test_forward_body_command_rotates_to_expected_world_direction(order, angle):
    w, z = math.cos(angle / 2), math.sin(angle / 2)
    values = [w, 0, 0, z] if order == "wxyz" else [0, 0, z, w]
    c, s = rotation(torch.tensor([values], dtype=torch.float64), order)
    # A unit forward body command must face east/north/south respectively.
    torch.testing.assert_close(c, torch.tensor([math.cos(angle)], dtype=torch.float64))
    torch.testing.assert_close(s, torch.tensor([math.sin(angle)], dtype=torch.float64))


def test_unknown_convention_is_rejected():
    with pytest.raises(ValueError, match="Quaternion order"):
        rotation(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), "unknown")


def test_startup_guard_rejects_wxyz_before_pose_writes():
    check = quaternion["assert_xyzw_identity"]
    check(torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]]))
    with pytest.raises(RuntimeError, match="xyzw identity"):
        check(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
