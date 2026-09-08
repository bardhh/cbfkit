"""Planar command rotation with explicit simulator quaternion conventions."""

IDENTITY_XYZW = (0.0, 0.0, 0.0, 1.0)


def assert_xyzw_identity(quaternions):
    """Fail before actuation if the configured identity pose uses another order.

    The warehouse initializes both assets without rotation. Checking their
    default root states catches a Lab convention change, including wxyz, before
    writing a cart pose or converting any robot command.
    """
    import torch

    expected = torch.tensor(IDENTITY_XYZW, device=quaternions.device, dtype=quaternions.dtype)
    if not torch.allclose(quaternions.abs(), expected.expand_as(quaternions), atol=1e-6):
        raise RuntimeError("Warehouse requires Isaac Lab xyzw identity root poses")


def quaternion_yaw_cos_sin(q, order):
    """Return planar rotation for Lab 2 (wxyz) or Lab 3 (xyzw) quaternions."""
    import torch

    if order == "xyzw":
        x, y, z, w = (q[:, i] for i in range(4))
    elif order == "wxyz":
        w, x, y, z = (q[:, i] for i in range(4))
    else:
        raise ValueError("Quaternion order must be wxyz or xyzw")
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return torch.cos(yaw), torch.sin(yaw)
