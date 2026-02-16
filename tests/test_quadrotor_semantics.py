
import jax.numpy as jnp
import pytest
from cbfkit.systems.quadrotor_6dof.models.quadrotor_6dof_dynamics import rotation_body_to_inertial

def test_quadrotor_rotation_semantics():
    """
    Verifies that the rotation matrix generator treats:
    phi -> Roll (rotation about X)
    theta -> Pitch (rotation about Y)
    psi -> Yaw (rotation about Z)

    This contradicts the incorrect docstring which claimed "yaw, pitch, and roll" order.
    """

    # Test 1: Phi (Roll) = 90 deg.
    # Should rotate Y axis to -Z axis (assuming Z down?) or Z to -Y.
    # Let's check X axis. Rotation about X leaves X invariant.
    phi = jnp.pi / 2
    theta = 0.0
    psi = 0.0

    R = rotation_body_to_inertial(phi, theta, psi)

    u_x = jnp.array([1.0, 0.0, 0.0])
    u_x_rot = jnp.matmul(R, u_x)

    # X axis should be unchanged
    assert jnp.allclose(u_x_rot, u_x, atol=1e-6)

    # Test 2: Theta (Pitch) = 90 deg.
    # Should rotate Z axis to X axis (or similar).
    # Rotation about Y leaves Y invariant.
    phi = 0.0
    theta = jnp.pi / 2
    psi = 0.0

    R = rotation_body_to_inertial(phi, theta, psi)

    u_y = jnp.array([0.0, 1.0, 0.0])
    u_y_rot = jnp.matmul(R, u_y)

    # Y axis should be unchanged
    assert jnp.allclose(u_y_rot, u_y, atol=1e-6)

    # Test 3: Psi (Yaw) = 90 deg.
    # Should rotate X axis to Y axis.
    # Rotation about Z leaves Z invariant (in magnitude/direction relative to frame).
    # Note: The frame definition involves a Z-axis flip (Body Z Down -> Inertial Z Up).
    # So u_z maps to -u_z. But it should NOT depend on psi.
    phi = 0.0
    theta = 0.0
    psi = jnp.pi / 2

    R = rotation_body_to_inertial(phi, theta, psi)

    u_z = jnp.array([0.0, 0.0, 1.0])
    u_z_rot = jnp.matmul(R, u_z)

    # Z axis should be mapped to -Z (due to frame def), but be stable under Yaw rotation.
    assert jnp.allclose(u_z_rot, -u_z, atol=1e-6)
