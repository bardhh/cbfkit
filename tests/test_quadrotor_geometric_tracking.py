"""Closed-loop regression tests for the 6-DOF quadrotor ``geometric_controller``.

History: the controller had two latent bugs that the pre-existing tests
(``test_quadrotor_geometric_controller.py``) could not catch, because they only
asserted that a *single* control evaluation was finite and shape ``(4,)`` — never
that a closed-loop simulation actually tracks a setpoint:

1. **Improper rotation (det = -1).** Both ``rotation_body_frame_to_inertial_frame``
   and the plant's ``rotation_body_to_inertial`` return an orthogonal but *improper*
   matrix (a reflection): the standard ZYX 3rd row is negated to encode the model's
   "body z-down, inertial h-up" convention. The geometric attitude error
   ``vee(Rdᵀ R - Rᵀ Rd)`` assumes ``R ∈ SO(3)``; feeding it a reflection loads the
   lateral attitude error onto the wrong axis, so the east position ``pe`` diverged
   (to ≈ -8 m for a +1.5 m setpoint) while ``pn`` and altitude tracked. The fix builds
   a proper rotation ``S @ R`` (``S = diag(1, 1, -1)``) inside the controller and runs
   the Lee 2010 law in the z-down frame — the plant itself is left untouched.

2. **Wrong body angular velocity.** The controller reconstructed ``omega`` from
   Euler-angle rates with a transform that is wrong even at hover (it placed the roll
   rate into the yaw slot), leaving roll undamped → integration NaN within ~1 s. The
   body angular velocity ``[p, q, r]`` is already the state slice ``x[9:12]``; the fix
   uses it directly.

These tests pin both corrections: convergence to an asymmetric setpoint (catches the
lateral-axis sign bug), absence of NaN/divergence, and that a pure roll rate produces
a roll-damping moment (catches the omega-axis bug).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import cbfkit.simulation.simulator as sim
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.systems.quadrotor_6dof.controllers.geometric import geometric_controller
from cbfkit.systems.quadrotor_6dof.models.quadrotor_6dof_dynamics import (
    quadrotor_6dof_dynamics,
    rotation_body_to_inertial,
)
from cbfkit.systems.quadrotor_6dof.utils.rotations import (
    rotation_body_frame_to_inertial_frame,
)

# Mass/inertia consistent between plant and the controller's default gains.
_M, _JX, _JY, _JZ = 4.34, 0.0820, 0.0845, 0.1377


def _plant():
    three_tuple = quadrotor_6dof_dynamics(m=_M, jx=_JX, jy=_JY, jz=_JZ)

    def dyn(x):
        f_val, g_mat, _s = three_tuple(x)
        return f_val, g_mat

    return dyn


def _run(desired, x0=None, dt: float = 0.01, tf: float = 6.0, use_jit: bool = False):
    dyn = _plant()
    if x0 is None:
        x0 = jnp.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    controller = geometric_controller(
        dynamics=dyn, desired_state=jnp.asarray(desired), dt=dt, m=_M, jx=_JX, jy=_JY, jz=_JZ
    )
    res = sim.execute(
        x0=x0,
        dt=dt,
        num_steps=int(tf / dt),
        dynamics=dyn,
        integrator=integrator,
        nominal_controller=controller,
        sensor=sensor,
        estimator=estimator,
        use_jit=use_jit,
    )
    return np.asarray(res["states"])


def test_geometric_controller_converges_without_nan():
    """Free-running setpoint regulation must reach the goal — pre-fix it NaN'd at ~0.95 s."""
    desired = jnp.array([2.0, 1.5, 3.0])
    states = _run(desired)

    assert not np.isnan(states).any(), "Integration produced NaN (omega-reconstruction bug?)."
    final_err = np.linalg.norm(states[-1, :3] - np.asarray(desired))
    assert (
        final_err < 0.1
    ), f"Final position error {final_err:.3f} m too large; controller did not track."


def test_no_lateral_divergence_on_asymmetric_setpoint():
    """An asymmetric (pn != pe) setpoint catches the improper-rotation lateral-axis sign bug.

    Pre-fix, the east position ``pe`` diverged to ≈ -8 m (wrong sign, growing) for a
    +1.5 m target while ``pn``/altitude tracked. A symmetric target cannot detect this.
    """
    desired = jnp.array([2.0, 1.5, 3.0])
    states = _run(desired)

    pe = states[:, 1]
    # pe must stay bounded and finish near +1.5 (not run away to large negative).
    assert np.abs(pe).max() < 3.0, f"|pe| peaked at {np.abs(pe).max():.2f} m — lateral divergence."
    assert (
        abs(float(states[-1, 1]) - 1.5) < 0.1
    ), f"Final pe={float(states[-1, 1]):.3f} did not reach +1.5; attitude-error axis sign regressed."
    # Attitude stays well away from gimbal/flip throughout.
    assert np.abs(states[:, 6:9]).max() < 1.0, "Attitude excursion too large; controller unstable."


def test_jit_and_nonjit_paths_agree():
    desired = jnp.array([2.0, 1.5, 3.0])
    s_nojit = _run(desired, use_jit=False)
    s_jit = _run(desired, use_jit=True)
    assert np.allclose(s_nojit[-1], s_jit[-1], atol=1e-3), "JIT and non-JIT trajectories diverge."


def test_pure_roll_rate_produces_roll_damping_moment():
    """A pure body roll rate p>0 (at the setpoint, level attitude) must yield a *roll*-damping
    moment M_x < 0 — pinning ``omega = x[9:12]``.

    Pre-fix, the Euler-rate reconstruction routed the roll rate into the yaw channel, so
    M_x ≈ 0 and M_z absorbed the (mis-attributed) damping. This isolates that bug from the
    rotation bug by placing the quadrotor exactly at its setpoint with level attitude.
    """
    desired = jnp.array([0.0, 0.0, 1.0])
    # At the setpoint, level, with only a roll rate p = 0.3 rad/s.
    x = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0])
    controller = geometric_controller(dynamics=_plant(), desired_state=desired, dt=0.01)
    control, _ = controller(0.0, x)

    m_x, m_y, m_z = float(control[1]), float(control[2]), float(control[3])
    assert (
        m_x < -0.1
    ), f"M_x={m_x:.3f} should be negative (damp roll); omega axis-mapping regressed."
    # The roll rate must NOT leak into the yaw moment.
    assert abs(m_z) < 1e-6, f"M_z={m_z:.3e} nonzero for a pure roll rate; omega routed to yaw."


def test_packaged_rotation_is_improper_but_controller_fix_is_proper():
    """Document the root cause and pin the fix's premise.

    The packaged body->inertial matrices are reflections (det = -1); the controller's
    ``S @ R`` correction (S = diag(1, 1, -1)) restores a proper SO(3) rotation.
    """
    S = np.diag([1.0, 1.0, -1.0])
    for phi, theta, psi in [(0.0, 0.0, 0.0), (0.3, -0.2, 0.5), (-0.4, 0.6, -0.7)]:
        x = jnp.array([0, 0, 0, 0, 0, 0, phi, theta, psi, 0, 0, 0.0])
        R = np.asarray(rotation_body_frame_to_inertial_frame(x))
        R_dyn = np.asarray(rotation_body_to_inertial(phi, theta, psi))
        assert np.allclose(R, R_dyn, atol=1e-9), "utils and dynamics rotation matrices must match."
        assert np.isclose(
            np.linalg.det(R), -1.0, atol=1e-6
        ), "packaged rotation should be improper."
        R_proper = S @ R
        assert np.isclose(np.linalg.det(R_proper), 1.0, atol=1e-6), "S @ R must be proper SO(3)."
        assert np.allclose(R_proper @ R_proper.T, np.eye(3), atol=1e-6), "S @ R must be orthogonal."
