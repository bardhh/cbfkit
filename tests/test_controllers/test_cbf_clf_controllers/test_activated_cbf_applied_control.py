"""The control a CBF-CLF-QP controller returns must be the control its QP solved for.

The barrier-activated generator deactivates a barrier by multiplying its whole constraint
row by a weight of exactly zero, so an all-zero row is present in every solve. The
generator used to read any such degenerate row as a signal to hand back the clipped
nominal control instead of the QP solution, while still reporting success -- the QP
computed a safe control and the plant received an unfiltered one. Closed-loop behaviour
became independent of every barrier parameter, which is what the parameter-sensitivity
test below pins against.
"""

import jax
import jax.numpy as jnp
import numpy as np

from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.controllers.cbf_clf.barrier_activated_cbf_clf_qp_control_laws import (
    barrier_activated_cbf_clf_qp_controller,
)
from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.controllers.cbf_clf.utils.barrier_activation import compute_activation_weights
from cbfkit.utils.user_types import CertificateCollection, ControllerData

CONTROL_LIMITS = jnp.array([1.0, 1.0])
OBSTACLES = jnp.array([[1.0, 0.0], [-4.0, 4.0], [4.0, -4.0]])
RADIUS = 0.5
KEY = jax.random.PRNGKey(0)


def _single_integrator(x):
    """2D single integrator: xdot = u."""
    return jnp.zeros(2), jnp.eye(2)


def _obstacle_barriers(gain: float = 1.0) -> CertificateCollection:
    """h_i(x) = ||x - o_i||^2 - RADIUS^2 >= 0 for each obstacle."""

    def make(o):
        return (
            lambda t, x: jnp.sum((x - o) ** 2) - RADIUS**2,
            lambda t, x: 2.0 * (x - o),
            lambda t, x: 2.0 * jnp.eye(2),
            lambda t, x: 0.0,
        )

    parts = [make(o) for o in OBSTACLES]
    return CertificateCollection(
        [p[0] for p in parts],
        [p[1] for p in parts],
        [p[2] for p in parts],
        [p[3] for p in parts],
        [(lambda val, g=gain: g * val) for _ in parts],
    )


def _activated_controller(gain: float = 1.0, k_closest: int = 1, **extra):
    return barrier_activated_cbf_clf_qp_controller(
        control_limits=CONTROL_LIMITS,
        dynamics_func=_single_integrator,
        barriers=_obstacle_barriers(gain),
        obstacle_positions=OBSTACLES,
        k_closest=k_closest,
        activation_radius=3.0,
        relaxable_cbf=True,
        **extra,
    )


def _call(controller, x, u_nom):
    return controller(0.0, x, u_nom, KEY, ControllerData())


# A state just outside obstacle 0 with the nominal control driving straight into it.
APPROACHING = jnp.array([0.4, 0.0])
U_TOWARD_OBSTACLE = jnp.array([1.0, 0.0])


def test_deactivated_barriers_produce_degenerate_rows():
    """Precondition for the rest of the suite: the zero-weight path is exercised."""
    weights = compute_activation_weights(APPROACHING, OBSTACLES, k=1, radius=3.0, smoothness=5.0)
    assert int(jnp.sum(weights == 0.0)) == len(OBSTACLES) - 1
    assert float(weights.max()) > 0.0


def test_applied_control_is_the_qp_solution():
    """u must equal the QP solution's control block, exactly, not approximately."""
    controller = _activated_controller()
    u, data = _call(controller, APPROACHING, U_TOWARD_OBSTACLE)

    assert not bool(data.error)
    assert int(np.asarray(data.error_data)) == 1
    np.testing.assert_array_equal(np.asarray(u), np.asarray(data.sol)[: CONTROL_LIMITS.shape[0]])


def test_applied_control_is_not_the_nominal_control():
    """The safety filter must actually reach the output when a barrier is active."""
    controller = _activated_controller()
    u, _ = _call(controller, APPROACHING, U_TOWARD_OBSTACLE)

    clipped_nominal = jnp.clip(U_TOWARD_OBSTACLE, -CONTROL_LIMITS, CONTROL_LIMITS)
    assert float(jnp.linalg.norm(u - clipped_nominal)) > 1e-3
    # Driving straight at the obstacle is what gets filtered out.
    assert float(u[0]) < float(clipped_nominal[0])


def test_vanilla_applied_control_is_the_qp_solution():
    """The vanilla path was already exact; keep it that way."""
    controller = vanilla_cbf_clf_qp_controller(
        control_limits=CONTROL_LIMITS,
        dynamics_func=_single_integrator,
        barriers=_obstacle_barriers(),
        relaxable_cbf=True,
    )
    u, data = _call(controller, APPROACHING, U_TOWARD_OBSTACLE)

    assert not bool(data.error)
    np.testing.assert_array_equal(np.asarray(u), np.asarray(data.sol)[: CONTROL_LIMITS.shape[0]])


def test_degenerate_row_does_not_mask_infeasibility():
    """A zero row whose right-hand side is negative is unsatisfiable, not ignorable."""

    def mock_cbf(control_limits, dynamics, barriers, lyapunovs, **kwargs):
        def compute(t, x):
            return jnp.zeros((1, 1)), jnp.array([-1.0]), {"complete": False}

        return compute

    def mock_clf(*args, **kwargs):
        return lambda t, x: (jnp.zeros((0, 1)), jnp.zeros((0,)), {})

    controller = cbf_clf_qp_generator(mock_cbf, mock_clf)(
        control_limits=jnp.array([1.0]),
        dynamics_func=lambda x: (jnp.zeros(1), jnp.eye(1)),
        barriers=([], [], [], [], []),
        lyapunovs=([], [], [], [], []),
        relaxable_cbf=False,
    )
    u, data = controller(0.0, jnp.zeros(1), jnp.array([1.0]), KEY, ControllerData())

    assert bool(data.error)
    assert bool(jnp.isnan(u).all())


def test_gradient_through_deactivated_barrier_is_finite():
    """Degenerate rows must not reintroduce NaN sensitivities (the old override's purpose)."""
    controller = _activated_controller()

    def loss(x):
        u, _ = _call(controller, x, U_TOWARD_OBSTACLE)
        return jnp.sum(u)

    grad = jax.grad(loss)(APPROACHING)
    assert bool(jnp.all(jnp.isfinite(grad)))


def _rollout(gain: float, steps: int = 40, dt: float = 0.05):
    """Closed loop on the single integrator, driving straight at obstacle 0."""
    controller = _activated_controller(gain=gain)
    x = jnp.array([-0.5, 0.0])
    traj = [x]
    for _ in range(steps):
        u, _ = _call(controller, x, U_TOWARD_OBSTACLE)
        x = x + dt * u
        traj.append(x)
    return np.asarray(jnp.stack(traj))


def test_closed_loop_responds_to_barrier_parameters():
    """Barrier tuning must move the trajectory.

    With the override in place, a 25-cell sweep over roots and slack bounds produced
    closed-loop trajectories that agreed to machine precision, because the QP solution
    never reached the plant. Identical trajectories under different class-K gains are the
    signature of that defect, so this asserts they differ.
    """
    slow = _rollout(gain=0.5)
    fast = _rollout(gain=5.0)

    assert np.abs(slow - fast).max() > 1e-3
    # The safe set is the same either way: both stay outside the obstacle.
    for traj in (slow, fast):
        assert np.linalg.norm(traj - np.asarray(OBSTACLES[0]), axis=1).min() >= RADIUS - 1e-6
