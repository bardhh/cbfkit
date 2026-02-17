import jax.numpy as jnp

from cbfkit.systems.quadrotor_6dof.controllers.geometric import geometric_controller
from cbfkit.systems.quadrotor_6dof.models.quadrotor_6dof_dynamics import quadrotor_6dof_dynamics
from cbfkit.utils.user_types import ControllerData


def _two_tuple_quadrotor_dynamics():
    three_tuple = quadrotor_6dof_dynamics()

    def dynamics(x):
        f_val, g_mat, _s = three_tuple(x)
        return f_val, g_mat

    return dynamics


def test_geometric_controller_accepts_three_tuple_dynamics():
    controller = geometric_controller(
        dynamics=quadrotor_6dof_dynamics(),
        desired_state=jnp.array([0.0, 0.0, 0.0]),
        dt=0.01,
    )

    state = jnp.array(
        [
            0.2,
            -0.1,
            1.0,
            0.1,
            -0.2,
            0.0,
            0.05,
            -0.04,
            0.03,
            0.01,
            -0.02,
            0.015,
        ]
    )
    control, data = controller(
        0.0,
        state,
        jnp.zeros((4,)),
        jnp.array([0, 1], dtype=jnp.uint32),
        ControllerData(),
    )

    assert control.shape == (4,)
    assert bool(jnp.all(jnp.isfinite(control)))
    assert isinstance(data, ControllerData)


def test_geometric_controller_accepts_two_tuple_dynamics():
    controller = geometric_controller(
        dynamics=_two_tuple_quadrotor_dynamics(),
        desired_state=jnp.array([0.0, 0.0, 0.0]),
        dt=0.01,
    )
    state = jnp.zeros((12,))
    control, _ = controller(0.0, state)
    assert control.shape == (4,)
    assert bool(jnp.all(jnp.isfinite(control)))
