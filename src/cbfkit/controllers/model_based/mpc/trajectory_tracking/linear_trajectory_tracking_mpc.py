import jax
import jax.numpy as jnp
from jax import jit, Array, lax, vmap
from typing import Tuple, Callable
from cbfkit.optimization.mpc import generate_mpc_solver_quadratic_cost_linear_dynamics
from cbfkit.utils.user_types import ControllerCallable, ControllerCallableReturns

jax.config.update("jax_enable_x64", True)


def dynamic_slice(arr, idx):
    def slice_fn(idx):
        idx = jnp.expand_dims(idx, axis=0).astype(int)
        return jnp.take(arr, idx, axis=1)

    return vmap(jit(slice_fn))(idx)


def linear_mpc(
    A: Array,
    B: Array,
    Q: Array,
    R: Array,
    waypoints: Tuple[Array, Array],
    horizon: float,
    dt: float,
) -> Callable[[float, Array], Tuple[Array, Array]]:
    """Computes solution to Linear MPC problem using jaxopt.OSQP"""
    # Number of timesteps to predict
    steps = int(horizon / dt)
    waypoints_jnp = jnp.array(waypoints)

    mpc_solver = generate_mpc_solver_quadratic_cost_linear_dynamics(A, B, Q, R, Q, steps)

    @jit
    def update_velocities(x: Array, xd: Array) -> Array:
        # Compute first-order derivatives
        x_dot_d = -jnp.diff(xd[0], axis=0, prepend=0)
        y_dot_d = -jnp.diff(xd[1], axis=0, prepend=0)

        # Compute second-order derivatives
        x_2dot_d = -jnp.diff(x_dot_d, axis=0, prepend=0)
        y_2dot_d = -jnp.diff(y_dot_d, axis=0, prepend=0)

        return jnp.vstack([xd, x_dot_d, y_dot_d, x_2dot_d, y_2dot_d])

    @jit
    def compute_waypoints(t: float, x: Array):

        def dynamic_slice_0(arr, idx):
            def slice_fn(idx):
                idx = jnp.expand_dims(idx, axis=0).astype(int)
                return jnp.take(arr, idx, axis=0)

            return vmap(jit(slice_fn))(idx)

        def cond_fn(carry: Tuple[Array, int]):
            return carry[2] < int(horizon / dt)

        def body_fn(carry: Tuple[Array, int]):
            arr, s0, idx = carry
            new_arr = jnp.squeeze(
                dynamic_slice(waypoints_jnp, jnp.expand_dims(carry[1] + carry[2], axis=0))
            )
            arr = arr.at[carry[2], :].set(new_arr)

            return (
                arr,
                carry[1],
                carry[2] + 1,
            )

        start_idx = (jnp.array([t]) / dt).astype(int)
        init_val = jnp.zeros((int(horizon / dt), 2)), start_idx, 0
        ret = lax.while_loop(cond_fn, body_fn, init_val)

        return jnp.array(ret[0])

    @jit
    def update_waypoints(t: float, x: Array) -> Array:
        nonlocal waypoints_jnp
        # Calculate indices for waypoints (time-based)
        # start = jnp.squeeze(dynamic_slice_0(x, jnp.array([-1])))
        # widx = jnp.arange(start, start + horizon, dt)
        # widx = widx.astype(int)[: int(horizon / dt)]
        # xd = jnp.squeeze(dynamic_slice(waypoints_jnp, ))

        # # State-based waypoint indices
        # deviations = jnp.abs(waypoints_jnp - jnp.expand_dims(x[:2], axis=1))
        # min_index = jnp.argmin(deviations) + int(1 / dt)

        # # Ensure that waypoint indices do not exceed size of waypoints_jnp
        # widx = jnp.linspace(
        #     min_index,
        #     min_index + int(horizon / dt),
        #     int(horizon / dt) + 1,
        # )

        # print(widx)
        # xd = jnp.squeeze(dynamic_slice(waypoints_jnp, widx)).T
        xd = compute_waypoints(t, x).T
        # import matplotlib.pyplot as plt

        # print(xd.shape)
        # plt.figure()
        # # plt.scatter(xbar[0, :], xbar[1, :], color="blue")
        # plt.scatter(xd[0, :], xd[0, :], color="green")
        # plt.show()
        return update_velocities(x, xd)

    @jit
    def solve_mpc(t: float, x: Array) -> Tuple[Array, Array]:
        # Calculate xd for all steps
        xd = update_waypoints(t, x)
        xd_supp = jnp.zeros((len(x) - xd.shape[0], xd.shape[1]))
        xd = jnp.vstack([xd, xd_supp]).T
        concatenated_x_xd = jnp.vstack([jnp.expand_dims(x, axis=0), xd])

        xbar, ubar = mpc_solver(concatenated_x_xd)
        # import matplotlib.pyplot as plt

        # print(xbar.shape)
        # print(xd.shape)
        # plt.figure()
        # plt.scatter(xbar[0, :], xbar[1, :], color="blue")
        # plt.scatter(xd[:, 0], xd[:, 1], color="green")
        # plt.show()

        return ubar.T, xbar

    return solve_mpc


def linear_mpc_controller(
    A: Array,
    B: Array,
    Q: Array,
    R: Array,
    waypoints: Tuple[Array, Array],
    horizon: float,
    dt: float,
) -> ControllerCallable:
    """Linear MPC Controller"""
    generate_control = linear_mpc(A, B, Q, R, waypoints, horizon, dt)

    @jit
    def controller(t: float, x: Array) -> ControllerCallableReturns:
        """Generates control input for linear MPC control law."""
        ubar, xbar = generate_control(t, x)
        data = {"xn_full": xbar, "un_full": ubar}

        return ubar[0, :], data

    return controller
