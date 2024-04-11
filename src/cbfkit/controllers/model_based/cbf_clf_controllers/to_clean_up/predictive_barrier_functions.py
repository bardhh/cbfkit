import jax.numpy as jnp
from jax import jit, jacfwd, jacrev, Array, grad, jacobian
from typing import Callable, Tuple


@jit
def predictive_cbf(h: Callable[[Array], Array]) -> Callable[[float, Array, Array], Array]:
    """Predictive Barrier Function.

    Arguments:
        x: concatenated (current) time and state vector -- [x, t]
        h: constraint function (for evaluating safety at current time)

    Returns:
        val: value of predictive CBF

    """

    def cbf(xbar: Array) -> Array:
        """Returns an Array of all of the CBF values over the state trajectory xbar."""

        return jnp.array([h(xx) for xx in xbar.T])

    return cbf


@jit
def grad_predictive_cbf(
    h: Callable[[float, Array], Array], grad_mpc: Callable[[float, Array], Array]
) -> Callable[[float, Array, Array], Array]:
    """Partial derivative of the predictive control barrier function with respect to the concatentated
    time and state vector.

    Arguments:
        x: concatenated (current) time and state vector -- [x, t]
        h: constraint function (for evaluating safety at current time)
        path: function to compute a path and control sequence based on the current time and state
        dt: timestep (in sec)

    Returns:
        dhdx: gradient of the predictive CBF

    """

    def grad_cbf(t: float, x: Array, xbar: Array) -> Array:
        return jnp.multiply(jacrev(h)(xbar), grad_mpc(t, x))

    return grad_cbf


# # @jit
# def d2hpdx2(
#     x: Array,
#     h: Callable[[float, Array], Array],
# ) -> Array:
#     """Second Partial derivative of the predictive control barrier function with respect to the
#     concatentated time and state vector.

#     Arguments:
#         x: concatenated (current) time and state vector -- [x, t]
#         h: constraint function (for evaluating safety at current time)
#         path: function to compute a path and control sequence based on the current time and state
#         dt: timestep (in sec)

#     Returns:
#         dhdx: gradient of the predictive CBF

#     """
#     return jacfwd(jacrev(hp))(x, h)


# ### Predictive Control Barrier Functions
# def predictive_func(
#     predictive: Callable[
#         [
#             Array,
#             Callable[[float, Array], Array],
#             Callable[[float, Array], Tuple[Array, Array]],
#             float,
#         ],
#         Array,
#     ],
#     nominal: Callable[[float, Array], Array],
# ) -> Callable[
#     [Callable[[float, Array], Tuple[Array, Array]], float], Callable[[float, Array], Array]
# ]:
#     def predictive_form(
#         mpc: Callable[[float, Array], Tuple[Array, Array]], dt: float
#     ) -> Callable[[float, Array], Array]:
#         def func(t: float, x: Array) -> Array:
#             return predictive(jnp.hstack([x, t]), nominal, mpc, dt)

#         return func

#     return predictive_form
