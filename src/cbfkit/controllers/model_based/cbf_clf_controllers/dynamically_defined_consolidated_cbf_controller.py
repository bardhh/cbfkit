"""
#! docstring
"""
from typing import Callable, Tuple
from jax import Array, jacfwd, jacrev
import jax.numpy as jnp
from cbfkit.utils.user_types import DynamicsCallable
from cbfkit.utils.matrix_vector_operations import invert_array


def generate_compute_w2dot_udot(
    augmented_cost_wdot: Callable[[Array], Array],
    augmented_cost_u: Callable[[Array], Array],
    dynamics: DynamicsCallable,
    idx_x: Array,
    idx_w: Array,
    idx_wdot: Array,
    idx_u: Array,
):
    jacobian_wdot = jacfwd(augmented_cost_wdot)
    hessian_wdot = jacrev(jacfwd(augmented_cost_wdot))
    jacobian_u = jacfwd(augmented_cost_u)
    hessian_u = jacrev(jacfwd(augmented_cost_u))
    n_cbfs = len(idx_w)

    def generate(pd_matrix: Array) -> Callable[[Array], Tuple[Array, Array]]:
        """ """

        def compute_w2dot_udot(z: Array) -> Tuple[Array, Array]:
            """ """
            f, g = dynamics(z)
            grad_wdot = jacobian_wdot(z)
            hess_wdot = hessian_wdot(z)
            grad_u = jacobian_u(z)
            hess_u = hessian_u(z)

            w2dot_f, w2dot_g = compute_w2dot(z, f, g, grad_wdot, hess_wdot, grad_u, hess_u)
            udot_f, udot_g = compute_udot(z, f, g, grad_wdot, hess_wdot, grad_u, hess_u)

            return jnp.hstack([w2dot_f, udot_f]), jnp.vstack([w2dot_g, udot_g])

        def compute_w2dot(
            z: Array,
            f: Array,
            g: Array,
            grad_wdot: Array,
            hess_wdot: Array,
            grad_u: Array,
            hess_u: Array,
        ) -> Tuple[Array, Array]:
            """_summary_

            Args:
                z (Array): _description_

            Returns:
                Array: _description_
            """

            term1 = -hess_wdot[idx_wdot, idx_wdot]
            term2a = jnp.array(
                [
                    jnp.matmul(pd_matrix[idx_u, idx_u], grad_u[idx_u])
                    + jnp.matmul(hess_u[idx_u, idx_x], f[idx_x] + jnp.matmul(g[idx_x], z[idx_u]))
                    + jnp.matmul(hess_u[idx_u, idx_w], z[idx_wdot])
                ]
            )
            term2 = jnp.array(
                [
                    jnp.matmul(pd_matrix[idx_wdot, idx_wdot], grad_wdot[idx_wdot])
                    + jnp.matmul(
                        hess_wdot[idx_wdot, idx_x],
                        f[idx_x] + jnp.matmul(g[idx_x], z[idx_u]),
                    )
                    + jnp.matmul(hess_wdot[idx_wdot, idx_w], z[idx_wdot])
                    - jnp.matmul(
                        hess_wdot[idx_wdot, idx_u],
                        jnp.matmul(invert_array(hess_u[idx_u, idx_u]), term2a),
                    )
                ]
            )
            term3 = jnp.eye(n_cbfs) - jnp.matmul(
                jnp.matmul(
                    invert_array(hess_wdot[idx_wdot, idx_wdot]),
                    hess_wdot[idx_wdot, idx_u],
                ),
                jnp.matmul(invert_array(hess_u[idx_u, idx_u]), hess_u[idx_u, idx_wdot]),
            )

            return (
                jnp.matmul(invert_array(term1), jnp.matmul(term2, invert_array(term3))),
                jnp.zeros((len(idx_wdot), len(idx_u))),
            )

        def compute_udot(
            z: Array,
            f: Array,
            g: Array,
            grad_wdot: Array,
            hess_wdot: Array,
            grad_u: Array,
            hess_u: Array,
        ) -> Tuple[Array, Array]:
            """_summary_

            Args:
                z (Array): _description_

            Returns:
                Array: _description_
            """
            term1 = -hess_u[idx_u, idx_u]
            term2a = jnp.array(
                [
                    jnp.matmul(pd_matrix[idx_wdot, idx_wdot], grad_wdot[idx_wdot])
                    + jnp.matmul(
                        hess_wdot[idx_wdot, idx_x],
                        f[idx_x] + jnp.matmul(g[idx_x], z[idx_u]),
                    )
                    + jnp.matmul(hess_wdot[idx_wdot, idx_w], z[idx_wdot])
                ]
            )
            term2 = jnp.array(
                [
                    jnp.matmul(pd_matrix[idx_u, idx_u], grad_u[idx_u])
                    + jnp.matmul(hess_u[idx_u, idx_x], f[idx_x] + jnp.matmul(g[idx_x], z[idx_u]))
                    + jnp.matmul(hess_u[idx_u, idx_w], z[idx_wdot])
                    - jnp.matmul(
                        hess_u[idx_u, idx_wdot],
                        jnp.matmul(invert_array(hess_wdot[idx_wdot, idx_wdot]), term2a),
                    )
                ]
            )
            term3 = jnp.eye(n_cbfs) - jnp.matmul(
                jnp.matmul(
                    invert_array(hess_u[idx_u, idx_u]),
                    hess_u[idx_u, idx_wdot],
                ),
                jnp.matmul(
                    invert_array(hess_wdot[idx_wdot, idx_wdot]),
                    hess_wdot[idx_wdot, idx_u],
                ),
            )

            return (
                jnp.matmul(invert_array(term1), jnp.matmul(term2, invert_array(term3))),
                jnp.zeros((len(idx_u), len(idx_u))),
            )

        return compute_w2dot_udot

    return generate
