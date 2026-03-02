from typing import Callable

import jax.numpy as jnp
from jax import Array, jacfwd, jacrev, jit

from cbfkit.utils.user_types import DynamicsCallable


def generate_predictor_corrector_dynamical_solution(
    augmented_cost: Callable[[Array], Array],
    pd_matrix: Array,
    dynamics: DynamicsCallable,
) -> Callable[[Array], Array]:
    """Build the predictor-corrector control dynamics from an augmented cost."""
    n_con = pd_matrix.shape[0]
    jacobian = jacfwd(augmented_cost)
    hessian = jacrev(jacfwd(augmented_cost))

    @jit
    def control_dynamics(z: Array) -> Array:
        f, _ = dynamics(z)
        jaco = jacobian(z)
        hess = hessian(z)

        matrix_term = -jnp.linalg.inv(hess[-n_con:-n_con:])
        vector_term = jnp.matmul(pd_matrix, jaco[-n_con]) + jnp.matmul(hess[-n_con:, :n_con], f)

        return jnp.matmul(matrix_term, vector_term)

    return control_dynamics
