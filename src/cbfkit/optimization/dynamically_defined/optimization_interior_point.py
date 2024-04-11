"""
#! docstring
"""
from typing import Callable
import jax.numpy as jnp
from jax import jit, Array, jacfwd, jacrev
from cbfkit.utils.user_types import DynamicsCallable


def generate_predictor_corrector_dynamical_solution(
    augmented_cost: Callable[[Array], Array],
    pd_matrix: Array,
    dynamics: DynamicsCallable,
) -> Callable[[Array], Array]:
    """Generator function for the dynamically defined control law.

    Args:
        augmented_cost (Callable[[Array], Array]): _description_
        pd_matrix (Array): _description_
        dynamics (DynamicsCallable): _description_

    Returns:
        Callable[[Array], Array]: _description_
    """
    n_con = pd_matrix.shape[0]
    jacobian = jacfwd(augmented_cost)
    hessian = jacrev(jacfwd(augmented_cost))

    @jit
    def control_dynamics(z: Array) -> Array:
        """_summary_

        Args:
            z (Array): _description_

        Returns:
            Array: _description_
        """
        dyn = dynamics(z)
        jaco = jacobian(z)
        hess = hessian(z)

        matrix_term = -jnp.linalg.inv(hess[-n_con:-n_con:])
        vector_term = jnp.matmul(pd_matrix, jaco[-n_con]) + jnp.matmul(hess[-n_con:, :n_con], dyn)

        return jnp.matmul(matrix_term, vector_term)

    return control_dynamics
