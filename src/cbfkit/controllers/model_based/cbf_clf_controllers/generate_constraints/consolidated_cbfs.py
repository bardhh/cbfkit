"""
#! docstring
"""

from typing import Callable, Tuple, Dict, Any, List
import jax.numpy as jnp
from jax import Array, jit, lax, vmap, jacfwd, jacrev
from cbfkit.utils.user_types import (
    DynamicsCallable,
    CertificateCollection,
    State,
)
from .unpack import unpack_for_cbf
from .generating_functions import (
    generate_compute_certificate_values_vmap as generate_compute_certificate_values,
)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.certificate_packager import (
    certificate_package,
)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.barrier_conditions.zeroing_barriers import (
    linear_class_k,
)


def ccbf(
    constraint_functions: List[Callable[[Array], Array]],
    **kwargs,
) -> Callable[[Array], Array]:
    n_bfs = len(constraint_functions)

    @jit
    def func(state_and_weights_and_time: Array) -> Array:
        t = state_and_weights_and_time[-1]
        x = state_and_weights_and_time[: -1 - 2 * n_bfs]
        w = state_and_weights_and_time[-1 - 2 * n_bfs : -1 - n_bfs]
        return 1 - jnp.sum(
            jnp.array([jnp.exp(-ww * cf(t, x)) for ww, cf in zip(w, constraint_functions)])
        )

    return func


def ccbf_grad(
    constraint_functions: List[Callable[[Array], Array]],
    **kwargs,
) -> Callable[[Array], Array]:
    """Jacobian for the constraint function defined by cbf.

    Args:
        #! kwargs -- manually populate

    Returns:
        ret (float): value of constraint function evaluated at time and state

    """
    jacobian = jacfwd(ccbf(constraint_functions, **kwargs))

    @jit
    def func(state_and_time: Array) -> Array:
        """_summary_

        Args:
            state_and_time (Array): concatenated state vector and time

        Returns:
            Array: cbf jacobian (gradient)
        """

        return jacobian(state_and_time)

    return func


def ccbf_hess(
    constraint_functions: List[Callable[[Array], Array]],
    **kwargs,
) -> Callable[[Array], Array]:
    """Hessian for the constraint function defined by cbf.

    Args:
        #! kwargs -- manually populate

    Returns:
        ret (float): value of constraint function evaluated at time and state

    """
    hessian = jacrev(jacfwd(ccbf(constraint_functions, **kwargs)))

    @jit
    def func(state_and_time: Array) -> Array:
        """_summary_

        Args:
            state_and_time (Array): concatenated state vector and time

        Returns:
            Array: cbf hessian
        """

        return hessian(state_and_time)

    return func


####################################################################################################
### Consolidated CBF: Lfh + Lgh * u + Lwh * wdot + \alpha(h) >= 0 ####################################################
def generate_compute_consolidated_cbf_constraints(
    control_limits: Array,
    dyn_func: DynamicsCallable,
    barriers: CertificateCollection = ([], [], [], [], []),
    lyapunovs: CertificateCollection = ([], [], [], [], []),
    **kwargs: Dict[str, Any],
) -> Callable[[float, State], Tuple[Array, Array]]:
    """
    #! To Do: docstring
    """
    if "alpha" not in kwargs:
        raise ValueError("Missing Class K function alpha from kwargs!")
    if "n_states" not in kwargs:
        raise ValueError("Missing n_states from kwargs!")
    bfs, _, _, _, _ = barriers
    consolidated_barrier_package = certificate_package(
        ccbf, ccbf_grad, ccbf_hess, kwargs["n_states"] + len(bfs)
    )
    consolidated_barriers = consolidated_barrier_package(
        certificate_conditions=linear_class_k(alpha=kwargs["alpha"]),
        constraint_functions=bfs,
    )
    compute_barrier_values = generate_compute_certificate_values(consolidated_barriers)

    n_con, n_bfs, _n_lfs, a_cbf, b_cbf, tunable = unpack_for_cbf(
        control_limits, barriers, lyapunovs, **kwargs
    )

    @jit
    def compute_cbf_constraints(t: float, x: State) -> Tuple[Array, Array]:
        """Computes CBF and CLF constraints."""
        nonlocal a_cbf, b_cbf
        data = {}
        dyn_f, dyn_g = dyn_func(x)

        if n_bfs > 0:
            bf_x, bj_x, _, dbf_t, bc_x = compute_barrier_values(t, x)

            a_cbf = a_cbf.at[:, :n_con].set(-jnp.matmul(bj_x, dyn_g))
            b_cbf = b_cbf.at[:].set(dbf_t + jnp.matmul(bj_x, dyn_f) + bc_x)
            if tunable:
                a_cbf = a_cbf.at[:, n_con:n_bfs].set(-bc_x)
                b_cbf = b_cbf.at[:].set(dbf_t + jnp.matmul(bj_x, dyn_f))

            violated = lax.cond(jnp.any(bf_x < 0), lambda _fake: True, lambda _fake: False, 0)

            data["bfs"] = bf_x
            data["violated"] = violated

        return a_cbf, b_cbf, data

    return compute_cbf_constraints
