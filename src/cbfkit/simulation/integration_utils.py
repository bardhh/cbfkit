from jax import Array
import jax.numpy as jnp

from cbfkit.integration import forward_euler
from cbfkit.integration.runge_kutta import runge_kutta_4
from cbfkit.utils.user_types import DynamicsCallable, IntegratorCallable, State


def integrate_with_cached_dynamics(
    x: State,
    u: Array,
    dt: float,
    dynamics: DynamicsCallable,
    integrator: IntegratorCallable,
    f: Array,
    g: Array,
    perturbation_value: Array,
    perturbation_is_increment: bool = False,
) -> State:
    """Integrates one simulation step reusing precomputed dynamics terms.

    Args:
        x: Current state.
        u: Current control input.
        dt: Simulation time step.
        dynamics: Dynamics callable returning ``(f(x), g(x))``.
        integrator: Integrator callable to use.
        f: Cached drift dynamics value at ``x``.
        g: Cached control matrix value at ``x``.
        perturbation_value: Cached perturbation value for this step.
        perturbation_is_increment: When True, ``perturbation_value`` is a
            discrete-time state increment (e.g. Euler-Maruyama stochastic term
            ``Σ(x)·sqrt(dt)·ξ``) and is added **once** to the integrated state,
            NOT folded into the vector field.  When False (default), the
            perturbation is a continuous-time rate ``d(x)`` that is folded into
            the drift as ``ẋ = f + g·u + d(x)`` (legacy rate-disturbance
            behavior for bounded/affine perturbations).

    Returns:
        Next integrated state.
    """
    if perturbation_is_increment:
        # Euler-Maruyama: integrate drift+control only, then add the increment once.
        if integrator == forward_euler:
            dx = f + jnp.matmul(g, u)
            return x + dx * dt + perturbation_value

        def vector_field_no_pert(s: State) -> Array:
            f_s, g_s = dynamics(s)
            return f_s + jnp.matmul(g_s, u)

        if integrator == runge_kutta_4:
            k1 = f + jnp.matmul(g, u)
            k2 = vector_field_no_pert(x + 0.5 * dt * k1)
            k3 = vector_field_no_pert(x + 0.5 * dt * k2)
            k4 = vector_field_no_pert(x + dt * k3)
            return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4) + perturbation_value

        return integrator(x, vector_field_no_pert, dt) + perturbation_value

    # Rate-disturbance path (default): perturbation folded into drift ×dt.
    if integrator == forward_euler:
        dx = f + jnp.matmul(g, u) + perturbation_value
        return x + dx * dt

    def vector_field(s: State) -> Array:
        f_s, g_s = dynamics(s)
        return f_s + jnp.matmul(g_s, u) + perturbation_value

    if integrator == runge_kutta_4:
        k1 = f + jnp.matmul(g, u) + perturbation_value
        k2 = vector_field(x + 0.5 * dt * k1)
        k3 = vector_field(x + 0.5 * dt * k2)
        k4 = vector_field(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return integrator(x, vector_field, dt)
