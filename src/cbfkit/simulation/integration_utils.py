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

    Returns:
        Next integrated state.
    """
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
