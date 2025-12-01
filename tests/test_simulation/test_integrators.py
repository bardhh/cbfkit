import jax.numpy as jnp
import pytest
from jax import jit

from cbfkit.integration import forward_euler, runge_kutta_4


def simple_linear_dynamics(x):
    """dx/dt = -x. Analytical solution: x(t) = x0 * e^-t"""
    return -x


def harmonic_oscillator_dynamics(x):
    """
    dx1/dt = x2
    dx2/dt = -x1
    Analytical solution for x0=[1, 0]: x1(t) = cos(t), x2(t) = -sin(t)
    """
    return jnp.array([x[1], -x[0]])


def test_integration_accuracy_linear():
    """Compare Forward Euler and RK4 accuracy on a simple linear system."""
    dt = 0.1
    t_final = 1.0
    steps = int(t_final / dt)
    x0 = jnp.array([1.0])

    # Forward Euler
    x_fe = x0
    for _ in range(steps):
        x_fe = forward_euler(x_fe, simple_linear_dynamics, dt)

    # RK4
    x_rk4 = x0
    for _ in range(steps):
        x_rk4 = runge_kutta_4(x_rk4, simple_linear_dynamics, dt)

    # Analytical
    x_true = x0 * jnp.exp(-t_final)

    error_fe = jnp.abs(x_fe - x_true)
    error_rk4 = jnp.abs(x_rk4 - x_true)

    print(f"Linear System (dt={dt}): FE Error = {error_fe[0]:.6f}, RK4 Error = {error_rk4[0]:.6f}")

    # RK4 should be significantly more accurate
    assert error_rk4 < error_fe
    assert error_rk4 < 1e-6  # RK4 is 4th order, error should be tiny


def test_integration_accuracy_harmonic():
    """Compare Forward Euler and RK4 accuracy on a harmonic oscillator (energy conservation)."""
    dt = 0.1
    t_final = 2 * jnp.pi  # One full period
    steps = int(t_final / dt)
    x0 = jnp.array([1.0, 0.0])

    # Forward Euler
    x_fe = x0
    for _ in range(steps):
        x_fe = forward_euler(x_fe, harmonic_oscillator_dynamics, dt)

    # RK4
    x_rk4 = x0
    for _ in range(steps):
        x_rk4 = runge_kutta_4(x_rk4, harmonic_oscillator_dynamics, dt)

    # Analytical at t=2*pi is x0
    x_true = x0

    error_fe = jnp.linalg.norm(x_fe - x_true)
    error_rk4 = jnp.linalg.norm(x_rk4 - x_true)

    print(f"Harmonic Oscillator (dt={dt}): FE Error = {error_fe:.6f}, RK4 Error = {error_rk4:.6f}")

    # Forward Euler is unstable for oscillatory systems (adds energy)
    # RK4 should be stable and accurate
    assert error_rk4 < error_fe
    assert error_rk4 < 0.1


def test_integrator_jit_compatibility():
    """Ensure integrators work within JIT compilation."""

    @jit
    def step_fe(x):
        return forward_euler(x, simple_linear_dynamics, 0.1)

    @jit
    def step_rk4(x):
        return runge_kutta_4(x, simple_linear_dynamics, 0.1)

    x0 = jnp.array([1.0])

    # Should not raise error
    step_fe(x0)
    step_rk4(x0)
