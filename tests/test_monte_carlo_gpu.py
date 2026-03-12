"""Tests for GPU-accelerated Monte Carlo simulation."""

import jax
import jax.numpy as jnp
from jax import random

from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.controllers.cbf_clf.generate_constraints import (
    generate_compute_vanilla_clf_constraints,
    generate_compute_zeroing_cbf_constraints,
)
from cbfkit.integration import forward_euler
from cbfkit.simulation.monte_carlo_gpu import (
    MonteCarloGPUResults,
    MonteCarloSetup,
    conduct_monte_carlo_gpu,
)
from cbfkit.simulation.safety_verification import SafetyStatistics, compute_safety_statistics
from cbfkit.utils.user_types import CertificateCollection, ControllerData, PlannerData


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _dynamics(x):
    return jnp.zeros(2), jnp.eye(2)


def _make_cbf(c, r):
    return (
        lambda _t, x: jnp.sum((x - c) ** 2) - r**2,
        lambda _t, x: 2 * (x - c),
        lambda _t, _x: 2 * jnp.eye(2),
        lambda _t, _x: 0.0,
        lambda h: 1.0 * h,
    )


def _default_sensor(t, x, *, sigma=None, key=None):
    return x


def _default_estimator(t, y, z, u, c):
    return y, c if c is not None else jnp.zeros((len(y), len(y)))


def _default_perturbation(x, u, f, g):
    def p(key):
        return jnp.zeros_like(x)

    return p


def _build_test_setup(n_obstacles: int = 3) -> MonteCarloSetup:
    """Build a small setup for testing."""
    centers = jnp.array([[3.0, 3.0], [5.0, 5.0], [7.0, 7.0]])[:n_obstacles]
    radii = jnp.array([0.5, 0.5, 0.5])[:n_obstacles]

    barrier_tuples = [_make_cbf(centers[i], radii[i]) for i in range(n_obstacles)]
    barriers = CertificateCollection(*[list(x) for x in zip(*barrier_tuples)])

    controller = cbf_clf_qp_generator(
        generate_compute_zeroing_cbf_constraints,
        generate_compute_vanilla_clf_constraints,
    )(
        control_limits=jnp.array([5.0, 5.0]),
        dynamics_func=_dynamics,
        barriers=barriers,
        relaxable_cbf=False,
        relaxable_clf=True,
    )

    def nominal_controller(t, x, _k, _r):
        return 2.0 * (jnp.array([10.0, 10.0]) - x), None

    def initial_state_sampler(key):
        return random.uniform(key, (2,), minval=-0.5, maxval=0.5)

    # Prime controller data
    prime_key = random.PRNGKey(0)
    x0 = jnp.zeros(2)
    u_nom = jnp.zeros(2)
    _, c_data = controller(0.0, x0, u_nom, prime_key, ControllerData())

    return MonteCarloSetup(
        dt=0.05,
        num_steps=20,
        dynamics=_dynamics,
        integrator=forward_euler,
        initial_state_sampler=initial_state_sampler,
        nominal_controller=nominal_controller,
        controller=controller,
        sensor=_default_sensor,
        estimator=_default_estimator,
        perturbation=_default_perturbation,
        sigma=jnp.zeros(0),
        controller_data=c_data,
        planner=None,
        planner_data=PlannerData(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConductMonteCarloGPU:
    def test_output_shapes(self):
        setup = _build_test_setup()
        n_trials = 4
        results = conduct_monte_carlo_gpu(setup, n_trials=n_trials, seed=0)

        assert isinstance(results, MonteCarloGPUResults)
        assert results.n_trials == n_trials
        assert results.states.shape == (n_trials, setup.num_steps, 2)
        assert results.controls.shape == (n_trials, setup.num_steps, 2)
        assert results.wall_time_s > 0

    def test_different_seeds_produce_different_states(self):
        setup = _build_test_setup()
        r1 = conduct_monte_carlo_gpu(setup, n_trials=4, seed=0)
        r2 = conduct_monte_carlo_gpu(setup, n_trials=4, seed=99)

        # Different seeds should (almost certainly) give different trajectories
        assert not jnp.allclose(r1.states, r2.states)

    def test_trials_are_diverse(self):
        setup = _build_test_setup()
        results = conduct_monte_carlo_gpu(setup, n_trials=4, seed=0)

        # Each trial should get a different initial state → different trajectory
        # Compare first and second trial
        assert not jnp.allclose(results.states[0], results.states[1])

    def test_wall_time_is_positive(self):
        setup = _build_test_setup()
        results = conduct_monte_carlo_gpu(setup, n_trials=2, seed=0)
        assert results.wall_time_s > 0


class TestSafetyStatistics:
    def test_computes_from_results(self):
        setup = _build_test_setup()
        results = conduct_monte_carlo_gpu(setup, n_trials=4, seed=0)
        stats = compute_safety_statistics(results)

        assert isinstance(stats, SafetyStatistics)
        assert stats.n_trials == 4
        assert 0.0 <= stats.violation_rate <= 1.0
        assert stats.per_trial_min_barrier.shape == (4,)
        assert stats.per_trial_violated.shape == (4,)

    def test_no_barriers_returns_zero_violations(self):
        """When controller_datas has no barrier data, violations should be zero."""
        # Create a minimal result with no sub_data
        dummy_results = MonteCarloGPUResults(
            states=jnp.zeros((2, 10, 2)),
            controls=jnp.zeros((2, 10, 2)),
            controller_datas=ControllerData(),
            planner_datas=PlannerData(),
            wall_time_s=0.1,
            n_trials=2,
        )
        stats = compute_safety_statistics(dummy_results)
        assert stats.violation_rate == 0.0
        assert stats.total_violation_steps == 0
