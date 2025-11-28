from unittest.mock import Mock

import jax.numpy as jnp
import pytest
from jax import random

from cbfkit.simulation.simulator import stepper
from cbfkit.utils.user_types import ControllerData, PlannerData


def test_stepper_basic():
    # Setup
    dt = 0.1
    key = random.PRNGKey(0)
    sigma = jnp.array([0.0])

    # Mocks
    mock_dynamics = Mock()
    mock_dynamics.return_value = (jnp.array([0.1]), jnp.array([[1.0]]))  # f, g

    mock_integrator = Mock()
    mock_integrator.side_effect = lambda x, xdot, dt: x + xdot * dt

    mock_planner = Mock()
    # returns u_planner, planner_data
    mock_planner.return_value = (jnp.array([1.0]), PlannerData(u_traj=jnp.array([1.0])))

    mock_nominal_controller = Mock()
    mock_nominal_controller.return_value = (jnp.array([1.0]), {})

    mock_controller = Mock()
    # returns u, controller_data
    mock_controller.return_value = (jnp.array([2.0]), ControllerData())

    mock_sensor = Mock()
    mock_sensor.side_effect = lambda t, x, sigma, key: x  # Pass through

    mock_estimator = Mock()
    # returns z, c
    mock_estimator.side_effect = lambda t, y, z, u, c: (y, c)  # Pass through

    mock_perturbation = Mock()
    mock_p_func = Mock()
    mock_p_func.return_value = jnp.array([0.0])
    mock_perturbation.return_value = mock_p_func

    # Create stepper
    step_fn = stepper(
        dt=dt,
        dynamics=mock_dynamics,
        integrator=mock_integrator,
        planner=mock_planner,
        nominal_controller=mock_nominal_controller,
        controller=mock_controller,
        sensor=mock_sensor,
        estimator=mock_estimator,
        perturbation=mock_perturbation,
        sigma=sigma,
        key=key,
        stl_trajectory_cost=None,
    )

    # Initial state
    t = 0.0
    x = jnp.array([1.0])
    u = jnp.array([0.0])
    z = jnp.array([1.0])
    c = jnp.array([0.0])
    controller_data = ControllerData()
    planner_data = PlannerData()

    # Call step
    x_next, u_next, z_next, c_next, cd_next, pd_next = step_fn(
        t, x, u, z, c, controller_data, planner_data
    )

    # Assertions
    # Dynamics: f=[0.1], g=[1.0]. u=[2.0] (from controller). p=[0.0].
    # xdot = f + g*u + p = 0.1 + 1.0*2.0 + 0 = 2.1
    # x_next = x + xdot*dt = 1.0 + 2.1 * 0.1 = 1.21

    assert jnp.allclose(x_next, jnp.array([1.21]))
    assert jnp.allclose(u_next, jnp.array([2.0]))
    assert isinstance(cd_next, ControllerData)
    assert isinstance(pd_next, PlannerData)

    # Verify calls
    mock_dynamics.assert_called()
    mock_sensor.assert_called()
    mock_estimator.assert_called()
    mock_planner.assert_called()
    mock_controller.assert_called()
    mock_perturbation.assert_called()
    mock_integrator.assert_called()


def test_stepper_controller_complete():
    # Setup
    dt = 0.1
    key = random.PRNGKey(0)
    sigma = jnp.array([0.0])

    mock_dynamics = Mock(return_value=(jnp.array([0.0]), jnp.array([[0.0]])))
    mock_integrator = Mock(side_effect=lambda x, xdot, dt: x)
    mock_planner = Mock(return_value=(jnp.array([0.0]), PlannerData()))
    mock_nominal_controller = Mock(return_value=(jnp.array([0.0]), {}))

    # Controller returns complete=True
    mock_controller = Mock(return_value=(jnp.array([0.0]), ControllerData(complete=True)))

    mock_sensor = Mock(side_effect=lambda t, x, sigma, key: x)
    mock_estimator = Mock(side_effect=lambda t, y, z, u, c: (y, c))
    mock_perturbation = Mock(return_value=lambda k: jnp.array([0.0]))

    step_fn = stepper(
        dt=dt,
        dynamics=mock_dynamics,
        integrator=mock_integrator,
        planner=mock_planner,
        nominal_controller=mock_nominal_controller,
        controller=mock_controller,
        sensor=mock_sensor,
        estimator=mock_estimator,
        perturbation=mock_perturbation,
        sigma=sigma,
        key=key,
        stl_trajectory_cost=None,
    )

    t = 0.0
    x = jnp.array([1.0])
    u = jnp.array([0.0])
    z = jnp.array([1.0])
    c = jnp.array([0.0])
    controller_data = ControllerData()
    planner_data = PlannerData()

    x_next, u_next, z_next, c_next, cd_next, pd_next = step_fn(
        t, x, u, z, c, controller_data, planner_data
    )

    assert cd_next.complete is True
    # Integrator should not be called if controller completes early
    mock_integrator.assert_not_called()


def test_stepper_planner_error():
    # Setup
    dt = 0.1
    key = random.PRNGKey(0)
    sigma = jnp.array([0.0])

    mock_dynamics = Mock(return_value=(jnp.array([0.0]), jnp.array([[0.0]])))

    # Planner returns error=True
    mock_planner = Mock(return_value=(jnp.array([0.0]), PlannerData(error=True)))

    mock_sensor = Mock(side_effect=lambda t, x, sigma, key: x)
    mock_estimator = Mock(side_effect=lambda t, y, z, u, c: (y, c))

    mock_integrator = Mock()
    mock_nominal_controller = Mock()
    mock_controller = Mock()
    mock_perturbation = Mock()

    step_fn = stepper(
        dt=dt,
        dynamics=mock_dynamics,
        integrator=mock_integrator,
        planner=mock_planner,
        nominal_controller=mock_nominal_controller,
        controller=mock_controller,
        sensor=mock_sensor,
        estimator=mock_estimator,
        perturbation=mock_perturbation,
        sigma=sigma,
        key=key,
        stl_trajectory_cost=None,
    )

    t = 0.0
    x = jnp.array([1.0])
    u = jnp.array([0.0])
    z = jnp.array([1.0])
    c = jnp.array([0.0])
    controller_data = ControllerData()
    planner_data = PlannerData()

    x_next, u_next, z_next, c_next, cd_next, pd_next = step_fn(
        t, x, u, z, c, controller_data, planner_data
    )

    assert pd_next.error is True
    # Integrator should not be called if planner errors out early
    mock_integrator.assert_not_called()
