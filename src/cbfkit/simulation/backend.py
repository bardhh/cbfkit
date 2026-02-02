from typing import Any, Callable, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array, random

from cbfkit.integration import forward_euler
from cbfkit.integration.runge_kutta import runge_kutta_4
from cbfkit.simulation.utils import SimulationStepData
from cbfkit.utils.user_types import (
    Control,
    ControllerCallable,
    ControllerData,
    Covariance,
    DynamicsCallable,
    Estimate,
    EstimatorCallable,
    IntegratorCallable,
    Key,
    NominalControllerCallable,
    PerturbationCallable,
    PlannerCallable,
    PlannerData,
    SensorCallable,
    State,
    Time,
)


def stepper(
    dt: float,
    dynamics: DynamicsCallable,
    integrator: IntegratorCallable,
    planner: Optional[PlannerCallable],
    nominal_controller: Optional[NominalControllerCallable],
    controller: Optional[ControllerCallable],
    sensor: SensorCallable,
    estimator: EstimatorCallable,
    perturbation: PerturbationCallable,
    sigma: Array,
    key: Key,
    stl_trajectory_cost,
) -> Callable[
    [
        Time,
        State,
        Optional[Control],
        Optional[Estimate],
        Optional[Covariance],
        Optional[ControllerData],
        Optional[PlannerData],
    ],
    Tuple[Array, Array, Array, Array, ControllerData, PlannerData],
]:
    """Creates a closure to step the simulation forward by one timestep.

    Moved from simulator.py to decouple logic.
    """

    def step(
        t: Time,
        x: State,
        u: Optional[Control],
        z: Optional[Estimate],
        c: Optional[Covariance],
        controller_data: Optional[ControllerData],
        planner_data: Optional[PlannerData],
    ) -> Tuple[Array, Array, Array, Array, ControllerData, PlannerData]:
        if controller_data is None:
            controller_data = ControllerData()
        if planner_data is None:
            planner_data = PlannerData()

        nonlocal key
        key, _ = random.split(key)  # type: ignore

        if z is None:
            z = x

        y = sensor(t, x, sigma=sigma, key=key)
        # Handle both 2-tuple (z, c) and 3-tuple (z, c, K) returns from estimator
        est_result = estimator(t, y, z, u, c)
        if len(est_result) == 3:
            z, c, _kalman_gain = est_result  # K available for risk-aware controllers via kwargs
        else:
            z, c = est_result
        f, g = dynamics(x)

        if planner is None and nominal_controller is None and controller is None:
            raise ValueError(
                "At least one of planner, nominal_controller, or controller must be specified."
            )

        if stl_trajectory_cost is not None:
            planner_data = planner_data._replace(
                prev_robustness=stl_trajectory_cost(dt, planner_data.xs)
            )
        else:
            planner_data = planner_data._replace(prev_robustness=None)

        if planner is not None:
            key, planner_key = random.split(key)  # type: ignore
            u_planner, planner_data = planner(t, z, None, planner_key, planner_data)
            if planner_data.error:
                return (
                    x,
                    u if u is not None else jnp.zeros(g.shape[1]),
                    z,
                    c if c is not None else jnp.zeros((len(z), len(z))),
                    controller_data,
                    planner_data,
                )
        else:
            planner_data = planner_data._replace(u_traj=None)

        if (planner is not None) and (planner_data.u_traj is not None):
            u = u_planner
        elif planner_data.x_traj is not None:
            timestep_idx = jnp.round(t / dt).astype(int)
            timestep_idx = jnp.clip(timestep_idx, 0, planner_data.x_traj.shape[1] - 1)
            key, nom_key = random.split(key)  # type: ignore
            u, _ = nominal_controller(t, z, nom_key, planner_data.x_traj[:, timestep_idx])
        else:
            if nominal_controller is None:
                u = jnp.zeros((g.shape[1],))
            else:
                key, nom_key = random.split(key)  # type: ignore
                u, _ = nominal_controller(t, z, nom_key, None)

        key, ctrl_key = random.split(key)  # type: ignore
        if controller is not None:
            u, controller_data = controller(t, z, u, ctrl_key, controller_data)
            if controller_data.error:
                return (
                    x,
                    u,
                    z,
                    c if c is not None else jnp.zeros((len(z), len(z))),
                    controller_data,
                    planner_data,
                )
            if controller_data.complete:
                return (
                    x,
                    u,
                    z,
                    c if c is not None else jnp.zeros((len(z), len(z))),
                    controller_data,
                    planner_data,
                )
        else:
            controller_data = ControllerData()

        p = perturbation(x, u, f, g)
        key, subkey = random.split(key)  # type: ignore

        if integrator == forward_euler:
            # Optimization (Bolt): Avoid re-evaluating dynamics for Forward Euler
            # We already computed f, g = dynamics(x) earlier
            dx = f + jnp.matmul(g, u) + p(subkey)
            x = x + dx * dt
        elif integrator == runge_kutta_4:
            # Optimization (Bolt): Reuse dynamics for RK4
            # k1 = f(x, u) which we already have components for
            k1 = f + jnp.matmul(g, u) + p(subkey)

            def vector_field(s: State) -> Array:
                f_s, g_s = dynamics(s)
                return f_s + jnp.matmul(g_s, u) + p(subkey)

            k2 = vector_field(x + 0.5 * dt * k1)
            k3 = vector_field(x + 0.5 * dt * k2)
            k4 = vector_field(x + dt * k3)

            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:

            def vector_field(s: State) -> Array:
                f_s, g_s = dynamics(s)
                return f_s + jnp.matmul(g_s, u) + p(subkey)

            x = integrator(x, vector_field, dt)

        u_ret = u
        c_ret = c if c is not None else jnp.zeros((len(z), len(z)))

        return x, u_ret, z, c_ret, controller_data, planner_data

    return step
