from typing import Callable, Optional, Tuple

import jax.numpy as jnp
from jax import Array, random

from cbfkit.simulation.integration_utils import integrate_with_cached_dynamics
from cbfkit.simulation.utils import resolve_nominal_control
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
    StlTrajectoryCostCallable,
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
    stl_trajectory_cost: Optional[StlTrajectoryCostCallable],
    plant=None,
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

    When ``plant`` is given, ``dynamics``/``integrator``/``perturbation`` are
    unused; the plant's opaque state is advanced with ``plant.step``. This eager
    plant path dispatches the whole state pytree from Python every step and is
    debug-only -- ``execute()`` warns when it is selected.
    """

    # Eager plant path (debug-only): keep the plant's opaque state alongside the
    # flat x the caller passes in. If the caller hands us an x that does not
    # match (fresh run, NaN clamp), rebuild from it.
    plant_state = None

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
        # One split per step, handing a dedicated subkey to each consumer. The
        # JIT backend (simulator_jit._advance) derives its subkeys the same way
        # in the same order, and test_rng_consistency pins the two streams
        # together -- change one scheme and you must change the other.
        key, sensor_key, planner_key, nom_key, ctrl_key, pert_key = random.split(key, 6)  # type: ignore

        if z is None:
            z = x

        y = sensor(t, x, sigma=sigma, key=sensor_key)
        # Handle both 2-tuple (z, c) and 3-tuple (z, c, K) returns from estimator
        est_result = estimator(t, y, z, u, c)
        if len(est_result) == 3:
            z, c, _kalman_gain = est_result  # K available for risk-aware controllers via kwargs
        else:
            z, c = est_result

        nonlocal plant_state
        if plant is None:
            f, g = dynamics(x)
        else:
            if plant_state is None or not bool(jnp.array_equal(plant.to_state(plant_state), x)):
                plant_state = plant.from_state(x)
            f = None
            g = jnp.zeros((x.shape[0], plant.nu))  # resolve_nominal_control reads only g.shape[1]

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
            u_planner = jnp.zeros(g.shape[1])
            planner_data = planner_data._replace(u_traj=None)

        u = resolve_nominal_control(
            t,
            z,
            dt,
            nom_key,
            g,
            nominal_controller,
            planner_data,
            u_planner,
            has_planner=(planner is not None),
        )

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

        if plant is None:
            p = perturbation(x, u, f, g)
            p_val = p(pert_key)
            x = integrate_with_cached_dynamics(
                x=x,
                u=u,
                dt=dt,
                dynamics=dynamics,
                integrator=integrator,
                f=f,
                g=g,
                perturbation_value=p_val,
                perturbation_is_increment=getattr(perturbation, "is_increment", False),
            )
        else:
            # pert_key stays split above even though unused: the split order is pinned.
            next_state = plant.step(plant_state, u)
            x_next = plant.to_state(next_state)
            if not bool(jnp.any(jnp.isnan(x_next))):
                plant_state = next_state  # commit only finite states; the caller clamps x
            x = x_next

        u_ret = u
        c_ret = c if c is not None else jnp.zeros((len(z), len(z)))

        return x, u_ret, z, c_ret, controller_data, planner_data

    return step
