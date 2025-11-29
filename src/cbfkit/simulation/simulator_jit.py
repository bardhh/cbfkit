from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax, random

from cbfkit.utils.user_types import (
    ControllerCallable,
    ControllerData,
    Covariance,
    DynamicsCallable,
    EstimatorCallable,
    IntegratorCallable,
    NominalControllerCallable,
    PerturbationCallable,
    PlannerCallable,
    PlannerData,
    SensorCallable,
    State,
)


def simulator_jit(
    dt: float,
    num_steps: int,
    dynamics: DynamicsCallable,
    integrator: IntegratorCallable,
    planner: Optional[PlannerCallable],
    nominal_controller: Optional[NominalControllerCallable],
    controller: Optional[ControllerCallable],
    sensor: SensorCallable,
    estimator: EstimatorCallable,
    perturbation: PerturbationCallable,
    sigma: jax.Array,
    key: jax.Array,
    initial_state: State,
    initial_controller_data: ControllerData,
    initial_planner_data: PlannerData,
    initial_covariance: Optional[Covariance] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, ControllerData, PlannerData]:
    """
    JIT-compiled simulation loop using jax.lax.scan.

    Requires:
    - planner_data and controller_data must be initialized with JAX-compatible arrays
      (no None) for any fields that will be used/updated.

    Returns:
        xs, us, zs, cs, c_datas (stacked), p_datas (stacked)
    """

    # Define the scan step function
    def scan_step(carry, step_idx):
        # Unpack carry
        key, t, x, u, z, c, controller_data, planner_data = carry

        # Split key for this step
        key, subkey = random.split(key)

        # 1. Sensor
        y = sensor(t, x, sigma=sigma, key=key)

        # 2. Estimator
        z, c = estimator(t, y, z, u, c)

        # 3. Dynamics (True)
        f, g = dynamics(x)

        # 4. Planner
        if planner is not None:
            key, planner_key = random.split(key)
            # Note: We assume planner signature matches and is JIT-able
            u_planner, planner_data = planner(t, z, None, planner_key, planner_data)
        else:
            u_planner = jnp.zeros((g.shape[1],))

        # 5. Nominal Controller
        # Logic: Check if planner provided u_traj or x_traj

        # Check for u_traj (Control Trajectory)
        # We assume if 'u_traj' field exists and is not None, we might use it.
        # We rely on the structure being static (determined by initial_planner_data).

        use_planner_u = False
        if planner is not None:
            # If planner exists, u_planner is valid.
            if planner_data.u_traj is not None:
                use_planner_u = True

        if use_planner_u:
            u_nom = u_planner
        else:
            # Check x_traj
            use_x_traj = False
            if planner_data.x_traj is not None:
                use_x_traj = True

            if use_x_traj:
                traj = planner_data.x_traj
                # Calculate index
                idx = jnp.round(t / dt).astype(int)
                idx = jnp.clip(idx, 0, traj.shape[1] - 1)
                x_des = traj[:, idx]

                key, nom_key = random.split(key)
                u_nom, _ = nominal_controller(t, z, nom_key, x_des)
            else:
                if nominal_controller is not None:
                    key, nom_key = random.split(key)
                    u_nom, _ = nominal_controller(t, z, nom_key, None)
                else:
                    u_nom = jnp.zeros((g.shape[1],))

        # 6. Controller (CBF/CLF filter)
        key, ctrl_key = random.split(key)
        if controller is not None:
            u, controller_data = controller(t, z, u_nom, ctrl_key, controller_data)
        else:
            u = u_nom

        # 7. Perturbation
        p = perturbation(x, u, f, g)

        # 8. Integration
        key, subkey = random.split(key)
        xdot = f + jnp.matmul(g, u) + p(subkey)
        x_next = integrator(x, xdot, dt)

        # Update time
        t_next = t + dt

        # Pack carry
        new_carry = (key, t_next, x_next, u, z, c, controller_data, planner_data)

        # Output (trajectory)
        output = (x, u, z, c, controller_data, planner_data)

        return new_carry, output

    # Initialize carry
    # u, z, c need initial values.
    # We use dummy values for the first step logic,
    # but x must be initial_state.

    # Initial control/estimate
    u0 = jnp.zeros((dynamics(initial_state)[1].shape[1],))
    z0 = initial_state  # Naive estimate

    if initial_covariance is not None:
        c0 = initial_covariance
    else:
        c0 = jnp.zeros((initial_state.shape[0], initial_state.shape[0]))

    carry_init = (
        key,
        0.0,  # t=0
        initial_state,
        u0,
        z0,
        c0,
        initial_controller_data,
        initial_planner_data,
    )

    # Run scan
    final_carry, trajectory = lax.scan(scan_step, carry_init, jnp.arange(num_steps))

    # Unpack trajectory
    xs, us, zs, cs, c_datas, p_datas = trajectory

    return xs, us, zs, cs, c_datas, p_datas
