from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.debug as jdebug
import jax.numpy as jnp
from jax import lax, random

# Hermes: Error code for NaN detected during integration
INTEGRATION_NAN_ERROR = -10

from cbfkit.utils.jit_monitor import JitMonitor
from cbfkit.simulation.integration_utils import integrate_with_cached_dynamics
from cbfkit.simulation.utils import resolve_nominal_control
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


def _make_scan_step(
    dynamics,
    integrator,
    planner,
    nominal_controller,
    controller,
    sensor,
    estimator,
    perturbation,
    sigma,
    dt,
    num_steps,
    enable_debug=True,
    progress_callback=None,
    progress_interval=1,
):
    """Factory that builds the lax.scan step function.

    When ``enable_debug=False``, host-side callbacks (NaN warning print and
    progress reporting) are omitted, making the returned ``scan_step``
    compatible with ``jax.vmap``.
    """

    def scan_step(carry, step_idx):
        # Unpack carry
        key, t, x, u, z, c, controller_data, planner_data = carry

        # Split key for this step
        key, subkey = random.split(key)

        # 1. Sensor
        y = sensor(t, x, sigma=sigma, key=key)

        # 2. Estimator - handle both 2-tuple (z, c) and 3-tuple (z, c, K) returns
        est_result = estimator(t, y, z, u, c)
        if len(est_result) == 3:
            z, c, _kalman_gain = est_result
        else:
            z, c = est_result

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

        # 5. Resolve nominal control from planner output
        u_nom, key = resolve_nominal_control(
            t,
            z,
            dt,
            key,
            g,
            nominal_controller,
            planner_data,
            u_planner,
            has_planner=(planner is not None),
        )

        # 6. Controller (CBF/CLF filter)
        key, ctrl_key = random.split(key)
        if controller is not None:
            u, controller_data = controller(t, z, u_nom, ctrl_key, controller_data)
        else:
            u = u_nom

        # Early stop conditions: planner/controller error or goal complete
        stop = (
            controller_data.error
            | controller_data.complete
            | (planner is not None and planner_data.error)
        )

        # 7. Perturbation and integration (skipped if stopped)
        def _integrate(_):
            p = perturbation(x, u, f, g)
            key_int, subkey = random.split(key)

            # Evaluate perturbation once per step.
            # This avoids repeated calls inside vector_field (e.g., 4 times for RK4),
            # reducing graph size and runtime if p is complex.
            p_val = p(subkey)
            x_next = integrate_with_cached_dynamics(
                x=x,
                u=u,
                dt=dt,
                dynamics=dynamics,
                integrator=integrator,
                f=f,
                g=g,
                perturbation_value=p_val,
            )
            return key_int, x_next

        def _hold(_):
            return key, x

        key, x_next_candidate = lax.cond(stop, _hold, _integrate, operand=None)

        # Check for NaNs in the next state to prevent divergent simulation
        nan_in_next = jnp.any(jnp.isnan(x_next_candidate))

        # If NaN is detected, revert to previous state to freeze simulation at last valid point
        x_next = jnp.where(nan_in_next, x, x_next_candidate)

        # If NaN is detected, force controller error to True.
        # This ensures the next iteration's 'stop' condition is triggered.
        current_error = controller_data.error
        new_error = current_error | nan_in_next
        controller_data = controller_data._replace(error=new_error)

        # Hermes: If NaN is detected and error_data exists, report Integration NaN error.
        if controller_data.error_data is not None:
            new_error_data = jnp.where(
                nan_in_next, INTEGRATION_NAN_ERROR, controller_data.error_data
            )
            controller_data = controller_data._replace(error_data=new_error_data)

        # Hermes: Print warning if NaN detected (disabled under vmap)
        if enable_debug:
            lax.cond(
                nan_in_next,
                lambda: jdebug.print(
                    "⚠️ Simulation stopped: NaN detected during integration at t={t}", t=t
                ),
                lambda: None,
            )

        if enable_debug and progress_callback is not None and progress_interval > 0:
            should_report = jnp.logical_or(
                step_idx == num_steps - 1, step_idx % progress_interval == 0
            )

            def _report(idx):
                # Host-side hook so we can surface progress without breaking JIT.
                def _do_report(step_value):
                    progress_callback(int(step_value))

                # ordered=True so progress updates are not reordered or dropped.
                jdebug.callback(_do_report, idx, ordered=True)

            lax.cond(should_report, _report, lambda _: None, step_idx)

        # Update time
        t_next = t + dt

        # Pack carry
        # Strip sampled_x_traj from carry to save bandwidth/memory
        planner_data_carry = planner_data._replace(sampled_x_traj=None)
        new_carry = (key, t_next, x_next, u, z, c, controller_data, planner_data_carry)

        # Output (trajectory)
        # Strip solver_params from logged data to save memory
        log_controller_data = controller_data
        if controller_data.sub_data is not None and "solver_params" in controller_data.sub_data:
            # Create a shallow copy and remove the key to avoid affecting carry
            log_sub_data = controller_data.sub_data.copy()
            del log_sub_data["solver_params"]
            log_controller_data = controller_data._replace(sub_data=log_sub_data)

        output = (x, u, z, c, log_controller_data, planner_data)

        return new_carry, output

    return scan_step


@partial(
    jax.jit,
    static_argnames=[
        "dynamics",
        "integrator",
        "planner",
        "nominal_controller",
        "controller",
        "sensor",
        "estimator",
        "perturbation",
        "progress_callback",
        "num_steps",
        "progress_interval",
    ],
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
    progress_callback: Optional[Callable[[int], None]] = None,
    progress_interval: int = 1,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, ControllerData, PlannerData]:
    """JIT-compiled simulation loop using jax.lax.scan.

    Requires:
    - planner_data and controller_data must be initialized with JAX-compatible arrays
      (no None) for any fields that will be used/updated.
    - Optional host-side progress reporting can be enabled via `progress_callback`.

    Returns
    -------
        xs, us, zs, cs, c_datas (stacked), p_datas (stacked)
    """
    print(f"JIT COMPILATION: simulator_jit (dt={dt}, num_steps={num_steps})")
    JitMonitor.increment("simulator_jit")

    scan_step = _make_scan_step(
        dynamics=dynamics,
        integrator=integrator,
        planner=planner,
        nominal_controller=nominal_controller,
        controller=controller,
        sensor=sensor,
        estimator=estimator,
        perturbation=perturbation,
        sigma=sigma,
        dt=dt,
        num_steps=num_steps,
        enable_debug=True,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
    )

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
