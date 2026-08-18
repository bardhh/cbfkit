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
    emit_estimates=True,
    log_planner_samples=False,
    plant=None,
):
    """Factory that builds the lax.scan step function.

    When ``enable_debug=False``, host-side callbacks (NaN warning print and
    progress reporting) are omitted, making the returned ``scan_step``
    compatible with ``jax.vmap``.

    ``log_planner_samples`` controls whether ``PlannerData.sampled_x_traj`` --
    one MPPI sample batch of shape ``(n_samples * state_dim, horizon)`` --
    survives the step. It is dropped by default: stacking it over the horizon
    dwarfs the trajectory itself (~320 MB for 1000 samples over 200 steps). Set
    it to ``True`` when a caller needs the rollout cloud, e.g. to animate the
    full-horizon MPPI overlay. The flag applies to the carry and the emitted
    output together, never one alone: the ``_held`` and ``_advance`` branches of
    the stop gate must return the same pytree structure, so a stripped carry
    paired with a populated output would be a structure mismatch. When it is
    ``True`` the initial carry's ``planner_data.sampled_x_traj`` must therefore
    be a correctly shaped array rather than ``None``.

    When ``emit_estimates=False``, the estimate ``z`` and covariance ``c`` are
    still carried (the estimator needs them) but are not emitted as per-step
    scan outputs, so ``lax.scan`` does not stack them over the horizon. Callers
    that discard them -- the vmapped Monte Carlo path -- then avoid
    materialising ``(num_steps, n)`` and ``(num_steps, n, n)`` buffers per
    trajectory. The emitted tuple is ``(x, u, controller_data, planner_data)``
    instead of ``(x, u, z, c, controller_data, planner_data)``.

    The carry holds a ``stopped`` flag: once a step reports goal completion, a
    controller/planner error, or a NaN, every later step skips the whole
    pipeline and re-emits the held values. Under ``jax.vmap`` the gate lowers to
    a select, so both branches still execute per lane and only the emitted
    values change; the held branch keeps the last valid (finite) values, so a
    lane that has stopped contributes no new data.

    When ``plant`` is given (a ``DiscretePlant`` such as ``MujocoPlant``), the
    state leaf of the carry is the plant's own opaque state (``mjx.Data``) and
    the flat vector fed to sensor/estimator/controller and emitted as the
    logged state is ``plant.to_state(state)``. ``dynamics``, ``integrator`` and
    ``perturbation`` are not used on that path.
    """

    if plant is None:
        _to_x = lambda s: s  # noqa: E731  -- carry leaf is the flat state
    else:
        _to_x = plant.to_state  # carry leaf is the plant's opaque state (mjx.Data)

    def _advance(key, t, s, x, u, z, c, controller_data, planner_data, stopped):
        """Run one full pipeline step and return the next carry values.

        ``s`` is the carry's state leaf; ``x`` is its flat projection (equal to
        ``s`` when there is no plant).
        """
        # One split per step, handing a dedicated subkey to each consumer. The
        # eager backend (backend.py) derives its subkeys the same way in the
        # same order, and test_rng_consistency pins the two streams together --
        # change one scheme and you must change the other.
        key, sensor_key, planner_key, nom_key, ctrl_key, pert_key = random.split(key, 6)

        # 1. Sensor
        y = sensor(t, x, sigma=sigma, key=sensor_key)

        # 2. Estimator - handle both 2-tuple (z, c) and 3-tuple (z, c, K) returns
        est_result = estimator(t, y, z, u, c)
        if len(est_result) == 3:
            z, c, _kalman_gain = est_result
        else:
            z, c = est_result

        # 3. Dynamics (True)
        if plant is None:
            f, g = dynamics(x)
        else:
            f = None
            # resolve_nominal_control reads only g.shape[1]; a zero placeholder keeps its signature.
            g = jnp.zeros((x.shape[0], plant.nu))

        # 4. Planner
        if planner is not None:
            # Note: We assume planner signature matches and is JIT-able
            u_planner, planner_data = planner(t, z, None, planner_key, planner_data)
        else:
            u_planner = jnp.zeros((g.shape[1],))

        # 5. Resolve nominal control from planner output
        u_nom = resolve_nominal_control(
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

        # 6. Controller (CBF/CLF filter)
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
            if plant is not None:
                return plant.step(s, u)

            p = perturbation(x, u, f, g)

            # Evaluate perturbation once per step.
            # This avoids repeated calls inside vector_field (e.g., 4 times for RK4),
            # reducing graph size and runtime if p is complex.
            p_val = p(pert_key)
            x_next = integrate_with_cached_dynamics(
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
            return x_next

        def _skip_integration(_):
            return s

        s_next_candidate = lax.cond(stop, _skip_integration, _integrate, operand=None)
        x_next_candidate = _to_x(s_next_candidate)

        # Check for NaNs in the next state to prevent divergent simulation
        nan_in_next = jnp.any(jnp.isnan(x_next_candidate))

        # If NaN is detected, revert to previous state to freeze simulation at last valid
        # point. Hold the *carry leaf*; tree.map covers both the flat array and mjx.Data.
        s_next = jax.tree.map(lambda a, b: jnp.where(nan_in_next, a, b), s, s_next_candidate)

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

        # Strip sampled_x_traj from carry to save bandwidth/memory. Retained
        # verbatim when the caller opted in, so the carry and the emitted output
        # agree with the initial carry's structure.
        if not log_planner_samples:
            planner_data = planner_data._replace(sampled_x_traj=None)

        # Latch the stop flag so every later step takes the held branch.
        stop_next = stopped | jnp.asarray(stop) | nan_in_next

        return (key, s_next, u, z, c, controller_data, planner_data, stop_next)

    def scan_step(carry, step_idx):
        # Unpack carry
        key, t, s, u, z, c, controller_data, planner_data, stopped = carry
        x = _to_x(s)

        # `stopped` is False on the first step, so the pipeline always runs at
        # least once. Both branches return the carry leaves, whose avals lax.scan
        # already pins to the incoming carry, so the structures agree by
        # construction.
        def _held(_):
            return (key, s, u, z, c, controller_data, planner_data, stopped)

        (
            key_next,
            s_next,
            u,
            z,
            c,
            controller_data,
            planner_data,
            stop_next,
        ) = lax.cond(
            stopped,
            _held,
            lambda _: _advance(key, t, s, x, u, z, c, controller_data, planner_data, stopped),
            operand=None,
        )

        if enable_debug and progress_callback is not None and progress_interval > 0:
            # Reported outside the stop gate so the progress bar still reaches
            # the end of the horizon after an early stop.
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
        new_carry = (
            key_next,
            t_next,
            s_next,
            u,
            z,
            c,
            controller_data,
            planner_data,
            stop_next,
        )

        # Output (trajectory)
        # Carry-only sub_data entries (solver warm starts, MPC knots) are not
        # stacked over the horizon; drop them from the emitted copy only.
        log_controller_data = controller_data
        if controller_data.sub_data is not None:
            dropped = [k for k in ("solver_params", "mpc") if k in controller_data.sub_data]
            if dropped:
                log_sub_data = controller_data.sub_data.copy()
                for k in dropped:
                    del log_sub_data[k]
                log_controller_data = controller_data._replace(sub_data=log_sub_data)

        # planner_data here is whatever _advance/_held produced, so its
        # sampled_x_traj follows log_planner_samples: absent by default, stacked
        # to (num_steps, n_samples * state_dim, horizon) when opted in. The
        # eager path applies the same rule before logging.
        if emit_estimates:
            output = (x, u, z, c, log_controller_data, planner_data)
        else:
            output = (x, u, log_controller_data, planner_data)

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
        "log_planner_samples",
        "plant",
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
    log_planner_samples: bool = False,
    plant=None,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, ControllerData, PlannerData]:
    """JIT-compiled simulation loop using jax.lax.scan.

    When ``plant`` is given, ``dynamics``/``integrator`` may be ``None``: the
    scan carries ``plant.from_state(initial_state)`` and logs
    ``plant.to_state(...)`` each step, so the returned ``xs`` keeps the flat
    ``(num_steps, plant.state_dim)`` shape.

    Requires:
    - planner_data and controller_data must be initialized with JAX-compatible arrays
      (no None) for any fields that will be used/updated.
    - Optional host-side progress reporting can be enabled via `progress_callback`.
    - When ``log_planner_samples=True``, ``initial_planner_data.sampled_x_traj``
      must already hold an array of the planner's sample-batch shape, since it
      becomes part of the scan carry.

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
        log_planner_samples=log_planner_samples,
        plant=plant,
    )

    # Initialize carry
    # u, z, c need initial values.
    # We use dummy values for the first step logic,
    # but x must be initial_state.

    # Initial control/estimate
    if plant is None:
        u0 = jnp.zeros((dynamics(initial_state)[1].shape[1],))
        s0 = initial_state
    else:
        u0 = jnp.zeros((plant.nu,))
        s0 = plant.from_state(initial_state)  # opaque plant state; fresh contact solve
    z0 = initial_state  # Naive estimate

    if initial_covariance is not None:
        c0 = initial_covariance
    else:
        c0 = jnp.zeros((initial_state.shape[0], initial_state.shape[0]))

    carry_init = (
        key,
        0.0,  # t=0
        s0,
        u0,
        z0,
        c0,
        initial_controller_data,
        initial_planner_data,
        jnp.asarray(False),  # stopped: the first step is never gated
    )

    # Run scan
    final_carry, trajectory = lax.scan(scan_step, carry_init, jnp.arange(num_steps))

    # Unpack trajectory
    xs, us, zs, cs, c_datas, p_datas = trajectory

    return xs, us, zs, cs, c_datas, p_datas
