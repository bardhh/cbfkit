"""The JIT scan must not stack per-step MPPI sample batches into its outputs.

``PlannerData.sampled_x_traj`` holds one MPPI sample batch per step
(``n_samples * state_dim, horizon``). Stacking it across the horizon dwarfs the
trajectory itself (measured 12,850x on a realistic MPPI shape) and OOMs GPU
runs; the scan emits the stripped planner data instead, matching the eager
path's logging behaviour.

Callers that need the rollout cloud -- the MPPI animations in
``examples/unicycle/reach_goal`` -- opt back in with ``log_planner_samples=True``
and pay the memory. Both directions are pinned here.
"""

import jax.numpy as jnp

import cbfkit.simulation.simulator as sim
from cbfkit.estimators.naive import naive as estimator
from cbfkit.integration import forward_euler
from cbfkit.sensors import perfect as sensor
from cbfkit.utils.user_types import PlannerData

N_SAMPLES, HORIZON, NUM_STEPS = 7, 4, 5


def _dynamics(x):
    return jnp.zeros_like(x), jnp.eye(len(x))


def _mppi_shaped_planner(t, x, u_prev, key, data):
    """Writes a fresh sample batch into planner data every step, like MPPI."""
    return jnp.zeros(2), data._replace(
        u_traj=jnp.zeros((2, HORIZON)),
        sampled_x_traj=jnp.zeros((N_SAMPLES * 2, HORIZON)),
    )


def test_jit_scan_does_not_stack_sampled_x_traj():
    planner_data = PlannerData(
        u_traj=jnp.zeros((2, HORIZON)),
        sampled_x_traj=jnp.zeros((N_SAMPLES * 2, HORIZON)),
    )
    results = sim.execute(
        x0=jnp.zeros(2),
        dt=0.01,
        num_steps=NUM_STEPS,
        dynamics=_dynamics,
        integrator=forward_euler,
        sensor=sensor,
        estimator=estimator,
        planner=_mppi_shaped_planner,
        planner_data=planner_data,
        use_jit=True,
        verbose=False,
    )
    _x, _u, _z, _p, _dk, _dv, planner_keys, planner_values = results

    stacked = [
        (key, value.shape)
        for key, value in zip(planner_keys, planner_values)
        if getattr(value, "shape", None) is not None
        and len(value.shape) >= 3
        and value.shape[0] == NUM_STEPS
        and value.shape[1] == N_SAMPLES * 2
    ]
    assert not stacked, f"sample batches stacked over the horizon: {stacked}"
    assert "sampled_x_traj" not in list(planner_keys), (
        "sampled_x_traj must be stripped from logged planner outputs, "
        f"got keys {list(planner_keys)}"
    )


def test_log_planner_samples_retains_stacked_sample_batches():
    """Opting in returns the sample batch stacked over the horizon."""
    planner_data = PlannerData(
        u_traj=jnp.zeros((2, HORIZON)),
        sampled_x_traj=jnp.zeros((N_SAMPLES * 2, HORIZON)),
    )
    results = sim.execute(
        x0=jnp.zeros(2),
        dt=0.01,
        num_steps=NUM_STEPS,
        dynamics=_dynamics,
        integrator=forward_euler,
        sensor=sensor,
        estimator=estimator,
        planner=_mppi_shaped_planner,
        planner_data=planner_data,
        use_jit=True,
        verbose=False,
        log_planner_samples=True,
    )
    planner_keys, planner_values = results.planner_keys, results.planner_values

    assert "sampled_x_traj" in planner_keys, (
        "log_planner_samples=True must keep sampled_x_traj in the logged planner "
        f"outputs, got keys {list(planner_keys)}"
    )
    sampled = planner_values[planner_keys.index("sampled_x_traj")]
    assert sampled.shape == (NUM_STEPS, N_SAMPLES * 2, HORIZON), (
        "expected the sample batch stacked over the horizon, got shape " f"{sampled.shape}"
    )
