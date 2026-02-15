
import time
import jax
import jax.numpy as jnp
from cbfkit.benchmarks.registry import register_scenario
from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.controllers.cbf_clf.generate_constraints import (
    generate_compute_vanilla_clf_constraints,
    generate_compute_zeroing_cbf_constraints,
)
from cbfkit.certificates.packager import certificate_package
from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
from cbfkit.simulation import simulator
from cbfkit.integration import forward_euler

@register_scenario("solver_stress", description="Dense obstacle field navigation")
def solver_stress(seed: int) -> dict:
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    # 20 Obstacles
    n_obs = 20
    obs_pos = jax.random.uniform(k1, (n_obs, 2), minval=1.0, maxval=9.0)
    radius = 0.5

    # Dynamics: Unicycle (x, y, theta) inputs (v, w)
    def dynamics(x):
        theta = x[2]
        f = jnp.zeros(3)
        g = jnp.array([[jnp.cos(theta), 0], [jnp.sin(theta), 0], [0, 1]])
        return f, g

    # Barriers
    barriers = []
    # Capture closure correctly by creating a factory
    def make_h(center):
        def h(x):
            return jnp.sum((x[:2] - center)**2) - radius**2
        return h

    for i in range(n_obs):
        h_func = make_h(obs_pos[i])
        # Package correctly:
        # 1. use_factory=False because h_func is the function itself
        # 2. input_style="state" because it takes only x
        # 3. Call the result with conditions
        pkg_factory = certificate_package(
            h_func,
            n=3,
            input_style="state",
            use_factory=False
        )
        barriers.append(pkg_factory(certificate_conditions=linear_class_k(1.0)))

    # Controller Setup
    gen_func = cbf_clf_qp_generator(
        generate_compute_zeroing_cbf_constraints,
        generate_compute_vanilla_clf_constraints
    )

    # We use a nominal controller wrapper to match signature
    def nominal_control(t, x):
        goal = jnp.array([10.0, 10.0])
        diff = goal - x[:2]
        dist = jnp.linalg.norm(diff)
        angle = jnp.arctan2(diff[1], diff[0])
        err_angle = angle - x[2]
        err_angle = jnp.arctan2(jnp.sin(err_angle), jnp.cos(err_angle))

        v = jnp.clip(dist, -1.0, 1.0)
        w = jnp.clip(4.0 * err_angle, -1.0, 1.0)
        return jnp.array([v, w])

    # Setup controller with 20 constraints
    controller = gen_func(
        control_limits=jnp.array([1.0, 1.0]),
        dynamics_func=dynamics,
        barriers=barriers,
        relaxable_cbf=False, # Hard constraints to force solver work
        slack_penalty_cbf=1e4,
    )

    # Wrap controller for simulator
    def controller_wrapper(t, x, u_nom, k, data):
        return controller(t, x, u_nom, k, data)

    def nominal_wrapper(t, x, k, ref):
        return nominal_control(t, x), None

    # Simulation
    x0 = jnp.array([0.0, 0.0, 0.0])
    dt = 0.05
    steps = 100 # Short but intense

    start_time = time.time()

    # Run with JIT to stress compilation and solver
    results = simulator.execute(
        x0=x0,
        dt=dt,
        num_steps=steps,
        dynamics=dynamics,
        integrator=forward_euler,
        nominal_controller=nominal_wrapper,
        controller=controller_wrapper,
        use_jit=True,
        verbose=False
    )

    duration = time.time() - start_time

    # Extract Metrics
    # Locate solver_iter key
    iter_key = None
    status_key = None

    keys = results.controller_keys
    values = results.controller_values

    # Find keys ending in 'solver_iter' or 'solver_status'
    for k in keys:
        if k.endswith("solver_iter"):
            iter_key = k
        if k.endswith("solver_status"):
            status_key = k

    avg_iter = 0.0
    max_iter = 0.0
    failures = 0

    if iter_key and len(values) > 0:
        idx = keys.index(iter_key)
        iters = values[idx]
        # Filter None or -99? JAX arrays shouldn't have None, but might have fill values
        # simulator.py uses -99 or NaN for missing.
        # But for JIT, it's dense.
        avg_iter = float(jnp.mean(iters))
        max_iter = float(jnp.max(iters))

    if status_key:
        idx = keys.index(status_key)
        statuses = values[idx]
        # Count non-1 statuses (assuming 1 is SOLVED)
        failures = int(jnp.sum(statuses != 1))

    return {
        "execution_time": duration,
        "avg_step_ms": (duration / steps) * 1000.0,
        "avg_solver_iter": avg_iter,
        "max_solver_iter": max_iter,
        "solver_failures": failures,
        "success": int(failures == 0), # Simple success definition
        "final_dist": float(jnp.linalg.norm(results.states[-1, :2] - jnp.array([10.0, 10.0])))
    }
