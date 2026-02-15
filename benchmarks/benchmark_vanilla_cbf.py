import time
import jax
import jax.numpy as jnp
import cbfkit.simulation.simulator as sim
import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
from cbfkit.certificates import concatenate_certificates, rectify_relative_degree
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller as cbf_controller
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.utils.user_types import PlannerData
from examples.unicycle.common.ellipsoidal_obstacle import cbf as ellipsoid_cbf

def benchmark():
    # Simulation parameters
    tf = 10.0 # Increased from 2.0
    dt = 0.01
    num_steps = int(tf / dt)

    init_state = jnp.array([0.0, 0.0, 0.0, jnp.pi / 4])
    desired_state = jnp.array([2.0, 4.0, 0.0, 0.0])
    actuation_constraints = jnp.array([100.0, 100.0])

    unicycle_dynamics = unicycle.plant(lam=1.0)
    unicycle_dynamics.a_max = actuation_constraints[0]
    unicycle_dynamics.omega_max = actuation_constraints[1]
    unicycle_dynamics.v_max = 1.0
    unicycle_dynamics.goal_tol = 0.25

    uniycle_nom_controller = unicycle.controllers.proportional_controller(
        dynamics=unicycle_dynamics,
        Kp_pos=1.0,
        Kp_theta=1.0,
    )

    obstacles = [
        (1, 2.0, 0.0),
        (3.0, 2.0, 0.0),
        (-1.0, 1.0, 0.0),
        (0.5, -1.0, 0.0),
    ]
    ellipsoids = [
        (0.5, 1.5),
        (0.75, 2.0),
        (1.0, 0.75),
        (0.75, 0.5),
    ]

    barriers = [
        rectify_relative_degree(
            function=ellipsoid_cbf(jnp.array(obs), jnp.array(ell)),
            system_dynamics=unicycle_dynamics,
            state_dim=len(init_state),
            form="high-order",
        )(
            certificate_conditions=zeroing_barriers.linear_class_k(5.0),
        )
        for obs, ell in zip(obstacles, ellipsoids)
    ]

    barrier_packages = concatenate_certificates(*barriers)

    # Enable tunable_class_k to trigger the code path we want to optimize
    controller = cbf_controller(
        control_limits=actuation_constraints,
        dynamics_func=unicycle_dynamics,
        barriers=barrier_packages,
        tunable_class_k=True,
    )

    print("Warming up...")
    # Warmup with SAME number of steps
    sim.execute(
        x0=init_state,
        dt=dt,
        num_steps=num_steps,
        dynamics=unicycle_dynamics,
        integrator=integrator,
        nominal_controller=uniycle_nom_controller,
        controller=controller,
        sensor=sensor,
        estimator=estimator,
        verbose=False,
        planner_data=PlannerData(
            x_traj=jnp.tile(desired_state.reshape(-1, 1), (1, num_steps + 1)),
            prev_robustness=None,
        ),
        use_jit=True,
    )

    print("Running benchmark...")
    start_time = time.time()
    num_runs = 20
    for _ in range(num_runs):
        sim.execute(
            x0=init_state,
            dt=dt,
            num_steps=num_steps,
            dynamics=unicycle_dynamics,
            integrator=integrator,
            nominal_controller=uniycle_nom_controller,
            controller=controller,
            sensor=sensor,
            estimator=estimator,
            verbose=False,
            planner_data=PlannerData(
                x_traj=jnp.tile(desired_state.reshape(-1, 1), (1, num_steps + 1)),
                prev_robustness=None,
            ),
            use_jit=True,
        )
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    print(f"Average execution time: {avg_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()
