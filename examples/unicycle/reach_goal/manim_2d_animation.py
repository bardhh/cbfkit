"""Tutorial: render a CBF-filtered unicycle reach-avoid run with the Manim 2D backend.

Runs the README quick-start simulation (unicycle drives to a goal while a CBF
safety filter keeps it clear of an obstacle), then renders the trajectory with
``CBFAnimator(backend="manim-medium")``.  This is the script that produced
``media/showcase/manim_2d_animator.gif``.

Requires the ``manim`` extra (``pip install cbfkit[manim]``) plus ffmpeg, and
on macOS the cairo/pango libraries (``brew install ffmpeg cairo pango``).
"""

import os

import jax.numpy as jnp
from jax import jit

from cbfkit.certificates import concatenate_certificates, rectify_relative_degree
from cbfkit.certificates.barrier_functions import ellipsoidal_barrier_factory
from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.estimators import naive
from cbfkit.integration import runge_kutta_4
from cbfkit.sensors import perfect
from cbfkit.simulation import simulator
from cbfkit.systems.unicycle.models.olfatisaber2002approximate.dynamics import (
    approx_unicycle_dynamics,
)
from cbfkit.utils.animators.deps import _HAS_MANIM

# Simulation Parameters
initial_state = jnp.array([0.0, 0.0, 0.0])
actuation_limits = jnp.array([5.0, jnp.pi])
goal = jnp.array([4.0, 0.0])
obstacle_center = jnp.array([2.0, 0.5, 0.0])
obstacle_radii = jnp.array([0.5, 0.5])
dt = 1e-2
num_steps = 500 if not os.getenv("CBFKIT_TEST_MODE") else 50

# Dynamics
dynamics = approx_unicycle_dynamics(lam=1.0)  # state: [x, y, theta]


# Nominal controller - drives toward the goal, ignorant of the obstacle
@jit
def nominal_controller(t, state, key, data):
    x, y, th = state
    heading = jnp.arctan2(goal[1] - y, goal[0] - x)
    return (
        jnp.array(
            [
                jnp.linalg.norm(jnp.array([x - goal[0], y - goal[1]])),  # speed
                jnp.arctan2(jnp.sin(heading - th), jnp.cos(heading - th)),  # steering
            ]
        ),
        {},
    )


# CBF barrier around the obstacle
cbf_factory, _, _ = ellipsoidal_barrier_factory(
    system_position_indices=(0, 1),
    obstacle_position_indices=(0, 1),
    ellipsoid_axis_indices=(0, 1),
)
barrier = rectify_relative_degree(
    function=cbf_factory(obstacle_center, obstacle_radii),
    system_dynamics=dynamics,
    state_dim=3,
    form="exponential",
)(certificate_conditions=linear_class_k(10.0))

# Safety-filtered controller
controller = vanilla_cbf_clf_qp_controller(
    control_limits=actuation_limits,
    dynamics_func=dynamics,
    barriers=concatenate_certificates(barrier),
)

results = simulator.execute(
    x0=initial_state,
    dt=dt,
    num_steps=num_steps,
    dynamics=dynamics,
    integrator=runge_kutta_4,
    nominal_controller=nominal_controller,
    controller=controller,
    sensor=perfect,
    estimator=naive,
)
print(f"Final position: ({results.states[-1, 0]:.2f}, {results.states[-1, 1]:.2f})")

# Render with the Manim 2D backend (skipped in test mode / without manim)
if os.getenv("CBFKIT_TEST_MODE"):
    print("CBFKIT_TEST_MODE set: skipping Manim render.")
elif not _HAS_MANIM:
    print("Manim not found. Rendering disabled. Install 'cbfkit[manim]' to enable it.")
else:
    import numpy as np

    from cbfkit.utils.animator import CBFAnimator

    anim = CBFAnimator(
        np.asarray(results.states),
        dt=dt,
        backend="manim-medium",  # or "manim" / "manim-low" for quicker renders
        title="CBF Safety Filter: Unicycle Reach-Avoid",
        aspect="equal",
        x_lim=(-0.7, 4.7),
        y_lim=(-1.2, 1.8),
    )
    anim.add_goal(np.asarray(goal), radius=0.25, color="g", label="Goal")
    anim.add_obstacle(np.asarray(obstacle_center[:2]), radius=float(obstacle_radii[0]), alpha=0.3)
    anim.add_agent(x_idx=0, y_idx=1, body_radius=0.12, body_color="tab:blue", trail=True)
    anim.show_time()
    out = anim.save("manim_2d_unicycle_reach_avoid.mp4")
    print(f"Animation saved to {out}")
