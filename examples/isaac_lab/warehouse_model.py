"""Planar moving-obstacle model used by the warehouse showcase (no Isaac imports).

State: robot x/y, cart x/y, cart vx/vy. The geometric margin is empirical;
this model does not certify the articulated robot or its tracking error.
"""

import jax.numpy as jnp

SAFE_RADIUS = 1.60
AISLE_LIMIT = 1.45
ALPHA = 1.25
GOAL = (6.0, 0.0)
CONTROL_LIMITS = (0.8, 0.35)


def dynamics(z):
    drift = jnp.array([0.0, 0.0, z[4], z[5], 0.0, 0.0], dtype=z.dtype)
    control = jnp.zeros((6, 2), dtype=z.dtype).at[:2, :].set(jnp.eye(2, dtype=z.dtype))
    return drift, control


def cart_barrier(z):
    delta = z[:2] - z[2:4]
    return jnp.dot(delta, delta) - SAFE_RADIUS**2


def residuals(z, u):
    """All three command-level CBF residuals; nonnegative is feasible."""
    delta = z[:2] - z[2:4]
    return jnp.array(
        [
            2 * jnp.dot(delta, u - z[4:6]) + ALPHA * cart_barrier(z),
            -u[1] + ALPHA * (AISLE_LIMIT - z[1]),
            u[1] + ALPHA * (AISLE_LIMIT + z[1]),
        ]
    )


def make_filter(num_envs, dt):
    from cbfkit.certificates import concatenate_certificates, generate_certificate
    from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
    from cbfkit.optimization.quadratic_program import get_solver
    from cbfkit.wrappers import BatchedSafetyFilter

    barriers = concatenate_certificates(
        *[
            generate_certificate(h, linear_class_k(ALPHA), input_style="state")
            for h in (cart_barrier, lambda z: AISLE_LIMIT - z[1], lambda z: AISLE_LIMIT + z[1])
        ]
    )
    return BatchedSafetyFilter.from_cbf_qp(
        num_envs=num_envs,
        dt=dt,
        dynamics=dynamics,
        barriers=barriers,
        # Favor yielding along the aisle over chasing the cart sideways. The
        # blind walking policy is less stable during sustained lateral motion.
        control_limits=jnp.array(CONTROL_LIMITS),
        p_mat=jnp.diag(jnp.array([1.0, 8.0])),
        solver=get_solver("fast"),
    )


def scenario_parameters(seed, count):
    """Fixed representative first scenario plus seeded variations, shared by all modes."""
    import numpy as np

    rng = np.random.default_rng(seed)
    result = np.stack(
        [
            rng.uniform(2.4, 3.0, count),
            rng.uniform(-2.7, -2.2, count),
            rng.uniform(0.55, 0.75, count),
            rng.uniform(-0.18, 0.18, count),
        ],
        axis=-1,
    ).astype(np.float32)
    if seed == 42:
        result[0] = [2.6, -2.4, 0.65, 0.0]
    return result
