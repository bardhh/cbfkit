import jax.numpy as jnp
from jax import jit, Array


@jit
def rotation_body_frame_to_inertial_frame(x: Array) -> Array:
    """Computes the rotation matrix from the body-fixed frame to the inertial
    frame based on the current state.

    Arguments:
        x (Array): current quadrotor state

    Returns:
        rotation_matrix (Array): rotation matrix from body-fixed to inertial frame

    """
    rotation = jnp.array(
        [
            [
                jnp.cos(x[7]) * jnp.cos(x[8]),
                jnp.sin(x[6]) * jnp.sin(x[7]) * jnp.cos(x[8]) - jnp.cos(x[6]) * jnp.sin(x[8]),
                jnp.cos(x[6]) * jnp.sin(x[7]) * jnp.cos(x[8]) + jnp.sin(x[6]) * jnp.sin(x[8]),
            ],
            [
                jnp.cos(x[7]) * jnp.sin(x[8]),
                jnp.sin(x[6]) * jnp.sin(x[7]) * jnp.sin(x[8]) + jnp.cos(x[6]) * jnp.cos(x[8]),
                jnp.cos(x[6]) * jnp.sin(x[7]) * jnp.sin(x[8]) - jnp.sin(x[6]) * jnp.cos(x[8]),
            ],
            [jnp.sin(x[7]), -jnp.sin(x[6]) * jnp.cos(x[7]), -jnp.cos(x[6]) * jnp.cos(x[7])],
        ]
    )

    return rotation
