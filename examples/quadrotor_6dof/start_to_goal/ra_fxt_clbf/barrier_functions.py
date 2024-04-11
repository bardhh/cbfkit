import jax.numpy as jnp
from cbfkit.models.quadrotor_6dof.certificate_functions.barrier_functions import attitude, altitude

# Define attitude CBF
attitude_limit = jnp.pi / 4
attitude_bf, attitude_bj, attitude_bh, attitude_bt = attitude(attitude_limit)

# Define altitude CBF
altitude_limit = 2.0
altitude_bf, altitude_bj, altitude_bh, altitude_bt = altitude(altitude_limit)


#! Accessible Objects
barrier_functions = [
    attitude_bf,
    altitude_bf,
]

barrier_jacobians = [
    attitude_bj,
    altitude_bj,
]

barrier_hessians = [
    attitude_bh,
    altitude_bh,
]

barrier_times = [
    attitude_bt,
    altitude_bt,
]


def barriers():
    return barrier_functions, barrier_jacobians, barrier_hessians, barrier_times
