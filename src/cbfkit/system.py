import jax.numpy as jnp
from jax import jit
from tail_recursion import tail_recursive


def step(state, dynamics, controller, dt):
    f, g = dynamics(state)
    u = controller(state)
    return jnp.add(state, jnp.multiply(dt, jnp.add(f, jnp.matmul(g, u))))


def simulate_iter(state, dynamics, controller, dt, num_steps):
    for _ in range(num_steps):
        state = step(state, dynamics, controller, dt)
        yield state


def simulate(state, dynamics, controller, dt, num_steps):
    return tuple(simulate_iter(state, dynamics, controller, dt, num_steps))


# @tail_recursive
# def simulate_system(state, dynamics, dt, num_steps, accumulator=None):
#     if accumulator is None:
#         accumulator = tuple()

#     if num_steps == 0:
#         return accumulator + (state,)
#     new_state = step(state, dynamics, dt)
#     recurse(new_state, dynamics, dt, num_steps - 1, accumulator + (new_state,))
