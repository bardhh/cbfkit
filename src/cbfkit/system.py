import jax.numpy as jnp
from jax import jit, random
from tail_recursion import tail_recursive

key = random.PRNGKey(0)


def step(state, dynamics, controller, dt):
    u = controller(state)
    dyn = dynamics(state)
    if len(dyn) == 2:
        f, g = dynamics(state)
        xdot = jnp.add(f, jnp.matmul(g, u))
        dx = jnp.multiply(dt, xdot)
    elif len(dyn) == 3:
        f, g, s = dynamics(state)
        dw = random.normal(key, shape=(s.shape[1],))
        xdot = jnp.add(f, jnp.matmul(g, u))
        dx = jnp.add(jnp.multiply(dt, xdot), jnp.matmul(s, dw))

    return jnp.add(state, dx)


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
