import numpy as np
import jax.numpy as jnp

x_max = 5.0
y_max = 5.0
x_rand = np.random.uniform(low=-x_max, high=x_max)
y_rand = np.random.uniform(low=-y_max, high=y_max)
a_rand = np.random.uniform(low=-jnp.pi, high=jnp.pi)
initial_state = jnp.array([x_rand, y_rand, a_rand])
desired_state = jnp.array([0.0, 0.0, 0])

Q = 0.5 * jnp.eye(len(initial_state))
R = None
