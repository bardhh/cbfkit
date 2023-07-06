import jax.numpy as jnp
from jax import random, Array
from typing import Any, Dict, Iterator, Optional, Tuple, List
from cbfkit.utils.logger import log, write_log
from cbfkit.utils.numerical_integration import forward_euler as integrator
from cbfkit.utils.user_types import ControllerCallable, DynamicsCallable, NumSteps, State, Time

global KEY
KEY = random.PRNGKey(0)


# This function computes the next state of a dynamical system given its current state,
# dynamics, controller, and time step.
def step(
    t: Time, state: State, dynamics: DynamicsCallable, controller: ControllerCallable, dt: Time
) -> Tuple[State, Dict[str, Any]]:
    global KEY
    u, data = controller(t, state)
    f, g, s = dynamics(state)

    # Deterministic Dynamics
    if s is None:
        xdot = f + jnp.matmul(g, u)

    # Stochastic Dynamics
    else:
        KEY, subkey = random.split(KEY)
        dw = random.normal(subkey, shape=(s.shape[1],))
        xdot = f + jnp.matmul(g, u) + jnp.matmul(s, dw)

    new_state = integrator(state, xdot, dt)

    return new_state, data


# This generator function simulates the dynamical system for a given number of steps,
# yielding the state at each step.
def simulate_iter(
    state: State,
    dynamics: DynamicsCallable,
    controller: ControllerCallable,
    dt: Time,
    num_steps: NumSteps,
) -> Iterator[Tuple[State, List[str], List[Array]]]:
    for s in range(num_steps):
        state, data = step(dt * s, state, dynamics, controller, dt)
        log(data)

        yield state, list(data.keys()), list(data.values())


# This function simulates the dynamical system for a given number of steps,
# returning a tuple of all states. Optionally, it can also write the logged data to a file.
def simulate(
    state: State,
    dynamics: DynamicsCallable,
    controller: ControllerCallable,
    dt: Time,
    num_steps: NumSteps,
    filepath: Optional[str] = None,
) -> Tuple[Array, List[str], List[Array]]:
    simulation_data = tuple(simulate_iter(state, dynamics, controller, dt, num_steps))
    if filepath is not None:
        write_log(filepath)

    state_trajectories = jnp.array([sim_data[0] for sim_data in simulation_data])
    data_keys = simulation_data[0][1]
    data_values = [sim_data[2] for sim_data in simulation_data]

    return state_trajectories, data_keys, data_values  # type: ignore[return-value]
