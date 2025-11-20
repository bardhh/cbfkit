import jax
import jax.numpy as jnp
from jax.random import multivariate_normal
from jax import jit, lax


def setup_mppi(
    dyn_func=None,
    trajectory_cost_func=None,
    stage_cost_func=None,
    terminal_cost_func=None,
    robot_state_dim=2,
    robot_control_dim=2,
    horizon=10,
    samples=10,
    control_bound=2,
    dt=0.05,
    use_GPU=True,
    costs_lambda=0.03,
    cost_perturbation_coeff=0.1,
):

    if dyn_func == None:
        print(f"Dynamics function not passed")
        exit()
    if trajectory_cost_func == None:
        if stage_cost_func == None:
            print(f"Neither Trajectury Cost nor Stage Cost function is passed.")
            exit()
        if terminal_cost_func == None:
            print(f"Neither Trajectory Cost nor Terminal Cost function is passed")
            exit()

    horizon = horizon
    samples = samples
    dt = dt
    use_gpu = use_GPU

    robot_state_dim = robot_state_dim
    robot_control_dim = robot_control_dim
    control_mu = jnp.zeros(robot_control_dim)  # .reshape(-1,1)
    control_cov = 4.0 * jnp.eye(robot_control_dim)
    control_cov_inv = jnp.linalg.inv(control_cov)
    control_bound = control_bound
    # TODO: implement separate lower and upper bounds
    # control_bound_lb = -jnp.array([1, 1]).reshape(-1, 1)
    # control_bound_ub = jnp.array([1, 1]).reshape(-1, 1)

    costs_lambda = costs_lambda
    cost_perturbation_coeff = cost_perturbation_coeff

    @jit
    def robot_dynamics_step(state, input):
        """
        Args: state and input vector
        Returns: next state
        """
        f, g = dyn_func(state[:, 0])  # , input)
        return state + (f.reshape(-1, 1) + g @ input) * dt

    @jit
    def weighted_sum(U, perturbation, costs):
        """
        Args:
            U: control input trajectory of all samples
            perturbation: random perturbation trajectory of all samples
            costs: cost of each sampled trajectory
        """
        costs = costs - jnp.min(costs)
        costs = costs / jnp.max(costs)
        lambd = costs_lambda
        weights = jnp.exp(-1.0 / lambd * costs)  # higher cost -> higher weight
        normalization_factor = jnp.sum(weights)

        def body(i, inputs):
            U = inputs
            U = U + perturbation[i] * weights[i] / normalization_factor
            return U

        return lax.fori_loop(0, samples, body, (U))

    @jit
    def single_sample_rollout(
        time, robot_states_init, perturbed_control, perturbation, x_prev, prev_robustness
    ):
        # Initialize robot_state
        robot_states = jnp.zeros((robot_state_dim, horizon))
        robot_states = robot_states.at[:, 0].set(robot_states_init)

        # loop over horizon
        cost_sample = 0

        def body(i, inputs):
            cost_sample, robot_states = inputs

            # get robot state
            robot_state = robot_states[:, [i]]

            # Get cost
            cost_sample = (
                cost_sample
                + cost_perturbation_coeff
                * (
                    (perturbed_control[:, [i]] - perturbation[:, [i]]).T
                    @ control_cov_inv
                    @ perturbation[:, [i]]
                )[0, 0]
            )
            if trajectory_cost_func == None:
                cost_sample = cost_sample + stage_cost_func(
                    robot_state[:, 0], perturbed_control[:, [i]][:, 0]
                )

            # Update robot states
            robot_states = robot_states.at[:, i + 1].set(
                robot_dynamics_step(robot_states[:, [i]], perturbed_control[:, [i]])[:, 0]
            )
            return cost_sample, robot_states

        cost_sample, robot_states = lax.fori_loop(0, horizon - 1, body, (cost_sample, robot_states))

        robot_state = robot_states[:, [horizon - 1]]
        cost_sample = (
            cost_sample
            + cost_perturbation_coeff
            * (
                (perturbed_control[:, [horizon]] - perturbation[:, [horizon]]).T
                @ control_cov_inv
                @ perturbation[:, [horizon]]
            )[0, 0]
        )
        # if trajectory_cost_func == None:
        #     cost_sample = cost_sample + terminal_cost_func(
        #         robot_state[:, 0], perturbed_control[:, [horizon]][:, 0]
        #     )
        if trajectory_cost_func != None:
            cost_sample = cost_sample + trajectory_cost_func(
                time, robot_states, perturbed_control, prev_robustness
            )

        return cost_sample, robot_states

    @jit
    def rollout_states_foresee(
        time, robot_init_state, perturbed_control, perturbation, x_prev, prev_robustness
    ):

        ##### Initialize

        # Robot
        robot_states = jnp.zeros((samples, robot_state_dim, horizon))
        robot_states = robot_states.at[:, :, 0].set(jnp.tile(robot_init_state.T, (samples, 1)))

        # Cost
        cost_total = jnp.zeros(samples)

        # Single sample rollout
        if use_gpu:

            @jit
            def body_sample(robot_states_init, perturbed_control_sample, perturbation_sample):
                cost_sample, robot_states_sample = single_sample_rollout(
                    time, robot_states_init, perturbed_control_sample.T, perturbation_sample.T
                )
                return cost_sample, robot_states_sample

            batched_body_sample = jax.vmap(body_sample, in_axes=0)
            (
                cost_total,
                robot_states,
            ) = batched_body_sample(robot_states[:, :, 0], perturbed_control, perturbation)
        else:

            @jit
            def body_samples(i, inputs):
                robot_states, cost_total = inputs

                # Get cost
                cost_sample, robot_states_sample = single_sample_rollout(
                    time,
                    robot_states[i, :, 0],
                    perturbed_control[i, :, :].T,
                    perturbation[i, :, :].T,
                    x_prev,
                    prev_robustness,
                )
                cost_total = cost_total.at[i].set(cost_sample)
                robot_states = robot_states.at[i, :, :].set(robot_states_sample)
                return robot_states, cost_total

            robot_states, cost_total = lax.fori_loop(
                0, samples, body_samples, (robot_states, cost_total)
            )

        return robot_states, cost_total

    @jit
    def compute_perturbed_control(subkey, control_mu, control_cov, control_bound, U):
        perturbation = multivariate_normal(
            subkey, control_mu, control_cov, shape=(samples, horizon)
        )  # K x T x nu

        perturbation = jnp.clip(perturbation, -3.0, 3.0)  # 0.3
        perturbed_control = U + perturbation

        perturbed_control = jnp.clip(perturbed_control, -control_bound, control_bound)
        perturbation = perturbed_control - U
        return perturbation, perturbed_control

    @staticmethod
    @jit
    def rollout_control(init_state, actions):
        states = jnp.zeros((robot_state_dim, horizon + 1))
        states = states.at[:, 0].set(init_state[:, 0])

        def body(i, inputs):
            states = inputs
            states = states.at[:, i + 1].set(
                robot_dynamics_step(states[:, [i]], actions[i, :].reshape(-1, 1))[:, 0]
            )
            return states

        states = lax.fori_loop(0, horizon, body, states)
        return states

    @jit
    def compute_rollout_costs(key, U, init_state, time, x_prev, prev_robustness):

        perturbation, perturbed_control = compute_perturbed_control(
            key, control_mu, control_cov, control_bound, U
        )

        sampled_robot_states, costs = rollout_states_foresee(
            time, init_state, perturbed_control, perturbation, x_prev, prev_robustness
        )

        U = weighted_sum(U, perturbation, costs)

        states_final = rollout_control(init_state, U)
        action = U[0, :].reshape(-1, 1)
        U = jnp.append(U[1:, :], U[[-1], :], axis=0)

        sampled_robot_states = sampled_robot_states.reshape((robot_state_dim * samples, horizon))

        return sampled_robot_states, states_final, action[:, 0], U

    return compute_rollout_costs
