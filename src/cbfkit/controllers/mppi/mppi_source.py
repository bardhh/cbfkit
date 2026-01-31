import inspect

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.random import multivariate_normal


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
    if dyn_func is None:
        print("Dynamics function not passed")
        exit()
    if trajectory_cost_func is None:
        if stage_cost_func is None:
            print("Neither Trajectury Cost nor Stage Cost function is passed.")
            exit()
        if terminal_cost_func is None:
            print("Neither Trajectory Cost nor Terminal Cost function is passed")
            exit()

    # Detect terminal_cost_func signature to support both (state) and (state, action) patterns
    terminal_cost_takes_action = False
    if terminal_cost_func is not None:
        try:
            sig = inspect.signature(terminal_cost_func)
            terminal_cost_takes_action = len(sig.parameters) >= 2
        except (ValueError, TypeError):
            # Fallback: assume 2 args (state, action) for compatibility with older code
            terminal_cost_takes_action = True

    horizon = horizon
    samples = samples
    dt = dt

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
        # Normalize costs in a numerically stable way
        costs_shifted = costs - jnp.min(costs)
        scale = jnp.maximum(jnp.max(costs_shifted), 1e-8)
        costs_norm = costs_shifted / scale
        lambd = costs_lambda
        log_weights = -costs_norm / jnp.maximum(lambd, 1e-8)
        max_logw = jnp.max(log_weights)
        weights = jnp.exp(log_weights - max_logw)
        normalization_factor = jnp.maximum(jnp.sum(weights), 1e-8)

        # Vectorized weighted sum
        weighted_perturbation = jnp.sum(perturbation * weights[:, None, None], axis=0)
        U = U + weighted_perturbation / normalization_factor
        return U

    @jit
    def single_sample_rollout(
        time, robot_states_init, perturbed_control, perturbation, x_prev, prev_robustness
    ):
        def step_fn(carry, inputs):
            current_state, current_cost = carry
            u_t, pert_t = inputs
            u_t_col = u_t.reshape(-1, 1)
            pert_t_col = pert_t.reshape(-1, 1)

            delta_cost = cost_perturbation_coeff * (
                (u_t_col - pert_t_col).T @ control_cov_inv @ pert_t_col
            )[0, 0]

            if trajectory_cost_func is None:
                delta_cost = delta_cost + stage_cost_func(current_state, u_t)

            next_state = robot_dynamics_step(current_state.reshape(-1, 1), u_t_col)[:, 0]
            new_cost = current_cost + delta_cost
            return (next_state, new_cost), current_state

        init_carry = (robot_states_init, 0.0)
        # Scan over 0 to horizon-2
        scan_inputs = (perturbed_control[:, :-1].T, perturbation[:, :-1].T)

        (final_carry_state, accumulated_cost), intermediate_states = lax.scan(
            step_fn, init_carry, scan_inputs
        )

        # intermediate_states contains x0 ... x_{H-2}
        # final_carry_state is x_{H-1}

        # Add final step cost (u_{H-1})
        u_final = perturbed_control[:, horizon - 1]
        pert_final = perturbation[:, horizon - 1]
        u_final_col = u_final.reshape(-1, 1)
        pert_final_col = pert_final.reshape(-1, 1)

        delta_cost_final = cost_perturbation_coeff * (
            (u_final_col - pert_final_col).T @ control_cov_inv @ pert_final_col
        )[0, 0]

        cost_sample = accumulated_cost + delta_cost_final

        # Note: We do NOT compute x_H.

        # Reconstruct robot_states
        # We need [x0, x1, ..., x_{H-1}]
        # intermediate_states has shape (H-1, dim) -> x0 ... x_{H-2}
        # final_carry_state has shape (dim,) -> x_{H-1}

        # Concatenate
        robot_states = jnp.vstack([intermediate_states, final_carry_state.reshape(1, -1)]).T
        # Shape (dim, H)

        if trajectory_cost_func is None:
            final_state = robot_states[:, horizon - 1]
            final_action = u_final
            if terminal_cost_takes_action:
                cost_sample = cost_sample + terminal_cost_func(final_state, final_action)
            else:
                cost_sample = cost_sample + terminal_cost_func(final_state)
        else:
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
        @jit
        def body_sample(robot_states_init, perturbed_control_sample, perturbation_sample):
            cost_sample, robot_states_sample = single_sample_rollout(
                time,
                robot_states_init,
                perturbed_control_sample.T,
                perturbation_sample.T,
                x_prev,
                prev_robustness,
            )
            return cost_sample, robot_states_sample

        batched_body_sample = jax.vmap(body_sample, in_axes=0)
        (
            cost_total,
            robot_states,
        ) = batched_body_sample(robot_states[:, :, 0], perturbed_control, perturbation)

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
        def step_fn(current_state, action):
            next_state = robot_dynamics_step(
                current_state.reshape(-1, 1), action.reshape(-1, 1)
            )[:, 0]
            return next_state, current_state

        final_state, intermediate_states = lax.scan(step_fn, init_state[:, 0], actions)

        states = jnp.vstack([intermediate_states, final_state.reshape(1, -1)]).T
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
