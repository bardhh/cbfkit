"""``SamplingMpc``: MPPI over control-spline knots with MJX rollouts.

Follows hydrax's ``SamplingBasedController`` + ``MPPI`` (MIT) in structure:
knots parameterise the control over ``plan_horizon``; each optimisation step
warm-starts by shifting the knot times to the current ``t``, samples Gaussian
knot perturbations, rolls each sample out through ``plant.step`` under
``lax.scan``, and takes a softmax-weighted average of the sampled knots.

Rollouts keep only per-sample costs -- never ``mjx.Data`` trajectories.
"""

from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from mujoco import mjx

from .spline import get_interp_func

RunningCost = Callable[[mjx.Data, Array, Any], Array]
TerminalCost = Callable[[mjx.Data, Any], Array]


class MpcState(NamedTuple):
    """Per-step MPC state, carried in ``ControllerData.sub_data["mpc"]``."""

    tk: Array  # (K,) absolute knot times
    mean: Array  # (K, nu) mean knots


class SamplingMpc:
    def __init__(
        self,
        plant: Any,
        running_cost: RunningCost,
        terminal_cost: TerminalCost,
        *,
        num_samples: int,
        plan_horizon: float,
        noise_level: float,
        temperature: float,
        num_knots: int = 4,
        spline_type: str = "zero",
        num_randomizations: int = 1,
        randomize_model: Optional[Callable[[mjx.Model, Array], dict]] = None,
        seed: int = 0,
    ) -> None:
        self.plant = plant
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost
        self.num_samples = int(num_samples)
        self.plan_horizon = float(plan_horizon)
        self.noise_level = float(noise_level)
        self.temperature = float(temperature)
        self.num_knots = int(num_knots)
        self.interp = get_interp_func(spline_type)
        self.dt = float(plant.dt)
        self.ctrl_steps = int(round(self.plan_horizon / self.dt))
        if self.ctrl_steps < 1:
            raise ValueError("plan_horizon must cover at least one plant step")
        self.num_randomizations = max(int(num_randomizations), 1)
        self.randomize_model = randomize_model
        self.seed = int(seed)
        # Domain randomisation is added in a later task.
        self.model = plant.model
        self.randomized_axes = None
        if self.num_randomizations > 1:
            raise NotImplementedError("num_randomizations > 1 is not implemented yet")

    # -- state -------------------------------------------------------------
    def init_state(self, initial_knots: Optional[Array] = None) -> MpcState:
        mean = (
            jnp.zeros((self.num_knots, self.plant.nu))
            if initial_knots is None
            else jnp.asarray(initial_knots)
        )
        if mean.shape != (self.num_knots, self.plant.nu):
            raise ValueError(
                f"initial_knots must have shape {(self.num_knots, self.plant.nu)}, got {mean.shape}"
            )
        tk = jnp.linspace(0.0, self.plan_horizon, self.num_knots)
        return MpcState(tk=tk, mean=mean)

    # -- rollouts ----------------------------------------------------------
    def _rollout(self, model: mjx.Model, data0: mjx.Data, controls: Array, aux: Any) -> Array:
        """Total cost of one control sequence ``controls: (H, nu)`` from ``data0``."""

        def body(d, u):
            d = self.plant.step(d, u, model=model)
            return d, self.dt * self.running_cost(d, u, aux)

        d_final, stage = jax.lax.scan(body, data0, controls)
        return jnp.sum(stage) + self.terminal_cost(d_final, aux)

    def _eval_batch(self, model: mjx.Model, data0: mjx.Data, controls: Array, aux: Any) -> Array:
        """Costs of a batch ``controls: (B, H, nu)`` under one model -> ``(B,)``."""
        return jax.vmap(lambda c: self._rollout(model, data0, c, aux))(controls)

    def rollout_cost(
        self, data0: mjx.Data, knots: Array, aux: Any = None, tk: Optional[Array] = None
    ) -> Array:
        """Public helper: cost of each knot sequence in ``knots: (B, K, nu)`` -> ``(B,)``."""
        tk = jnp.linspace(0.0, self.plan_horizon, self.num_knots) if tk is None else tk
        tq = jnp.linspace(tk[0], tk[-1], self.ctrl_steps)
        controls = self.interp(tq, tk, knots)
        return self._eval_batch(self.model, data0, controls, aux)

    # -- MPPI --------------------------------------------------------------
    def optimize(
        self, data0: mjx.Data, t: Array, state: MpcState, key: Array, aux: Any = None
    ) -> Tuple[MpcState, Array]:
        # Warm start: shift knot times to start at t, re-evaluate old spline there.
        new_tk = jnp.linspace(0.0, self.plan_horizon, self.num_knots) + t
        clamped = jnp.clip(new_tk, state.tk[0], state.tk[-1])
        mean = self.interp(clamped, state.tk, state.mean[None])[0]

        # Sample knots.
        noise = jax.random.normal(key, (self.num_samples, self.num_knots, self.plant.nu))
        knots = jnp.clip(mean + self.noise_level * noise, self.plant.u_min, self.plant.u_max)

        # Roll out.
        tq = jnp.linspace(new_tk[0], new_tk[-1], self.ctrl_steps)
        controls = self.interp(tq, new_tk, knots)  # (N, H, nu)
        costs = self._eval_batch(self.model, data0, controls, aux)  # (N,)

        # Softmax-weighted average (jax.nn.softmax subtracts the baseline).
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        new_mean = jnp.sum(weights[:, None, None] * knots, axis=0)
        return MpcState(tk=new_tk, mean=new_mean), costs

    def get_action(self, state: MpcState, t: Array) -> Array:
        return self.interp(jnp.atleast_1d(jnp.asarray(t)), state.tk, state.mean[None])[0, 0]

    def step(
        self, data0: mjx.Data, t: Array, state: MpcState, key: Array, aux: Any = None
    ) -> Tuple[Array, MpcState]:
        state, _ = self.optimize(data0, t, state, key, aux)
        return self.get_action(state, t), state
