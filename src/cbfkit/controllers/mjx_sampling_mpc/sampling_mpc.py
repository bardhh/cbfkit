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
        # Domain randomisation: a batched mjx.Model with a leading axis of size
        # num_randomizations on every randomised field, plus the matching
        # vmap in_axes pytree (hydrax convention).
        self.model = plant.model
        self.randomized_axes = None
        if self.num_randomizations > 1:
            if randomize_model is None:
                raise ValueError("num_randomizations > 1 requires randomize_model=")
            keys = jax.random.split(jax.random.PRNGKey(self.seed), self.num_randomizations)
            randomizations = jax.vmap(lambda k: randomize_model(plant.model, k))(keys)
            self.model = plant.model.tree_replace(randomizations)
            axes = jax.tree.map(lambda _: None, plant.model)
            self.randomized_axes = axes.tree_replace({k: 0 for k in randomizations})

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
        """Costs of a batch ``controls: (B, H, nu)`` -> ``(B,)``, averaged over randomised models."""

        def one(m: mjx.Model) -> Array:
            return jax.vmap(lambda c: self._rollout(m, data0, c, aux))(controls)

        if self.randomized_axes is None:
            return one(model)
        costs = jax.vmap(one, in_axes=(self.randomized_axes,))(model)  # (R, B)
        return jnp.mean(costs, axis=0)

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

    # -- CBFKit controller adapter ----------------------------------------
    def as_controller(self):
        """Return a ``ControllerCallable``: ``(t, x, u_nom, key, data) -> (u, data)``.

        ``MpcState`` is carried in ``data.sub_data["mpc"]`` and created on the
        first call (the simulator's priming call), so the JIT carry always holds
        a concrete state. ``u_nom`` is forwarded to the cost functions as ``aux``.
        The rollout root is ``plant.from_state(x)`` -- a fresh contact solve, as
        in hydrax's deterministic loop.
        """
        plant = self.plant

        def controller(t, x, u_nom, key, data):
            sub = dict(data.sub_data) if data.sub_data is not None else {}
            state = sub.get("mpc")
            if state is None:
                state = self.init_state()
            data0 = plant.from_state(x)
            u, state = self.step(data0, t, state, key, aux=u_nom)
            sub["mpc"] = state
            return u, data._replace(sub_data=sub, u=u, u_nom=u_nom)

        # Already canonical 5-arg form; tell setup_controller not to wrap it.
        controller.__cbfkit_controller_adapter__ = True  # type: ignore[attr-defined]
        return controller
