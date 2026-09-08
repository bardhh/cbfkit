"""``SamplingMpc``: MPPI over control-spline knots with MJX rollouts.

Follows hydrax's ``SamplingBasedController`` + ``MPPI`` (MIT) in structure:
knots parameterise the control over ``plan_horizon``; each optimisation step
warm-starts by shifting the knot times to the current ``t``, samples Gaussian
knot perturbations, rolls each sample out through ``plant.step`` under
``lax.scan``, and takes a softmax-weighted average of the sampled knots.

Rollouts keep only per-sample costs -- never ``mjx.Data`` trajectories.
"""

import warnings
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from mujoco import mjx

from .spline import get_interp_func

RunningCost = Callable[[mjx.Data, Array, Any], Array]
TerminalCost = Callable[[mjx.Data, Any], Array]


class MpcState(NamedTuple):
    """Per-step MPC state, carried in ``ControllerData.sub_data["_mpc"]``."""

    tk: Array  # (K,) absolute knot times
    mean: Array  # (K, nu) mean knots
    cov: Optional[Array] = None  # (K, nu, nu) knot covariance; None for plain MPPI


class SamplingMpc:
    def __init__(
        self,
        plant: Any,
        running_cost: RunningCost,
        terminal_cost: TerminalCost,
        *,
        num_samples: int,
        plan_horizon: float,
        noise_level: Any,
        temperature: float,
        num_knots: int = 4,
        spline_type: str = "zero",
        num_randomizations: int = 1,
        randomize_model: Optional[Callable[[mjx.Model, Array], dict]] = None,
        seed: int = 0,
        update: str = "mppi",
        min_noise_level: Optional[float] = None,
        cov_adaptation_rate: float = 0.5,
        iterations: int = 1,
    ) -> None:
        self.plant = plant
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost
        self.num_samples = int(num_samples)
        self.plan_horizon = float(plan_horizon)
        # Scalar, or a (nu,) vector for per-actuator exploration scales.
        self.noise_level = jnp.asarray(noise_level, dtype=float)
        self.temperature = float(temperature)
        self.num_knots = int(num_knots)
        # "mppi": fixed Gaussian noise. "cma": MPPI-CMA (block-diagonal, per-knot
        # covariance adapted from the weighted samples, eigenvalues floored at
        # min_noise_level^2 -- hydrax's MppiCma).
        if update not in ("mppi", "cma"):
            raise ValueError("update must be 'mppi' or 'cma'")
        self.update = update
        self.min_noise_level = float(min_noise_level) if min_noise_level is not None else None
        self.alpha = float(cov_adaptation_rate)
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        self.iterations = int(iterations)
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
        if self.num_randomizations == 1 and randomize_model is not None:
            warnings.warn(
                "randomize_model= given but num_randomizations=1: no randomisation is applied.",
                UserWarning,
                stacklevel=2,
            )
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
        cov = None
        if self.update == "cma":
            sig = jnp.broadcast_to(self.noise_level, (self.plant.nu,))
            cov = jnp.tile(jnp.diag(sig**2)[None], (self.num_knots, 1, 1))
        return MpcState(tk=tk, mean=mean, cov=cov)

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
    def _optimize_once(
        self, data0: mjx.Data, t: Array, state: MpcState, key: Array, aux: Any = None
    ) -> Tuple[MpcState, Array]:
        # Warm start: shift knot times to start at t, re-evaluate old spline there.
        new_tk = jnp.linspace(0.0, self.plan_horizon, self.num_knots) + t
        clamped = jnp.clip(new_tk, state.tk[0], state.tk[-1])
        mean = self.interp(clamped, state.tk, state.mean[None])[0]

        # Sample knots.
        if self.update == "cma":
            assert state.cov is not None  # init_state fills cov whenever update == "cma"
            noise = jax.random.multivariate_normal(
                key,
                mean=jnp.zeros(self.plant.nu),
                cov=state.cov,
                shape=(self.num_samples, self.num_knots),
            )  # (N, K, nu)
            knots = jnp.clip(mean + noise, self.plant.u_min, self.plant.u_max)
        else:
            noise = jax.random.normal(key, (self.num_samples, self.num_knots, self.plant.nu))
            knots = jnp.clip(mean + self.noise_level * noise, self.plant.u_min, self.plant.u_max)

        # Roll out. Query times span [t, t + plan_horizon] in ctrl_steps points,
        # i.e. spacing plan_horizon/(H-1) rather than exactly dt -- byte-for-byte
        # hydrax (alg_base.py), kept for parity with its tuned configurations.
        tq = jnp.linspace(new_tk[0], new_tk[-1], self.ctrl_steps)
        controls = self.interp(tq, new_tk, knots)  # (N, H, nu)
        costs = self._eval_batch(self.model, data0, controls, aux)  # (N,)

        # Softmax-weighted average (jax.nn.softmax subtracts the baseline).
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        new_mean = jnp.sum(weights[:, None, None] * knots, axis=0)
        new_cov = state.cov
        if self.update == "cma":
            assert state.cov is not None
            dev = knots - new_mean[None]  # (N, K, nu)
            sample_cov = jnp.einsum("n,nki,nkj->kij", weights, dev, dev)
            cov = (1.0 - self.alpha) * state.cov + self.alpha * sample_cov
            floor = self.min_noise_level if self.min_noise_level is not None else 0.0
            eigvals, eigvecs = jnp.linalg.eigh(cov)
            eigvals = jnp.maximum(eigvals, floor**2)
            new_cov = jnp.einsum("kij,kj,klj->kil", eigvecs, eigvals, eigvecs)
        return MpcState(tk=new_tk, mean=new_mean, cov=new_cov), costs

    def optimize(
        self, data0: mjx.Data, t: Array, state: MpcState, key: Array, aux: Any = None
    ) -> Tuple[MpcState, Array]:
        """``iterations`` MPPI updates from the same state; returns the last iteration's costs."""
        if self.iterations == 1:
            return self._optimize_once(data0, t, state, key, aux)
        keys = jax.random.split(key, self.iterations)

        def body(st, k):
            st, costs = self._optimize_once(data0, t, st, k, aux)
            return st, costs

        state, costs = jax.lax.scan(body, state, keys)
        return state, costs[-1]

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

        ``MpcState`` is carried in ``data.sub_data["_mpc"]`` (the leading
        underscore marks it carry-only: the simulator does not stack it over the
        horizon) and created on the first call (the simulator's priming call),
        so the JIT carry always holds a concrete state. ``u_nom`` is forwarded
        to the cost functions as ``aux``. The rollout root is
        ``plant.from_state(x)`` -- a fresh contact solve, as in hydrax's
        deterministic loop.

        The same closure is returned on repeated calls: ``controller`` is a
        static JIT argument keyed by identity, so a fresh closure per call
        would recompile the whole simulation.
        """
        if getattr(self, "_controller", None) is not None:
            return self._controller
        plant = self.plant

        def controller(t, x, u_nom, key, data):
            sub = dict(data.sub_data) if data.sub_data is not None else {}
            state = sub.get("_mpc")
            if state is None:
                state = self.init_state()
            # Absolute time on the rollout root so time-dependent costs (gait phase) work.
            data0 = plant.from_state(x).replace(time=jnp.asarray(t, dtype=float))
            u, state = self.step(data0, t, state, key, aux=u_nom)
            sub["_mpc"] = state
            return u, data._replace(sub_data=sub, u=u, u_nom=u_nom)

        # Already canonical 5-arg form; tell setup_controller not to wrap it.
        controller.__cbfkit_controller_adapter__ = True  # type: ignore[attr-defined]
        self._controller = controller
        return controller
