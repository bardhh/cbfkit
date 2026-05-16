"""Standalone CBF safety filter for action filtering."""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array, random

from cbfkit.utils.user_types import (
    EMPTY_CERTIFICATE_COLLECTION,
    CertificateCollection,
    ControllerCallable,
    ControllerData,
    DynamicsCallable,
)


class SafetyFilter:
    """CBF-based safety filter that wraps a ControllerCallable.

    Manages per-step state (time, PRNG key, solver warm-start) and provides
    fallback behavior on solver failure. Usable standalone or via the
    Gymnasium SafetyFilterWrapper.

    Not thread-safe. Not compatible with gymnasium.vector environments.
    """

    def __init__(
        self,
        controller: ControllerCallable,
        dt: float,
        seed: int,
        fallback: Union[str, Callable] = "passthrough",
        barriers: Optional[CertificateCollection] = None,
    ):
        self._controller = controller
        self._dt = dt
        self._fallback = fallback
        self._barriers = barriers
        self._t = 0.0
        self._key = random.PRNGKey(seed)
        self._data = ControllerData()
        self._initial_seed = seed

    @classmethod
    def from_controller(
        cls,
        controller: ControllerCallable,
        dt: float = 0.01,
        seed: int = 0,
        fallback: Union[str, Callable] = "passthrough",
        barriers: Optional[CertificateCollection] = None,
    ) -> "SafetyFilter":
        """Create a SafetyFilter from a pre-built ControllerCallable."""
        return cls(controller=controller, dt=dt, seed=seed, fallback=fallback, barriers=barriers)

    @classmethod
    def from_cbf_qp(
        cls,
        dynamics: DynamicsCallable,
        barriers,
        control_limits: Array,
        variant: str = "vanilla",
        lyapunovs=None,
        fallback: Union[str, Callable] = "passthrough",
        dt: float = 0.01,
        seed: int = 0,
        **kwargs: Any,
    ) -> "SafetyFilter":
        """Create a SafetyFilter from CBF-QP specification.

        Args:
            dynamics: System dynamics (x) -> (f, g).
            barriers: Barrier certificate(s) (CertificateInput).
            control_limits: Symmetric actuation limits.
            variant: "vanilla", "robust", or "stochastic".
            lyapunovs: Optional Lyapunov certificate(s).
            fallback: Fallback strategy on solver failure.
            dt: Timestep for internal time tracking.
            seed: PRNG seed.
            **kwargs: Forwarded to the QP generator.
        """
        from cbfkit.controllers.cbf_clf import (
            robust_cbf_clf_qp_controller,
            stochastic_cbf_clf_qp_controller,
            vanilla_cbf_clf_qp_controller,
        )

        generators = {
            "vanilla": vanilla_cbf_clf_qp_controller,
            "robust": robust_cbf_clf_qp_controller,
            "stochastic": stochastic_cbf_clf_qp_controller,
        }
        if variant not in generators:
            raise ValueError(
                f"Unknown variant {variant!r}. Choose from: {list(generators.keys())}. "
                f"For risk-aware variants, use SafetyFilter.from_controller() with a "
                f"pre-built controller."
            )

        generator = generators[variant]
        controller = generator(
            control_limits=control_limits,
            dynamics_func=dynamics,
            barriers=barriers,
            lyapunovs=lyapunovs if lyapunovs is not None else EMPTY_CERTIFICATE_COLLECTION,
            **kwargs,
        )

        # Normalize barriers to CertificateCollection for barrier_values()
        normalized_barriers = None
        if barriers is not None and barriers != EMPTY_CERTIFICATE_COLLECTION:
            if isinstance(barriers, tuple) and hasattr(barriers, "functions"):
                normalized_barriers = barriers
            elif isinstance(barriers, (list, tuple)):
                from cbfkit.certificates import concatenate_certificates

                normalized_barriers = concatenate_certificates(*barriers)
            else:
                normalized_barriers = barriers

        return cls(
            controller=controller,
            dt=dt,
            seed=seed,
            fallback=fallback,
            barriers=normalized_barriers,
        )

    @property
    def time(self) -> float:
        """Current internal time."""
        return self._t

    def filter(self, state, action) -> Tuple[Array, Dict[str, Any]]:
        """Filter an action through the CBF safety controller.

        Args:
            state: Current system state.
            action: Proposed action (u_nom).

        Returns:
            (u_applied, info) where u_applied is the safe action and info
            contains diagnostics.
        """
        state = jnp.asarray(state, dtype=jnp.float64)
        u_nom = jnp.asarray(action, dtype=jnp.float64)

        self._key, subkey = random.split(self._key)
        u_qp, updated_data = self._controller(self._t, state, u_nom, subkey, self._data)

        # Detect solver failure (explicit error flag or NaN output)
        solver_failed = bool(updated_data.error) or bool(jnp.any(jnp.isnan(u_qp)))

        if solver_failed:
            u_applied = self._apply_fallback(state, u_nom)
            fallback_used = True
        else:
            u_applied = u_qp
            fallback_used = False

        intervened = bool(not jnp.allclose(u_nom, u_applied, atol=1e-4))

        # Extract barrier values from controller sub_data or recompute
        barrier_values = None
        if updated_data.sub_data and "bfs" in updated_data.sub_data:
            barrier_values = updated_data.sub_data["bfs"]
        elif self._barriers is not None and not solver_failed:
            barrier_values = self.barrier_values(state)

        info = {
            "u_nom": u_nom,
            "u_qp": u_qp,
            "u_applied": u_applied,
            "intervened": intervened,
            "barrier_values": barrier_values,
            "solver_status": (
                int(updated_data.error_data) if updated_data.error_data is not None else None
            ),
            "controller_error": bool(updated_data.error),
            "fallback_used": fallback_used,
        }

        self._t += self._dt
        self._data = updated_data

        return u_applied, info

    def _apply_fallback(self, state: Array, u_nom: Array) -> Array:
        """Apply fallback strategy on solver failure."""
        if self._fallback == "passthrough":
            return u_nom
        elif self._fallback == "zero":
            return jnp.zeros_like(u_nom)
        elif callable(self._fallback):
            return self._fallback(state, u_nom)
        else:
            raise ValueError(f"Unknown fallback strategy: {self._fallback!r}")

    def barrier_values(self, state, t=None) -> Optional[Array]:
        """Evaluate barrier functions at the given state.

        Args:
            state: System state.
            t: Time (defaults to internal time counter).

        Returns:
            Array of barrier values, or None if no barriers available.
        """
        if self._barriers is None:
            return None
        if t is None:
            t = self._t
        state = jnp.asarray(state, dtype=jnp.float64)
        return jnp.array([f(t, state) for f in self._barriers[0]])

    def reset(self, seed=None):
        """Reset filter state for a new episode.

        Args:
            seed: Optional new PRNG seed.
        """
        self._t = 0.0
        self._data = ControllerData()
        if seed is not None:
            self._key = random.PRNGKey(seed)
        else:
            self._key = random.PRNGKey(self._initial_seed)
