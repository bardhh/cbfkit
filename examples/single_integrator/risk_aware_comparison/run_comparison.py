"""Reproduces ACC 2026 Fig. 1: occupancy heatmaps + empirical p_fail for four controllers."""
import os
import sys

# Allow running directly as `python examples/.../run_comparison.py` (puts repo root on sys.path).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
from jax import Array, random

import cbfkit.simulation.simulator as sim
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.simulation.monte_carlo import conduct_monte_carlo

from examples.single_integrator.risk_aware_comparison import config as c
from examples.single_integrator.risk_aware_comparison.controllers import (
    CONTROLLER_NAMES,
    build_controller,
)
from examples.single_integrator.risk_aware_comparison.dynamics import (
    make_dynamics,
    make_perturbation,
)
from examples.single_integrator.risk_aware_comparison.nominal import outward_drive

_DYNAMICS = make_dynamics()
_NOMINAL = outward_drive(c.V_MAX)
_PERTURBATION = make_perturbation(c.SIGMA, c.DT)
_X0 = jnp.array(c.X0)


def _make_execute(controller):
    def execute(trial_no: int = 0, key: Array = None) -> Array:
        if key is None:
            key = random.PRNGKey(trial_no)
        states, *_ = sim.execute(
            x0=_X0,
            dynamics=_DYNAMICS,
            sensor=sensor,
            controller=controller,
            nominal_controller=_NOMINAL,
            estimator=estimator,
            integrator=integrator,
            perturbation=_PERTURBATION,
            dt=c.DT,
            num_steps=c.N_STEPS,
            key=key,
            verbose=False,
        )
        return states  # (N_STEPS, 2)

    return execute


def run_one(name: str, n_trials: int, seed: int = 0) -> Tuple[float, np.ndarray]:
    """Return (p_fail, all_xy) for a single controller across n_trials rollouts.

    A fixed ``seed`` makes the run reproducible (each trial still gets an
    independent PRNG key derived from it); without it conduct_monte_carlo falls
    back to entropy and results vary run-to-run."""
    results: List[Array] = conduct_monte_carlo(
        _make_execute(build_controller(name)), n_trials=n_trials, seed=seed
    )
    failed = 0
    pts = []
    for states in results:
        norms = jnp.linalg.norm(states, axis=-1)
        if bool(jnp.any(norms >= c.R_C)):
            failed += 1
        pts.append(np.asarray(states))
    return failed / n_trials, np.concatenate(pts, axis=0)


def run_all(n_trials: int = c.N_TRIALS, seed: int = 0) -> Dict[str, Tuple[float, np.ndarray]]:
    return {name: run_one(name, n_trials, seed=seed) for name in CONTROLLER_NAMES}


def plot(results: Dict[str, Tuple[float, np.ndarray]], out_path: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    titles = {
        "nominal": "Nominal",
        "s_cbf": "S-CBF",
        "ra_cbf_dt": "RA-CBF-DT",
        "ra_cbf_ct": "RA-CBF-CT",
    }
    for ax, name in zip(axes, CONTROLLER_NAMES):
        p_fail, xy = results[name]
        ax.hist2d(xy[:, 0], xy[:, 1], bins=120, range=[[-1.2, 1.2], [-1.2, 1.2]], norm=LogNorm())
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(c.R_C * np.cos(theta), c.R_C * np.sin(theta), "r--", lw=1.5)
        ax.set_title(f"{titles[name]} | p_fail = {p_fail:.4f}")
        ax.set_aspect("equal")
    fig.suptitle(f"State Occupancy Heatmaps (rho_d = {c.RHO_D})")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    n = 20 if os.environ.get("CBFKIT_TEST_MODE") else c.N_TRIALS
    res = run_all(n_trials=n)
    for name in CONTROLLER_NAMES:
        print(f"{name:10s} p_fail = {res[name][0]:.4f}")
    if not os.environ.get("CBFKIT_TEST_MODE"):
        plot(res, "media/showcase/risk_aware_comparison.png")
