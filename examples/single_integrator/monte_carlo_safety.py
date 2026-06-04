"""GPU/vmap Monte Carlo safety verification: 200 stochastic single-integrator CBF-QP rollouts."""
import os
import sys

# Add the project root to the path so we can import cbfkit + examples.
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import contextlib

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.controllers.cbf_clf.generate_constraints import (
    generate_compute_vanilla_clf_constraints,
    generate_compute_zeroing_cbf_constraints,
)
from cbfkit.integration import forward_euler
from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation
from cbfkit.simulation.monte_carlo_gpu import MonteCarloSetup, conduct_monte_carlo_gpu
from cbfkit.utils.user_types import CertificateCollection, ControllerData, PlannerData

# --- Scenario (verified-clean: low alpha keeps the jaxopt QP stable under vmap) ---
GOAL = jnp.array([4.0, 4.0])
OBS = jnp.array([2.0, 2.0])
R = 0.6
ALPHA = 1.0
NOISE = 0.4
DT = 0.05
# CBFKIT_TEST_MODE: short horizon, few trials, and skip the GIF render entirely.
TEST_MODE = bool(os.getenv("CBFKIT_TEST_MODE"))
NSTEPS = 20 if TEST_MODE else 100
N_TRIALS = 20 if TEST_MODE else 200

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
GIF_PATH = os.path.join(RESULTS_DIR, "monte_carlo_safety.gif")


def dynamics(x):
    return jnp.zeros(2), jnp.eye(2)


# h(x) = ||x - c||^2 - r^2, relative-degree-1 zeroing barrier (single integrator).
f_h = lambda _t, x: jnp.sum((x - OBS) ** 2) - R**2  # noqa: E731
j_h = lambda _t, x: 2.0 * (x - OBS)  # noqa: E731
h_h = lambda _t, _x: 2.0 * jnp.eye(2)  # noqa: E731
p_h = lambda _t, _x: 0.0  # noqa: E731
a_h = lambda h: ALPHA * h  # noqa: E731
barriers = CertificateCollection([f_h], [j_h], [h_h], [p_h], [a_h])

controller = cbf_clf_qp_generator(
    generate_compute_zeroing_cbf_constraints,
    generate_compute_vanilla_clf_constraints,
)(
    control_limits=jnp.array([8.0, 8.0]),
    dynamics_func=dynamics,
    barriers=barriers,
    relaxable_cbf=False,
    relaxable_clf=True,
)


def nominal_controller(t, x, _key, _ref):
    return 2.0 * (GOAL - x), None


def initial_state_sampler(key):
    return jnp.array([0.0, 0.0]) + 0.18 * random.normal(key, (2,))


def _sensor(t, x, *, sigma=None, key=None):
    return x


def _estimator(t, y, z, u, c):
    return y, (c if c is not None else jnp.zeros((len(y), len(y))))


# The CBF-QP controller emits batched jax.debug.print spam under vmap (every branch of
# its status lax.switch fires); silence it at the fd level around the kernel run.
@contextlib.contextmanager
def _silence_fds():
    saved = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(devnull)
        os.close(saved[0])
        os.close(saved[1])


def main():
    # Pass the perturbation UNWRAPPED so its `.is_increment` flag survives (Euler-Maruyama).
    perturbation = generate_stochastic_perturbation(sigma=lambda x: NOISE * jnp.eye(2), dt=DT)

    _, c_data = controller(0.0, jnp.zeros(2), jnp.zeros(2), random.PRNGKey(0), ControllerData())
    setup = MonteCarloSetup(
        dt=DT,
        num_steps=NSTEPS,
        dynamics=dynamics,
        integrator=forward_euler,
        initial_state_sampler=initial_state_sampler,
        nominal_controller=nominal_controller,
        controller=controller,
        sensor=_sensor,
        estimator=_estimator,
        perturbation=perturbation,
        sigma=jnp.zeros(0),
        controller_data=c_data,
        planner=None,
        planner_data=PlannerData(),
    )

    print(f"[monte_carlo_safety] running {N_TRIALS} vmap'd stochastic rollouts...")
    with _silence_fds():
        results = conduct_monte_carlo_gpu(setup, n_trials=N_TRIALS, seed=0)
    states = np.asarray(results.states)  # (N_TRIALS, NSTEPS, 2)
    print(f"[monte_carlo_safety] kernel wall time: {results.wall_time_s:.2f}s")

    # Geometric safety check (independent of the controller's internal barrier bookkeeping).
    dist = np.linalg.norm(states - np.asarray(OBS), axis=-1)  # (N, NSTEPS)
    inside = dist < R  # (N, NSTEPS)
    ever_inside = inside.any(axis=1)  # (N,)
    # Cumulative empirical violation rate up to each step.
    cum_viol_rate = np.array([float(inside[:, : k + 1].any(axis=1).mean()) for k in range(NSTEPS)])
    overall_rate = float(ever_inside.mean())
    print(
        f"[monte_carlo_safety] overall empirical violation rate: {overall_rate:.3f} "
        f"(min dist to obstacle center {dist.min():.3f}, R={R})"
    )

    if TEST_MODE:
        # Fast path: skip the GIF render, just report the safety metric.
        print(f"[monte_carlo_safety] CBFKIT_TEST_MODE: skipping GIF render.")
        return overall_rate

    # Draw a representative subset to keep the GIF small (the full 200-line translucent tangle
    # bloats the palette), but ALWAYS include every breaching trial so the red paths shown stay
    # consistent with the empirical-risk counter, which is computed over ALL N_TRIALS.
    N_DRAW = 60
    rng = np.random.default_rng(0)
    viol_idx = np.flatnonzero(ever_inside)
    safe_idx = np.flatnonzero(~ever_inside)
    n_safe_draw = min(len(safe_idx), max(0, N_DRAW - len(viol_idx)))
    safe_draw = rng.choice(safe_idx, size=n_safe_draw, replace=False)
    draw_idx = np.concatenate([safe_draw, viol_idx]).astype(int)
    draw_states = states[draw_idx]  # (N_DRAW, NSTEPS, 2)
    draw_colors = ["tab:red" if ever_inside[i] else "tab:blue" for i in draw_idx]

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.add_patch(plt.Circle((float(OBS[0]), float(OBS[1])), R, color="red", alpha=0.3, zorder=1))
    ax.add_patch(
        plt.Circle((float(OBS[0]), float(OBS[1])), R, fill=False, color="red", lw=1.5, zorder=2)
    )
    ax.plot(float(GOAL[0]), float(GOAL[1]), "g*", markersize=18, zorder=6)
    ax.plot(0.0, 0.0, "ks", markersize=6, zorder=6)

    lc = LineCollection([], colors=draw_colors, linewidths=0.5, alpha=0.3, zorder=3)
    ax.add_collection(lc)
    dots = ax.scatter(
        draw_states[:, 0, 0], draw_states[:, 0, 1], s=6, c=draw_colors, alpha=0.7, zorder=4
    )
    txt = ax.text(
        0.03,
        0.97,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    legend_handles = [
        Line2D([0], [0], color="tab:blue", lw=1.5, label="safe rollout"),
        Line2D(
            [0], [0], marker="*", color="w", markerfacecolor="g", markersize=12, lw=0, label="Goal"
        ),
        Line2D(
            [0], [0], marker="s", color="w", markerfacecolor="k", markersize=7, lw=0, label="Start"
        ),
    ]
    if len(viol_idx) > 0:
        legend_handles.insert(1, Line2D([0], [0], color="tab:red", lw=1.5, label="breached"))

    ax.set_xlim(-1.0, 5.0)
    ax.set_ylim(-1.0, 5.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Monte Carlo safety verification — {N_TRIALS} stochastic CBF rollouts", fontsize=9
    )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    stride = max(1, NSTEPS // 30)
    frame_idx = list(range(0, NSTEPS, stride))

    def update(k):
        lc.set_segments([draw_states[i, : k + 1, :] for i in range(len(draw_idx))])
        dots.set_offsets(draw_states[:, k, :])
        rate = cum_viol_rate[k]
        n_viol = int(round(rate * N_TRIALS))
        txt.set_text(
            f"step {k + 1:3d}/{NSTEPS}\n"
            f"trials         {N_TRIALS}\n"
            f"violations     {n_viol}\n"
            f"empirical risk {rate * 100:4.1f}%"
        )
        return lc, dots, txt

    os.makedirs(RESULTS_DIR, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=frame_idx, interval=100, blit=False)
    anim.save(GIF_PATH, writer=PillowWriter(fps=10), dpi=80)
    plt.close(fig)
    print(f"[monte_carlo_safety] saved GIF -> {GIF_PATH}")
    return overall_rate


if __name__ == "__main__":
    main()
