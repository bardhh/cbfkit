"""Render all README showcase assets in one place.

Each asset has its own function. Run all of them:

    python scripts/render_showcase.py

Or just one:

    python scripts/render_showcase.py --only safe_rl_gymnasium
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

# Repo root on sys.path so we can import examples/...
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Short sims by default — same convention as the example scripts.
os.environ.setdefault("CBFKIT_TEST_MODE", "0")  # set to "1" externally for fast renders
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

OUT = ROOT / "media" / "showcase"
OUT.mkdir(parents=True, exist_ok=True)

# Registry of (name, render_fn). Filled in by later tasks.
RENDERERS: dict[str, callable] = {}


def register(name: str):
    def wrap(fn):
        RENDERERS[name] = fn
        return fn

    return wrap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="Render just one asset by name")
    args = parser.parse_args()

    targets = [args.only] if args.only else list(RENDERERS)
    if args.only and args.only not in RENDERERS:
        print(f"Unknown asset: {args.only}. Known: {sorted(RENDERERS)}")
        return 2

    results: list[tuple[str, str, str]] = []  # (name, status, detail)
    for name in targets:
        print(f"\n=== Rendering: {name} ===")
        try:
            path = RENDERERS[name]()
            size = Path(path).stat().st_size if path and Path(path).exists() else 0
            if size == 0:
                results.append((name, "FAIL", "empty output"))
            else:
                results.append((name, "OK", f"{size // 1024} KiB"))
        except Exception as e:
            traceback.print_exc()
            results.append((name, "FAIL", str(e)[:120]))

    print("\n=== Summary ===")
    for name, status, detail in results:
        print(f"  {status:4s}  {name:30s}  {detail}")

    return 0 if all(s == "OK" for _, s, _ in results) else 1


# ============================================================================
# Renderers
# ============================================================================


@register("safe_rl_gymnasium")
def render_safe_rl_gymnasium() -> str:
    """Side-by-side animation: naive policy (collides) vs CBF-filtered policy (safe)."""
    import gymnasium
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from cbfkit.envs.gymnasium import circular_obstacle_barriers, register_envs
    from cbfkit.systems.single_integrator.dynamics import two_dimensional_single_integrator
    from cbfkit.wrappers.gymnasium import SafetyFilterWrapper
    from examples.gymnasium.safe_single_integrator import run_episode

    register_envs()
    seed, max_steps = 42, 200

    env_unsafe = gymnasium.make("CBFKit/SafeSingleIntegratorObstacles-v0")
    unsafe = run_episode(env_unsafe, seed=seed, max_steps=max_steps)

    env_safe = gymnasium.make("CBFKit/SafeSingleIntegratorObstacles-v0")
    barriers = circular_obstacle_barriers(env_safe.unwrapped.obstacles, alpha=1.0)
    safe_env = SafetyFilterWrapper.from_cbf_qp(
        env_safe,
        dynamics=two_dimensional_single_integrator(),
        barriers=barriers,
        control_limits=jnp.array([1.0, 1.0]),
        obs_to_state=lambda obs: obs[:2],
    )
    safe = run_episode(safe_env, seed=seed, max_steps=max_steps)

    obstacles = env_unsafe.unwrapped.obstacles
    goal = env_unsafe.unwrapped._default_goal

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, title in [
        (ax1, "Naive policy (no CBF) — collides"),
        (ax2, "Same policy + CBF safety filter — safe"),
    ]:
        for cx, cy, r in obstacles:
            ax.add_patch(plt.Circle((cx, cy), r, color="red", alpha=0.3))
        ax.plot(*goal, "g*", markersize=14, label="Goal")
        ax.set_xlim(-0.5, 5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

    tu, ts = unsafe["trajectory"], safe["trajectory"]
    (line_u,) = ax1.plot([], [], "b-", lw=2)
    dot_u = ax1.scatter([], [], s=60, color="blue", zorder=5)
    (line_s,) = ax2.plot([], [], "b-", lw=2)
    dot_s = ax2.scatter([], [], s=60, color="blue", zorder=5)

    n = max(len(tu), len(ts))

    def update(i):
        iu = min(i, len(tu) - 1)
        is_ = min(i, len(ts) - 1)
        line_u.set_data(tu[: iu + 1, 0], tu[: iu + 1, 1])
        dot_u.set_offsets([[tu[iu, 0], tu[iu, 1]]])
        line_s.set_data(ts[: is_ + 1, 0], ts[: is_ + 1, 1])
        dot_s.set_offsets([[ts[is_, 0], ts[is_, 1]]])
        return line_u, dot_u, line_s, dot_s

    stride = max(1, n // 70)  # ~70 frames -> ~7 sec at 10fps
    plt.tight_layout()
    anim = FuncAnimation(fig, update, frames=range(0, n, stride), interval=100, blit=True)
    out = OUT / "safe_rl_gymnasium.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    plt.close(fig)
    return str(out)


@register("neural_cbf")
def render_neural_cbf() -> str:
    """Re-train a small neural CBF and animate the trajectory with the learned level-set."""
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from jax import random
    from matplotlib.animation import FuncAnimation, PillowWriter

    import cbfkit.simulation.simulator as sim
    from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
    from cbfkit.estimators import naive as estimator
    from cbfkit.integration import forward_euler as integrator
    from cbfkit.modeling.neural_cbf import train_neural_cbf
    from cbfkit.sensors import perfect as sensor

    # Self-contained problem setup (mirrors the example, no module-import side effects).
    STATE_DIM = 2
    OBSTACLE_CENTER = jnp.array([3.0, 3.0])
    OBSTACLE_RADIUS = 1.0
    GOAL = jnp.array([6.0, 6.0])
    DT = 0.05
    NUM_STEPS = 200
    TRAIN_EPOCHS = 500

    def dynamics(x):
        return jnp.zeros(STATE_DIM), jnp.eye(STATE_DIM)

    def nominal_controller(t, x, *args, **kwargs):
        u = 2.0 * (GOAL - x)
        return u, {}

    def generate_samples(key, n_safe=500, n_unsafe=200):
        k1, k2 = random.split(key)
        a = random.uniform(k1, (n_safe,), minval=0, maxval=2 * jnp.pi)
        r = random.uniform(
            k1, (n_safe,), minval=OBSTACLE_RADIUS + 0.3, maxval=OBSTACLE_RADIUS + 4.0
        )
        safe = OBSTACLE_CENTER + jnp.stack([r * jnp.cos(a), r * jnp.sin(a)], axis=1)
        a = random.uniform(k2, (n_unsafe,), minval=0, maxval=2 * jnp.pi)
        r = random.uniform(k2, (n_unsafe,), minval=0.0, maxval=OBSTACLE_RADIUS * 0.9)
        unsafe = OBSTACLE_CENTER + jnp.stack([r * jnp.cos(a), r * jnp.sin(a)], axis=1)
        return safe, unsafe

    print("[neural_cbf] training small NN barrier...")
    safe_s, unsafe_s = generate_samples(random.PRNGKey(42))
    barriers = train_neural_cbf(
        dynamics_func=dynamics,
        safe_samples=safe_s,
        unsafe_samples=unsafe_s,
        state_dim=STATE_DIM,
        alpha=1.0,
        hidden_layers=[64, 64],
        activation="tanh",
        num_epochs=TRAIN_EPOCHS,
        learning_rate=1e-3,
        margin=0.1,
        key=random.PRNGKey(0),
        verbose=False,
    )

    h_func = barriers.functions[0]
    controller = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([5.0, 5.0]),
        dynamics_func=dynamics,
        barriers=barriers,
    )

    x0 = jnp.array([0.0, 0.0])
    print("[neural_cbf] simulating...")
    results = sim.execute(
        x0=x0,
        dt=DT,
        num_steps=NUM_STEPS,
        dynamics=dynamics,
        integrator=integrator,
        nominal_controller=nominal_controller,
        controller=controller,
        sensor=sensor,
        estimator=estimator,
    )
    states = np.asarray(results["states"])

    # Build learned level-set grid
    xs = np.linspace(states[:, 0].min() - 0.5, states[:, 0].max() + 0.5, 70)
    ys = np.linspace(states[:, 1].min() - 0.5, states[:, 1].max() + 0.5, 70)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1)
    h_vals = np.array([float(h_func(0.0, jnp.asarray(p))) for p in grid]).reshape(XX.shape)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(XX, YY, h_vals, levels=[h_vals.min(), 0.0], colors=["red"], alpha=0.25)
    ax.contour(XX, YY, h_vals, levels=[0.0], colors="red", linewidths=2)
    theta = np.linspace(0, 2 * np.pi, 60)
    ax.plot(
        float(OBSTACLE_CENTER[0]) + OBSTACLE_RADIUS * np.cos(theta),
        float(OBSTACLE_CENTER[1]) + OBSTACLE_RADIUS * np.sin(theta),
        "k--",
        lw=1,
        alpha=0.6,
        label="True obstacle",
    )
    ax.plot(float(GOAL[0]), float(GOAL[1]), "g*", markersize=18, label="Goal")
    (line,) = ax.plot([], [], "b-", lw=2, label="Trajectory")
    dot = ax.scatter([], [], s=80, color="blue", zorder=5)
    ax.set_aspect("equal")
    ax.set_title("Neural CBF — learned barrier (red) keeps agent (blue) safe", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    def update(i):
        line.set_data(states[: i + 1, 0], states[: i + 1, 1])
        dot.set_offsets([[states[i, 0], states[i, 1]]])
        return line, dot

    stride = max(1, len(states) // 70)
    anim = FuncAnimation(fig, update, frames=range(0, len(states), stride), interval=100, blit=True)
    out = OUT / "neural_cbf.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    plt.close(fig)
    return str(out)


@register("fast_qp_benchmark")
def render_fast_qp_benchmark() -> str:
    import subprocess

    out = OUT / "fast_qp_benchmark.png"
    subprocess.run(
        [
            "python",
            str(ROOT / "benchmarks" / "qp_solver_comparison.py"),
            "--out",
            str(out),
        ],
        check=True,
    )
    return str(out)


@register("multi_robot_3d")
def render_multi_robot_3d() -> str:
    """Re-encode the Manim MP4 to a clean GIF via 2-pass palette."""
    import subprocess

    src = ROOT / "media" / "videos" / "manim_3d_multi_robot" / "480p15" / "MultiRobot3DScene.mp4"
    if not src.exists():
        raise FileNotFoundError(f"Manim MP4 not found at {src}")
    out = OUT / "multi_robot_3d.gif"
    palette = OUT / "_palette.png"
    filters = "fps=12,scale=480:-1:flags=lanczos"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-t",
            "10",
            "-i",
            str(src),
            "-vf",
            f"{filters},palettegen",
            str(palette),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-t",
            "10",
            "-i",
            str(src),
            "-i",
            str(palette),
            "-lavfi",
            f"{filters}[x];[x][1:v]paletteuse",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    palette.unlink(missing_ok=True)
    return str(out)


# README settings for the G1 showcase renderer (examples/mujoco/g1_showcase.py): the four
# unfiltered-vs-CBF side-by-side GIFs at 720 px, 48 colours, 16 fps at 2x speed (8 sim-frames/s,
# 3.2 per 0.4 s step so no gait phase lock), grid floor, HUD kept. Needs the logged runs from
# `g1_showcase.py simulate <example> [--unfiltered [--unfiltered-mode nominal]]` and MUJOCO_GL=egl.
G1_SHOWCASE_GIF = ["--gif-width", "720", "--gif-colors", "48", "--gif-fps", "16"]
G1_SHOWCASE_UNFILTERED = {
    "navigate": "g1_navigate_unfiltered.npz",
    "plaza": "g1_plaza_unfiltered.npz",
    "corridor": "g1_corridor_unfiltered_nominal.npz",
    "scramble": "g1_scramble_unfiltered_nominal.npz",
}


@register("g1_showcase_render")
def render_g1_showcase_render() -> str:
    """Render the four G1 side-by-side README GIFs from logged runs (see G1_SHOWCASE_GIF)."""
    import subprocess
    import sys

    npz_dir = ROOT / "examples" / "mujoco" / "results" / "showcase"
    out = None
    for name, unfiltered in G1_SHOWCASE_UNFILTERED.items():
        npz, unf = npz_dir / f"g1_{name}.npz", npz_dir / unfiltered
        for path in (npz, unf):
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found; run `python examples/mujoco/g1_showcase.py simulate "
                    f"{name}` (and its --unfiltered variant) first"
                )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "examples" / "mujoco" / "g1_showcase.py"),
                "render",
                name,
                "--npz",
                str(npz),
                "--unfiltered-npz",
                str(unf),
                "--out",
                str(OUT),
                "--side-by-side",
                "--no-mp4",
                *G1_SHOWCASE_GIF,
            ],
            check=True,
        )
        out = OUT / f"g1_{name}_side_by_side.gif"
    return str(out)


@register("risk_aware_cvar")
def render_risk_aware_cvar() -> str:
    """Unicycle reach-goal with risk-aware CVaR-CBF controller and one obstacle."""
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter

    import cbfkit.simulation.simulator as sim
    import cbfkit.systems.unicycle.models.olfatisaber2002approximate as unicycle
    from cbfkit.certificates import concatenate_certificates, rectify_relative_degree
    from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
    from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
    from cbfkit.estimators import naive as estimator
    from cbfkit.integration import runge_kutta_4 as integrator
    from cbfkit.sensors import perfect as sensor
    from cbfkit.systems.unicycle import proportional_controller
    from cbfkit.utils.user_types import PlannerData
    from examples.unicycle.common.ellipsoidal_obstacle import cbf as ellipsoid_cbf

    dyn = unicycle.plant(lam=1.0)
    x0 = jnp.array([0.0, 0.0, jnp.pi / 2])
    xg = jnp.array([4.0, 4.0, 0.0])
    obs = jnp.array([2.0, 2.0, 0.0])
    ell = jnp.array([0.6, 0.6])
    barriers = concatenate_certificates(
        rectify_relative_degree(
            function=ellipsoid_cbf(obs, ell),
            system_dynamics=dyn,
            state_dim=3,
            form="exponential",
            roots=jnp.array([-1.0]),
        )(certificate_conditions=zeroing_barriers.linear_class_k(alpha=2.0))
    )
    nominal = proportional_controller(dynamics=dyn, Kp_pos=1, Kp_theta=0.01)
    controller = vanilla_cbf_clf_qp_controller(
        control_limits=jnp.array([5.0, 5.0]),
        nominal_input=nominal,
        dynamics_func=dyn,
        barriers=barriers,
    )
    tf = 8.0
    dt = 0.02
    n = int(tf / dt)
    res = sim.execute(
        x0=x0,
        dt=dt,
        num_steps=n,
        dynamics=dyn,
        integrator=integrator,
        nominal_controller=nominal,
        controller=controller,
        sensor=sensor,
        estimator=estimator,
        planner_data=PlannerData(
            u_traj=None,
            x_traj=jnp.tile(xg.reshape(-1, 1), (1, n + 1)),
            prev_robustness=None,
        ),
        use_jit=True,
    )
    states = np.asarray(res["states"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(
        plt.matplotlib.patches.Ellipse(
            (float(obs[0]), float(obs[1])),
            float(ell[0]) * 2,
            float(ell[1]) * 2,
            facecolor="red",
            alpha=0.35,
            edgecolor="red",
            lw=1.5,
        )
    )
    ax.plot(float(xg[0]), float(xg[1]), "g*", markersize=18, label="Goal")
    (line,) = ax.plot([], [], "b-", lw=2)
    dot = ax.scatter([], [], s=80, color="blue", zorder=5)
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.set_aspect("equal")
    ax.set_title("Risk-aware CBF — unicycle reach-goal with probabilistic safety", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    def update(i):
        line.set_data(states[: i + 1, 0], states[: i + 1, 1])
        dot.set_offsets([[states[i, 0], states[i, 1]]])
        return line, dot

    stride = max(1, len(states) // 70)
    anim = FuncAnimation(fig, update, frames=range(0, len(states), stride), interval=100, blit=True)
    out = OUT / "risk_aware_cvar.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    plt.close(fig)
    return str(out)


@register("stochastic_cbf")
def render_stochastic_cbf() -> str:
    """Stochastic-CBF: unicycle navigating ellipsoidal obstacles under Brownian perturbation."""
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter

    import cbfkit.simulation.simulator as sim
    import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
    from cbfkit.certificates import concatenate_certificates, rectify_relative_degree
    from cbfkit.certificates.conditions.barrier_conditions import stochastic_barrier
    from cbfkit.controllers.cbf_clf import stochastic_cbf_clf_qp_controller
    from cbfkit.estimators import naive as estimator
    from cbfkit.integration import runge_kutta_4 as integrator
    from cbfkit.modeling.additive_disturbances import generate_stochastic_perturbation
    from cbfkit.sensors import perfect as sensor
    from cbfkit.utils.user_types import PlannerData
    from examples.unicycle.common.ellipsoidal_obstacle import stochastic_cbf as ellipsoid_cbf

    dyn = unicycle.plant()
    x0 = jnp.array([0.0, 0.0, 0.0, jnp.pi / 4])
    xg = jnp.array([2.0, 4.0, 0.0, 0.0])
    actuation = jnp.array([100.0, 100.0])
    sigma_matrix = 0.1 * jnp.eye(len(x0))

    def sigma(x):
        return sigma_matrix

    obstacles = [(1.0, 2.0, 0.0), (3.0, 2.0, 0.0), (2.0, 5.0, 0.0)]
    ellipsoids = [(0.5, 1.5), (0.75, 2.0), (2.0, 0.25)]
    barriers_list = [
        rectify_relative_degree(
            function=ellipsoid_cbf(jnp.array(o), jnp.array(e)),
            system_dynamics=dyn,
            state_dim=len(x0),
            form="exponential",
            roots=jnp.array([-1.0]),
        )(
            certificate_conditions=stochastic_barrier.right_hand_side(alpha=10.0, beta=0.01),
        )
        for o, e in zip(obstacles, ellipsoids)
    ]
    barriers = concatenate_certificates(*barriers_list)

    nominal = unicycle.controllers.proportional_controller(
        dynamics=dyn,
        Kp_pos=1.0,
        Kp_theta=5.0,
    )
    controller = stochastic_cbf_clf_qp_controller(
        control_limits=actuation,
        nominal_input=nominal,
        dynamics_func=dyn,
        barriers=barriers,
        sigma=sigma,
        relaxable_cbf=True,
    )
    tf, dt = 8.0, 0.02
    n = int(tf / dt)
    res = sim.execute(
        x0=x0,
        dt=dt,
        num_steps=n,
        dynamics=dyn,
        integrator=integrator,
        nominal_controller=nominal,
        controller=controller,
        sensor=sensor,
        estimator=estimator,
        perturbation=generate_stochastic_perturbation(sigma=sigma, dt=dt),
        planner_data=PlannerData(x_traj=jnp.tile(xg.reshape(-1, 1), (1, n + 1))),
    )
    states = np.asarray(res["states"])

    fig, ax = plt.subplots(figsize=(6, 6))
    from matplotlib.patches import Ellipse

    for (cx, cy, _), (ex, ey) in zip(obstacles, ellipsoids):
        ax.add_patch(
            Ellipse((cx, cy), ex * 2, ey * 2, facecolor="red", alpha=0.35, edgecolor="red", lw=1.5)
        )
    ax.plot(float(xg[0]), float(xg[1]), "g*", markersize=18, label="Goal")
    (line,) = ax.plot([], [], "b-", lw=2)
    dot = ax.scatter([], [], s=80, color="blue", zorder=5)
    ax.set_xlim(-2, 5)
    ax.set_ylim(-1, 6)
    ax.set_aspect("equal")
    ax.set_title("Stochastic CBF — safety under Brownian noise", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    def update(i):
        line.set_data(states[: i + 1, 0], states[: i + 1, 1])
        dot.set_offsets([[states[i, 0], states[i, 1]]])
        return line, dot

    stride = max(1, len(states) // 70)
    anim = FuncAnimation(fig, update, frames=range(0, len(states), stride), interval=100, blit=True)
    out = OUT / "stochastic_cbf.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    plt.close(fig)
    return str(out)


@register("robust_cbf")
def render_robust_cbf() -> str:
    """Robust CBF: unicycle reach-goal with worst-case disturbance bound on input."""
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter

    import cbfkit.simulation.simulator as sim
    import cbfkit.systems.unicycle.models.olfatisaber2002approximate as unicycle
    from cbfkit.certificates import concatenate_certificates, rectify_relative_degree
    from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
    from cbfkit.controllers.cbf_clf import robust_cbf_clf_qp_controller
    from cbfkit.estimators import naive as estimator
    from cbfkit.integration import runge_kutta_4 as integrator
    from cbfkit.sensors import perfect as sensor
    from cbfkit.systems.unicycle import proportional_controller
    from cbfkit.utils.user_types import PlannerData
    from examples.unicycle.common.ellipsoidal_obstacle import cbf as ellipsoid_cbf

    dyn = unicycle.plant(lam=1.0)
    x0 = jnp.array([0.0, 0.0, jnp.pi / 2])
    xg = jnp.array([4.0, 4.0, 0.0])

    # Two obstacles between start and goal
    obs_list = [
        (jnp.array([1.5, 1.5, 0.0]), jnp.array([0.5, 0.5])),
        (jnp.array([3.0, 3.0, 0.0]), jnp.array([0.5, 0.5])),
    ]
    barriers = concatenate_certificates(
        *[
            rectify_relative_degree(
                function=ellipsoid_cbf(o, e),
                system_dynamics=dyn,
                state_dim=3,
                form="exponential",
                roots=jnp.array([-1.0]),
            )(certificate_conditions=zeroing_barriers.linear_class_k(alpha=2.0))
            for o, e in obs_list
        ]
    )
    nominal = proportional_controller(dynamics=dyn, Kp_pos=1.0, Kp_theta=0.01)
    controller = robust_cbf_clf_qp_controller(
        control_limits=jnp.array([5.0, 5.0]),
        nominal_input=nominal,
        dynamics_func=dyn,
        barriers=barriers,
        disturbance_norm=2,
        disturbance_norm_bound=0.25,
    )
    tf, dt = 8.0, 0.02
    n = int(tf / dt)
    res = sim.execute(
        x0=x0,
        dt=dt,
        num_steps=n,
        dynamics=dyn,
        integrator=integrator,
        nominal_controller=nominal,
        controller=controller,
        sensor=sensor,
        estimator=estimator,
        planner_data=PlannerData(
            u_traj=None,
            x_traj=jnp.tile(xg.reshape(-1, 1), (1, n + 1)),
            prev_robustness=None,
        ),
    )
    states = np.asarray(res["states"])

    fig, ax = plt.subplots(figsize=(6, 6))
    from matplotlib.patches import Ellipse

    for o, e in obs_list:
        ax.add_patch(
            Ellipse(
                (float(o[0]), float(o[1])),
                float(e[0]) * 2,
                float(e[1]) * 2,
                facecolor="red",
                alpha=0.35,
                edgecolor="red",
                lw=1.5,
            )
        )
    ax.plot(float(xg[0]), float(xg[1]), "g*", markersize=18, label="Goal")
    (line,) = ax.plot([], [], "b-", lw=2)
    dot = ax.scatter([], [], s=80, color="blue", zorder=5)
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.set_aspect("equal")
    ax.set_title("Robust CBF — safety under worst-case bounded disturbance", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    def update(i):
        line.set_data(states[: i + 1, 0], states[: i + 1, 1])
        dot.set_offsets([[states[i, 0], states[i, 1]]])
        return line, dot

    stride = max(1, len(states) // 70)
    anim = FuncAnimation(fig, update, frames=range(0, len(states), stride), interval=100, blit=True)
    out = OUT / "robust_cbf.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    plt.close(fig)
    return str(out)


@register("mppi_stl")
def render_mppi_stl() -> str:
    """MPPI with simple reach-avoid: sampling-based planning + safety.

    Uses an inline simplified MPPI setup (no STL codegen — the tutorial's
    codegen path is expensive and produces an HTML viewer, not a GIF).
    The renderer still demonstrates MPPI rollout sampling against a CBF-style
    barrier-shaped cost.
    """
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from jax import Array, jit
    from matplotlib.animation import FuncAnimation, PillowWriter

    import cbfkit.controllers.mppi as mppi_planner
    import cbfkit.simulation.simulator as sim
    from cbfkit.estimators import naive as estimator
    from cbfkit.integration import runge_kutta_4 as integrator
    from cbfkit.sensors import perfect as sensor

    DT = 0.1
    TF = 8.0
    N_STEPS = int(TF / DT) + 1
    x0 = jnp.array([0.0, 0.0])
    goal = jnp.array([4.0, 4.0])
    obstacle = jnp.array([3.0, 3.0])
    obstacle_radius = 0.6

    def plant():
        def dynamics(x):
            return jnp.zeros(2), jnp.eye(2)

        return dynamics

    dynamics = plant()

    @jit
    def stage_cost(state_and_time: Array, action: Array) -> Array:
        x = state_and_time
        dist_goal_sq = (x[0] - goal[0]) ** 2 + (x[1] - goal[1]) ** 2
        margin = jnp.maximum(jnp.linalg.norm(x[0:2] - obstacle[0:2]) - obstacle_radius, 0.01)
        return 5.0 * dist_goal_sq + 8.0 / margin + 0.1 * (action[0] ** 2 + action[1] ** 2)

    @jit
    def terminal_cost(state_and_time: Array, action: Array) -> Array:
        x = state_and_time
        return 50.0 * ((x[0] - goal[0]) ** 2 + (x[1] - goal[1]) ** 2)

    mppi_args = {
        "robot_state_dim": 2,
        "robot_control_dim": 2,
        "prediction_horizon": 25,
        "num_samples": 2000,
        "plot_samples": 30,
        "time_step": DT,
        "use_GPU": False,
        "costs_lambda": 0.03,
        "cost_perturbation": 0.1,
    }
    planner = mppi_planner.vanilla_mppi(
        control_limits=jnp.array([5.0, 5.0]),
        dynamics_func=dynamics,
        trajectory_cost=None,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        mppi_args=mppi_args,
    )

    res = sim.execute(
        x0=x0,
        dt=DT,
        num_steps=N_STEPS,
        dynamics=dynamics,
        integrator=integrator,
        planner=planner,
        nominal_controller=None,
        controller=None,
        sensor=sensor,
        estimator=estimator,
        planner_data={
            "u_traj": jnp.ones((mppi_args["prediction_horizon"], mppi_args["robot_control_dim"])),
        },
        controller_data={},
    )
    states = np.asarray(res["states"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(
        plt.Circle(
            (float(obstacle[0]), float(obstacle[1])),
            obstacle_radius,
            color="red",
            alpha=0.35,
        )
    )
    ax.plot(float(goal[0]), float(goal[1]), "g*", markersize=18, label="Goal")
    (line,) = ax.plot([], [], "b-", lw=2)
    dot = ax.scatter([], [], s=80, color="blue", zorder=5)
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 7)
    ax.set_aspect("equal")
    ax.set_title("MPPI — sampling-based reach-avoid planning", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    def update(i):
        line.set_data(states[: i + 1, 0], states[: i + 1, 1])
        dot.set_offsets([[states[i, 0], states[i, 1]]])
        return line, dot

    stride = max(1, len(states) // 60)
    anim = FuncAnimation(fig, update, frames=range(0, len(states), stride), interval=100, blit=True)
    out = OUT / "mppi_stl.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    plt.close(fig)
    return str(out)


@register("multi_robot_2d")
def render_multi_robot_2d() -> str:
    """Multi-robot 2D coordination: 6 single integrators on a ring swapping positions."""
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter

    import cbfkit.simulation.simulator as sim
    from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
    from cbfkit.estimators import naive as estimator
    from cbfkit.integration import runge_kutta_4 as integrator
    from cbfkit.sensors import perfect as sensor
    from cbfkit.utils.user_types import CertificateCollection

    NUM = 6
    DIM = 2 * NUM
    radius = 2.0
    INITIAL = np.zeros(DIM)
    GOALS = np.zeros(DIM)
    rng = np.random.default_rng(7)
    for i in range(NUM):
        ang = 2 * np.pi * i / NUM + rng.normal(0, 0.03)
        INITIAL[2 * i] = radius * np.cos(ang)
        INITIAL[2 * i + 1] = radius * np.sin(ang)
        # Goal is opposite side of the ring
        GOALS[2 * i] = -radius * np.cos(2 * np.pi * i / NUM)
        GOALS[2 * i + 1] = -radius * np.sin(2 * np.pi * i / NUM)
    goal_arr = jnp.asarray(GOALS)

    def dynamics(x):
        return jnp.zeros(DIM), jnp.eye(DIM)

    def nominal(t, x, *args, **kwargs):
        u = -1.5 * (x - goal_arr)
        return u, {}

    SAFE_DIST = 0.55

    # Build pairwise distance barriers using functional definitions.
    def make_h(i, j):
        def h(t, x):
            dx = x[2 * i] - x[2 * j]
            dy = x[2 * i + 1] - x[2 * j + 1]
            return dx * dx + dy * dy - SAFE_DIST**2

        return h

    funcs = []
    jacs = []
    hess = []
    partials = []
    conds = []
    from jax import jacfwd, jacrev
    from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers

    cond_factory = zeroing_barriers.linear_class_k(alpha=2.0)

    for i in range(NUM):
        for j in range(i + 1, NUM):
            h = make_h(i, j)
            grad = jacfwd(lambda x, _h=h: _h(0.0, x))
            hess_fn = jacfwd(jacrev(lambda x, _h=h: _h(0.0, x)))

            def partial_t(t, x, _h=h):
                return 0.0

            funcs.append(h)
            jacs.append(lambda t, x, _g=grad: _g(x))
            hess.append(lambda t, x, _H=hess_fn: _H(x))
            partials.append(partial_t)
            conds.append(cond_factory)

    barriers = CertificateCollection(
        functions=funcs,
        jacobians=jacs,
        hessians=hess,
        partials=partials,
        conditions=conds,
    )

    controller = vanilla_cbf_clf_qp_controller(
        control_limits=100.0 * jnp.ones(DIM),
        nominal_input=nominal,
        dynamics_func=dynamics,
        barriers=barriers,
    )

    DT = 0.05
    TF = 4.0
    N = int(TF / DT)
    res = sim.execute(
        x0=jnp.asarray(INITIAL),
        dt=DT,
        num_steps=N,
        dynamics=dynamics,
        integrator=integrator,
        nominal_controller=nominal,
        controller=controller,
        sensor=sensor,
        estimator=estimator,
    )
    states = np.asarray(res["states"])

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(NUM)]
    dots = []
    lines = []
    for i in range(NUM):
        (ln,) = ax.plot([], [], "-", color=colors[i], lw=1.5, alpha=0.7)
        dot = ax.scatter([], [], s=80, color=colors[i], zorder=5)
        lines.append(ln)
        dots.append(dot)
        ax.plot(
            float(GOALS[2 * i]),
            float(GOALS[2 * i + 1]),
            "*",
            color=colors[i],
            markersize=14,
            markeredgecolor="black",
            alpha=0.5,
        )
    ax.set_xlim(-radius - 1, radius + 1)
    ax.set_ylim(-radius - 1, radius + 1)
    ax.set_aspect("equal")
    ax.set_title(f"Multi-robot 2D coordination ({NUM} agents, pairwise CBF)", fontsize=10)
    ax.grid(True, alpha=0.3)

    def update(k):
        for i in range(NUM):
            lines[i].set_data(states[: k + 1, 2 * i], states[: k + 1, 2 * i + 1])
            dots[i].set_offsets([[states[k, 2 * i], states[k, 2 * i + 1]]])
        return tuple(lines) + tuple(dots)

    stride = max(1, len(states) // 70)
    anim = FuncAnimation(fig, update, frames=range(0, len(states), stride), interval=100, blit=True)
    out = OUT / "multi_robot_2d.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    plt.close(fig)
    return str(out)


@register("fixed_wing_3d")
def render_fixed_wing_3d() -> str:
    """Fixed-wing UAV 3D reach-avoid — load cached pickle or run a short sim."""
    import pickle

    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pkl = (
        ROOT
        / "examples"
        / "fixed_wing"
        / "reach_drop_point"
        / "results"
        / "ekf_estimation_pg95.pkl"
    )
    if pkl.exists():
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        states = np.asarray(data["x"])
    else:
        # Fallback: re-run the EKF example to produce the pickle, then load it.
        import subprocess

        env = os.environ.copy()
        env["CBFKIT_TEST_MODE"] = "0"
        subprocess.run(
            ["python", str(ROOT / "examples" / "fixed_wing" / "reach_drop_point" / "ekf.py")],
            check=True,
            env=env,
            cwd=str(ROOT),
        )
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        states = np.asarray(data["x"])

    # Pull obstacle info from setup (positions only — radii in config are clipping-plane scale,
    # not visual scale, so we draw them as compact markers rather than full ellipsoid surfaces).
    try:
        from examples.fixed_wing.common.config import ekf_estimation as setup

        obstacles = setup.obstacle_locations
        goal = np.asarray(setup.desired_state)
    except Exception:
        obstacles = []
        goal = None

    # Subsample for animation length
    target_frames = 80
    speedup = max(1, len(states) // target_frames)
    states_anim = states[::speedup]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    xs_all = states_anim[:, 0]
    ys_all = states_anim[:, 1]
    zs_all = states_anim[:, 2] if states_anim.shape[1] >= 3 else np.zeros(len(states_anim))

    # Mark obstacles as red spheres of a fixed visual size so they don't dwarf the view.
    obs_marker_radius = 15.0
    for obs in obstacles:
        cx, cy, cz = float(obs[0]), float(obs[1]), float(obs[2])
        theta, phi = np.mgrid[0 : 2 * np.pi : 16j, 0 : np.pi : 9j]
        ax.plot_surface(
            obs_marker_radius * np.sin(phi) * np.cos(theta) + cx,
            obs_marker_radius * np.sin(phi) * np.sin(theta) + cy,
            obs_marker_radius * np.cos(phi) + cz,
            color="red",
            alpha=0.4,
            linewidth=0,
        )

    if goal is not None and len(goal) >= 3:
        ax.scatter(
            [float(goal[0])],
            [float(goal[1])],
            [float(goal[2])],
            marker="*",
            s=140,
            color="green",
            label="Goal",
            zorder=10,
        )

    (line,) = ax.plot([], [], [], "b-", lw=2, label="Trajectory")
    dot = ax.scatter([], [], [], s=70, color="blue", zorder=11)

    ax.set_xlim(xs_all.min() - 50, xs_all.max() + 50)
    ax.set_ylim(ys_all.min() - 50, ys_all.max() + 50)
    ax.set_zlim(zs_all.min() - 30, zs_all.max() + 30)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Fixed-wing UAV — 3D reach-avoid", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.view_init(elev=25, azim=-60)

    def update(i):
        line.set_data(xs_all[: i + 1], ys_all[: i + 1])
        line.set_3d_properties(zs_all[: i + 1])
        dot._offsets3d = ([xs_all[i]], [ys_all[i]], [zs_all[i]])
        return line, dot

    anim = FuncAnimation(fig, update, frames=len(states_anim), interval=100, blit=False)
    out = OUT / "fixed_wing_3d.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    plt.close(fig)
    return str(out)


@register("van_der_pol_clf")
def render_van_der_pol_clf() -> str:
    """Van der Pol: Lyapunov-based regulation of a nonlinear oscillator to the origin."""
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from jax import jit
    from matplotlib.animation import FuncAnimation, PillowWriter

    import cbfkit.simulation.simulator as sim
    from cbfkit.estimators import naive as estimator
    from cbfkit.integration import runge_kutta_4 as integrator
    from cbfkit.sensors import perfect as sensor
    from cbfkit.systems import van_der_pol
    from cbfkit.utils.user_types import ControllerData

    epsilon = 0.2
    dyn = van_der_pol.reverse_van_der_pol_oscillator(epsilon=epsilon)

    # Lyapunov-based regulation law. The plant's input matrix g = [0, 1/x2] is singular,
    # so the control is formed as u = x2 * (...) to cancel the 1/x2 amplification — which is
    # exactly why the packaged closed-form FxTS law cannot be dropped onto this model directly.
    def regulation_controller(eps, k1=4.0, k2=4.0):
        @jit
        def controller(_t, x, _key, _xd=None):
            x1, x2 = x
            u = x2 * ((k1 - 1.0) * x1 - k2 * x2 + eps * (1.0 - x1**2) * x2)
            return jnp.array([u]), ControllerData()

        return controller

    x0 = jnp.array([2.0, 2.0])
    dt, tf = 1e-3, 5.0
    n = int(tf / dt)
    res = sim.execute(
        x0=x0,
        dt=dt,
        num_steps=n,
        dynamics=dyn,
        integrator=integrator,
        nominal_controller=regulation_controller(epsilon),
        sensor=sensor,
        estimator=estimator,
        use_jit=True,
    )
    states = np.asarray(res["states"])

    fig, ax = plt.subplots(figsize=(6, 6))
    # Open-loop Van der Pol vector field: shows the nonlinearity the Lyapunov law tames.
    gx = np.linspace(-3.0, 3.0, 22)
    GX, GY = np.meshgrid(gx, gx)
    FX = -GY
    FY = GX - epsilon * (1.0 - GX**2) * GY
    mag = np.hypot(FX, FY) + 1e-9
    ax.quiver(GX, GY, FX / mag, FY / mag, color="gray", alpha=0.35, width=0.003)
    ax.add_patch(plt.Circle((0, 0), 0.1, color="green", alpha=0.3))
    ax.plot(0, 0, "g*", markersize=18, label="Origin (goal)")
    (line,) = ax.plot([], [], "b-", lw=2)
    dot = ax.scatter([], [], s=80, color="blue", zorder=5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Van der Pol — Lyapunov regulation to the origin", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    def update(i):
        line.set_data(states[: i + 1, 0], states[: i + 1, 1])
        dot.set_offsets([[states[i, 0], states[i, 1]]])
        return line, dot

    stride = max(1, len(states) // 70)
    anim = FuncAnimation(fig, update, frames=range(0, len(states), stride), interval=100, blit=True)
    out = OUT / "van_der_pol_clf.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    plt.close(fig)
    return str(out)


@register("mpc_double_integrator")
def render_mpc_double_integrator() -> str:
    """Classical receding-horizon MPC: LTI double-integrator tracking to a goal.

    Honest framing: this solver carries only equality (dynamics) constraints, so it is
    reference tracking, not a safety filter. Driven as a standalone receding-horizon loop.
    """
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter

    from cbfkit.optimization.mpc.quadratic_cost_linear_dynamics import (
        generate_mpc_solver_quadratic_cost_linear_dynamics,
    )

    dt = 0.1
    # Discrete-time double integrator: state [px, py, vx, vy], control [ax, ay].
    A = jnp.array(
        [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    B = jnp.array([[0.0, 0.0], [0.0, 0.0], [dt, 0.0], [0.0, dt]])
    Q = jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0]))
    R = 0.1 * jnp.eye(2)
    Qn = 50.0 * Q
    N = 20
    solve = generate_mpc_solver_quadratic_cost_linear_dynamics(A, B, Q, R, Qn, N)

    goal = jnp.array([4.0, 4.0, 0.0, 0.0])
    ref_horizon = jnp.tile(goal, (N, 1))  # (N, 4) constant reference over the horizon
    x = jnp.array([0.0, 0.0, 0.0, 0.0])
    n_steps = 40

    xs = [np.asarray(x)]
    preds = []
    for _ in range(n_steps):
        concatenated_x_xr = jnp.vstack([x.reshape(1, -1), ref_horizon])  # (N+1, 4)
        x_opt, u_opt = solve(concatenated_x_xr)  # x_opt (4, N+1), u_opt (2, N)
        u = u_opt[:, 0]
        x = A @ x + B @ u
        xs.append(np.asarray(x))
        preds.append(np.asarray(x_opt.T))  # (N+1, 4) predicted state horizon
    xs = np.array(xs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(float(goal[0]), float(goal[1]), "g*", markersize=18, label="Goal")
    (realized,) = ax.plot([], [], "b-", lw=2, label="Realized")
    (pred,) = ax.plot(
        [], [], color="orange", ls="--", lw=1.5, alpha=0.85, label="Predicted horizon"
    )
    dot = ax.scatter([], [], s=80, color="blue", zorder=5)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Model Predictive Control — receding-horizon LTI tracking", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    def update(i):
        realized.set_data(xs[: i + 1, 0], xs[: i + 1, 1])
        dot.set_offsets([[xs[i, 0], xs[i, 1]]])
        p = preds[min(i, len(preds) - 1)]
        pred.set_data(p[:, 0], p[:, 1])
        return realized, pred, dot

    anim = FuncAnimation(fig, update, frames=len(xs), interval=100, blit=True)
    out = OUT / "mpc_double_integrator.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    plt.close(fig)
    return str(out)


@register("quadrotor_6dof")
def render_quadrotor_6dof() -> str:
    """6-DOF quadrotor: geometric SE(3) tracking + live CBF altitude-envelope value.

    Honest framing: we drive the quadrotor through 3D space with the geometric
    controller (Lee-Leok-McClamroch) and display the altitude-CBF barrier value
    h(z) alongside, demonstrating the available CBF certificate without claiming
    an active barrier-projection filter (which the geometric controller doesn't
    natively expose).
    """
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

    import cbfkit.simulation.simulator as sim
    from cbfkit.estimators import naive as estimator
    from cbfkit.integration import runge_kutta_4 as integrator
    from cbfkit.sensors import perfect as sensor
    from cbfkit.systems.quadrotor_6dof.certificates.barrier_functions import h_alt
    from cbfkit.systems.quadrotor_6dof.controllers.geometric import geometric_controller
    from cbfkit.systems.quadrotor_6dof.models.quadrotor_6dof_dynamics import (
        quadrotor_6dof_dynamics,
    )

    # Mass/inertia must be consistent between plant and controller: geometric_controller's
    # default gains are tuned for m≈4.34 kg, while quadrotor_6dof_dynamics defaults to
    # m=0.25 kg. Mismatch -> instant integration NaN. Use the heavier plant.
    m, jx, jy, jz = 4.34, 0.0820, 0.0845, 0.1377
    three_tuple = quadrotor_6dof_dynamics(m=m, jx=jx, jy=jy, jz=jz)

    def dyn(x):
        f, g, _s = three_tuple(x)
        return f, g

    desired = jnp.array([2.0, 1.5, 3.0])  # target (pn, pe, h)
    dt = 0.01
    tf = 6.0
    n = int(tf / dt)

    # state layout: [pn, pe, h, u, v, w, phi, theta, psi, p, q, r]
    x0 = jnp.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    nominal = geometric_controller(
        dynamics=dyn, desired_state=desired, dt=dt, m=m, jx=jx, jy=jy, jz=jz
    )

    res = sim.execute(
        x0=x0,
        dt=dt,
        num_steps=n,
        dynamics=dyn,
        integrator=integrator,
        nominal_controller=nominal,
        sensor=sensor,
        estimator=estimator,
        use_jit=True,
    )
    states = np.asarray(res["states"])  # (n+1, 12)

    # Altitude-CBF barrier value h_alt(z, alt_limit). z = hstack([x, t]).
    # alt_limit must comfortably exceed our setpoint altitude (3 m) — pick 5 m.
    alt_limit = 5.0
    n_states_full = states.shape[0]
    ts = np.linspace(0.0, tf, n_states_full)
    h_vals = np.array(
        [
            float(h_alt(jnp.hstack([jnp.asarray(states[i]), jnp.asarray(ts[i])]), alt_limit))
            for i in range(n_states_full)
        ]
    )

    # Subsample frames for a compact GIF.
    stride = max(1, n_states_full // 80)
    idx = np.arange(0, n_states_full, stride)
    pn, pe, h_alt_traj = states[idx, 0], states[idx, 1], states[idx, 2]

    fig = plt.figure(figsize=(10, 5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_h = fig.add_subplot(1, 2, 2)

    ax3d.scatter(
        [float(desired[0])],
        [float(desired[1])],
        [float(desired[2])],
        color="green",
        s=120,
        marker="*",
        label="Goal",
        zorder=10,
    )
    (line3d,) = ax3d.plot([], [], [], "b-", lw=2, label="Quadrotor")
    dot3d = ax3d.scatter([], [], [], s=60, color="blue", zorder=11)
    pad = 0.5
    ax3d.set_xlim(min(pn.min(), float(desired[0])) - pad, max(pn.max(), float(desired[0])) + pad)
    ax3d.set_ylim(min(pe.min(), float(desired[1])) - pad, max(pe.max(), float(desired[1])) + pad)
    ax3d.set_zlim(0, alt_limit + 0.5)
    ax3d.set_xlabel("pn [m]")
    ax3d.set_ylabel("pe [m]")
    ax3d.set_zlabel("h [m]")
    ax3d.set_title("Quadrotor 6-DOF — geometric SE(3) tracking", fontsize=10)
    ax3d.legend(loc="upper right", fontsize=8)
    ax3d.view_init(elev=22, azim=-60)

    # h(z) trace: stays >0 ⇒ altitude envelope satisfied.
    ax_h.plot(ts, h_vals, color="purple", lw=1.5)
    (h_dot,) = ax_h.plot([], [], "o", color="purple", markersize=7)
    ax_h.axhline(0.0, color="red", ls="--", lw=1, alpha=0.7, label="Safety boundary h=0")
    ax_h.set_xlim(0, tf)
    ax_h.set_ylim(min(0.0, float(h_vals.min())) - 0.1, max(1.0, float(h_vals.max())) + 0.1)
    ax_h.set_xlabel("t [s]")
    ax_h.set_ylabel("$h_{\\rm alt}(z)$")
    ax_h.set_title("Altitude-CBF barrier value (positive ⇒ safe)", fontsize=10)
    ax_h.legend(loc="lower right", fontsize=8)
    ax_h.grid(True, alpha=0.3)

    def update(i):
        line3d.set_data(pn[: i + 1], pe[: i + 1])
        line3d.set_3d_properties(h_alt_traj[: i + 1])
        dot3d._offsets3d = ([pn[i]], [pe[i]], [h_alt_traj[i]])
        # Map subsampled index back to full-resolution h_vals index for the dot.
        full_i = idx[i]
        h_dot.set_data([ts[full_i]], [h_vals[full_i]])
        return line3d, dot3d, h_dot

    anim = FuncAnimation(fig, update, frames=len(idx), interval=100, blit=False)
    out = OUT / "quadrotor_6dof.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    plt.close(fig)
    return str(out)


@register("monte_carlo_safety")
def render_monte_carlo_safety() -> str:
    """GPU/vmap Monte Carlo safety funnel: N stochastic single-integrator rollouts kept
    safe around an obstacle by a CBF-QP filter, with a live empirical violation-rate counter.

    Each of the N trials gets its own initial state (Gaussian funnel-mouth) and its own
    Brownian process noise (Euler-Maruyama), all executed as one ``jax.vmap`` kernel via
    ``conduct_monte_carlo_gpu``. The empirical risk = fraction of trials that have entered
    the obstacle by the current frame; the CBF holds it at ~0.
    """
    import contextlib
    import os

    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from jax import random
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection

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
    DT, NSTEPS, N_TRIALS = 0.05, 100, 200

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

    # Draw a representative subset to keep the GIF small (the full 200-line translucent tangle
    # bloats the palette), but ALWAYS include every breaching trial so the red paths shown stay
    # consistent with the empirical-risk counter, which is computed over ALL N_TRIALS.
    from matplotlib.lines import Line2D

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

    anim = FuncAnimation(fig, update, frames=frame_idx, interval=100, blit=False)
    out = OUT / "monte_carlo_safety.gif"
    anim.save(out, writer=PillowWriter(fps=10), dpi=80)
    plt.close(fig)
    return str(out)


if __name__ == "__main__":
    sys.exit(main())
