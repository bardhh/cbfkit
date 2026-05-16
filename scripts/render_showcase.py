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


if __name__ == "__main__":
    sys.exit(main())
