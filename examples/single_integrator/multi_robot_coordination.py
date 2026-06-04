"""Multi-robot 2D coordination: 6 single integrators on a ring swapping to opposite positions via pairwise distance CBFs."""
import os
import sys

# Add the project root to the path so we can import cbfkit.
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jacrev

import cbfkit.simulation.simulator as sim
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.estimators import naive as estimator
from cbfkit.integration import runge_kutta_4 as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.utils.user_types import CertificateCollection

# CBFKIT_TEST_MODE: short horizon + skip the GIF render/save entirely.
TEST_MODE = bool(os.getenv("CBFKIT_TEST_MODE"))

NUM = 6
DIM = 2 * NUM
RADIUS = 2.0
SAFE_DIST = 0.55
DT = 0.05
TF = 0.5 if TEST_MODE else 4.0

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
GIF_PATH = os.path.join(RESULTS_DIR, "multi_robot_coordination.gif")

INITIAL = np.zeros(DIM)
GOALS = np.zeros(DIM)
rng = np.random.default_rng(7)
for i in range(NUM):
    ang = 2 * np.pi * i / NUM + rng.normal(0, 0.03)
    INITIAL[2 * i] = RADIUS * np.cos(ang)
    INITIAL[2 * i + 1] = RADIUS * np.sin(ang)
    # Goal is opposite side of the ring.
    GOALS[2 * i] = -RADIUS * np.cos(2 * np.pi * i / NUM)
    GOALS[2 * i + 1] = -RADIUS * np.sin(2 * np.pi * i / NUM)
goal_arr = jnp.asarray(GOALS)


def dynamics(x):
    return jnp.zeros(DIM), jnp.eye(DIM)


def nominal(t, x, *args, **kwargs):
    u = -1.5 * (x - goal_arr)
    return u, {}


# Build pairwise distance barriers: h = dx^2 + dy^2 - SAFE_DIST^2.
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


def main():
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

    # Minimum pairwise distance over the whole run (safety metric, SAFE_DIST is the bound).
    min_pair_dist = np.inf
    for i in range(NUM):
        for j in range(i + 1, NUM):
            dx = states[:, 2 * i] - states[:, 2 * j]
            dy = states[:, 2 * i + 1] - states[:, 2 * j + 1]
            min_pair_dist = min(min_pair_dist, float(np.sqrt(dx * dx + dy * dy).min()))
    print(
        f"[multi_robot_coordination] min pairwise distance over run: {min_pair_dist:.3f} "
        f"(safety bound SAFE_DIST={SAFE_DIST})"
    )

    if TEST_MODE:
        # Fast path: skip the GIF render, just report the safety metric.
        print("[multi_robot_coordination] CBFKIT_TEST_MODE: skipping GIF render.")
        return min_pair_dist

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

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
    ax.set_xlim(-RADIUS - 1, RADIUS + 1)
    ax.set_ylim(-RADIUS - 1, RADIUS + 1)
    ax.set_aspect("equal")
    ax.set_title(f"Multi-robot 2D coordination ({NUM} agents, pairwise CBF)", fontsize=10)
    ax.grid(True, alpha=0.3)

    def update(k):
        for i in range(NUM):
            lines[i].set_data(states[: k + 1, 2 * i], states[: k + 1, 2 * i + 1])
            dots[i].set_offsets([[states[k, 2 * i], states[k, 2 * i + 1]]])
        return tuple(lines) + tuple(dots)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stride = max(1, len(states) // 70)
    anim = FuncAnimation(fig, update, frames=range(0, len(states), stride), interval=100, blit=True)
    anim.save(GIF_PATH, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"[multi_robot_coordination] saved GIF -> {GIF_PATH}")
    return min_pair_dist


if __name__ == "__main__":
    main()
