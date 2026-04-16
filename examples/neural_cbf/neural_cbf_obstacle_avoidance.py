"""Neural CBF obstacle avoidance — single integrator.

Demonstrates learning a barrier function from data and using it for
safe control.  A 2D single integrator must reach a goal while avoiding
a circular obstacle.  Instead of hand-crafting h(x) = ||x-c||^2 - r^2,
we *learn* h(x) from samples of the safe and unsafe regions.

Requirements::

    pip install cbfkit[neural]   # adds flax + optax

Run::

    python examples/neural_cbf/neural_cbf_obstacle_avoidance.py
"""

import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
from jax import random

import cbfkit.simulation.simulator as sim
from cbfkit.controllers.cbf_clf import vanilla_cbf_clf_qp_controller
from cbfkit.integration import forward_euler as integrator
from cbfkit.modeling.neural_cbf import train_neural_cbf
from cbfkit.sensors import perfect as sensor
from cbfkit.estimators import naive as estimator

# ── Problem setup ──────────────────────────────────────────────────────

STATE_DIM = 2
OBSTACLE_CENTER = jnp.array([3.0, 3.0])
OBSTACLE_RADIUS = 1.0
GOAL = jnp.array([6.0, 6.0])
DT = 0.05
NUM_STEPS = 200 if not os.getenv("CBFKIT_TEST_MODE") else 20
TRAIN_EPOCHS = 1000 if not os.getenv("CBFKIT_TEST_MODE") else 50


def dynamics(x):
    """Single integrator: xdot = u."""
    return jnp.zeros(STATE_DIM), jnp.eye(STATE_DIM)


def nominal_controller(t, x, *args, **kwargs):
    """Proportional drive toward the goal."""
    u = 2.0 * (GOAL - x)
    return u, {}


# ── Step 1: Generate training data ────────────────────────────────────


def generate_samples(key, n_safe=500, n_unsafe=200):
    """Sample points inside (unsafe) and outside (safe) the obstacle."""
    k1, k2 = random.split(key)

    # Safe: ring around obstacle [r+0.3, r+4.0]
    angles = random.uniform(k1, (n_safe,), minval=0, maxval=2 * jnp.pi)
    radii = random.uniform(
        k1, (n_safe,), minval=OBSTACLE_RADIUS + 0.3, maxval=OBSTACLE_RADIUS + 4.0
    )
    safe = OBSTACLE_CENTER + jnp.stack([radii * jnp.cos(angles), radii * jnp.sin(angles)], axis=1)

    # Unsafe: inside obstacle [0, r*0.9]
    angles = random.uniform(k2, (n_unsafe,), minval=0, maxval=2 * jnp.pi)
    radii = random.uniform(k2, (n_unsafe,), minval=0.0, maxval=OBSTACLE_RADIUS * 0.9)
    unsafe = OBSTACLE_CENTER + jnp.stack([radii * jnp.cos(angles), radii * jnp.sin(angles)], axis=1)

    return safe, unsafe


# ── Step 2: Train the neural CBF ──────────────────────────────────────

print("Generating training samples...")
safe_samples, unsafe_samples = generate_samples(random.PRNGKey(42))
print(f"  Safe: {safe_samples.shape[0]} points, Unsafe: {unsafe_samples.shape[0]} points")

print("\nTraining neural CBF...")
barriers = train_neural_cbf(
    dynamics_func=dynamics,
    safe_samples=safe_samples,
    unsafe_samples=unsafe_samples,
    state_dim=STATE_DIM,
    alpha=1.0,
    hidden_layers=[64, 64],
    activation="tanh",
    num_epochs=TRAIN_EPOCHS,
    learning_rate=1e-3,
    margin=0.1,
    key=random.PRNGKey(0),
    verbose=True,
)
print("Training complete. CertificateCollection ready.")

# ── Step 3: Verify the learned barrier ────────────────────────────────

h_func = barriers.functions[0]

# Check: safe point should have h > 0
safe_point = jnp.array([0.0, 0.0])
h_safe = h_func(0.0, safe_point)
print(f"\nh(safe_point={safe_point}) = {float(h_safe):.4f}  (should be > 0)")

# Check: unsafe point (obstacle center) should have h < 0
h_unsafe = h_func(0.0, OBSTACLE_CENTER)
print(f"h(obstacle_center={OBSTACLE_CENTER}) = {float(h_unsafe):.4f}  (should be < 0)")

# ── Step 4: Build controller and simulate ─────────────────────────────

print("\nBuilding CBF-CLF-QP controller with learned barrier...")
controller = vanilla_cbf_clf_qp_controller(
    control_limits=jnp.array([5.0, 5.0]),
    dynamics_func=dynamics,
    barriers=barriers,
)

x0 = jnp.array([0.0, 0.0])

print(f"Simulating: x0={x0}, goal={GOAL}, obstacle=({OBSTACLE_CENTER}, r={OBSTACLE_RADIUS})")
print(f"  dt={DT}, steps={NUM_STEPS}")

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

states = results["states"]

# ── Step 5: Check results ─────────────────────────────────────────────

final_state = states[-1]
final_dist_goal = float(jnp.linalg.norm(final_state - GOAL))
min_dist_obstacle = float(jnp.min(jnp.linalg.norm(states - OBSTACLE_CENTER, axis=1)))

print(f"\nResults:")
print(f"  Final state: [{float(final_state[0]):.2f}, {float(final_state[1]):.2f}]")
print(f"  Distance to goal: {final_dist_goal:.2f}")
print(f"  Min distance to obstacle center: {min_dist_obstacle:.2f} (radius={OBSTACLE_RADIUS})")

if min_dist_obstacle > OBSTACLE_RADIUS:
    print("  SAFE: Robot avoided the obstacle!")
else:
    print("  VIOLATION: Robot entered the obstacle region.")

if final_dist_goal < 1.0:
    print("  REACHED: Robot is near the goal.")
else:
    print("  NOT REACHED: Robot did not reach the goal (may need more steps).")

# Optional: plot if matplotlib is available
try:
    if not os.getenv("CBFKIT_TEST_MODE"):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(states[:, 0], states[:, 1], "b-", linewidth=1.5, label="Trajectory")
        ax.plot(x0[0], x0[1], "go", markersize=10, label="Start")
        ax.plot(GOAL[0], GOAL[1], "r*", markersize=15, label="Goal")

        theta = jnp.linspace(0, 2 * jnp.pi, 100)
        ax.fill(
            OBSTACLE_CENTER[0] + OBSTACLE_RADIUS * jnp.cos(theta),
            OBSTACLE_CENTER[1] + OBSTACLE_RADIUS * jnp.sin(theta),
            alpha=0.3,
            color="red",
            label="Obstacle",
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Neural CBF Obstacle Avoidance")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(os.path.dirname(__file__), "neural_cbf_trajectory.png"),
            dpi=150,
        )
        print("\nPlot saved to examples/neural_cbf/neural_cbf_trajectory.png")
        plt.show()
except ImportError:
    pass
