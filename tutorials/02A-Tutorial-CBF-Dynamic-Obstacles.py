from sympy import symbols
import numpy as np
import matplotlib.pyplot as plt
import control as control
from cbfkit.tutorial import cbf as cbf, cbf_utils, sys_and_ctrl
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse
from sympy import symbols, Matrix, sin, cos, lambdify, exp, sqrt, log, diff, Mul, srepr
import platform

# import conditional if system is mac m1
if platform.system() == "Darwin" and platform.machine() == "arm64":
    from kvxopt import solvers
else:
    from cvxopt import solvers

# Robot Goal
x_goal = np.array([20, 20])

# The bad set defines the radii of the eclipse. The position is part of the dynamics of the system.
bad_sets = [[3, 3]]

# Parameters for reference controller
ctrl_param = 2

# Symbols and equations for the CBF
xr_0, xr_1, xo_0, xo_1, a, b, xr_0_dot, xr_1_dot = symbols(
    "xr_0 xr_1 xo_0 xo_1 a b xr_0_dot xr_1_dot"
)
symbs = (xr_0, xr_1, xo_0, xo_1, a, b)

# Barrier function - distance of robot to obstacle
B = ((xr_0 - xo_0) / a) ** 2 + ((xr_1 - xo_1) / b) ** 2 - 1

# dx = f(x) + g(x)u
f = Matrix([0, 0, -0.5, 0, -0.5, -0.5, 1, -0.5, 1.6, 0, -2.5, 3.1])
g = Matrix([xr_0, xr_1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
degree = 2

# Initialize CBF
my_CBF = cbf.CBF(
    B=B,
    f=f,
    g=g,
    states=(xr_0, xr_1, xo_0, xo_1),
    bad_sets=bad_sets,
    symbs=symbs,
    degree=1,
    alpha=3,
)

# intial condition
x_0 = np.array([0, 2, 6, 5.5, 9.5, 11, 2, 13, -5, 15, 26, -9])

# Simulation settings
T_max = 20
n_samples = 250
T = np.linspace(0, T_max, n_samples)
dt = T[1] - T[0]
params = {
    "x_goal": x_goal,
    "x_start": x_0,
    "bad_sets": bad_sets,
    "ctrl_param": ctrl_param,
    "CBF": my_CBF,
}


# Disable cvxopt optimiztaion output
solvers.options["show_progress"] = False
solvers.options["max_iter"] = 1000

# Init Plot
fig, ax = plt.subplots()

# Simulate system
print("\nComputing trajectories for the initial condition:")
print("x_0\t x_1")
print(x_0[0], "\t", x_0[1])

# Compute output on the nimble ant system for given initial conditions and timesteps T
x = np.zeros((np.size(x_0), len(T)))
x[:, 0] = x_0

# Simulate system
for i in range(len(T) - 1):
    x[:, i + 1] = x[:, i] + dt * np.array(
        sys_and_ctrl.nimble_ant_with_agent_f(T[i], x[:, i], [], params)
    )

print("\n*Simulation Done. See plot for animation...")

# Animate
plt.xlim(0, 22)
plt.ylim(0, 22)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
(line1,) = ax.plot([], [], lw=2)


nrObs = int((x_0.size - 2) / 2)

ells = []
for i in range(0, nrObs):
    ells.append(
        Ellipse(
            (x[2 * i + 2][0], x[2 * i + 3][0]),
            bad_sets[0][0],
            bad_sets[0][1],
            color="g",
            alpha=0.3,
        )
    )

goal_square = plt.Rectangle(
    x_goal - np.array([0.5, 0.5]), 0.5, 0.5, color="r", alpha=0.5
)

time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)


def init():
    patches = []
    patches.append(ax.add_patch(goal_square))
    return patches


def animate(i):
    patches = []
    for j in range(0, nrObs):
        ells[j].center = (x[2 * j + 2][i], x[2 * j + 3][i])
        patches.append(ax.add_patch(ells[j]))
    line1.set_data((x[0][0:i], x[1][0:i]))
    time_text.set_text("t = " + "{:.2f}".format(i * dt))
    patches.append(ax.add_patch(goal_square))
    patches.append(
        line1,
    )
    patches.append(
        time_text,
    )
    return patches


ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=30, blit=True, frames=n_samples
)

# writergif = PillowWriter(fps=30)
# ani.save("demo.gif", writer=writergif)

plt.show()

print("\n*Animation Complete. Exiting...\n")
