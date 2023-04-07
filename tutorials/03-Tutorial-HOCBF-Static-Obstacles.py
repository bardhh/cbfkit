import platform

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse
from sympy import Matrix, Mul, Symbol, cos, diff, exp, lambdify, log, sin, sqrt, srepr, symbols

from cbfkit.tutorial import cbf as cbf
from cbfkit.tutorial import cbf_utils, sys_and_ctrl

# import conditional if system is mac m1
if platform.system() == "Darwin" and platform.machine() == "arm64":
    from kvxopt import solvers
else:
    from cvxopt import solvers

# Robot Goal
x_goal = np.array([5, 5])

# Undesired areas in ellipse format (x,y,rad_x,rad_y) - Use example(0) through example(3)
bad_sets = cbf_utils.example(3)

# Parameters for reference controller
ctrl_param = [10]

# Symbols and equations for the CBF
xr0, xr1, xr2, cx, cy, rad_x, rad_y, xr2_dot, u = symbols("xr0 xr1 xr2 cx cy rad_x rad_y xr2_dot u")
symbs = (cx, cy, rad_x, rad_y, xr0, xr1, xr2)

# Barrier function - distance of robot to obstacle
B = ((xr0 - cx) / rad_x) ** 2 + ((xr1 - cy) / rad_y) ** 2 - 1

# dx = f(x) + g(x)u
f = Matrix([cos(xr2), sin(xr2), 0])
g = Matrix([0, 0, 1])
states_dot = Matrix([cos(xr2), sin(xr2), xr2_dot])

# Initialize CBF
myCBF = cbf.CBF(
    B=B,
    f=f,
    g=g,
    symbs=symbs,
    states=(xr0, xr1, xr2),
    bad_sets=bad_sets,
    states_dot=states_dot,
    degree=2,
    alpha=[10, 10],
)

# ? Simulation settings
T_max = 10
n_samples = 500
T = np.linspace(0, T_max, n_samples)
dt = T[1] - T[0]
params = {
    "goal_x": x_goal,
    "bad_sets": bad_sets,
    "ctrl_param": ctrl_param,
    "myCBF": myCBF,
}

solvers.options["show_progress"] = False

x_0 = np.array([0.5, 0.5, 0])

# Loop through initial conditions
print("\nComputing trajectories for the initial condition:")
print("x_0\t x_1")
print(x_0[0], "\t", x_0[1])

# If initial condition is inside the bad set, error out.
curr_bs = []
for idxj, j in enumerate(bad_sets):
    curr_bs = bad_sets[idxj]
    assert (
        cbf_utils.is_inside_ellipse([x_0[0], x_0[1]], bad_sets[idxj]) == 0
    ), "Initial condition is inside ellipse"


# Compute output on the unicycle system for given initial conditions and timesteps T
x = np.zeros((np.size(x_0), len(T)))
x[:, 0] = x_0
for i in range(len(T) - 1):
    x[:, i + 1] = x[:, i] + dt * np.array(sys_and_ctrl.unicycle_f(T[i], x[:, i], [], params))

# Animate
fig, ax = plt.subplots()
ax = cbf_utils.plot_cbf_elements(ax, bad_sets, x_goal)

plt.xlim(-1, 6)
plt.ylim(-1, 6)

(line1,) = ax.plot([], [], lw=2)
goal_square = plt.Rectangle(x_goal - np.array([0.5, 0.5]), 0.2, 0.2, color="r", alpha=0.5)


def init():
    line1.set_data([], [])
    return line1


def animate(i):
    line1.set_data((x[0][0:i], x[1][0:i]))
    return line1, goal_square


ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=5, frames=n_samples, repeat=False
)

# writergif = PillowWriter(fps=30)
# ani.save("03.gif", writer=writergif)

plt.show()

print("\n*Animation Complete. Exiting...\n")
