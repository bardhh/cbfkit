# This code is a Python program that simulates the movement of a robot in an environment with obstacles. It uses the Sympy, Numpy, Matplotlib, and Control libraries to perform calculations and generate visualizations. The program defines a goal for the robot to reach, as well as undesired areas in ellipse format that the robot should avoid. It also sets parameters for a reference controller and initializes a CBF (Control Barrier Function) to ensure safety during the simulation. The program then simulates the system and generates an animation of the robot's movement towards the goal while avoiding the obstacles. Finally, it displays the animation and exits.

import platform

import control as control
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from sympy import Matrix, symbols

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
bad_sets = cbf_utils.example(1)

# Parameters for reference controller
ctrl_param = 0.8

# Symbols and equations for the CBF
x_0, x_1, z_0, z_1, a, b = symbols("x_0 x_1 z_0 z_1 a b")
symbs = (x_0, x_1, z_0, z_1, a, b)
# Barrier function - distance of robot to obstacle
B = ((x_0 - z_0) / a) ** 2 + ((x_1 - z_1) / b) ** 2 - 1

f = 0
g = Matrix([0, 0, 1.0])

# Initialize CBF
my_CBF = cbf.CBF(B, f, g, states=(x_0, x_1), bad_sets=bad_sets, symbs=symbs, degree=1, alpha=0.6)

# Simulation settings
T_max = 20
n_samples = 200
T = np.linspace(0, T_max, n_samples)
dt = T[1] - T[0]
params = {
    "x_goal": x_goal,
    "bad_sets": bad_sets,
    "ctrl_param": ctrl_param,
    "CBF": my_CBF,
}

# Disable cvxopt optimiztaion output
solvers.options["show_progress"] = False

# intial condition
x_0 = np.array([0.5, 1.5])

# Simulate system
print("\nComputing trajectories for the initial condition:")
print("x_0\t x_1")
print(x_0[0], "\t", x_0[1])

# If initial condition is inside the bad set, error out.
for idxj, j in enumerate(bad_sets):
    curr_bs = bad_sets[idxj]
    assert (
        cbf_utils.is_inside_ellipse([x_0[0], x_0[1]], bad_sets[idxj]) == 0
    ), "Initial condition is inside ellipse"

# Compute output on the nimble ant system for given initial conditions and timesteps T
x = np.zeros((np.size(x_0), len(T)))
x[:, 0] = x_0

# Simulate system
for i in range(len(T) - 1):
    x[:, i + 1] = x[:, i] + dt * np.array(sys_and_ctrl.nimble_ant_f(T[i], x[:, i], [], params))

print("\n*Simulation Done. See plot for animation...")

# Init Plot
fig, ax = plt.subplots()

# Animate
ax = cbf_utils.plot_cbf_elements(ax, bad_sets, x_goal)

plt.xlim(0, 6)
plt.ylim(0, 6)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

(line1,) = ax.plot([], [], lw=2)
goal_square = plt.Rectangle(x_goal - np.array([0.5, 0.5]), 0.2, 0.2, color="r", alpha=0.5)


def init():
    line1.set_data([], [])
    return line1


def animate(i):
    line1.set_data((x[0][0:i], x[1][0:i]))
    return line1


ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=20, frames=n_samples, repeat=False
)

# writergif = PillowWriter(fps=30)
# ani.save("01.gif", writer=writergif)

plt.show()

print("\n*Animation Complete. Exiting...\n")
