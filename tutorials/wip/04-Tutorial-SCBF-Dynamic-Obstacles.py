import platform

import control as control

# from cbfkit.examples.tutorials.cbflib import tut_scbf, sys_and_ctrl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sympy import Matrix, cos, exp, sin, symbols

from examples.tutorials.cbflib import sys_and_ctrl, tut_scbf

# import conditional if system is mac m1
if platform.system() == "Darwin" and platform.machine() == "arm64":
    from kvxopt import solvers
else:
    from cvxopt import solvers


def strList2SympyMatrix(str_list):
    sympy_symbols_lst = []
    for istr in str_list:
        sympy_symbol = symbols(istr)
        sympy_symbols_lst.append(sympy_symbol)
    sympy_matrix = Matrix(sympy_symbols_lst)
    return sympy_matrix


def appr_unicycle(states, inputs, l):
    """This function defines approximate unicycle model

    Args:
        f_r_states_str (list): name list of system states
        inputs_str (list): name list of system inputs

    Returns:
        f, g (symbolic expressions): to describe model of the system as dx = f+g*input
    """

    if states.shape[0] != 3 or inputs.shape[0] != 2:
        raise ValueError("appr_unicycle model has 3 states and 2 inputs")

    f = Matrix([0, 0, 0])
    g = Matrix(
        [
            [cos(states[2]), -l * sin(states[2])],
            [sin(states[2]), l * cos(states[2])],
            [0, 1],
        ]
    )
    return f, g


# Robot Goal
x_goal = np.array([10, 3.5])

# The bad set defines the radii of the eclipse. The position is part of the dynamics of the system.
bad_sets = [[3, 2]]

# Parameters for reference controller
ctrl_param = [2]

# Symbols and equations for the CBF
xr_0, xr_1, xo_0, xo_1, a, b, xr_0_dot, xr_1_dot = symbols(
    "xr_0 xr_1 xo_0 xo_1 a b xr_0_dot xr_1_dot"
)
symbs = (xr_0, xr_1, xo_0, xo_1, a, b)

f_r_states_str = ["xr_0", "xr_1", "xr_2"]
f_r_inputs_str = ["ur_0", "ur_1"]
C = Matrix([[1, 0, 0], [0, 1, 0]])

f_r_states = strList2SympyMatrix(f_r_states_str)
f_r_inputs = strList2SympyMatrix(f_r_inputs_str)

l = 0.1
f_r, g_r = appr_unicycle(f_r_states, f_r_inputs, l)

G = Matrix(np.eye(len(f_r_states)))
D = Matrix(np.eye(2))


# Barrier function - distance of robot to obstacle
h = ((xr_0 - xo_0)) ** 2 + ((xr_1 - xo_1)) ** 2 - 1.0

B = exp(-2 * h)

f_o_states_str = ["xo_0", "xo_1"]
f_o_states = strList2SympyMatrix(f_o_states_str)

f_o = Matrix([-1, 0])
g_o = Matrix([0.2, 0.0])

# Initialize CBF
# my_CBF = cbf.CBF(B=B, f=f, g=g, states=(
# xr_0, xr_1, xo_0, xo_1), bad_sets=bad_sets, symbs=symbs, degree=1, alpha = 0.6)

my_CBF = tut_scbf.tut_scbf(
    B=B,
    f_r=f_r,
    g_r=g_r,
    f_o=f_o,
    g_o=g_o,
    f_r_states=f_r_states,
    f_r_inputs=f_r_inputs,
    f_o_states=f_o_states,
    bad_sets=bad_sets,
    symbs=symbs,
    alpha=0.6,
)

# Simulation settings
T_max = 10
n_samples = 250
T = np.linspace(0, T_max, n_samples)
dt = T[1] - T[0]
params = {
    "x_goal": x_goal,
    "bad_sets": bad_sets,
    "ctrl_param": ctrl_param,
    "CBF": my_CBF,
}

# intial condition
xr_0 = np.array([0, 2.4, 0])

xo_0 = np.array([10, 3.8])

# Disable cvxopt optimiztaion output
solvers.options["show_progress"] = False
solvers.options["max_iter"] = 5000

# Init Plot
fig, ax = plt.subplots()

# Simulate system
print("\nComputing trajectories for the initial condition:")
print("x_0\t x_1")
print(xr_0[0], "\t", xr_0[1])

# Compute output on the nimble ant system for given initial conditions and timesteps T
xr_sim = np.zeros((np.size(xr_0), len(T)))
xr_sim[:, 0] = xr_0

xo_sim = np.zeros((np.size(xo_0), len(T)))
xo_sim[:, 0] = xo_0

# Simulate system
for i in range(len(T) - 1):
    xr_sim[:, i + 1] = (
        xr_sim[:, i]
        + dt * (np.array(sys_and_ctrl.approx_unicycle_agent_f(T[i], xr_sim[:, i], [], params))).T
    )
    xo_sim[:, i + 1] = (
        xo_sim[:, i]
        + dt * (np.array(sys_and_ctrl.simple_agent_left_f(T[i], xo_sim[:, i], [], params))).T
    )

print("\n*Simulation Done. See plot for animation...")

# Animate
plt.xlim(0, 12)
plt.ylim(0, 6)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
(line1,) = ax.plot([], [], lw=2)
(line2,) = ax.plot([], [], lw=2)

ell = Ellipse((xo_sim[0][0], xo_sim[0][0]), 1, 1, color="g", alpha=0.3)

goal_square = plt.Rectangle(x_goal - np.array([0.5, 0.5]), 0.2, 0.2, color="r", alpha=0.5)


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    ell.center = (xo_sim[0][0], xo_sim[1][0])
    return line1, line2


def animate(i):
    line1.set_data((xr_sim[0][0:i], xr_sim[1][0:i]))
    line2.set_data((xo_sim[0][0:i], xo_sim[1][0:i]))
    ell.center = (xo_sim[0][i], xo_sim[1][i])
    return line1, line2, ax.add_patch(ell)


ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=10, blit=False, frames=n_samples
)

# writergif = PillowWriter(fps=30)
# ani.save("02.gif", writer=writergif)


plt.show()

print("\n*Animation Complete. Exiting...\n")
