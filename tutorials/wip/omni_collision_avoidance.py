import math
import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from sympy import Matrix, cos, sin, symbols


def n_omni_robots_f(t, x, u, params):
    # f = np.zeros((3, 3))
    f = []
    sum_1 = 0
    sum_2 = 0
    # k = 0.025

    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[0]):
            if i != j:
                # define control parameter k for omnidirectional robot
                k = max((0.5 - (np.linalg.norm(x[i, 0:2] - x[j, 0:2]))) / 20, 0.0001)
                # k = 0.025
                # print('k ', k, 'norm', np.linalg.norm(x[i,0:2] - x[j,0:2]))
                sum_1 = sum_1 + k * (x[i, 0] - x[j, 0]) / (
                    np.linalg.norm(x[i, 0:2] - x[j, 0:2]) + 0.000001
                )
                # print(x[i, 0] - x[j, 0],np.linalg.norm(x[i,0:2] - x[j,0:2]) + 0.000001, sum_1)
                sum_2 = sum_2 + k * (x[i, 1] - x[j, 1]) / (
                    np.linalg.norm(x[i, 0:2] - x[j, 0:2]) + 0.000001
                )
        f.append([])
        f[i].append(sum_1)
        f[i].append(sum_2)
        f[i].append(0)
        sum_1 = 0
        sum_2 = 0

    f = np.array(f)
    print(f)
    # dynamics of the robot
    R = 0.02
    L = 0.2
    newg = np.zeros((x.shape[0], 3, 3), dtype=float)
    for i in range(0, x.shape[0]):
        # np.append(g,[])
        newg[i] = np.array(
            [
                [
                    [cos(x[i][2]), -1 * sin(x[i][2]), 0],
                    [sin(x[i][2]), cos(x[i][2]), 0],
                    [0, 0, 1],
                ]
            ]
        )

    B = np.array(
        [
            [0, cos(math.pi / 6), -1 * cos(math.pi / 6)],
            [-1, sin(math.pi / 6), sin(math.pi / 6)],
            [L, L, L],
        ],
        dtype=float,
    )

    u = np.zeros((3, 1))
    u[0] = 0
    u[1] = -1
    u[2] = 1

    dx = np.zeros((x_0.shape[0], x_0.shape[1]))

    for i in range(0, x.shape[0]):
        if i == 0:
            u[0] = 0
            u[1] = 1
            u[2] = -1
        if i == 1:
            u[0] = 0
            u[1] = -1
            u[2] = 1
        temp = np.matmul(newg[i], np.linalg.inv(B.T)) * R
        temp2 = np.matmul(temp, u)
        dx[i] = f[i] + temp2.T

    return dx


def example(i):
    # Examples of different bad sets
    # The x,y,z,d where x,y represent the center and z,d represents the major, minor axes of the ellipse
    switcher = {
        0: [[3, 2.0, 1.0, 1.0]],
        1: [
            [1.0, 2.0, 0.5, 0.5],
            [4.0, 1.0, 0.5, 0.5],
            [3.0, 2.0, 0.5, 0.5],
            [4.5, 4.2, 0.5, 0.5],
        ],
        2: [[3.5, 1.0, 0.2, 2.0], [2.0, 2.5, 1.0, 0.2], [1.5, 1.0, 0.5, 0.5]],
        3: [[3.5, 3.0, 0.2, 2.0], [2.0, 2.5, 1.0, 0.2], [1.5, 1.0, 0.5, 0.5]],
    }
    return switcher.get(i, "Invalid")


def is_inside_ellipse(x, x_e):
    if ((x[0] - x_e[0]) / x_e[2]) ** 2 + ((x[1] - x_e[1]) / x_e[3]) ** 2 <= 1:
        return 1
    else:
        return 0


# Robot Goal
goal_x = np.array([5, 5])

# Elipse format (x,y,rad_x,rad_y)
bad_sets = example(3)

# Parameters for reference controller
ctrl_param = [5]

xr0, xr1, xr2, cx, cy, rad_x, rad_y, xr2_dot, u = symbols("xr0 xr1 xr2 cx cy rad_x rad_y xr2_dot u")

B = ((xr0 - cx) / rad_x) ** 2 + ((xr1 - cy) / rad_y) ** 2 - 1
# B = (xr0 - cx)**2 + (xr1 - cx)**2 - 1
f = Matrix([cos(xr2), sin(xr2), 0])
g = Matrix([0, 0, 1])
states_dot = Matrix([cos(xr2), sin(xr2), xr2_dot])

# expr_bs_dx0 = diff(expr_bs,xr0)
# expr_bs_dx1 = diff(expr_bs,xr1)


# ? Simulation settings
T_max = 200
n_samples = 100
T = np.linspace(0, T_max, n_samples)
dt = T[1] - T[0]
params = {
    "goal_x": goal_x,
    "bad_sets": bad_sets,
    "ctrl_param": ctrl_param,
}

# System definition using the control toolbox
# nimble_car_sys = control.NonlinearIOSystem(
#     nimble_car_f, None, inputs=None, outputs=None, dt=None,
#     states=('x0', 'x1', 'x2'), name='nimble_car',
#  params=params)

# ? Initial conditions
# ? min, max of x,y values for initial conditions
min_x, min_y, max_x, max_y = -0.5, -0.5, 0.5, 0.5
nx, ny = 10, 10  # number of initial conditions in each axis

# Vectors of initial conditions in each axis
xx = np.linspace(min_x, max_x, nx)
yy = np.linspace(min_y, max_y, ny)

# ? Uncomment the following for specific intial conditions
xx = [0.5]
yy = [0.5]


# exmample 1
# x1 = np.array([1, 2, 0], dtype=np.fl oat32)
# x2 = np.array([4, 2.1, 0])
# x_0 = np.vstack([x1, x2])

# exmample 2
x1 = np.array([2, 1, math.pi / 4], dtype=np.float32)
x2 = np.array([3, 4, 3 * math.pi / 4])
x3 = np.array([4, 2, 7 * math.pi / 4])
x4 = np.array([4, 4, math.pi / 4])
x_0 = np.vstack([x1, x2, x3, x4])


for idxi, i in enumerate(xx):
    for idxk, k in enumerate(yy):
        print(round(i, 2), "\t", round(k, 2), "\t... ", end="", flush=True)
        # x_0 = np.array([i, k, 0])

        # Compute output on the silly bug system for given initial conditions and timesteps T
        x = np.zeros((len(T), x_0.shape[0], x_0.shape[1]), dtype=np.float64)
        x[0] = x_0
        for i in range(len(T) - 1):
            x[i + 1] = x[i] + dt * np.array(n_omni_robots_f(T[i], x[i], [], params))

        # # Plot initial conditions and path of system
        # plt.plot(i, k, 'x-', markersize=5, color=[0, 0, 0, 1])
        # plt.plot(x[0], x[1], 'o-', markersize=2,
        #          color=next(colors, [1, 1, 1, 1]))

        colors = ["red", "green", "purple"]

        # plot motion of robot for matrix x
        # for i in range(len(T)):
        #     for j in range(3):
        #         plt.plot(x[i][j][0], x[i][j][1], 'o-', markersize=2, color=colors[j])

        # plt.show()
        # print("Done")

# Plot
fig, ax = plt.subplots()
jet = plt.get_cmap("jet")
# colors = iter(jet(np.linspace(0, 1, len(xx)*len(yy))))

plt.xlim(0, 5)
plt.ylim(0, 5)
# plt.xlim(0, 12)
# plt.ylim(0, 6)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

(line1,) = ax.plot([], [], lw=2)
(line2,) = ax.plot([], [], lw=2)
(line3,) = ax.plot([], [], lw=2)
(line4,) = ax.plot([], [], lw=2)

lines = [line1, line2, line3, line4]


# animate path of two robots
def animate(i):
    ims = []
    # iterate over lines
    for j in range(len(lines)):
        lines[j].set_data((x[0:i, j, 0], x[0:i, j, 1]))
        ims.append(lines[j])

    # line1.set_data((x[0:i,0,0], x[0:i,0,1]))
    # line2.set_data((x[0:i,1,0], x[0:i,1,1]))
    # ims.append(line1)
    # ims.append(line2)
    return ims


# n_samples = 98

ani = animation.FuncAnimation(fig, animate, interval=50, blit=True, frames=n_samples)

plt.show()
