import numpy as np
import matplotlib.pyplot as plt
import control as control
import cvxopt as cvxopt
from matplotlib.patches import Ellipse
import numpy.random as rnd
import math as math
from sympy import symbols, Matrix, sin, cos, lambdify, exp, sqrt, log


def nimble_ant_c(x, params):
    # Controller for silly bug

    goal_x = params['goal_x']
    bad_sets = params['bad_sets']
    ctrl_param = params['ctrl_param']

    # Reference controller

    uref_0 = ctrl_param[0] * math.atan((goal_x[1]-x[1])/(goal_x[0]-x[0]))

    ############################
    # cvxopt quadratic program
    # minimize  0.5 x'Px + q'x
    # s.t       Gx<=h
    ############################

    # P matrix
    P = cvxopt.matrix(np.eye(1))
    P = .5 * (P + P.T)  # symmetric

    # q matrix
    q = cvxopt.matrix(np.array([-uref_0]), (1, 1))

    # Parameters for the CBF
    G = []
    h = []

    # For each bad set, a separate CBF constraint should be added
    for idxi, _ in enumerate(bad_sets):
        curr_bs = bad_sets[idxi]

        g1 = -1 * (2*math.sin(x[2])) * (x[0]-curr_bs[0]) / curr_bs[2]**2 - \
            2 * math.cos(x[2]) * (x[1] - curr_bs[1]) / curr_bs[3]**2
        g2 = 4 * ((x[0] - curr_bs[0])/curr_bs[2])**2 + 4*((x[1] - curr_bs[1])/curr_bs[3])**2 + 4*math.cos(
            x[2]) * ( 2*x[0] - 2*curr_bs[0] ) / curr_bs[2]**2 + 4 * ( math.sin(x[2]) ) * ( 2*x[0] - 2 * curr_bs[1] ) / curr_bs[3]**2

        # ((2*x_0 - cx)/rad_x)^2 + ((2*x_1 - cy)/rad_y)^2 - 1
        g3 = ((x[0] - curr_bs[0])/curr_bs[2])**2 + \
            ((x[1] - curr_bs[1])/curr_bs[3])**2 - 1

        G.append([-g1])
        h.append([g3+g2])

    # Convert lists to cvxopt.matrix object
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)

    # Run optimizer and return solution
    sol = cvxopt.solvers.qp(P, q, G.T, h.T, None, None)
    x_sol = sol['x']
    return x_sol[0:1]


def nimble_ant_f(t, x, u, params):
    # Function for a silly bug

    # if goal reached, do nothing
    # goal_x = params['goal_x']
    # if np.linalg.norm(x-goal_x) <= 0.05:
    #     return [0, 0]

    # compute control given current position
    u_0 = nimble_ant_c(x, params)

    v = 1

    # compute change in xy direction
    dx0 = v*math.cos(x[2])
    dx1 = v*math.sin(x[2])
    dx2 = u_0[0]

    return [dx0, dx1, dx2]


def example(i):
    # Examples of different bad sets
    # The x,y,z,d where x,y represent the center and z,d represents the major, minor axes of the ellipse
    switcher = {
        0: [[3., 2., 1., 1.]],
        1: [[1., 2., 0.5, 0.5], [4., 1., 0.5, 0.5],
            [3., 2., 0.5, 0.5], [4.5, 4.2, 0.5, 0.5]],
        2: [[3.5, 1., 0.2, 2.], [2., 2.5, 1., 0.2], [1.5, 1., 0.5, 0.5]],
        3: [[3.5, 3., 0.2, 2.], [2., 2.5, 1., 0.2], [1.5, 1., 0.5, 0.5]]
    }
    return switcher.get(i, "Invalid")


def is_inside_ellipse(x, x_e):
    if ((x[0] - x_e[0])/x_e[2])**2 + ((x[1] - x_e[1])/x_e[3])**2 <= 1:
        return 1
    else:
        return 0


# Robot Goal
goal_x = np.array([5, 5])

# Elipse format (x,y,rad_x,rad_y)
bad_sets = example(0)

# Parameters for reference controller
ctrl_param = [1]

# Simulation settings
T_max = 20
n_samples = 100
T = np.linspace(0, T_max, n_samples)

# System definition using the control toolbox
silly_bug_sys = control.NonlinearIOSystem(
    nimble_ant_f, None, inputs=None, outputs=None, dt=None,
    states=('x0', 'x1', 'x2'), name='silly_bug',
    params={'goal_x': goal_x, 'bad_sets': bad_sets, 'ctrl_param': ctrl_param})

# Initial conditions
# min, max of x,y values for initial conditions
min_x, min_y, max_x, max_y = -0.5, -0.5, 0.5, 0.5
nx, ny = 4, 4              # number of initial conditions in each axis

# Vectors of initial conditions in each axis
xx = np.linspace(min_x, max_x, nx)
yy = np.linspace(min_y, max_y, ny)

# Uncomment the following for specific intial conditions
xx = [4]
yy = [0]

# Disable cvxopt optimiztaion output
cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['max_iter'] = 1000


# Plot
fig, ax = plt.subplots()
jet = plt.get_cmap('tab20b')
colors = iter(jet(np.linspace(0, 1, len(xx)*len(yy))))

# Loop through initial conditions
print('Computing trajectories for initial conditions:')
print('x_0\t x_1')
for idxi, i in enumerate(xx):
    for idxk, k in enumerate(yy):
        # If initial condition is inside the bad set, skip it.
        bool_val = 0
        curr_bs = []

        for idxj, j in enumerate(bad_sets):
            curr_bs = bad_sets[idxj]
            if is_inside_ellipse([i, k], bad_sets[idxj]):
                print('Skip (Invalid):\t', i, k)
                bool_val = 1
        if bool_val == 1:
            continue

        print(round(i, 2), '\t', round(k, 2), "\t... ", end="", flush=True)
        x_0 = np.array([i, k])

        # Compute output on the silly bug system for given initial conditions and timesteps T
        t, y, x = control.input_output_response(sys=silly_bug_sys, T=T, U=0, X0=[
                                                i, k,0], return_x=True, method='BDF')

        # Plot initial conditions and path of system
        plt.plot(i, k, 'x-', markersize=5, color=[0, 0, 0, 1])
        plt.plot(x[0], x[1], 'o-', markersize=2,
                 color=next(colors, [1, 1, 1, 1]))

        print("Done")

curr_bs = []
for idxi, _ in enumerate(bad_sets):
    curr_bs = bad_sets[idxi]
    ell = Ellipse((curr_bs[0], curr_bs[1]), 2 *
                  curr_bs[2], 2 * curr_bs[3], color='r')
    ax.add_patch(ell)

goal_square = plt.Rectangle(goal_x-np.array([.1, .1]), .2, .2, color='g')
ax.add_patch(goal_square)

plt.xlim(0, 6)
plt.ylim(0, 6)
plt.show()
