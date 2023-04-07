import numpy as np
import matplotlib.pyplot as plt
import cvxopt as cvxopt
from matplotlib.patches import Ellipse
import numpy.random as rnd
from sympy import symbols, Matrix, sin, cos, lambdify, exp, sqrt, log, diff, Mul, srepr
from sympy.diffgeom import LieDerivative
from sympy.diffgeom.rn import R2_r
import math

class CBF:
    def __init__(self, B, f, g, states, bad_sets, states_dot):
        self.B = B
        self.phi = []
        self.states = states
        self.G = []
        self.h = []
        self.expr_bs = []
        self.lamb_G = []

        a1 = 7
        a2 = 10

        expr = self.get_expr(B, f, g, states, states_dot)

        G, h = self.decompose_G_h(expr, g, states_dot)
        self.lamb_G.append(
            lambdify([(cx, cy, rad_x, rad_y, xr0, xr1, xr2, xr3)], G, "math"))
#!  h+B is incorrect when second order system,
        self.lamb_h = lambdify(
            # [(cx, cy, rad_x, rad_y, xr0, xr1, xr2)], (h+B), "math")
            [(cx, cy, rad_x, rad_y, xr0, xr1, xr2, xr3)], h , "math")

    def compute_G_h(self, x):
        self.G = []
        self.h = []
        for idxi, _ in enumerate(bad_sets):
            curr_bs = bad_sets[idxi]
            tmp_g = []
            self.G.append([])
            for lamb in self.lamb_G:
                tmp_g = lamb(tuple(np.hstack((curr_bs, x))))
                self.G[idxi].append(tmp_g)
            self.h.append(self.lamb_h(tuple(np.hstack((curr_bs, x)))))
        return self.G, self.h

    def get_expr(self, B, f, g, states, states_dot):
        a = 10
        B_dot_var = []
        for i in states:
            B_dot_var.append(diff(B, i))
        B_dot = Matrix(B_dot_var)
        B_dot_f = B_dot.T * states_dot
        phi = B_dot_f[0] + a * B
        self.phi.append(phi)
        if xr2_dot in phi.free_symbols and xr3_dot in phi.free_symbols:  #! This needs to be revised
            return phi
        elif xr2_dot in phi.free_symbols or xr3_dot in phi.free_symbols:
            Warning('mixed relative degrees')
        else:
            return self.get_expr(phi, f, g, states, states_dot)

    def decompose_G_h(self, expr, g, states_dot):
        G = []
        h = 0
        for arg in expr.args:
            if xr2_dot in arg.free_symbols:
                G.append(- arg.subs(xr2_dot, 1))                   
            elif xr3_dot in arg.free_symbols:
                G.append(- arg.subs(xr3_dot, 1))
            else:
                h = h + arg
        return G, h


def nimble_car_c(x, params):
    # Controller for nimble car
    goal_x = params['goal_x']
    bad_sets = params['bad_sets']
    ctrl_param = params['ctrl_param']
    myCBF = params['myCBF']

    # Reference controller
    theta_ref = math.atan((goal_x[1]-x[1])/(goal_x[0]-x[0]))
    uref_w = ctrl_param[0] * (theta_ref - x[2])
    uref_v = 1  # hard coded here but can be replaced later

    ############################
    # cvxopt quadratic program
    # minimize  0.5 x'Px + q'x
    # s.t       Gx<=h
    ############################
    # P matrix
    P = cvxopt.matrix(np.eye(2))
    # P = .5 * (P + P.T)  # symmetric

    # q matrix
    q = cvxopt.matrix(np.array([-1*uref_w, -1*uref_v]), (2, 1))

    G, h = myCBF.compute_G_h(x)
    G = [g[0] for g in G]

    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)

    # Run optimizer and return solution
    # sol = cvxopt.solvers.qp(P, q, G.T, h, None, None)
    try:
        sol = cvxopt.solvers.qp(P, q, G.T, h, None, None)
        x_sol = sol['x']
    except:
        x_sol = [0]
        print("bad")
    # print(x, ' G: ', G, ' h: ', h, ' x_sol: ', x_sol)
    return x_sol[0:2]


def nimble_car_f(t, x, u, params):
    # Function for a silly bug
    # if goal reached, do nothing
    goal_x = params['goal_x']
    if (x[0] - goal_x[0])**2 + (x[1] - goal_x[1])**2 <= 0.1**2:
        return [0, 0, 0, 0]

    # compute control given current position
    u_0 = nimble_car_c(x, params)

    # compute change in xy direction
    dx0 = math.cos(x[2]) 
    dx1 = math.sin(x[2])
    dx2 = u_0[0]
    dx3 = u_0[1]

    return [dx0, dx1, dx2, dx3]


def example(i):
    # Examples of different bad sets
    # The x,y,z,d where x,y represent the center and z,d represents the major, minor axes of the ellipse
    switcher = {
        0: [[3, 2., 1., 1.]],
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
bad_sets = example(3)

# Parameters for reference controller
ctrl_param = [5]

xr0, xr1, xr2, xr3, cx, cy, rad_x, rad_y, xr2_dot, xr3_dot, u = symbols(
    'xr0 xr1 xr2 xr3 cx cy rad_x rad_y xr2_dot xr3_dot u')

B = ((xr0 - cx)/rad_x)**2 + ((xr1 - cy)/rad_y)**2 - 1
# B = (xr0 - cx)**2 + (xr1 - cx)**2 - 1
f = Matrix([xr3*cos(xr2), xr3*sin(xr2), 0, 0])
g = Matrix([[0, 0, 1, 0], [0, 0, 0, 1]])
states_dot = Matrix([xr3*cos(xr2), xr3*sin(xr2), xr2_dot, xr3_dot])

# expr_bs_dx0 = diff(expr_bs,xr0)
# expr_bs_dx1 = diff(expr_bs,xr1)

myCBF = CBF(B, f, g, (xr0, xr1, xr2, xr3), bad_sets, states_dot)

#? Simulation settings
T_max = 10
n_samples = 1000
T = np.linspace(0, T_max, n_samples)
dt = T[1]-T[0]
params={'goal_x': goal_x, 'bad_sets': bad_sets, 'ctrl_param': ctrl_param, 'myCBF': myCBF}

# System definition using the control toolbox
# nimble_car_sys = control.NonlinearIOSystem(
#     nimble_car_f, None, inputs=None, outputs=None, dt=None,
#     states=('x0', 'x1', 'x2'), name='nimble_car',
#  params=params)

#? Initial conditions
#? min, max of x,y values for initial conditions
min_x, min_y, max_x, max_y = -0.5, -0.5, 0.5, 0.5
nx, ny = 10, 10              # number of initial conditions in each axis

# Vectors of initial conditions in each axis
xx = np.linspace(min_x, max_x, nx)
yy = np.linspace(min_y, max_y, ny)

#? Uncomment the following for specific intial conditions
xx = [0.5]
yy = [0.5]

# Disable cvxopt optimiztaion output
cvxopt.solvers.options['show_progress'] = False
# cvxopt.solvers.options['max_iter'] = 1000

# Plot
fig, ax = plt.subplots()
jet = plt.get_cmap('tab20b')
colors = iter(jet(np.linspace(0, 1, len(xx)*len(yy))))

# Loop through initial conditions
print('Computing trajectories for initial conditions:')
print('x_0\t x_1')


curr_bs = []
for idxi, _ in enumerate(bad_sets):
    curr_bs = bad_sets[idxi]
    ell = Ellipse((curr_bs[0], curr_bs[1]), 2 *
                  curr_bs[2], 2 * curr_bs[3], color='r')
    ax.add_patch(ell)

# goal_square = plt.Rectangle(goal_x-np.array([.1, .1]), .2, .2, color='g')
# ax.add_patch(goal_square)

goal_circle =  Ellipse((goal_x[0], goal_x[1]), 2*0.1, 2*0.1, color='g')
ax.add_patch(goal_circle)

plt.xlim(-2, 7)
plt.ylim(-2, 7)

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
        x_0 = np.array([i, k, 0, 0])  

        # Compute output on the silly bug system for given initial conditions and timesteps T
        x = np.zeros((np.size(x_0), len(T)))
        x[:,0] = x_0
        for i in range(len(T)-1):
            x[:,i+1] = x[:,i] + dt*  np.array( nimble_car_f(T[i], x[:,i], [], params))
            


        # Plot initial conditions and path of system
        plt.plot(i, k, 'x-', markersize=5, color=[0, 0, 0, 1])
        plt.plot(x[0], x[1], 'o-', markersize=2,
                 color=next(colors, [1, 1, 1, 1]))
        plt.show()
        print("Done")


