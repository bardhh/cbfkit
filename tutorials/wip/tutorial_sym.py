from matplotlib.patches import Ellipse
from sympy import symbols, lambdify, diff
import numpy as np
import matplotlib.pyplot as plt
import control as control
import cvxopt as cvxopt


class CBF:
    def __init__(self, B, g, states, bad_sets):
        """ This initializes the CBF and computes functions for the G and h matrices for convex optimization later on.
        Args:
            B (sympy expression):   The expression for the bad set representation
            g (sympy expression):   The expression for the g(x) function in the dynamics of the system
            states (tuple):         A tuple with the states of the system
            bad_sets (list):        A list of bad sets, with each row consisting of x,y,z,d
                                    where x,y represent the center and z,d represents the major,
                                    minor axes of the ellipse
        """
        self.B = B
        self.states = states
        self.G = []                 # G matrix for CVXopt
        self.h = []                 # h matrix for CVXopt
        self.expr_bs = []           # symbolic expressions for bad sets
        self.lamb_G = []            # function for computation of symbolic expression for G matrix
        for i in self.states:
            temp_expr = diff(B, i)
            self.expr_bs.append(temp_expr)
            self.lamb_G.append(
                lambdify([(cx, cy, rad_x, rad_y, xr0, xr1)], temp_expr, "math"))
        # function for computation of symbolic expression for h matrix
        self.lamb_h = lambdify([(cx, cy, rad_x, rad_y, xr0, xr1)], B, "math")

    def compute_G_h(self, x):
        """ The method computes the G and h matrices for convex optimization given current state

        Args:
            x (numpy.ndarray): array with the current state of the system

        Returns:
            list: returns G matrix
            list: returns h matrix
        """
        self.G = []
        self.h = []

        # for each bad set, given current state, compute the G and h matrices
        for idxi, _ in enumerate(bad_sets):
            curr_bs = bad_sets[idxi]
            tmp_g = []
            self.G.append([])
            for lamb in self.lamb_G:
                tmp_g = lamb(tuple(np.hstack((curr_bs, x))))
                self.G[idxi].append(-1*tmp_g)
            self.h.append(self.lamb_h(tuple(np.hstack((curr_bs, x)))))
        return self.G, self.h


def nimble_ant_c(x, params):
    """ Controller for nimble ant

    Args:
        x (numpy.ndarray): current state of the system
        params (dict): Dict keys:
                        goal_x: the goal or target state
                        bad_sets: list of elippses defining bad sets
                        ctrl_param: parameters for the controller
                        CBF: the CBF object 
    Returns:
        cvxopt.base.matrix: the control for the system
    """
    goal_x = params['goal_x']
    bad_sets = params['bad_sets']
    ctrl_param = params['ctrl_param']
    myCBF = params['CBF']

    # Reference controller
    uref_0 = ctrl_param[0] * ((goal_x[0]-x[0]))
    uref_1 = ctrl_param[1] * ((goal_x[1]-x[1]))

    ############################
    # cvxopt quadratic program
    # minimize  0.5 x'Px + q'x
    # s.t       Gx<=h
    ############################

    # P matrix
    P = cvxopt.matrix(np.eye(2))
    P = .5 * (P + P.T)  # symmetric

    # q matrix
    q = cvxopt.matrix(np.array([-uref_0, -uref_1]), (2, 1))

    G, h = myCBF.compute_G_h(x)

    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)

    # Run optimizer and return solution
    sol = cvxopt.solvers.qp(P, q, G.T, h, None, None)
    x_sol = sol['x']
    return x_sol[0:2]


def nimble_ant_f(t, x, u, params):
    """ Function for the nimble_ant system

    Args:
        t (float): [description]
        x (numpy.ndarray): [description]
        u (numpy.ndarray): [description]
        params (dict): Dict keys:
                        goal_x: the goal or target state
                        bad_sets: list of elippses defining bad sets
                        ctrl_param: parameters for the controller
                        CBF: the CBF object 

    Returns:
        list: dx
    """

    # compute control given current position
    u_0 = nimble_ant_c(x, params)

    # compute change in xy direction
    dx0 = u_0[0]
    dx1 = u_0[1]

    return [dx0, dx1]


def example(i):
    # Examples of different bad sets
    # The x,y,z,d where x,y represent the center and z,d represents the major, minor axes of the ellipse
    switcher = {
        0: [[3., 3., 1., 1]],
        1: [[1., 2., 0.5, 0.5], [4., 1., 0.5, 0.5],
            [3., 2., 0.5, 0.5], [4.5, 4.2, 0.5, 0.5]],
        2: [[3.5, 1., 0.2, 2.], [2., 2.5, 1., 0.2], [1.5, 1., 0.5, 0.5]],
        3: [[3.5, 3., 0.2, 2.], [2., 2.5, 1., 0.2], [1.5, 1., 0.5, 0.5]]
    }
    return switcher.get(i, "Invalid")


def is_inside_ellipse(x, x_e):
    # Check if state is inside ellipse
    if ((x[0] - x_e[0])/x_e[2])**2 + ((x[1] - x_e[1])/x_e[3])**2 <= 1:
        return 1
    else:
        return 0


# Robot Goal
goal_x = np.array([5, 5])

# Elipse format (x,y,rad_x,rad_y)
bad_sets = example(2)

# Parameters for reference controller
ctrl_param = [0.3, 0.3]

# Symbols for the CBF
xr0, xr1, cx, cy, rad_x, rad_y, u = symbols('xr0 xr1 cx cy rad_x rad_y u')
B = ((xr0 - cx)/rad_x)**2 + ((xr1 - cy)/rad_y)**2 - 1
g = u

# expr_bs_dx0 = diff(expr_bs,xr0)
# expr_bs_dx1 = diff(expr_bs,xr1)

myCBF = CBF(B, g, (xr0, xr1), bad_sets)

# Simulation settings
T_max = 35
n_samples = 100
T = np.linspace(0, T_max, n_samples)

# System definition using the control toolbox
nimble_ant_sys = control.NonlinearIOSystem(
    nimble_ant_f, None, inputs=None, outputs=None, dt=None,
    states=('x0', 'x1'), name='nimble_ant',
    params={'goal_x': goal_x, 'bad_sets': bad_sets, 'ctrl_param': ctrl_param, 'CBF': myCBF})

# Initial conditions
# min, max of x,y values for initial conditions
min_x, min_y, max_x, max_y = 0, 0, 0.5, 0.5
nx, ny = 3, 3              # number of initial conditions in each axis

# Vectors of initial conditions in each axis
xx = np.linspace(min_x, max_x, nx)
yy = np.linspace(min_y, max_y, ny)

# Uncomment the following for specific intial conditions
xx = [0]
yy = [0]

# Disable cvxopt optimiztaion output
cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['max_iter'] = 1000


# Plot
fig, ax = plt.subplots()
jet = plt.get_cmap('jet')
colors = iter(jet(np.linspace(0, 1, len(xx)*len(yy))))

# Loop through initial conditions and simulate system
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
        t, y, x = control.input_output_response(sys=nimble_ant_sys, T=T, U=0, X0=[
                                                i, k], return_x=True, method='BDF')

        # Plot initial conditions and path of system
        plt.plot(i, k, 'x-', markersize=3, color=[0, 0, 0, 1])
        plt.plot(x[0], x[1], 'o-', markersize=1,
                 color=next(colors, [1, 1, 1, 1]))

        print("Done")

curr_bs = []
for idxi, _ in enumerate(bad_sets):
    curr_bs = bad_sets[idxi]
    ell = Ellipse((curr_bs[0], curr_bs[1]), 2 *
                  curr_bs[2], 2 * curr_bs[3], color='r', alpha=0.3)
    ax.add_patch(ell)

goal_square = plt.Rectangle(goal_x-np.array([.1, .1]), .2, .2, color='g', alpha=0.5)
ax.add_patch(goal_square)

plt.xlim(0, 6)
plt.ylim(0, 6)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()
