import numpy as np
import matplotlib.pyplot as plt
import control as control
import cvxopt as cvxopt


def silly_bug_c(x, params):
    # Control function
    goal_x = params['goal_x']
    bad_sets = params['bad_sets']

    # Control parameters
    k1 = 0.3
    k2 = 0.3

    # Reference controller
    uref_0 = k1 * ((goal_x[0]-x[0]))
    uref_1 = k2 * ((goal_x[1]-x[1]))

    # Disable cvxopt optimiztaion output
    cvxopt.solvers.options['show_progress'] = False

    # cvxopt quadratic program
    # minimize  0.5 x'Px + q'x
    # s.t       Gx<=h

    # P matrix
    P = cvxopt.matrix(np.eye(2))
    P = .5 * (P + P.T)  # symmetric

    # q matrix
    q = cvxopt.matrix(np.array([-uref_0, -uref_1]), (2, 1))

    # Parameters for the CBF
    G = []
    h = []

    # For each bad set, a separate CBF constraint should be added
    curr_bs = []
    for idxi, _ in enumerate(bad_sets):
        curr_bs = bad_sets[idxi]

        g1 = 2*x[0] - 2*curr_bs[0]
        g2 = 2*x[1] - 2*curr_bs[1]
        g3 = (x[0] - curr_bs[0])**2 + (x[1] - curr_bs[1])**2 - curr_bs[2]

        G.append([-g1, -g2])
        h.append([g3])

    # Convert lists to cvxopt.matrix object
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)

    # Run optimizer and return solution
    sol = cvxopt.solvers.qp(P, q, G.T, h.T, None, None)
    x_sol = sol['x']
    return x_sol[0:2]


def silly_bug_f(t, x, u, params):
    # a function for a silly bug

    # compute control given current position
    u_0 = silly_bug_c(x, params)

    # compute change in xy direction
    dx0 = u_0[0]
    dx1 = u_0[1]

    return [dx0, dx1]


# Robot Goal
goal_x = np.array([5, 5])
bad_sets = [[1., 2., 0.5], [4, 1, 0.5], [3, 3.5, 0.5], [4.5, 4.2, 0.5]]

# Simulation settings
T_max = 20
n_samples = 100
T = np.linspace(0, T_max, n_samples)

# Initial conditions
# min, max of x,y values for initial conditions
min_x, min_y, max_x, max_y = -1, -1, 3, 3
nx, ny = 4, 4              # number of initial conditions in each axis

# Vectors of initial conditions in each axis
xx = np.linspace(min_x, max_x, nx)
yy = np.linspace(min_y, max_y, ny)

# Uncomment the following for specific intial conditions
# xx = [0]
# yy = [1]

# System definition using the control toolbox
silly_bug_sys = control.NonlinearIOSystem(
    silly_bug_f, None, inputs=None, outputs=None, dt=None,
    states=('x0', 'x1'), name='silly_bug', params={'goal_x': goal_x, 'bad_sets': bad_sets})

# Plot
fig, ax = plt.subplots()

# Plot colors
jet = plt.get_cmap('jet')
colors = iter(jet(np.linspace(0, 1, len(xx)*len(yy))))

# Loop through initial conditions
for idxi, i in enumerate(xx):
    for idxk, k in enumerate(yy):
        # If initial condition is inside the bad set, skip it.
        bool_val = 0
        curr_bs = []
        for idxj, j in enumerate(bad_sets):
            curr_bs = bad_sets[idxj]
            if ((i-curr_bs[0])**2+(k-curr_bs[1])**2 <= curr_bs[2]):
                print('Skip (Invalid):\t', i, k)
                bool_val = 1
        if bool_val == 1:
            continue

        print('Computing:\t', i, k)
        x_0 = np.array([i, k])

        # Compute output on the silly bug system for given initial conditions and timesteps T
        t, y, x = control.input_output_response(sys=silly_bug_sys, T=T, U=0, X0=[
                                                i, k], return_x=True, method='BDF')

        # Plot initial conditions and path of system
        plt.plot(i, k, 'x-', markersize=5, color=[0, 0, 0, 1])
        plt.plot(x[0], x[1], 'o-', markersize=2,
                 color=next(colors, [1, 1, 1, 1]))

curr_bs = []
for idxi, _ in enumerate(bad_sets):
    curr_bs = bad_sets[idxi]
    circle = plt.Circle((curr_bs[0], curr_bs[1]), curr_bs[2], color='r')
    ax.add_patch(circle)

# Plot bad set
# circle1 = plt.Circle((3, 3), 1, color='r')
# ax.add_patch(circle1)

# Plot goal set
goal_square = plt.Rectangle(goal_x-np.array([.1, .1]), .2, .2, color='g')
ax.add_patch(goal_square)

plt.xlim(0, 6)
plt.ylim(0, 6)
plt.show()
