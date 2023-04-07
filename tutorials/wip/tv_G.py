from sympy import symbols
import numpy as np
import matplotlib.pyplot as plt
import control as control
import cvxopt as cvxopt
from cbfkit.tutorial import cbf as cbf, cbf_utils, sys_and_ctrl
import matplotlib.animation as animation
from sympy import symbols, Matrix, sin, cos, lambdify, exp, sqrt, log, diff, Mul, srepr
from cbf_create import *

# Robot Goal
x_goal = np.array([10, 5])

# intial condition
x_0 = np.array([0, 0])

# Undesired areas in ellipse format (x,y,rad_x,rad_y) - Use example(0) through example(3)
bad_sets = [] #cbf_utils.example(1)

# Parameters for reference controller
ctrl_param = []

# Symbols and equations for the CBF
xr0, xr1, u, t, xr0_dot, xr1_dot, k = symbols(
    'xr0 xr1 u t xr0_dot xr1_dot, k')


pred = 10
h = pred - sqrt((xr0-x_goal[0])**2 + (xr1-x_goal[1])**2)
#B  = -5/15*t + 5 + h

# calculate norm between two points 
# B_exp =  11*exp(-0.4796*t) + 9 - sqrt((xr0-x_goal[0])**2 + (xr1-x_goal[1])**2)
# B = B_exp
B_exp =  k*exp(-0.8*t) + h
k0 = cbf_find_param(x_0,x_goal,h,B_exp,5,15)
B = B_exp.subs(k,k0)

# dx = g(x)u - not used
f = 0
g = Matrix([0, 0, 1.0])

# Initialize CBF
my_CBF = cbf.CBF(B, f, g, states=(xr0,xr1), states_dot=[xr0_dot,xr1_dot], symbs=(xr0,xr1), degree=1, time_varying = 1)

# Simulation settings
T_max = 15
n_samples = 200
T = np.linspace(0, T_max, n_samples)
dt = T[1]-T[0]
params = {'x_goal': x_goal, 'bad_sets': bad_sets,
          'ctrl_param': ctrl_param, 'CBF': my_CBF}

# Disable cvxopt optimiztaion output
cvxopt.solvers.options['show_progress'] = False

# Simulate system
print('\nComputing trajectories for the initial condition:')
print('x_0\t x_1')
print(x_0[0], '\t', x_0[1])

# If initial condition is inside the bad set, error out.
for idxj, j in enumerate(bad_sets):
    curr_bs = bad_sets[idxj]
    assert cbf_utils.is_inside_ellipse(
        [x_0[0], x_0[1]], bad_sets[idxj]) == 0, "Initial condition is inside ellipse"

# Compute output on the nimble ant system for given initial conditions and timesteps T
x = np.zeros((np.size(x_0), len(T)))
x[:, 0] = x_0

# Simulate system
for i in range(len(T)-1):
    x[:, i+1] = x[:, i] + dt * \
        np.array(sys_and_ctrl.nimble_ant_tv_f(T[i], x[:, i], [], params))
    # if B.subs([(t,T[i]),(xr0,x[0,i]),(xr1,x[1,i])]) > h.subs([(xr0,x[0,i]),(xr1,x[1,i])]) and T[i]>=5:
    #     print(T[i])
    #     print(float(B.subs([(t,T[i]),(xr0,x[0,i]),(xr1,x[1,i])])))
    #     print(float(h.subs([(xr0,x[0,i]),(xr1,x[1,i])])))
    #     print('\n')


print("\n*Simulation Done. See plot for animation...")

# Init Plot
fig1, ax1 = plt.subplots()

# Animate
ax1 = cbf_utils.plot_cbf_elements_circle(ax1, bad_sets, x_goal, 10, 'r')

plt.xlim(-2, 12)
plt.ylim(-2, 12)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

line1, = ax1.plot([], [], lw=2)
goal_square = plt.Rectangle(
    x_goal-np.array([1, 1]), 1, 1, color='g', alpha=0.5)
time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

robot_circ = plt.Circle(
        (x[0][-1], x[1][-1]), 0.15, color='r', fill=True, alpha=0.5)

def init():
    line1.set_data([], [])
    robot_circ.center = (x[0][0], x[1][0])
    ax1.add_patch(robot_circ)
    return line1, robot_circ


def animate(i):
    line1.set_data((x[0][0:i], x[1][0:i]))
    robot_circ.center = (x[0][i], x[1][i])
    time_text.set_text(i*dt)
    return line1, time_text, robot_circ

ani = animation.FuncAnimation(
    fig1, animate, init_func=init, interval=10, frames=n_samples, repeat=False)

fig2, ax2 = plt.subplots()

dist_to_goal = np.zeros(len(T))

# get distance to goal
for i in range(len(T)):
    dist_to_goal[i] = sqrt((x[0][i]-x_goal[0])**2 + (x[1][i]-x_goal[1])**2)

# Plot distance to goal
ax2.plot(T, dist_to_goal)
plt.ylim(-2, 20)
plt.xlim(0, 20)
plt.axvline(x=5, color='r')
plt.axvline(x=15, color='g')

plt.xlabel('Time (s)')
plt.ylabel('Distance to goal (m)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# find index of array closest to value
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

plt.axhline(y=dist_to_goal[find_nearest(T, 5)], color='y')
dist_text = ax2.text(0, 1.02, '', transform=ax2.transAxes)
dist_text.set_text('dist(||x - [10 5]^T||< 10, t = 5) = '+ "{:.2f}".format(dist_to_goal[find_nearest(T, 5)]))

dist_text_3 = ax2.text(0.2, 1.10, 'G_[5,15](||x - [10 5]^T||< 10)', transform=ax2.transAxes, fontsize=12)

# plot patch of halfspace x bellow 0
patch_x_below_0 = plt.Rectangle(
    (0, pred), 20, -20, color='r', alpha=0.2)
ax2.add_patch(patch_x_below_0)

plt.show()
print(x[0][-1], x[1][-1])
print((((x[0][-1]-x_goal[0])**2 + (x[1][-1]-x_goal[1])**2)**0.5))

print("\n*Animation Complete. Exiting...\n")
