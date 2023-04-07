import numpy as np
import math
import datetime
from sympy import symbols, Matrix
from sympy.solvers import solve
from sympy.polys.polytools import count_roots
import platform

# import conditional if system is mac m1
if platform.system() == "Darwin" and platform.machine() == "arm64":
    from kvxopt import matrix, solvers
else:
    from cvxopt import matrix, solvers


def nimble_ant_c(x, params):
    """Controller for nimble ant

    Args:
        t (float): [description]
        x (numpy.ndarray): [description]
        u (numpy.ndarray): the input to the controller is the state of the system x
        params (dict): Dict keys:
                        goal_x: the goal or target state
                        bad_sets: list of elippses defining bad sets
                        ctrl_param: parameters for the controller
                        CBF: the CBF object
    Returns:
        cvxopt.base.matrix: the control for the system
    """
    x_goal = params["x_goal"]
    ctrl_param = params["ctrl_param"]
    my_CBF = params["CBF"]

    # Reference controller
    #! note that u, the input to the controller, is the state of the system
    # u_ref = ctrl_param * ((x_goal-x[0:2]))

    k = symbols("k")

    k_param = float(solve(k * max((x_goal - x[0:2])) - ctrl_param, k)[0])

    u_ref = k_param * ((x_goal - x[0:2]))

    if np.all((x == 0)):
        return u_ref

    ############################
    # cvxopt quadratic program
    # minimize  0.5 x'Px + q'x
    # s.t       Gx<=h
    ############################

    # P matrix
    P = matrix(np.eye(2))
    P = 0.5 * (P + P.T)  # symmetric

    # q matrix
    q = matrix(-1 * np.array(u_ref), (2, 1))

    G, h = my_CBF.compute_G_h(x)

    G = matrix(G)
    h = matrix(h)

    # Run optimizer and return solution
    sol = solvers.qp(P, q, G.T, h, None, None)
    x_sol = sol["x"]
    return x_sol[0:2]


def nimble_ant_f(t, x, u, params):
    """Function for the nimble_ant system

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

    x_goal = params["x_goal"]
    if (x[0] - x_goal[0]) ** 2 + (x[1] - x_goal[1]) ** 2 <= 0.1**2:
        return [0, 0]

    # compute control given current position
    u = nimble_ant_c(x, params)

    # dynamics
    dx0 = u[0]
    dx1 = u[1]

    return [dx0, dx1]


def nimble_ant_with_agent_c(x, params):
    """Controller for nimble ant with agent

    Args:
        t (float): [description]
        x (numpy.ndarray): [description]
        u (numpy.ndarray): the input to the controller is the state of the system x
        params (dict): Dict keys:
                        goal_x: the goal or target state
                        bad_sets: list of elippses defining bad sets
                        ctrl_param: parameters for the controller
                        CBF: the CBF object
    Returns:
        cvxopt.base.matrix: the control for the system
    """

    x_goal = params["x_goal"]
    ctrl_param = params["ctrl_param"]
    my_CBF = params["CBF"]

    # Reference controller
    #! note that u, the input to the controller, is the state of the system

    k = symbols("k")
    k_param = float(solve(k * max((x_goal - x[0:2])) - ctrl_param, k)[0])
    u_ref = k_param * ((x_goal - x[0:2]))

    if np.all((x == 0)):
        return u_ref

    ############################
    # cvxopt quadratic program
    # minimize  0.5 x'Px + q'x
    # s.t       Gx<=h
    ############################

    # P matrix
    P = matrix(np.eye(4))
    P = 0.5 * (P + P.T)  # symmetric

    # q matrix

    #! TEMPORARY - FIX NEEDED
    # q = matrix(-1 * np.array(u_ref), (2, 1))
    u_ref = np.append(u_ref, [0, 0], axis=0)
    q = matrix(-1 * np.array(u_ref), (4, 1))

    nrObs = int((x.size - 2) / 2)
    G = np.empty((nrObs, 4))
    h = np.empty(nrObs)
    for i in range(0, nrObs):
        tempG, tempH = my_CBF.compute_G_h(x[np.r_[0:2, 2 * i + 2 : 2 * i + 4]])
        G[i] = tempG[0]
        h[i] = tempH[0]

    G = matrix(G)
    h = matrix(h)

    # Disable cvxopt optimiztaion output
    # solvers.options['show_progress'] = False
    # solvers.options['max_iter'] = 5000

    # Run optimizer and return solution
    sol = solvers.qp(P, q, G, h, None, None)
    x_sol = sol["x"]
    return x_sol[0:2]


def nimble_ant_with_agent_f(t, x, u, params):
    """Function for nimble ant with agent that moves to the left

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

    # x_goal = params['x_goal']
    # if (x[0] - x_goal[0])**2 + (x[1] - x_goal[1])**2 <= 0.1**2:
    #     return [0, 0]

    # compute control given current position
    x_goal = params["x_goal"]
    if np.linalg.norm(x[0:2] - x_goal) <= 0.1:
        return np.zeros(x.size)

    u = nimble_ant_with_agent_c(x, params)

    # dynamics
    dx = [u[0], u[1]]
    for i in range(2, x.size):
        dx.append(params["CBF"].f[i])

    return dx


def unicycle_f(t, x, u, params):
    # Function for a silly bug
    # if goal reached, do nothing
    goal_x = params["goal_x"]
    if np.linalg.norm(x[0:2] - goal_x) <= 0.1:
        return [0, 0, 0]

    # compute control given current position
    u_0 = unicycle_c(x, params)

    # compute change in xy direction
    dx0 = math.cos(x[2])
    dx1 = math.sin(x[2])
    dx2 = u_0[0]

    return [dx0, dx1, dx2]


def unicycle_c(x, params):
    # Controller for nimble car
    goal_x = params["goal_x"]
    bad_sets = params["bad_sets"]
    ctrl_param = params["ctrl_param"]
    myCBF = params["myCBF"]

    # Reference controller
    theta_ref = math.atan((goal_x[1] - x[1]) / (goal_x[0] - x[0]))
    uref_0 = ctrl_param[0] * (theta_ref - x[2])

    # math.atan2(sin(theta_ref-x[2]), cos(theta_ref-x[2]))

    ############################
    # cvxopt quadratic program
    # minimize  0.5 x'Px + q'x
    # s.t       Gx<=h
    ############################
    # P matrix
    P = matrix(np.eye(1))
    # P = .5 * (P + P.T)  # symmetric

    # q matrix
    q = matrix(np.array([-1 * uref_0]), (1, 1))

    G, h = myCBF.compute_G_h(x)

    G = matrix(G)
    h = matrix(h)

    # Run optimizer and return solution
    # sol = solvers.qp(P, q, G.T, h, None, None)
    try:
        sol = solvers.qp(P, q, G.T, h, None, None)
        x_sol = sol["x"]
    except:
        x_sol = [0]
        print("no sol")
    # print(x, ' G: ', G, ' h: ', h, ' x_sol: ', x_sol)
    return x_sol[0:1]


def unicycle_agent_f(t, x, u, params):
    # Function for a silly bug
    # if goal reached, do nothing
    goal_x = params["goal_x"]
    if np.linalg.norm(x[0:2] - goal_x) <= 0.1:
        return [0, 0, 0]

    # compute control given current position
    u_0 = unicycle_c(x, params)

    # compute change in xy direction
    dx0 = math.cos(x[2])
    dx1 = math.sin(x[2])
    dx2 = u_0[0]
    dx3 = -1
    dx4 = 0

    return [dx0, dx1, dx2, dx3, dx4]


def unicycle_agent_c(x, params):
    # Controller for nimble car
    goal_x = params["goal_x"]
    bad_sets = params["bad_sets"]
    ctrl_param = params["ctrl_param"]
    myCBF = params["myCBF"]

    # Reference controller
    theta_ref = math.atan((goal_x[1] - x[1]) / (goal_x[0] - x[0]))
    uref_0 = ctrl_param[0] * (theta_ref - x[2])

    # math.atan2(sin(theta_ref-x[2]), cos(theta_ref-x[2]))

    ############################
    # cvxopt quadratic program
    # minimize  0.5 x'Px + q'x
    # s.t       Gx<=h
    ############################
    # P matrix
    P = matrix(np.eye(3))
    # P = .5 * (P + P.T)  # symmetric

    # q matrix
    uref_0 = np.append(uref_0, [0, 0], axis=0)
    q = matrix(np.array([-1 * uref_0]), (3, 1))

    G, h = myCBF.compute_G_h(x)

    G = matrix(G)
    h = matrix(h)

    # Run optimizer and return solution
    # sol = solvers.qp(P, q, G.T, h, None, None)
    try:
        sol = solvers.qp(P, q, G.T, h, None, None)
        x_sol = sol["x"]
    except:
        x_sol = [0]
        print(["No sol" + datetime.datetime.now().time()])
    # print(x, ' G: ', G, ' h: ', h, ' x_sol: ', x_sol)
    return x_sol[0:1]


# def unicycle_c(x, params):
#     # Controller for nimble car
#     goal_x = params['goal_x']
#     bad_sets = params['bad_sets']
#     ctrl_param = params['ctrl_param']
#     myCBF = params['myCBF']

#     # Reference controller
#     theta_ref = math.atan((goal_x[1]-x[1])/(goal_x[0]-x[0]))
#     uref_0 = ctrl_param[0] * (theta_ref - x[2])

#     ############################
#     # cvxopt quadratic program
#     # minimize  0.5 x'Px + q'x
#     # s.t       Gx<=h
#     ############################

#     # P matrix
#     P = matrix(np.eye(1))
#     # P = .5 * (P + P.T)  # symmetric

#     # q matrix
#     q = matrix(np.array([-1*uref_0]), (1, 1))

#     G, h = myCBF.compute_G_h(x)

#     G = matrix(G)
#     h = matrix(h)

#     # Run optimizer and return solution
#     # sol = solvers.qp(P, q, G.T, h, None, None)
#     try:
#         sol = solvers.qp(P, q, G.T, h, None, None)
#         x_sol = sol['x']
#     except:
#         x_sol = [0]
#         print("QP iteration fail. Trying again...")
#     # print(x, ' G: ', G, ' h: ', h, ' x_sol: ', x_sol)
#     return x_sol[0:1]


# def unicycle_f(t, x, u, params):
#     # Function for a silly bug
#     # if goal reached, do nothing
#     goal_x = params['goal_x']
#     if (x[0] - goal_x[0])**2 + (x[1] - goal_x[1])**2 <= 0.1**2:
#         return [0, 0, 0]

#     # compute control given current position
#     u_0 = unicycle_c(x, params)

#     # compute change in xy direction
#     dx0 = math.cos(x[2])
#     dx1 = math.sin(x[2])
#     dx2 = u_0[0]

#     return [dx0, dx1, dx2]


################ TV Stuff


def nimble_ant_tv_c(x, t, params):
    """Controller for nimble ant

    Args:
        t (float): [description]
        x (numpy.ndarray): [description]
        u (numpy.ndarray): the input to the controller is the state of the system x
        params (dict): Dict keys:
                        goal_x: the goal or target state
                        bad_sets: list of elippses defining bad sets
                        ctrl_param: parameters for the controller
                        CBF: the CBF object
    Returns:
        cvxopt.base.matrix: the control for the system
    """
    x_goal = params["x_goal"]
    ctrl_param = params["ctrl_param"]
    my_CBF = params["CBF"]

    # Reference controller
    #! note that u, the input to the controller, is the state of the system

    ############################
    # cvxopt quadratic program
    # minimize  0.5 x'Px + q'x
    # s.t       Gx<=h
    ############################

    # P matrix
    P = matrix(np.eye(2))
    # P = .5 * (P + P.T)  # symmetric

    # q matrix
    q = matrix(np.array([0.0, 0]), (2, 1))

    # F_[5,15](||x-x_g||)<=5)
    # h(x) = 5 - ||x-x_g||
    # b(x,t) = y(t) -  ||x-x_g||
    # y(t) = -5/15*t + 10
    # t = symbols('t')
    # my_CBF.B = my_CBF.B_TV.subs('t', time)

    # G, h = my_CBF.compute_G_h(x)
    G, h = my_CBF.compute_G_h(x, t)

    G = matrix(G)
    h = matrix(h)
    print("G: ", G, " h; ", h)
    # Run optimizer and return solution
    sol = solvers.qp(P, q, G.T, h, None, None)
    x_sol = sol["x"]
    print("sol: ", x_sol)

    return x_sol[0:2]


def nimble_ant_tv_f(t, x, u, params):
    """Function for the nimble_ant system

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

    x_goal = params["x_goal"]
    if (x[0] - x_goal[0]) ** 2 + (x[1] - x_goal[1]) ** 2 <= 0.1**2:
        return [0, 0]

    # compute control given current position
    u = nimble_ant_tv_c(x, t, params)

    # dynamics
    dx0 = u[0]
    dx1 = u[1]

    return [dx0, dx1]


def approx_unicycle_agent_f(t, x, u, params):
    # Function for a silly bug
    # if goal reached, do nothing
    x_goal = params["x_goal"]
    if np.linalg.norm(x[0:2] - x_goal) <= 0.1:
        return [0, 0, 0]

    # compute control given current position
    u = approx_unicycle_agent_c(x, params)

    l = 0.1

    g = Matrix(
        [[np.cos(x[2]), -l * np.sin(x[2])], [np.sin(x[2]), l * np.cos(x[2])], [0, 1]]
    )

    dx = g * Matrix([1, u[0]])

    # compute change in xy direction
    # dx0 = math.cos(x[2])
    # dx1 = math.sin(x[2])
    # dx2 = u_0[0]

    return dx


def approx_unicycle_agent_c(x, params):
    # Controller for nimble car
    x_goal = params["x_goal"]
    bad_sets = params["bad_sets"]
    ctrl_param = params["ctrl_param"]
    my_CBF = params["CBF"]

    # Reference controller
    theta_ref = math.atan((x_goal[1] - x[1]) / (x_goal[0] - x[0]))
    uref_1 = ctrl_param[0] * (theta_ref - x[2])

    # math.atan2(sin(theta_ref-x[2]), cos(theta_ref-x[2]))

    ############################
    # cvxopt quadratic program
    # minimize  0.5 x'Px + q'x
    # s.t       Gx<=h
    ############################
    # P matrix
    P = matrix(np.eye(1))
    # P = .5 * (P + P.T)  # symmetric

    # q matrix
    q = matrix(-1 * uref_1)
    # q = matrix(np.array([0,-1*uref_1]), (2, 1))

    G, h = my_CBF.compute_G_h(x)

    G = matrix(G)
    h = matrix(h)

    # Run optimizer and return solution
    # sol = solvers.qp(P, q, G.T, h, None, None)
    try:
        sol = solvers.qp(P, q, G.T, h, None, None)
        x_sol = sol["x"]
    except:
        x_sol = [0.1, 0]
        print(["No sol" + str(datetime.datetime.now().time())])
    # print(x, ' G: ', G, ' h: ', h, ' x_sol: ', x_sol)
    return x_sol[0:2]


def simple_agent_left_f(t, x, u, params):
    # Controller for nimble car
    return [-1, 0]
