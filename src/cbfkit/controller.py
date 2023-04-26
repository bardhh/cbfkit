# Bardh Hoxha, Tom Yamaguchi


import math

import cvxopt as cvxopt
import matplotlib.pyplot as plt
import numpy as np
import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateRequest, GetWorldProperties

# def print_cli(x_min, x_max, value):
#     # calculate total length of CLI based on the range between x_min and x_max
#     cli_length = 20
#     value_range = x_max - x_min
#     cli_unit_length = value_range / cli_length

#     # calculate the position of the value within the CLI
#     value_pos = int((value - x_min) / cli_unit_length)

#     # create the CLI string
#     cli_string = '[' + '-' * cli_length + ']\n'
#     cli_string += '{:>{}}{}{}'.format('{:.2f}'.format(x_min), value_pos, '', '{:.2f}'.format(x_max)) + '\n'
#     cli_string += ' ' * (value_pos + 1) + '(^_^)\n'

#     # print the CLI string
#     print(cli_string)


def print_cli_range(x_min, x_max, val1, val2):
    # calculate the range between x_min and x_max
    r = x_max - x_min

    # calculate the position of val1 and val2 within the range
    pos1 = int((val1 - x_min) / r * 10)
    pos2 = int((val2 - x_min) / r * 10)

    # create the CLI bar with pointers for val1 and val2
    cli = ""
    for i in range(11):
        if i == pos1:
            cli += "{:.2f}".format(val1)
        elif i == pos2:
            cli += "{:.2f}".format(val2)
        else:
            cli += "-"
    cli += (
        "\n"
        + str(x_min)
        + " " * pos1
        + "^"
        + " " * (pos2 - pos1 - 1)
        + "^"
        + " " * (10 - pos2)
        + str(x_max)
    )

    print(cli)


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    # function to solve the quadratic program
    P = 0.5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    cvxopt.solvers.options["show_progress"] = False
    cvxopt.solvers.options["maxiters"] = 1000
    sol = cvxopt.solvers.qp(*args)
    if "optimal" not in sol["status"]:
        return None
    return np.array(sol["x"]).reshape((P.shape[1],))


class ReferenceControl:
    """Object for reference (desired) control."""

    def __init__(self, goal_info):
        self.goal_info = goal_info

    def ref_calculate(self, x_r):
        x_c = self.goal_info.goal_center
        dy = x_r[1] - x_c[1]
        dx = x_r[0] - x_c[0]
        theta_ref = np.arctan2(dy, dx)
        u_ref = [0.3, 0]
        if dx > 0 and dy >= 0:
            if x_r[2] < 0:
                u_ref[1] = -3.142 - (x_r[2] - theta_ref)
            else:
                u_ref[1] = +3.142 - (x_r[2] - theta_ref)
        elif dx <= 0 and dy > 0:
            # Checked
            u_ref[1] = theta_ref - (3.142 + x_r[2])
        elif dx < 0 and dy < 0:
            # Checked
            u_ref[1] = theta_ref + 3.142 - x_r[2]
        else:
            # Checked
            if x_r[2] < 0:
                u_ref[1] = -3.142 - (x_r[2] - theta_ref)
            else:
                u_ref[1] = +3.142 - (x_r[2] - theta_ref)

        # print([math.degrees((x_r[2])),math.degrees(theta_ref),math.degrees(u_ref[1])])
        return u_ref


class Controller:
    """Controller class which take the connected system (ego, CBF_list, ROS) and combine it with map_cbf to generate controller as a qp

    Args:
        connected_system (Connected_system): (ego, CBF_list, ROS)
        goal_info ([type]): [description]
        map_info
        include_radius (int): horizon for CBF consideration

    Returns:
        [type]: [description]
    """

    # make sure ego is the system with which CBF is created
    def __init__(
        self,
        system,
        goal_info,
        map_info,
        cbf_list,
        include_radius=30,
        method="lyapunov",
    ):
        self.system = system
        self.goal_info = goal_info
        self.map_info = map_info
        self.include_radius = include_radius
        self.count = 0  # num of times control_callback is called
        # assert method is in ["lyapunov", "reference_control"], "method should be either lyapunov or reference_control"
        if method == "reference_control":
            try:
                reference_control(goal_info)
            except:
                raise ValueError("reference computation function required")
            self.control_func = self.cbf_controller_compute_reference
            ref_control = reference_control(goal_info)
            self.ref_func = ref_control.ref_calculate
        else:
            self.control_func = self.cbf_controller_compute_lyap

    def controller_callback(self, event):
        # this controller loop call back.
        self.count += 1

        u = self.control_func()

        self.system.publish(u)

        if self.goal_info.set(self.system.ego.curr_state) < 0:
            rospy.loginfo("reached Goal set!!")
            rospy.signal_shutdown("reached goal_info h")

    def cbf_controller_compute_reference(self):
        ego = self.system.ego
        cbf_list = self.system.cbf_list
        x_r = ego.curr_state
        u_s = ego.inputs

        cbf_list = cbf_list
        goal_info = self.goal_info
        my_map = self.map_info

        unsafe_list = []
        dists = []
        for CBF in cbf_list:
            dist = CBF.constraint.h(x_r, CBF.agent.curr_state)
            print(dist)
            dists.append(dist)
            if dist < self.include_radius:
                unsafe_list.append(CBF)

        u_ref = self.ref_func(x_r)

        num_qp_var = len(u_s) + len(unsafe_list) + len(my_map.constraint.h)
        num_constraints = 2 * len(u_s) + len(unsafe_list) + len(my_map.constraint.h)

        A = np.zeros((num_constraints, num_qp_var))
        b = np.zeros(num_constraints)

        for j in range(len(unsafe_list)):
            # CBF Constraints
            x_o = unsafe_list[j].agent.curr_state
            try:
                dx = np.array(unsafe_list[j].agent.state_traj[-5:-1][-1][1]) - np.array(
                    unsafe_list[j].agent.state_traj[-5:-1][0][1]
                )
                dt = (
                    unsafe_list[j].agent.state_traj[-5:-1][-1][0]
                    - unsafe_list[j].agent.state_traj[-5:-1][0][0]
                )
                mean_inp = dx / dt
            except:
                mean_inp = [0, 0, 0]

            A[j, np.arange(len(u_s))] = unsafe_list[j].constraint.LHS(x_r, x_o)[0]
            A[j, len(u_s) + j] = -1
            b[j] = unsafe_list[j].constraint.RHS(x_r, x_o, mean_inp)

        # Adding U constraint
        A[len(unsafe_list), 0] = 1
        b[len(unsafe_list)] = ego.input_range[0, 1]
        A[len(unsafe_list) + 1, 0] = -1
        b[len(unsafe_list) + 1] = -ego.input_range[0, 0]
        A[len(unsafe_list) + 2, 1] = 1
        b[len(unsafe_list) + 2] = ego.input_range[1, 1]
        A[len(unsafe_list) + 3, 1] = -1
        b[len(unsafe_list) + 3] = -ego.input_range[1, 0]

        # Adding map constraints
        for j in range(len(my_map.constraint.h)):
            A[len(unsafe_list) + 2 * len(u_s) + j, np.arange(len(u_s))] = my_map.constraint.LHS[j](
                x_r
            )[0]
            A[len(unsafe_list) + 2 * len(u_s) + j, len(u_s) + len(unsafe_list) + j] = -1
            b[len(unsafe_list) + 2 * len(u_s) + j] = my_map.constraint.RHS[j](x_r)

        H = np.zeros((num_qp_var, num_qp_var))
        H[0, 0] = 50
        H[1, 1] = 1

        ff = np.zeros((num_qp_var, 1))
        for j in range(len(unsafe_list)):
            # ff[len(u_s)+j] = 3      # To reward not using the slack variables when not required
            if unsafe_list[j].constraint.type is "exp":
                H[len(u_s) + j, len(u_s) + j] = 20  # Use for exponential barrier functions
            else:
                H[
                    len(u_s) + j, len(u_s) + j
                ] = 50  # To reward not using the slack variables when not required

        for j in range(len(my_map.constraint.h)):
            # ff[len(u_s)+len(unsafe_list)+j] = 2
            H[len(u_s) + len(unsafe_list) + j, len(u_s) + len(unsafe_list) + j] = 4
        ff[0] = -100 * u_ref[0]
        ff[1] = -2 * u_ref[1]

        try:
            uq = cvxopt_solve_qp(H, ff, A, b)
            print("{:.2f}".format(uq))
        except ValueError:
            uq = [0, 0]
            rospy.loginfo("Domain Error in cvx")

        if uq is None:
            uq = [0, 0]
            rospy.loginfo("infeasible QP")
        return uq

    def cbf_controller_compute_lyap(self):
        ego = self.system.ego
        cbf_list = self.system.cbf_list
        x_r = ego.curr_state
        u_s = ego.inputs

        cbf_list = cbf_list
        goal_info = self.goal_info
        my_map = self.map_info

        unsafe_list = []
        dists = []
        for CBF in cbf_list:
            dist = CBF.constraint.h(x_r, CBF.agent.curr_state)
            dists.append(dist)
            if dist < self.include_radius:
                unsafe_list.append(CBF)
        if min(dists) < 0:
            in_unsafe = 1
        else:
            in_unsafe = 0
        mindist = min(dists)
        minJ = np.where(dists == mindist)

        num_qp_var = len(u_s) + len(unsafe_list) + len(my_map.constraint.h) + 1
        numConstraints = 2 * len(u_s) + len(unsafe_list) + len(my_map.constraint.h) + 2

        A = np.zeros((numConstraints, num_qp_var))
        b = np.zeros(numConstraints)

        for j in range(len(unsafe_list)):
            # CBF Constraints
            x_o = unsafe_list[j].agent.curr_state
            try:
                dx = np.array(unsafe_list[j].agent.state_traj[-5:-1][-1][1]) - np.array(
                    unsafe_list[j].agent.state_traj[-5:-1][0][1]
                )
                dt = (
                    unsafe_list[j].agent.state_traj[-5:-1][-1][0]
                    - unsafe_list[j].agent.state_traj[-5:-1][0][0]
                )
                mean_inp = dx / dt
            except:
                mean_inp = [0, 0, 0]

            A[j, np.arange(len(u_s))] = unsafe_list[j].constraint.LHS(x_r, x_o)[0]
            A[j, len(u_s) + j] = -1
            b[j] = unsafe_list[j].constraint.RHS(x_r, x_o, mean_inp)

        #! These are hard coded
        # Adding U constraint
        A[len(unsafe_list), 0] = 1
        b[len(unsafe_list)] = ego.input_range[0, 1]
        A[len(unsafe_list) + 1, 0] = -1
        b[len(unsafe_list) + 1] = -ego.input_range[0, 0]
        A[len(unsafe_list) + 2, 1] = 1
        b[len(unsafe_list) + 2] = ego.input_range[1, 1]
        A[len(unsafe_list) + 3, 1] = -1
        b[len(unsafe_list) + 3] = -ego.input_range[1, 0]

        # Adding map constraints
        for j in range(len(my_map.constraint.h)):
            A[len(unsafe_list) + 2 * len(u_s) + j, np.arange(len(u_s))] = my_map.constraint.LHS[j](
                x_r
            )[0]
            A[len(unsafe_list) + 2 * len(u_s) + j, len(u_s) + len(unsafe_list) + j] = -1
            b[len(unsafe_list) + 2 * len(u_s) + j] = my_map.constraint.RHS[j](x_r)

        #! Appears to be hard-coded
        # Adding goal_info based Lyapunov !!!!!!!!!!!!!!!!! Needs to be changed for a different example
        A[len(unsafe_list) + 2 * len(u_s) + len(my_map.constraint.h), 0:2] = [
            goal_info.Lyap(x_r, [1, 0]),
            goal_info.Lyap(x_r, [0, 1]),
        ]
        A[len(unsafe_list) + 2 * len(u_s) + len(my_map.constraint.h), -1] = -1
        b[len(unsafe_list) + 2 * len(u_s) + len(my_map.constraint.h)] = 0
        A[len(unsafe_list) + 2 * len(u_s) + len(my_map.constraint.h) + 1, -1] = 1
        b[len(unsafe_list) + 2 * len(u_s) + len(my_map.constraint.h) + 1] = np.finfo(float).eps + 1

        H = np.zeros((num_qp_var, num_qp_var))
        H[0, 0] = 0.1
        H[1, 1] = 0.1

        ff = np.zeros((num_qp_var, 1))
        for j in range(len(unsafe_list)):
            # ff[len(u_s)+j] = 3      # To reward not using the slack variables when not required
            if unsafe_list[j].constraint.type is "exp":
                H[len(u_s) + j, len(u_s) + j] = 20  # Use for exponential barrier functions
            else:
                H[
                    len(u_s) + j, len(u_s) + j
                ] = 50  # To reward not using the slack variables when not required

        for j in range(len(my_map.constraint.h)):
            # ff[len(u_s)+len(unsafe_list)+j] = 2
            H[len(u_s) + len(unsafe_list) + j, len(u_s) + len(unsafe_list) + j] = 15
        # ff[-1] = np.ceil(self.count/100.0)
        ff[-1] = 1

        try:
            uq = cvxopt_solve_qp(H, ff, A, b)
            print_cli_range(-0.3, 0.3, uq[0], uq[1])
            # print(["{:.2f}".format(uq[0]),"{:.2f}".format(uq[1])])
        except ValueError:
            uq = [0, 0]
            rospy.loginfo("Domain Error in cvx")

        if uq is None:
            uq = [0, 0]
            rospy.loginfo("infeasible QP")
        return uq
