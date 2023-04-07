from sympy import exp
import time
import numpy as np
from cbfkit.models.bicycle import appr_unicycle
from cbfkit.models.human import careless_agent
from cbfkit.system import *
from cbfkit.controller import *
from cbfkit.cbf import *
from cbfkit.utils import strList2SympyMatrix

# check if rospy is installed, if not, raise error
try:
    import rospy
except ImportError:
    raise ImportError("rospy is not installed. Plaase install ROS first")


# from src.models.bicycle import appr_unicycle
# from src.models.human import careless_agent
# from src.system import *
# from src.controller import *
# from src.cbf import *
# from src.utils import strList2SympyMatrix
# import rospy


if __name__ == "__main__":
    """This is main"""
    # assume we have read the names of agents from ROS and stored them here

    agent_names = ["agent", "agent1"]

    states = strList2SympyMatrix(["xr_0", "xr_1", "xr_2"])
    inputs = strList2SympyMatrix(["ur_0", "ur_1"])

    l = 0.05
    f, g = appr_unicycle(states, inputs, l)
    # C = Matrix([[1, 0, 0], [0, 1, 0]])
    input_range = np.array([[-0.3, 0.3], [-0.3, 0.3]])
    # ego_system = System('ego', states, inputs, f, g)
    ego_system = System("HSR", states, inputs, f, g, None, input_range)
    print(f"{ego_system.system_details()}\n")

    # AGENTS #

    states = strList2SympyMatrix(["xo_0", "xo_1", "xo_2"])
    inputs = strList2SympyMatrix(["uo_0", "uo_1", "uo_2"])
    f = careless_agent(states, inputs)
    g = None

    # C = Matrix([[1, 0, 0, 0], [0, 1, 0, 0]])
    # G = Matrix(np.eye(len(states)))
    # D = Matrix(np.eye(2))
    # agent = Stochastic('agent',states, inputs, f, None, C, G = G , D= D)

    agent_1 = System("agent1", states, inputs, f)
    print(f"{agent_1.system_details()}\n")

    # agent2
    agent_2 = System("agent2", states, inputs, f)
    print(f"{agent_1.system_details()}\n")

    unsafe_radius = 1
    # Define h such that h(x)<=0 defines unsafe region
    def h(x, y, unsafe_radius):
        return (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 - (unsafe_radius + l) ** 2

    def h1(x, y):
        return h(x, y, unsafe_radius)

    # B initially negative, so Bdot<= -aB

    exp_var = True

    if exp_var:

        def B(x, y):
            return exp(-h(x, y, unsafe_radius)) - 1

        B_type = "exp"
    else:

        def B(x, y):
            return -h(x, y, unsafe_radius)

        B_type = "lin"

    cbf_1 = CBF(h1, B, B_type, ego_system, agent_1)
    print(f"{cbf_1.details()}\n")

    cbf_2 = CBF(h1, B, B_type, ego_system, agent_2)
    print(f"{cbf_2.details()}\n")

    # Enviroment Bounds
    env_bounds = type("", (), {})()
    env_bounds.y_min = -1.2
    env_bounds.y_max = 1
    corridor_map = Map_CBF(env_bounds, ego_system)

    # Goal set description
    goal_center = np.array([0, 0])
    r_goal = np.power(0.5, 2)

    def goal_set_func(x):
        return (x[0] - goal_center[0]) ** 2 + (x[1] - goal_center[1]) ** 2 - r_goal

    goal_func = Goal_Lyap(goal_center, goal_set_func, ego_system)

    rospy.init_node("HSR")
    cbf_list = [cbf_1, cbf_2]
    connected_HSR = ConnectedSystem(ego_system, cbf_list)
    my_cbf_controller = Controller(
        connected_HSR, goal_func, corridor_map, 5, "reference_control"
    )
    # [sec] we can change controll priod with this parameter.
    control_period = 0.05
    time.sleep(1)
    rospy.Timer(rospy.Duration(control_period), my_cbf_controller.controller_callback)
    rospy.spin()
