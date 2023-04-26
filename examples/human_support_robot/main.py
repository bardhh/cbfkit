""""module docstring here""" ""

import time
import numpy as np

# from sympy import Matrix
from src.models.bicycle import appr_unicycle
from src.models.human_agent import agent_stop_when_close, careless_agent
from src.system import DeterministicSystem, StochasticSystem, ConnectedSystem
from src.controller import Controller
from src.cbf import Cbf, ZeroingCbf, StochasticCbf, RiskAwareCbf, MapCbf, GoalLyap
from src.utils import strList2SympyMatrix
import rospy

from .specifications import h, b, clf, LENGTH, GOAL

# assume we have read the names of agents from ROS and stored them here
agent_names = ["HSR", "agent1", "agent2"]

# Unicycle Agent
ego_stochastic = True
states_str = ["xr_0", "xr_1", "xr_2"]
inputs_str = ["ur_0", "ur_1"]
states = strList2SympyMatrix(states_str)
inputs = strList2SympyMatrix(inputs_str)
input_range = np.array([[-0.3, 0.3], [-0.3, 0.3]])
f, g = appr_unicycle(states, inputs, LENGTH)
if ego_stochastic:
    ego_agent = StochasticSystem(
        agent_names[0], states, inputs, f, g, None, input_range, G=G, D=D
    )
else:
    ego_agent = DeterministicSystem(
        agent_names[0], states, inputs, f, g, None, input_range
    )
print(ego_agent.system_details())

# Human Agents
humans_stochastic = True
states_str = ["xo_0", "xo_1", "xo_2"]
inputs_str = ["uo_0", "uo_1", "uo_2"]
states = strList2SympyMatrix(states_str)
inputs = strList2SympyMatrix(inputs_str)
f = careless_agent(states, inputs)
g = None
if humans_stochastic:
    human_agent_1 = StochasticSystem(
        agent_names[1], states, inputs, f, input_range=None, G=G, D=D
    )
    human_agent_2 = StochasticSystem(
        agent_names[2], states, inputs, f, input_range=None, G=G, D=D
    )
else:
    human_agent_1 = DeterministicSystem(agent_names[1], states, inputs, f)
    human_agent_2 = DeterministicSystem(agent_names[2], states, inputs, f)
print(human_agent_1.system_details())
print(human_agent_2.system_details())
other_agents = [human_agent_1, human_agent_2]

# # define state and input names
# states_str = ["xr_0", "xr_1", "xr_2"]
# inputs_str = ["ur_0", "ur_1"]
# states = strList2SympyMatrix(states_str)
# inputs = strList2SympyMatrix(inputs_str)

# l = 0.05
# f, g = appr_unicycle(states, inputs, l)
# # C = Matrix([[1, 0, 0], [0, 1, 0]])
# input_range = np.array([[-0.3, 0.3], [-0.3, 0.3]])
# # ego_system = System('ego', states, inputs, f, g)
# ego_system = System("HSR", states, inputs, f, g, None, input_range)
# print(ego_system.system_details())

# # Define Agents
# states_str = ["xo_0", "xo_1", "xo_2"]
# inputs_str = ["uo_0", "uo_1", "uo_2"]

# states = strList2SympyMatrix(states_str)
# inputs = strList2SympyMatrix(inputs_str)
# f = careless_agent(states, inputs)
# g = None
# C = Matrix([[1, 0, 0, 0], [0, 1, 0, 0]])
# G = Matrix(np.eye(len(states)))
# D = Matrix(np.eye(2))
# # agent = Stochastic('agent',states, inputs, f, None, C, G = G , D= D)
# agent_system = System("agent1", states, inputs, f)
# # One agent instance is enough for all agents of the same type, if we have other types of agents,
# # we can create that, we may need to think about a way to assign agents to system
# print(agent_system.system_details())

# # agent2
# agent_system2 = System(agent_names[2], states, inputs, f)
# print(agent_system.system_details())

# Create CBF Objects
cbf_list = []
cbf_class = RiskAwareCbf
for agent in other_agents:
    new_cbf = cbf_class(h, b, ego_agent, agent)
    print(new_cbf.details())
    cbf_list.append(new_cbf)

# Environment Bounds
env_bounds = type("", (), {})()
env_bounds.y_min = -1.2
env_bounds.y_max = 1
corridor_map = MapCbf(env_bounds, ego_agent)

# Define goal-reaching object
goal_func = GoalLyap(GOAL, clf, ego_agent)

try:
    rospy.init_node("HSR")
    connected_HSR = ConnectedSystem(ego_agent, cbf_list)
    my_cbf_controller = Controller(
        connected_HSR, goal_func, corridor_map, 5, "reference_control"
    )
    # [sec] we can change controll priod with this parameter.
    control_period = 0.05
    time.sleep(1)
    rospy.Timer(rospy.Duration(control_period), my_cbf_controller.controller_callback)
    rospy.spin()
except rospy.ROSInterruptException:
    pass
