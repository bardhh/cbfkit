import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

# ROS others
import tf


class System:

    """System class object

    Args:
        name (string): name of the system
        states (sympy list): states of system
        inputs (sympy list): inputs of the system
        input_Range (List of tuples) : (min, max) input range for each input
        f (sympy expression): function f(x)
        g (sympy expression): function g(x)
        C (sympy expression): function C(x)

    Returns:
        system object: the model describes dxdt = f(x) + g(x)*inputs , y = Cx where x is the system states
    """

    def __init__(self, name, states, inputs, f, g=None, C=None, input_range=None):
        # TODO: Check the observability given C, the assert part may need more attention too
        self.name = name
        self.states = states
        self.inputs = inputs
        self.input_range = input_range
        self.f = f
        self.state_traj = []
        self.control_traj = []
        self.curr_state = []
        self.curr_inputs = []
        self.time = 0.0
        self.dt = 0.0
        self.nDim = len(states)
        self.full_observability = True

        if g is not None:
            self.g = g
            self.dx = self.f + self.g * self.inputs
        else:
            self.g = None
            self.dx = self.f

        #! Error checking
        assert self.f.shape == len(
            self.states
        ), f"Incorrect dimensions of drift term f: received {self.f.shape}, should be {(len(self.states),)}"
        assert self.g.shape == len(
            self.states
        ), f"Incorrect dimensions of control matrix term g: received {self.g.shape}, should be {(len(self.states),len(self.inputs))}"

        self.C = C
        if self.C is not None:
            if np.array(C).shape != np.eye(self.nDim).shape or not np.allclose(
                np.eye(self.nDim), C
            ):
                assert (
                    np.array(C).shape[1] == self.nDim
                ), "inappropriate C shape"  # y = CX
                self.full_observability = False
        else:
            self.C = np.eye(self.nDim)

    def add_state_traj(self, state, time):
        # record state trajectory
        self.dt = time - self.time
        self.time = time
        self.curr_state = state
        self.state_traj.append([time, state[:]])

    def add_control_traj(self, control, time):
        # record control trajectory
        self.curr_inputs = control
        self.control_traj.append([time, control[:]])

    def system_details(self):
        return (
            "Name: {}\n"
            + "States: {}\n"
            + "Inputs: {}\n"
            + "f: {}\n"
            + "g: {}\n"
            + "C: {}\n"
            + "Obs.: {}\n".format(
                self.name,
                self.states,
                self.inputs,
                self.f,
                self.g,
                self.C,
                self.full_observability,
            )
        )


class DeterministicSystem(System):
    """Deterministic system model object. Inherits from System.

    Args:
        System (_type_): _description_
    """

    def __init__(self, name, states, inputs, f, g=None, C=None, input_range=None):
        super().__init__(name, states, inputs, f, g, C)


class StochasticSystem(System):
    """Stochastic system model object. Inherits from System.

    Args:
        System (_type_): _description_
    """

    def __init__(
        self, name, states, inputs, f, g=None, C=None, input_range=None, G=None, D=None
    ):
        super().__init__(name, states, inputs, f, g, C, input_range)
        if G is None and D is None:
            raise ValueError("Did you mean to create a deterministic system?")

        self.G = G
        if self.G is not None:
            assert np.array(self.G).shape[0] == self.nDim, "inappropriate G shape"

        self.D = D
        if self.D is not None:
            assert np.array(self.D).shape[0] == self.C.shape[0]
            self.full_observability = False  #! MB: I don't know why this is here

    def system_details(self):
        superOut = super().system_details()
        out = superOut + "{}\n {}\n".format(self.D, self.G)
        return out


class ClosedLoopSystem(object):
    def __init__(self, ego_system, controller, ros_used=False):
        # Initialize the ClosedLoopSystem object with the given ego system and controller
        self.ego = ego_system
        self.controller = controller
        self.ros_used = ros_used

    def simulate(self, sim_time=10, dt=0.01):
        # Simulate the closed loop system for the given simulation time and time step
        if dt <= 0:
            raise ValueError("dt must be positive")
        if sim_time < 0:
            raise ValueError("sim_time must be non-negative")
        self.sim_time = sim_time
        self.dt = dt
        T = [
            t * dt for t in range(int(sim_time / dt) + 1)
        ]  # Create a list of time steps
        for t in T:
            self.step(self.dt)  # Take a simulation step with the given time step

    def step(self, control, dt=0.01):
        # Take a single simulation step with the given control input and time step
        self.time += dt  # Update the current time
        self.curr_state = (
            self.curr_state + dt * self.f + dt * self.g * control
        )  # Update the current state using the dynamics equations
        self.state_traj.append(
            [self.time, self.curr_state[:]]
        )  # Append the current time and state to the state trajectory
        self.control_traj.append(
            [self.time, control[:]]
        )  # Append the current time and control input to the control trajectory


class ConnectedSystem(object):
    def __init__(self, ego_system, cbf_list, ros_used=False):
        self.ego = ego_system
        self.cbf_list = cbf_list
        self.ros_used = ros_used
        self.vw_publisher = rospy.Publisher(
            "/hsrb/command_velocity", Twist, queue_size=10
        )

        # # subscliber to get odometry of HSR & agents
        #! TEMPORARY FIX!!!

        rospy.Subscriber(
            "/transformed_hsr", Odometry, self.tOdometry_callback, queue_size=10
        )

        # rospy.Subscriber('/hsrb/odom_ground_truth', Odometry,
        #                  self.tOdometry_callback, queue_size=10)

        # rospy.Subscriber('/global_pose', PoseStamped, odometry_callback, queue_size=10)

        # assume we have read the names of agents from ROS and stored them here
        self.i = 0
        for cbf in self.cbf_list:
            agentname = cbf.agent.name
            #! TEMPORARY FIX!!!

            rospy.Subscriber(
                "/transformed_agent",
                Odometry,
                self.agent_callback,
                callback_args=agentname,
                queue_size=10,
            )

            # rospy.Subscriber('/'+agentname+'pose', PoseStamped,
            #                  self.agent_callback, callback_args=agentname, queue_size=10)

    def tOdometry_callback(self, odometry):
        now = rospy.get_rostime()
        time = now.secs + now.nsecs * pow(10, -9)
        p = odometry.pose.pose.position
        # transfer orientaton(quaternion)->agular(euler)
        angular = orientation2angular(odometry.pose.pose.orientation)
        state = [p.x, p.y, angular.z]
        self.ego.add_state_traj(state, time)

    def odometry_callback(self, poseStamped):
        poseStamped = poseStamped

    def agent_callback(self, agentPose, agentname):
        now = rospy.get_rostime()
        time = now.secs + now.nsecs * pow(10, -9)
        p = agentPose.pose.pose.position
        # transfer orientaton(quaternion)->agular(euler)
        angular = orientation2angular(agentPose.pose.pose.orientation)
        state = [p.x, p.y, angular.z]
        for i in range(len(self.cbf_list)):
            if self.cbf_list[i].agent.name == agentname:
                self.cbf_list[i].agent.add_state_traj(state, time)

    def publish(self, u):
        now = rospy.get_rostime()
        time = now.secs + now.nsecs * pow(10, -9)
        vel_msg = Twist()
        vel_msg.linear.x = u[0]
        vel_msg.angular.z = u[1]
        self.ego.add_control_traj(u, time)
        self.vw_publisher.publish(vel_msg)


def orientation2angular(orientation):
    quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)
    angular = Vector3(euler[0], euler[1], euler[2])
    return angular
