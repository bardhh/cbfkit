from matplotlib.patches import Ellipse, Rectangle
import matplotlib.pyplot as plt
import numpy as np

class Obstacle:
    def __init__(self, xo_0, xo_1, a, b, type):
        """ Obstacle as an ellipse
        Args: where z_0, z_1 represent the center and a, b represents the major, minor axes of the ellipse
            xo_0, xo_1:   center of ellipse
            a, b:   major and minor axes
            dynamic: equal to 1 if obstacle is dynamic
        """
        self.xo_0 = xo_0
        self.xo_1 = xo_1
        self.a = a
        self.b = b
        self.type = type.lower()

def example(i):
    # Examples of different bad sets as ellipses
    # Defined as [z_0 ,z_1 , a, b] where z_0, z_1 represent the center and a, b represents the major, minor axes of the ellipse
    switcher = {
        0: [[3., 3., 1., 1]],
        1: [[1., 2., 0.5, 0.5], [4., 1., 0.5, 0.5],
            [3., 2., 0.5, 0.5], [4.5, 4.2, 0.5, 0.5]],
        2: [[3.5, 1., 0.2, 2.], [2., 2.5, 1., 0.2], [1.5, 1., 0.5, 0.5]],
        3: [[3.5, 3., 0.2, 2.], [2., 2.5, 1., 0.2], [1.5, 1., 0.5, 0.5]],
        4: [[10, 3.5, 3, 2]],
        5: [[10, 3.5, 5, 4]],
        6: [[1., 2., 0.5, 0.5], [4., 1., 0.5, 0.5],
            [3., 2., 0.5, 0.5], [4.5, 4.2, 0.5, 0.5],
            [4., 7., 0.5, 1], [3, 5, 1, 0.5],
            [8., 8., 0.5, 0.5], [10, 8.8, 0.5, 0.5],
            [10.8, 10., 1, 0.5], [13, 12, 2, 1],
            [16., 17., 3, 1]],
        # Random location on obstacles
        7:  np.concatenate(( np.random.uniform(low=2, high=18, size=(20,2)) , np.full((20, 2), 1)),axis=1)
    }
    return switcher.get(i, "Invalid")


def is_inside_ellipse(x, ell_params):
    # Check if state is inside ellipse
    if ((x[0] - ell_params[0])/ell_params[2])**2 + ((x[1] - ell_params[1])/ell_params[3])**2 <= 1:
        return 1
    else:
        return 0


def plot_cbf_elements(ax, bad_sets, goal_x):
    # Plot the bad sets and the goal region
    for idxi, _ in enumerate(bad_sets):
        curr_bs = bad_sets[idxi]
        ell = Ellipse((curr_bs[0], curr_bs[1]), 2 *
                      curr_bs[2], 2 * curr_bs[3], color='r', alpha=0.3)
        ax.add_patch(ell)

    goal_square = plt.Rectangle(
        goal_x-np.array([.1, .1]), .2, .2, color='g', alpha=0.5)

    ax.add_patch(goal_square)

    return ax

def plot_cbf_elements_rectangle(ax, bad_sets, goal_x):
    # Plot the bad sets and the goal region
    for idxi, _ in enumerate(bad_sets):
        curr_bs = bad_sets[idxi]
        ell = Rectangle((curr_bs[0]-curr_bs[2], curr_bs[1]-curr_bs[3]), 2 *
                      curr_bs[2], 2 * curr_bs[3], color='g', alpha=0.3)
        ax.add_patch(ell)

    goal_square = plt.Rectangle(
        goal_x-np.array([.1, .1]), .2, .2, color='g', alpha=0.5)

    ax.add_patch(goal_square)

    return ax

def plot_cbf_elements_circle(ax, bad_sets, goal_x, rad, c):
    # Plot the bad sets and the goal region
    for idxi, _ in enumerate(bad_sets):
        curr_bs = bad_sets[idxi]
        ell = Rectangle((curr_bs[0]-curr_bs[2], curr_bs[1]-curr_bs[3]), 2 *
                      curr_bs[2], 2 * curr_bs[3], color='g', alpha=0.3)
        ax.add_patch(ell)

    goal_square = plt.Circle(
        goal_x-np.array([.1, .1]), rad, color=c, alpha=0.2)
    ax.add_patch(goal_square)

    return ax
