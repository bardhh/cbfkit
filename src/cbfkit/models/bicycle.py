from sympy import Matrix, cos, sin
import unittest


def accel_unicycle(states, inputs):
    """This function defines acceleration unicycle model

    Args:
        states (list): list of system states
        inputs (list): list of system inputs

    Returns:
        f, g (symbolic expressions): to describe model of the system as dx = f+g*input
    """

    assert (
        len(states) == 4 and len(inputs) == 2
    ), f"Expected states to have length 4 and inputs to have length 2, but got {len(states)} and {len(inputs)}, respectively."

    f = Matrix([states[2] * cos(states[3]), states[2] * sin(states[3]), 0, 0])
    g = Matrix([[0, 0], [0, 0], [0, 1], [1, 0]])
    return f, g


def appr_unicycle(states, inputs, l):
    """This function defines approximate unicycle model

    Args:
        states (list): list of system states as [x, y, theta]
        inputs (list): list of system inputs as [v, w], where v is velocity and w is angular velocity
        l (float): distance between wheels

    Returns:
        (f, g) (tuple): to describe model of the system as dx = f+g*input
    """

    assert (
        states.shape[0] == 3 and inputs.shape[0] == 2
    ), "appr_unicycle model has 3 states and 2 inputs"

    g = Matrix(
        [
            [cos(states[2]), -l * sin(states[2])],
            [sin(states[2]), l * cos(states[2])],
            [0, 1],
        ]
    )

    return Matrix([0, 0, 0]), g
