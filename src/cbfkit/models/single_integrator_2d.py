from sympy import Matrix, cos, sin
import unittest


def single_integrator(states, inputs):
    """This function defines a 2D single integrator model

    dxdt = vx
    dydt = vy

    Args:
        states (list): list of system states
        inputs (list): list of system inputs

    Returns:
        f, g (symbolic expressions): to describe model of the system as dx = f+g*input
    """

    assert (
        len(states) == 2 and len(inputs) == 2
    ), f"Expected states to have length 2 and inputs to have length 2, but got {len(states)} and {len(inputs)}, respectively."

    f = Matrix([0, 0])
    g = Matrix([[1, 0], [0, 1]])
    return f, g
