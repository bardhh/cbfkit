"""Generates the constraint functions for the human support robot (HSR) example.
"""
from numpy import array
from sympy import exp

# Constraint function parameters
UNSAFE_RADIUS = 1
LENGTH = 0.05

# Goal set description
GOAL = array([0, 0])
R_GOAL = 0.5

# State Constraint(s)
def h(x, y) -> float:
    """Barrier-like constraint defining set of safe states as zero super-level set of this function.

    Args:
        x (_type_): ego agent state
        y (_type_): other agent state

    Returns:
        float: constraint function value
    """
    return (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 - (UNSAFE_RADIUS + LENGTH) ** 2


def b(x, y):
        return exp(-h(x, y))


def clf(x) -> float:
    """Control Lyapunov function encoding a goal specification.

    Args:
        x (_type_): ego agent state

    Returns:
        float: CLF value
    """
    return (x[0] - GOAL[0]) ** 2 + (x[1] - GOAL[1]) ** 2 - R_GOAL**2