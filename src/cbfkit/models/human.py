from sympy import Matrix, exp


def careful_agent(states, inputs, radi, multi):
    """This function defines agent model with the assumption that the agent maintains its velocities
    in the x and y direction unless it is close to the ego when it slows down

    Args:
        states (Sympy matrix): vector of symbolic system states
        inputs (Sympy matrix): vector of symbolic system inputs

    Returns:
        f (symbolic expressions): to describe model of the system as dx = f
    """
    if states.shape[0] != 3 or inputs.shape[0] != 2:
        raise ValueError("agent_break model has 3 states and 3 inputs")

    c = multi
    dx = (states[0] - inputs[0]) * exp(c * (radi - (states[0] - inputs[0]) ** 2))
    dy = (states[1] - inputs[1]) * exp(c * (radi - (states[1] - inputs[1]) ** 2))
    dtheta = 1 / (1 + (dy / dx) ** 2)

    f = Matrix([dx, dy, dtheta])
    return f


def careless_agent(states, inputs):
    """This function defines agent model with the assumption that the agent maintains its velocities
    in the x and y direction unless it is close to the ego when it slows down

    Args:
        states (Sympy matrix): vector of symbolic system states
        inputs (Sympy matrix): vector of symbolic system inputs

    Returns:
        f (symbolic expressions): to describe model of the system as dx = f
    """
    if states.shape[0] != 3 or inputs.shape[0] != 3:
        raise ValueError("careless_agent model has 3 states and 3 inputs")

    dx = inputs[0]
    dy = inputs[1]
    dtheta = inputs[2]

    f = Matrix([dx, dy, dtheta])
    return f
