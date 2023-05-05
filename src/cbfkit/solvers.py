from cvxopt import matrix, solvers


def qp_solver(H, f, A, b, G=None, h=None):
    """
    Solve a quadratic program using the cvxopt solver.

    Args:
    H: quadratic cost matrix.
    f: linear cost vector.
    A: linear constraint matrix.
    b: linear constraint vector.
    G: quadratic constraint matrix.
    h: quadratic constraint vector.

    Returns:
    sol['x']: Solution to the QP
    """
    # Use the cvxopt library to solve the quadratic program
    P = matrix(H)
    q = matrix(f)
    A = matrix(A)
    b = matrix(b)
    options = {"show_progress": False}

    if G is None and h is None:
        sol = solvers.qp(P, q, A, b, options=options)
    else:
        G = matrix(G)
        h = matrix(h)
        sol = solvers.qp(P, q, G=G, h=h, A=A, b=b, options=options)

    return sol["x"]
