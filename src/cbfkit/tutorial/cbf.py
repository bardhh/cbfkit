import numpy as np
from sympy import (
    symbols,
    Matrix,
    sin,
    cos,
    lambdify,
    exp,
    sqrt,
    log,
    diff,
    Mul,
    srepr,
    Symbol,
)


class CBF:
    def __init__(
        self,
        B,
        f,
        g,
        states,
        symbs,
        degree,
        bad_sets=[],
        states_dot=[],
        time_varying=0,
        alpha=1,
    ):
        """This initializes the CBF and computes functions for the G and h matrices for convex optimization later on.
        Args:
            B (sympy expression):   The expression for the bad set representation
            f (sympy expression):   The expression for the f(x) function
            g (sympy expression):   The expression for the g(x) function
            states (tuple):         A tuple with the states of the system
            bad_sets (list):        A list of bad sets, with each row consisting of z_0,z_1,a,b
                                    where z_0,z_1 represent the center and a,b represents the major,
                                    minor axes of the ellipse
        """
        self.B = B
        self.B_TV = B
        self.phi = []
        self.f = f
        self.g = g
        self.states = states
        self.bad_sets = bad_sets
        self.symbs = symbs
        self.G = []  # G matrix for CVXopt
        self.h = []  # h matrix for CVXopt
        self.expr_bs = []  # symbolic expressions for bad sets
        self.lamb_G = []
        # function for computation of symbolic expression for G matrix
        self.degree = degree
        self.time_varying = time_varying
        self.alpha = alpha

        if self.degree == 1:
            if self.time_varying == 1:
                t = Symbol("t")
                self.lamb_h = lambdify([symbs + (t,)], B, "math")
                for i in self.states:
                    temp_expr = diff(B, i)
                    self.expr_bs.append(temp_expr)
                    self.lamb_G.append(lambdify([symbs + (t,)], temp_expr, "math"))
            else:
                self.lamb_h = lambdify([symbs], B, "math")
                for i in self.states:
                    temp_expr = diff(B, i)
                    self.expr_bs.append(temp_expr)
                    self.lamb_G.append(lambdify([symbs], temp_expr, "math"))
        elif self.degree == 2:
            expr = self.get_expr(B, f, g, states, states_dot, alpha, 0)

            G, h = self.decompose_G_h(expr, g, states_dot)

            self.lamb_G.append(
                # lambdify((cx, cy, rad_x, rad_y, xr0, xr1, xr2), G, "math"))
                lambdify([symbs], G, "math")
            )
            self.lamb_h = lambdify(
                # (cx, cy, rad_x, rad_y, xr0, xr1, xr2), h, "math")
                [symbs],
                h,
                "math",
            )
        else:
            raise ValueError("degree > 2 not implemented yet")

    def compute_G_h(self, x, t=0):
        """The method computes the G and h matrices for convex optimization given current state

        Args:
            x (numpy.ndarray): array with the current state of the system

        Returns:
            list: returns G matrix
            list: returns h matrix
        """
        self.G = []
        self.h = []

        if self.degree == 1:
            if self.time_varying == 1:
                for lamb in self.lamb_G:
                    tmp_g = lamb(tuple(x) + (t,))
                    self.G.append(-1 * tmp_g)
                self.h.append((10 * self.lamb_h(tuple(x) + (t,))))  #
            else:
                if self.bad_sets == []:
                    for lamb in self.lamb_G:
                        tmp_g = lamb(tuple(x))
                        self.G.append(-1 * tmp_g)
                    self.h.append(self.lamb_h(x))
                else:
                    for idxi, _ in enumerate(self.bad_sets):
                        curr_bs = self.bad_sets[idxi]
                        tmp_g = []
                        self.G.append([])
                        for lamb in self.lamb_G:
                            tmp_g = lamb(tuple(np.hstack((x, curr_bs))))
                            self.G[idxi].append(-1 * tmp_g)
                        self.h.append(
                            self.alpha * self.lamb_h(tuple(np.hstack((x, curr_bs))))
                        )
        elif self.degree == 2:
            # for each bad set, given current state, compute the G and h matrices
            for idxi, _ in enumerate(self.bad_sets):
                curr_bs = self.bad_sets[idxi]
                tmp_g = []
                self.G.append([])
                for lamb in self.lamb_G:
                    tmp_g = lamb(tuple(np.hstack((curr_bs, x))))
                    self.G[idxi].append(tmp_g)
                self.h.append(self.lamb_h(tuple(np.hstack((curr_bs, x)))))
        else:
            raise ValueError("degree > 2 not implemented yet")
        return self.G, self.h

    def get_expr(self, B, f, g, states, states_dot, alpha, count):

        B_dot_var = []
        for i in states:
            B_dot_var.append(diff(B, i))
        B_dot = Matrix(B_dot_var)

        B_dot_f = B_dot.T * states_dot
        phi = B_dot_f[0] + alpha[count] * B
        self.phi.append(phi)
        if states_dot[2] in phi.free_symbols:  # ! This needs to be revised
            return phi
        else:
            return self.get_expr(phi, f, g, states, states_dot, alpha, count=count + 1)

    def decompose_G_h(self, expr, g, states_dot):
        G = []
        h = 0
        for arg in expr.args:
            if states_dot[2] in arg.free_symbols:
                G = -arg.subs(states_dot[2], 1)
            else:
                h = h + arg
        return G, h
