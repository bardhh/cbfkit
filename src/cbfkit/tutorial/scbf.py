import numpy as np
from sympy import symbols, Matrix, sin, cos, lambdify, exp, sqrt, log, diff, Mul, srepr, Symbol


class tut_scbf:
    def __init__(self, B, f_r, g_r, f_o, g_o, f_r_states, f_r_inputs, f_o_states, bad_sets, symbs, alpha=1):
        """ This initializes the CBF and computes functions for the G and h matrices for convex optimization later on.
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

        self.f_r = f_r
        self.g_r = g_r
        self.f_r_states = f_r_states

        self.f_o = f_o
        self.g_o = g_o
        self.f_o_states = f_o_states
        self.f_r_inputs = f_r_inputs

        self.bad_sets = bad_sets
        self.symbs = symbs
        self.G = []                 # G matrix for CVXopt
        self.h = []                 # h matrix for CVXopt
        self.expr_bs = []           # symbolic expressions for bad sets
        self.lamb_G = []
        self.lamb_h = []

        self.alpha = alpha
        self.BFsym = B
        # self.B(*([self.f_r_states, self.f_o_states]))
        # function for computation of symbolic expression for G matrix

        CBF_d = self.BFsym.diff(Matrix([f_r_states, f_o_states]))
        CBF_d2 = self.BFsym.diff(f_o_states,2)

        a = 1
        b = 1

        self.lamb_G = lambdify([f_r_states,f_o_states,f_r_inputs], (CBF_d.T*Matrix([g_r*Matrix(f_r_inputs[0:2]), Matrix(np.zeros((len(f_o),1)))]))[0])
        self.lamb_h = lambdify([f_r_states,f_o_states], -a*self.BFsym + b - (CBF_d.T*Matrix([f_r,f_o])+0.5*(g_o.T*Matrix([[Matrix(CBF_d2[0,0]),Matrix(CBF_d2[1,0])]])*g_o))[0])

        # self.lamb_G = lambdify([f_r_states,f_o_states], (CBF_d.T*Matrix([f_r,f_o])+0.5*(g_o.T*Matrix([[Matrix(CBF_d2[0,0]),Matrix(CBF_d2[1,0])]])*g_o))[0])

    def compute_G_h(self, x, x_o=[], u=[0, 1], t=0):
        """ The method computes the G and h matrices for convex optimization given current state

        Args:
            x (numpy.ndarray): array with the current state of the system

        Returns:
            list: returns G matrix
            list: returns h matrix
        """
        self.G = []
        self.h = []

        if self.bad_sets == []:
            for lamb in self.lamb_G:
                tmp_g = lamb(tuple(x))
                self.G.append(-1*tmp_g)
            self.h.append(self.lamb_h(x))
        else:
            for idxi, _ in enumerate(self.bad_sets):
                curr_bs = self.bad_sets[idxi]
                tmp_g = []
                self.G.append([])
                tmp_g = self.lamb_G(x, curr_bs, u)
                self.G[idxi].append(-1*tmp_g)
                self.h.append(self.alpha * self.lamb_h(x, curr_bs))

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
            return self.get_expr(phi, f, g, states, states_dot, alpha, count=count+1)

    def decompose_G_h(self, expr, g, states_dot):
        G = []
        h = 0
        for arg in expr.args:
            if states_dot[2] in arg.free_symbols:
                G = - arg.subs(states_dot[2], 1)
            else:
                h = h + arg
        return G, h
