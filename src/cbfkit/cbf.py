# Bardh Hoxha, Tom Yamaguchi
from typing import Callable

import numpy as np
from sympy import Matrix, cos, diff, exp, lambdify, log, sin, sqrt, symbols


class ConstraintFunction(object):
    """ConstraintFunction class for defining constraint to be enforced via CBF

    Args:
        h (list): lambdafied expression #! Verify This
        B (list): lambdafied expression #! Verify This
        LHS (list): lambdafied expression #! Verify This
        RHS (list): lambdafied expression #! Verify This
    """

    def __init__(self, h=[], B=[], inputs=[], convention="superlevel"):
        self.h = h
        self.B = B
        self.inputs = inputs
        self.LHS = []
        self.RHS = []

        # Convention (Sublevel vs. Superlevel)
        self.convention = convention


class Cbf:
    """Parent class for CBF objects -- not meant to stand on its own.

    Args:
        h (list): lambdafied expression #! Verify This
        B (list): lambdafied expression #! Verify This
        ego (system): ego system
        agent (system): agent system
    """

    def __init__(self, h, B, ego_system, agent_system):
        self.states = [ego_system.states, agent_system.states]
        self.constraint = ConstraintFunction(h, B, ego_system.inputs)
        self.compute_lhs_rhs(ego_system, agent_system)
        self.ego_system = ego_system
        self.agent_system = agent_system
        self.integrated_barrier_value = 0
        self._lfb = None
        self._lgb = None

        self.compute_integrator_update()

    def compute_integrator_update(self) -> None:
        """Calls child method to compute the update rule for the integrated barrier value.

        Returns:
            None
        """
        self._compute_integrator_update()

    def compute_lhs_rhs(self, ego, agent):
        """
        Computes "B_dot <= -alpha*B(x)" if B<=0 is safe
        LHS*ego.inputs <= RHS

        Args:
            ego <class System>
            agent <class System>
        """
        # call child method
        return self._compute_lhs_rhs(ego, agent)

        # # Would need to account for another agents' actions to guarantee safety
        # negative_expr = (cf_d.T * Matrix([ego.f + ego.g * self.constraint.inputs, agent.f]))[0] + alpha * cf_sym
        # input_absent = 1
        # while input_absent:
        #     #TODO: Put a warning for mixed relative degrees, and work on a function that checks for the relative degree
        #     for input in self.constraint.inputs:
        #         if input not in negative_expr.free_symbols:
        #             cf_d = cf_d.diff(Matrix([ego.states, agent.states]))
        #             negative_expr = (cf_d.T *  Matrix([ego.f+ ego.g * self.constraint.inputs, agent.f]))[0]+alpha*cf_sym
        #             input_absent = 1
        #             break
        #         else:
        #             input_absent = 0

        # BF_d2 =  self.constraint.diff(self.x_o_s,2)
        # UnsafeInfo.CBF = lambdify([ego.states,self.x_o_s], CBF)

    def details(self):
        return "{}\n {}\n {}\n".format(
            self.constraint.h(*self.states),
            self.constraint.B(*self.states),
            self.states,
        )

    def update_integrated_barrier_value(self) -> None:
        """Updates integrated barrier value.

        Arguments:
            u: control input array

        Returns:
            None

        """
        lfb = self._lfb(
            self.ego_system.curr_state,
            self.agent_system.curr_state,
            self.agent_system.curr_inputs,
        )
        lgb = self._lgb(self.ego_system.curr_state, self.agent_system.curr_state)
        dB = (lfb + lgb * self.ego_system.curr_inputs) * self.ego_system.dt

        self.integrated_barrier_value += dB


class ZeroingCbf(Cbf):
    """Class for Zeroing CBFs

    Args:
        h (list): lambdafied expression #! Verify This
        B (list): lambdafied expression #! Verify This
        ego (system): ego system
        agent (system): agent system
    """

    def __init__(self, h, B, ego_system, agent_system):
        super().__init__(h, B, ego_system, agent_system)
        self.alpha = 1  # Linear class K function

    def _compute_integrator_update(self, ego, agent):
        """
        To Do.
        """
        # Extract symbolic constraint function and its derivative
        cf_sym = self.constraint.B(*self.states)
        cf_d = cf_sym.diff(Matrix([ego.states, agent.states]))

        self._lfb = lambdify(
            [ego.states, agent.states, agent.inputs],
            (cf_d.T * Matrix([ego.f, agent.f]))[0],
        )
        self._lgb = lambdify([ego.states, agent.states], Matrix(cf_d[: ego.nDim]).T * ego.g)

    def _compute_lhs_rhs(self, ego, agent):
        """
        Computes "B_dot <= -alpha*B(x)" if B<=0 is safe
        LHS*ego.inputs <= RHS

        Args:
            ego <class System>
            agent <class System>
        """
        # Extract symbolic constraint function and its derivative
        cf_sym = self.constraint.B(*self.states)
        cf_d = cf_sym.diff(Matrix([ego.states, agent.states]))

        #! TO DO: account for actions of other agents
        self.constraint.RHS = lambdify(
            [ego.states, agent.states, agent.inputs],
            -self.alpha * cf_sym - (cf_d.T * Matrix([ego.f, agent.f]))[0],
        )
        self.constraint.LHS = lambdify(
            [ego.states, agent.states], (Matrix(cf_d[: ego.nDim]).T * ego.g)
        )


class StochasticCbf(Cbf):
    """Class for Stochastic CBFs

    Args:
        h (list): lambdafied expression #! Verify This
        B (list): lambdafied expression #! Verify This
        ego (system): ego system
        agent (system): agent system
    """

    def __init__(self, h, B, ego_system, agent_system):
        super().__init__(h, B, ego_system, agent_system)
        self.alpha = 1  # Parameter multiplying constraint value
        self.beta = 0.1  # Independent additive parameter

    def _compute_integrator_update(self, ego, agent):
        """
        To Do.
        """
        # Extract symbolic constraint function and its derivative
        cf_sym = self.constraint.B(*self.states)
        cf_d = cf_sym.diff(Matrix([ego.states, agent.states]))

        self._lfb = lambdify(
            [ego.states, agent.states, agent.inputs],
            (cf_d.T * Matrix([ego.f, agent.f]))[0],
        )
        self._lgb = lambdify([ego.states, agent.states], Matrix(cf_d[: ego.nDim]).T * ego.g)

    def _compute_lhs_rhs(self, ego, agent):
        """
        Computes "B_dot <= -alpha*B(x)" if B<=0 is safe
        LHS*ego.inputs <= RHS

        Args:
            ego <class System>
            agent <class System>
        """
        # Extract symbolic constraint function and its derivative
        cf_sym = self.constraint.B(*self.states)
        cf_d = cf_sym.diff(Matrix([ego.states, agent.states]))
        cf_2d = cf_d.diff(Matrix([ego.states, agent.states]))

        #! TO DOs:
        # account for actions of other agents
        # create sigma for agent dynamics
        self.constraint.RHS = lambdify(
            [ego.states, agent.states, agent.inputs],
            -self.beta
            + self.alpha * cf_sym
            + (cf_d.T * Matrix([ego.f, agent.f]))[0]
            + 0.5
            * np.matrix.trace(
                Matrix([ego.sigma, agent.sigma]).T @ cf_2d @ Matrix([ego.sigma, agent.sigma])
            ),
        )
        self.constraint.LHS = lambdify(
            [ego.states, agent.states], (-Matrix(cf_d[: ego.nDim]).T * ego.g)
        )


class RiskAwareCbf(Cbf):
    """Class for Risk-Aware CBFs

    Args:
        h (list): lambdafied expression #! Verify This
        B (list): lambdafied expression #! Verify This
        ego (system): ego system
        agent (system): agent system
    """

    def __init__(self, h, B, ego_system, agent_system):
        super().__init__(h, B, ego_system, agent_system)

        self.alpha = 1  # Linear class K parameter

        self._compute_integrator_update(ego_system, agent_system)

    def _compute_integrator_update(self, ego, agent):
        """
        To Do.
        """
        # Extract symbolic constraint function and its derivative
        cf_sym = self.constraint.B(*self.states)
        cf_d = cf_sym.diff(Matrix([ego.states, agent.states]))

        self._lfb = lambdify(
            [ego.states, agent.states, agent.inputs],
            (cf_d.T * Matrix([ego.f, agent.f]))[0],
        )
        self._lgb = lambdify([ego.states, agent.states], Matrix(cf_d[: ego.nDim]).T * ego.g)

    def _compute_lhs_rhs(self, ego, agent):
        """
        Computes "B_dot <= -alpha*B(x)" if B<=0 is safe
        LHS*ego.inputs <= RHS

        Args:
            ego <class System>
            agent <class System>
        """
        # Extract symbolic constraint function and its derivative
        cf_sym = self.constraint.B(*self.states)
        cf_d = cf_sym.diff(Matrix([ego.states, agent.states]))

        #! TO DOs:
        # account for actions of other agents
        self.constraint.RHS = lambdify(
            [ego.states, agent.states, agent.inputs],
            -self.alpha * self.integrated_barrier_value + (cf_d.T * Matrix([ego.f, agent.f]))[0],
        )
        self.constraint.LHS = lambdify(
            [ego.states, agent.states], (-Matrix(cf_d[: ego.nDim]).T * ego.g)
        )


class MapCbf(object):
    def __init__(self, env_bounds, ego):
        # TODO: add checks on the passed argument
        """
        Computes "B_dot <= -alpha*B(x)" if B<=0 is safe
        LHS*ego.inputs <= RHS

        Args:
            env_bounds: object that has either or all of these attributes {'x_min','x_max','y_min','y_max',''}
            ego <class System>
        """

        self.states = ego.states
        self.constraint = ConstraintFunction()
        self.constraint.inputs = ego.inputs
        alpha = 6

        for attr in ["x_min", "x_max", "y_min", "y_max"]:
            if hasattr(env_bounds, attr):
                if attr == "x_min":
                    h = -(-ego.states[0] + getattr(env_bounds, attr))
                elif attr == "x_max":
                    h = -(ego.states[0] - env_bounds.x_max)
                elif attr == "y_min":
                    h = -(-ego.states[1] + env_bounds.y_min)
                elif attr == "y_max":
                    h = -(ego.states[1] - env_bounds.y_max)
                CBF = -h
                BF_d = CBF.diff(Matrix([ego.states]))
                self.constraint.h.append(lambdify([ego.states], h))
                self.constraint.B.append(lambdify([ego.states], CBF))
                self.constraint.RHS.append(
                    lambdify([ego.states], -alpha * CBF - (BF_d.T * ego.f)[0])
                )
                self.constraint.LHS.append(lambdify([ego.states], (BF_d.T * ego.g)))

    def add_map_cbf():
        return sympyMatrix


class GoalLyap(object):
    def __init__(self, goal_center, goal_set_func, ego):
        self.set = goal_set_func
        self.goal_center = goal_center
        GoalSym = goal_set_func(ego.states)
        self.Lyap = lambdify([ego.states, ego.inputs], GoalSym.diff(ego.states).T * ego.dx)
