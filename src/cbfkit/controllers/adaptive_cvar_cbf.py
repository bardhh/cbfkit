from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np

try:
    import casadi as ca
except ImportError:
    ca = None
from jax import Array, random
from numpy.random import Generator

from cbfkit.utils.uncertainty import generate_uncertainty_pmf
from cbfkit.utils.user_types import (
    Control,
    ControllerCallable,
    ControllerData,
    Key,
    State,
    Time,
)


class AdaptiveCVaRBarrierSolver:
    """
    Internal solver class for Adaptive CVaR-CBF.
    Handles the CasADi NLP formulation and solving.
    """

    def __init__(
        self,
        dynamics_model: Dict[str, Any],  # A, B matrices, limits, etc.
        params: Dict[str, Any],
        obstacles: List[Any],
        noise_params: List[List[float]],
    ):
        if ca is None:
            raise ImportError(
                "CasADi is not installed. Please install it with `pip install cbfkit[casadi]`."
            )

        self.dt = dynamics_model["dt"]
        self.A = dynamics_model["A"]
        self.B = dynamics_model["B"]
        self.u_min = dynamics_model["u_min"]
        self.u_max = dynamics_model["u_max"]
        self.x_min = dynamics_model.get("x_min", -np.inf * np.ones((4, 1)))
        self.x_max = dynamics_model.get("x_max", np.inf * np.ones((4, 1)))
        self.radius = dynamics_model["radius"]
        self.m = self.B.shape[1]
        self.n = self.A.shape[0]

        self.htype = params.get("htype", "dist_cone")
        self.S = params.get("S", 15)
        self.beta_candidates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        self.gamma = params.get("gamma", 0.1)
        self.w_a = params.get("w_a", 1.0)

        self.obstacles = obstacles
        self.noise_params = noise_params

        self.prev_solu = None

        # PMF Caches (updated per step if needed, or assume static for now?
        # Original code updated PMF at init. But PMF depends on u and x.
        # Ideally we update PMF at every step based on current/nominal u?
        # Original code: updated at init. Wait, if PMF depends on u, it must be updated?
        # In original code `update_pmf` was called in `__init__` using `all_robots[i].u`.
        # If `u` changes, PMF changes. But in `solve_opt`, it uses `self.robot_wu_samples`.
        # The samples were fixed at init in the original implementation?
        # Let's check original code snippet if I can...
        # "self.update_pmf(obstacles, all_robots)" in __init__.
        # So it used the initial u? That seems like a simplification or bug in original.
        # I will re-generate PMF at every solve step using u_nom to be more correct/robust.

    def dynamics_uncertain(self, x, u, wu, wx):
        # x, u are casadi or numpy
        # wu, wx are numpy samples
        return self.A @ x + self.B @ (u + wu) + wx

    def dynamics(self, x, u):
        return self.A @ x + self.B @ u

    def solve(
        self,
        t: float,
        x: np.ndarray,
        u_nom: np.ndarray,
        rng: Optional[Generator] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solves the optimization problem.
        """
        # 1. Update PMF based on current state and nominal input
        # For self (robot)
        pmf, wu_samples, wx_samples = generate_uncertainty_pmf(
            u_nom, x, self.noise_params, self.S, rng=rng
        )

        # For obstacles (assuming they have their own noise params or similar)
        # Simplified: use same noise for obs for now, or extract from obs if available
        obs_pmfs = []
        obs_wu_samples = []
        obs_wx_samples = []
        for obs in self.obstacles:
            # obs.velocity_xy, obs.x_curr required
            # Assuming obs has .noise attribute
            obs_noise = getattr(obs, "noise", [[0.01] * 4, [0.01] * 4])
            o_pmf, o_wu, o_wx = generate_uncertainty_pmf(
                obs.velocity_xy, obs.x_curr, obs_noise, self.S, rng=rng
            )
            obs_pmfs.append(o_pmf)
            obs_wu_samples.append(o_wu)
            obs_wx_samples.append(o_wx)

        # 2. Iterate Beta
        x_curr = x.reshape(-1, 1)
        u_nom = u_nom.reshape(-1, 1)

        u_opt = None
        log_data = {}

        for beta_val in self.beta_candidates:
            self.beta = beta_val
            res, stat = self._solve_one_iter(
                x_curr, u_nom, pmf, wu_samples, wx_samples, obs_pmfs, obs_wu_samples, obs_wx_samples
            )

            if stat in ["Solve_Succeeded", "Solved_To_Acceptable_Level"]:
                sol = res["x"].full().flatten()
                u_opt = sol[0 : self.m]
                self.prev_solu = sol.reshape(-1, 1)

                # Store logs
                log_data = {
                    "beta": beta_val,
                    "cost": res["f"].full().flatten()[0],
                    "solver_status": stat,
                    "h_vals": [],  # Populate if needed
                }
                break
            else:
                log_data = {
                    "beta": beta_val,
                    "cost": np.nan,
                    "solver_status": stat,
                }

        if u_opt is None:
            # Fallback? Return nominal or zeros?
            u_opt = np.zeros(self.m)

        return u_opt, log_data

    def _solve_one_iter(self, x, u_nom, robot_pmf, robot_wu, robot_wx, obs_pmfs, obs_wu, obs_wx):
        u = ca.MX.sym("u", self.m, 1)
        n_zeta = len(self.obstacles)  # + len(all_robots) - 1 (assuming single robot for now)
        n_eta = n_zeta * self.S

        zeta = ca.MX.sym("zeta", n_zeta)
        eta = ca.MX.sym("eta", n_eta)

        constraints = []
        lbg = []
        ubg = []

        # Cost
        cost = ca.mtimes([(u - u_nom).T, self.w_a * np.eye(self.m), (u - u_nom)])

        zeta_idx = 0

        for iObs, obs in enumerate(self.obstacles):
            hsk1_list = []
            offset = iObs * self.S

            # Current H (barrier at k) - calculated deterministically?
            # Original code: h_k, d_h = agent_barrier_dt(x, x_k1_nom, ...)
            x_k1_nom = self.dynamics(x, u)
            pre_obs_pos_nom = obs.x_curr[:2] + obs.velocity_xy[:2] * self.dt
            pre_obs_state_nom = np.vstack((pre_obs_pos_nom, obs.velocity_xy[:2])).reshape(-1, 1)

            h_k, _ = self.agent_barrier_dt(
                x, x_k1_nom, pre_obs_state_nom, obs.radius + self.radius, self.htype
            )

            # Get samples for this obstacle
            obs_wu_arr = obs_wu[iObs]
            obs_wx_arr = obs_wx[iObs]

            for s in range(self.S):
                # Robot Uncertainty
                wu_s = robot_wu[s].reshape(-1, 1)
                wx_s = robot_wx[s].reshape(-1, 1)
                x_k1 = self.dynamics_uncertain(x, u, wu_s, wx_s)

                # Obs Uncertainty
                wu_s_obs = obs_wu_arr[s].reshape(-1, 1)
                wx_s_obs = obs_wx_arr[s].reshape(-1, 1)

                pre_obs_pos = (
                    obs.x_curr[:2] + (obs.velocity_xy[:2] + wu_s_obs) * self.dt + wx_s_obs[:2]
                )
                pre_obs_state = np.vstack((pre_obs_pos, obs.velocity_xy[:2])).reshape(-1, 1)

                # Barrier at k+1
                _, d_h = self.agent_barrier_dt(
                    x, x_k1, pre_obs_state, obs.radius + self.radius, self.htype
                )
                # h(x_{k+1}) = h(x_k) + d_h.
                # Wait, agent_barrier_dt returns h_k and d_h = h_k1 - h_k.
                # So h_k1 = h_k + d_h.

                # Note: h_k here should technically be h(x, obs_state_now). It doesn't depend on noise.
                # But d_h depends on x_k1 (noisy).
                # Using the h_k from nominal calculation above.

                hs_k1 = h_k + d_h
                hsk1_list.append(hs_k1)

            # CVaR Constraints
            # 1. -h_s - zeta - eta_s <= 0
            constraints.append(
                -ca.vertcat(*hsk1_list)
                - zeta[zeta_idx] * ca.DM.ones((self.S, 1))
                - eta[offset : offset + self.S]
            )
            lbg.extend([-ca.inf] * self.S)
            ubg.extend([0] * self.S)

            # 2. eta_s >= 0
            constraints.append(eta[offset : offset + self.S])
            lbg.extend([0] * self.S)
            ubg.extend([ca.inf] * self.S)

            # 3. CVaR Condition
            # psi1_k = -zeta - 1/beta * E[eta] + (gamma - 1)*h_k <= 0 (if using <= 0 form)
            # Original code used: psi1_k = -(zeta + ... ) + (-1 + gamma)*h_k
            # And constraints.append(psi1_k), lbg=0, ubg=inf.
            # So psi1_k >= 0.
            # => -(CVaR(h_next)) + (gamma-1)h_k >= 0
            # => CVaR(h_next) <= (gamma-1)h_k ... wait.
            # Usual form: h_next >= (1-gamma)h_curr.
            # Let's stick to original code logic:
            psi1_k = (
                -(
                    zeta[zeta_idx]
                    + (1 / self.beta) * ca.dot(obs_pmfs[iObs], eta[offset : offset + self.S])
                )
                + (-1 + self.gamma) * h_k
            )
            constraints.append(psi1_k)
            lbg.append(0)
            ubg.append(ca.inf)

            zeta_idx += 1

        # Input constraints
        constraints.append(u - self.u_min)
        lbg.extend([0] * self.m)
        ubg.extend([ca.inf] * self.m)

        constraints.append(self.u_max - u)
        lbg.extend([0] * self.m)
        ubg.extend([ca.inf] * self.m)

        # NLP
        opt_vars = ca.vertcat(u, zeta, eta)
        nlp = {"x": opt_vars, "f": cost, "g": ca.vertcat(*constraints)}
        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        # Initial guess
        n_vars = opt_vars.numel()
        x0 = np.zeros((n_vars, 1))
        if self.prev_solu is not None and self.prev_solu.shape[0] == n_vars:
            x0 = self.prev_solu
        else:
            x0[0 : self.m] = u_nom

        lbx = -ca.inf * np.ones((n_vars, 1))
        ubx = ca.inf * np.ones((n_vars, 1))

        return solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg), solver.stats()["return_status"]

    def agent_barrier_dt(self, x_k, x_k1, obs_state, radius, htype):
        # Re-implementation of barrier logic
        # obs_state: [px, py, vx, vy]

        if htype == "dist_cone":
            return self._h_dist_cone(x_k, x_k1, obs_state, radius)
        else:
            # Default to dist
            return self._h_dist(x_k, x_k1, obs_state, radius)

    def _h_dist(self, x_k, x_k1, obs_state, radius, beta=1.05):
        def compute(x):
            # dist^2 - r^2
            return (x[0] - obs_state[0]) ** 2 + (x[1] - obs_state[1]) ** 2 - beta * (radius**2)

        h_k = compute(x_k)
        h_k1 = compute(x_k1)
        return h_k, h_k1 - h_k

    def _h_dist_cone(self, x_k, x_k1, obs_state, radius, rate=10.0):
        def collision_cone(x, o):
            p_rel = o[0:2] - x[0:2]
            v_rel = o[2:4] - x[2:4]
            norm_p = ca.norm_2(p_rel) if isinstance(p_rel, ca.MX) else np.linalg.norm(p_rel)
            norm_v = ca.norm_2(v_rel) if isinstance(v_rel, ca.MX) else np.linalg.norm(v_rel)

            dot = 0.0
            if isinstance(norm_p, ca.MX):
                dot = ca.dot(p_rel, v_rel) / (norm_p * norm_v + 1e-6)
            else:
                if norm_p > 1e-5 and norm_v > 1e-5:
                    dot = np.dot(p_rel.flatten(), v_rel.flatten()) / (norm_p * norm_v)
            return dot

        def compute(x):
            # h_dist part
            h_dist = (x[0] - obs_state[0]) ** 2 + (x[1] - obs_state[1]) ** 2 - radius**2

            # h_vel part
            dot = collision_cone(x, obs_state)
            w = (
                ca.norm_2(obs_state[2:4])
                if isinstance(obs_state, ca.MX)
                else np.linalg.norm(obs_state[2:4])
            )
            # v_obs_est = 1.0 (hardcoded in original)
            w = w / 1.0

            if isinstance(dot, ca.MX):
                theta = (1.0 / rate) * ca.log(1.0 + ca.exp(-rate * dot))
            else:
                theta = (1.0 / rate) * np.log(1.0 + np.exp(-rate * dot))

            h_vel = (radius**2) * w * theta
            return h_dist - h_vel

        h_k = compute(x_k)
        h_k1 = compute(x_k1)
        return h_k, h_k1 - h_k


def adaptive_cvar_cbf_controller(
    dynamics_model: Dict[str, Any],
    obstacles: List[Any],
    params: Optional[Dict[str, Any]] = None,
    noise_params: Optional[List[List[float]]] = None,
) -> ControllerCallable:
    """
    Factory function for Adaptive CVaR-CBF Controller.

    Args:
        dynamics_model: Dictionary containing 'A', 'B', 'dt', 'u_min', 'u_max', 'radius'.
        obstacles: List of obstacle objects (must have .x_curr, .velocity_xy, .radius).
        params: Controller parameters (htype, S, beta, etc.).
        noise_params: Noise parameters for uncertainty generation.

    Returns:
        A cbfkit-compatible controller function.
    """
    if params is None:
        params = {}
    if noise_params is None:
        noise_params = [[0.01] * 4, [0.01] * 4]

    # Instantiate solver (holds state via closure)
    solver = AdaptiveCVaRBarrierSolver(dynamics_model, params, obstacles, noise_params)

    def controller(
        t: Time,
        x: State,
        u_nom: Optional[Control],
        key: Key,
        data: ControllerData,
    ) -> Tuple[Array, ControllerData]:
        # Check if we need to retrieve state from data?
        # For now, solver keeps its own state (prev_solu).
        # Ideally, we should load/save prev_solu from/to `data.sub_data`.

        if data.sub_data is not None and "prev_solu" in data.sub_data:
            solver.prev_solu = data.sub_data["prev_solu"]

        # Convert JAX arrays to numpy for CasADi
        x_np = np.array(x)
        u_nom_np = np.array(u_nom) if u_nom is not None else np.zeros(solver.m)

        # Seed random number generator
        seed = int(random.randint(key, (), 0, 2**31 - 1))
        rng = np.random.default_rng(seed)

        # Solve
        # CasADi expects float time
        t_float = float(t) if isinstance(t, (float, int)) else float(t[0]) if t.shape else 0.0

        u_opt, log_info = solver.solve(t_float, x_np, u_nom_np, rng=rng)

        # Return
        u_jax = jnp.array(u_opt).flatten()

        # Update data
        new_sub_data = data.sub_data.copy() if data.sub_data is not None else {}
        new_sub_data["prev_solu"] = solver.prev_solu
        new_sub_data.update(log_info)

        new_data = data._replace(u=u_jax, sub_data=new_sub_data)

        return u_jax, new_data

    return controller
