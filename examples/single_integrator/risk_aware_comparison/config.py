# examples/single_integrator/risk_aware_comparison/config.py
"""Constants for the ACC 2026 Fig. 1 risk-aware CBF comparison (2D single integrator)."""

# --- From Fig. 1 / paper section II.2 ---
SIGMA = 0.1  # isotropic diffusion scale
R_C = 1.0  # circular keep-in radius; safe set {||x|| < R_C}
RHO_D = 0.30  # design risk bound
V_MAX = 0.4  # nominal outward "task-drive" speed
N_TRIALS = 1000  # Monte-Carlo rollouts

# --- Derived (Eq. 6): eta = sup ||dB/dx . sigma|| = 2*sigma/R_c on the safe set ---
ETA = 2.0 * SIGMA / R_C  # = 0.2

# --- Assumptions (not pinned by Fig. 1; reasonable defaults, overridable) ---
T = 5.0  # horizon (s); reaches boundary in ~R_c/V_MAX = 2.5 s
DT = 0.05  # timestep (s)  => N_STEPS = 100
X0 = (0.1, 0.0)  # initial condition; reframed barrier h(x0) = 0.99 (B(x0) = ||x0||^2/R_c^2 = 0.01)
ACTUATION_LIMIT = 1.0  # per-axis control bound for the safety filter

# --- S-CBF baseline parameterization (paper does not specify; conservative by design) ---
SCBF_ALPHA = 1.0
SCBF_BETA = 1.0  # stochastic robustness margin; tuned so S-CBF is the safest controller (Fig. 1)

N_STEPS = int(T / DT)
