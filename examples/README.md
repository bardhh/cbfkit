# Examples

Ready-to-run scripts that demonstrate CBFKit using **pre-built systems** from `cbfkit.systems`. No code generation required — just install `cbfkit` and run.

## Systems

| Directory | Description |
|---|---|
| `unicycle/` | Unicycle robot navigation with CBF, MPPI, stochastic CBF, and risk-aware controllers |
| `single_integrator/` | Single integrator with various state estimation methods (EKF, UKF) |
| `differential_drive/` | Differential-drive obstacle avoidance and human-aware navigation |
| `fixed_wing/` | Fixed-wing aerial vehicle control |
| `pedestrian/` | Robot navigation among pedestrians |
| `van_der_pol/` | Van der Pol oscillator regulation |
| `adaptive_cvar_cbf/` | Adaptive CVaR-CBF for risk-aware control |
| `benchmarking/` | Parameter sweep and benchmarking utilities |
| `diagnostics/` | Diagnostic and debugging tools |

## Quick Start

```bash
# Unicycle reach-avoid with CBF safety filter
python examples/unicycle/reach_goal/unicycle_reach_avoid_cbf.py

# MPPI planner with CBF obstacle avoidance
python examples/unicycle/reach_goal/mppi_cbf.py

# Parameter sweep for tuning CBF controllers
python examples/benchmarking/parameter_sweep.py
```

## Looking to build your own system?

See the [`tutorials/`](../tutorials/README.md) directory for step-by-step guides using code generation.
