# Examples

Ready-to-run scripts that demonstrate CBFKit using **pre-built systems** from `cbfkit.systems`. No code generation required — just install `cbfkit` and run.

## Recommended Order

| # | Directory | Description |
|---|---|---|
| 1 | `unicycle/` | Start here: CBF basics with reach-avoid, MPPI, stochastic and risk-aware controllers |
| 2 | `single_integrator/` | State estimation (EKF, UKF) with risk-aware CBF control |
| 3 | `differential_drive/` | Dynamic and static obstacle avoidance, barrier activation, human-aware navigation |
| 4 | `pedestrian/` | Multi-agent navigation among pedestrians |
| 5 | `fixed_wing/` | 3D fixed-wing aerial vehicle control with EKF |
| 6 | `van_der_pol/` | Oscillator regulation with CBF-CLF constraints |
| 7 | `adaptive_cvar_cbf/` | Adaptive CVaR-CBF for risk-aware control |
| 8 | `parameter_sweep/` | Parameter sweep and benchmarking utilities |

## Quick Start

```bash
# Unicycle reach-avoid with CBF safety filter
python examples/unicycle/reach_goal/unicycle_reach_avoid_cbf.py

# MPPI planner with CBF obstacle avoidance
python examples/unicycle/reach_goal/mppi_cbf.py

# Parameter sweep for tuning CBF controllers
python examples/parameter_sweep/parameter_sweep.py
```

## Looking to build your own system?

See the [`tutorials/`](../tutorials/README.md) directory for step-by-step guides using code generation.
