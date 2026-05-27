# Risk-Aware CBF Comparison (ACC 2026, Fig. 1)

Reproduces Figure 1 of Hoxha et al., *"Bayesian Risk-Aware CBFs for Discrete-Time
Stochastic Systems with Learned Dynamics"* (ACC 2026): state-occupancy heatmaps and
empirical failure probability `p_fail` for four controllers on a 2D stochastic single
integrator (`dx = u dt + sigma dW`) inside a circular keep-in set `{||x|| < R_c}`. An
outward "task-drive" nominal pushes the robot toward the boundary; each safety filter
keeps it (probabilistically) inside.

## Run
```bash
python examples/single_integrator/risk_aware_comparison/run_comparison.py   # 1000 rollouts + heatmaps
CBFKIT_TEST_MODE=1 python examples/single_integrator/risk_aware_comparison/run_comparison.py  # fast (20 rollouts, no plot)
```
Output figure: `media/showcase/risk_aware_comparison.png`. Reproducible (`run_all` uses `seed=0`).

## The four controllers
| Name | Type | Behavior |
|---|---|---|
| Nominal | no filter | outward drive, exits almost surely |
| S-CBF | per-step stochastic (supermartingale) CBF | most conservative -> safest |
| RA-CBF-DT | accumulating-budget RA-CBF, discrete-time margin (Eq. 13) | conservative |
| RA-CBF-CT | accumulating-budget RA-CBF, continuous-time margin (Eq. 8) | tightest -> rides near boundary, fails ~rho_d |

S-CBF and RA-CBF are *different controller families*: a per-step pointwise condition vs. a
finite-horizon **accumulated-drift budget** `I_L(t) <= 1 - gamma - Delta_rho`. The accumulating
form is essential — a per-step RA-CBF re-zeros the budget each step and damps the very noise
that should drive boundary crossings, collapsing `p_fail` to 0.

## Code -> paper map
| File | Paper |
|---|---|
| `barrier.py` | `B = ||x||^2/R_c^2` (cost, for RA-CBF) and `h = 1 - B` (reframed, for S-CBF), sec II.2 |
| `risk_aware_margins.py` (lib) | Eq. 8 (CT margin), Eq. 13 (DT margin) |
| `accumulating_risk_aware_cbf_controller` (lib, `controllers/cbf_clf/`) | the discrete-time RA-CBF (accumulated `I_L` budget) |
| `controllers.py` | builds Nominal / S-CBF / RA-CBF-DT / RA-CBF-CT |
| `run_comparison.py` | Fig. 1 heatmaps + `p_fail` |

## Two core CBFKit fixes this reproduction required
Reproducing the figure surfaced (and this branch fixes) two genuine core bugs:
1. **Stochastic noise mis-scaling** — `generate_stochastic_perturbation` returns a Brownian
   *increment* `sigma*sqrt(dt)*xi`, but the integrator was multiplying it by `dt` again
   (~`dt`x too little noise). Fixed in `simulation/integration_utils.py` (Euler-Maruyama).
2. **Path-integral RA-CBF stub** — the shipped `risk_aware_path_integral_cbf_clf_qp_controller`
   never accumulated its integral. Replaced by the JIT-safe `accumulating_risk_aware_cbf_controller`
   (carries `I_L` in `ControllerData.sub_data`).

S-CBF runs through the fast PDIPM QP solver (`get_solver("fast")`) to avoid OSQP thrashing
near the origin.

## Results
Empirical `p_fail` over 1000 rollouts (seed=0), vs. the paper's reported values:

| Controller | This reproduction (1000 rollouts, seed=0) | Paper Fig. 1 |
|---|---|---|
| Nominal | 1.000 | 1.000 |
| S-CBF | 0.000 | 0.004 |
| RA-CBF-DT | 0.015 | 0.06 |
| RA-CBF-CT | 0.136 | 0.258 |

The ordering `Nominal > RA-CBF-CT > RA-CBF-DT > S-CBF` matches the paper, and the heatmaps
reproduce its qualitative story (RA-CBF variants concentrate mass near the boundary; S-CBF
stays well inside; Nominal escapes). The safety guarantee is an upper bound, so empirical
values fall at or below the design risk `rho_d=0.30`. Exact magnitudes differ from the paper
because `T`, `dt`, `x0`, `ACTUATION_LIMIT`, and `SCBF_BETA` are reasonable assumptions, not
the paper's (unpublished) exact regime — see `config.py`.

## Constants & assumptions
See `config.py`. Fig. 1 fixes `sigma=0.1`, `R_c=1`, `rho_d=0.30`, `v_max=0.4`, 1000 rollouts;
`T`, `dt`, `x0`, `ACTUATION_LIMIT`, and `SCBF_BETA` are documented reasonable assumptions
(the paper does not pin them down).
