# Tutorial Entrypoint Migration Map

Status date: 2026-02-17

This document maps removed/renamed tutorial commands to the maintained
entrypoints. Legacy paths listed here are retained as compatibility shims.

| Legacy command | Replacement command | Compatibility behavior |
|---|---|---|
| `python tutorials/mppi_cbf_reach_avoid.py` | `python tutorials/simulate_mppi_cbf.py` | Legacy script prints deprecation warning and forwards to replacement |
| `python tutorials/mppi_stl_reach_avoid.py` | `python tutorials/simulate_mppi_stl.py` | Legacy script prints deprecation warning and forwards to replacement |
| `python tutorials/single_integrator_dynamic_obstacles.py` | `python tutorials/single_integrator_reach_avoid_dyn_obs.py` | Legacy script prints deprecation warning and forwards to replacement |

Notes:
- For migration tests and CI speed, aliases honor `CBFKIT_TUTORIAL_ALIAS_ONLY=1`
  to print mapping without executing the replacement.
- New documentation should reference replacement commands only.
