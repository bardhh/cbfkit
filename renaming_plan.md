# Naming Convention for Examples and Tutorials

## Convention

### Directory Hierarchy

```
examples/<system>/<task>/<variant>.py
```

| Level | Rule | Examples |
|:---|:---|:---|
| **System** | Dynamical system name, `snake_case` | `unicycle/`, `differential_drive/`, `van_der_pol/` |
| **Task** | Control objective (not method), `snake_case` | `reach_goal/`, `regulation/`, `obstacle_avoidance/` |
| **Shared dirs** | Fixed names for shared code/output | `common/`, `visualizations/`, `results/` |

### File Naming

Use **descriptive names** — no forced suffix. The name should convey what distinguishes each variant.

| Variant type | Approach | Examples |
|:---|:---|:---|
| Estimation method | Name of the estimator | `ekf.py`, `ukf.py`, `hybrid_ekf_ukf.py` |
| Perfect state access | Describes the sensing | `perfect_sensing.py` |
| Controller/planner | Name of the method | `vanilla_cbf.py`, `mppi_cbf.py`, `stochastic_cbf.py` |
| Monte Carlo | Method + `_monte_carlo` | `risk_aware_cbf_monte_carlo.py` |
| Config/helpers | What they provide | `config.py`, `barrier_functions.py` |

**Avoid:** `_control` / `_main` / `_demo` suffixes, `simulate_` prefix, abbreviations, method names as directory names, redundant path info in file names.

### Tutorials

| Type | Pattern | Examples |
|:---|:---|:---|
| Python scripts | `<system>_<task>.py` or `<method>_<task>.py` | `unicycle_reach_avoid.py`, `mppi_cbf_reach_avoid.py` |
| Notebooks | `<descriptive_topic>.ipynb` | `code_generation_tutorial.ipynb` |

---

## Completed Renames

### Examples

#### Fixed Wing (`examples/fixed_wing/`)

| Original | Final | Status |
|:---|:---|:---|
| `reach_drop_point/ekf_estimation.py` | `reach_drop_point/ekf.py` | Done |

#### Single Integrator (`examples/single_integrator/`)

| Original | Final | Status |
|:---|:---|:---|
| `ra_fxt_clf/` | `reach_goal/` | Done |
| `ra_fxt_clf/perfect_measurements.py` | `reach_goal/perfect_sensing.py` | Done |
| `ra_fxt_clf/ekf_estimation.py` | `reach_goal/ekf.py` | Done |
| `ra_fxt_clf/ukf_estimation.py` | `reach_goal/ukf.py` | Done |

#### Unicycle (`examples/unicycle/`)

| Original | Final | Status |
|:---|:---|:---|
| `start_to_goal/` | `reach_goal/` | Done |
| `risk_aware_cbf/` | `reach_goal/` | Done (merged) |
| `start_to_goal/perfect_measurements.py` | `reach_goal/perfect_sensing.py` | Done |
| `start_to_goal/ekf_estimation.py` | `reach_goal/ekf.py` | Done |
| `start_to_goal/ukf_estimation.py` | `reach_goal/ukf.py` | Done |
| `start_to_goal/ekf_ukf_hybrid_estimation.py` | `reach_goal/hybrid_ekf_ukf.py` | Done |
| `start_to_goal/mppi_cbf_start_to_goal_main.py` | `reach_goal/mppi_cbf.py` | Done |
| `start_to_goal/stochastic_cbf_start_to_goal_main.py` | `reach_goal/stochastic_cbf.py` | Done |
| `start_to_goal/vanilla_cbf_start_to_goal_main.py` | `reach_goal/vanilla_cbf_accel_unicycle.py` | Done |
| `risk_aware_cbf/start_to_goal_montecarlo.py` | `reach_goal/risk_aware_cbf_monte_carlo.py` | Done |
| `risk_aware_cbf/start_to_goal.py` | `reach_goal/risk_aware_cbf.py` | Done |
| `reach_goal/vanilla_cbf_control.py` | `reach_goal/vanilla_cbf.py` | Done |
| Stale: `nominal_control_ukf_estimation_main.py` | Deleted (duplicate of `ukf.py`) | Done |
| Stale: `risk_aware_cbf_monte_carlo_main.py` | Deleted (duplicate) | Done |

#### Van der Pol (`examples/van_der_pol/`)

| Original | Final | Status |
|:---|:---|:---|
| `ra_fxt_clbf/` | `regulation/` | Done |
| `ra_fxt_clbf/perfect_measurements.py` | `regulation/perfect_sensing.py` | Done |
| `ra_fxt_clbf/ukf_estimation.py` | `regulation/ukf.py` | Done |
| Stale: `regulation/ukf_estimation.py` | Deleted (duplicate) | Done |
| Stale: `regulation/perfect_state_measurements/main.py` | Deleted (duplicate) | Done |

#### Differential Drive (`examples/differential_drive/`) — Reorganized

| Original | Final | Status |
|:---|:---|:---|
| `single_robot_cbf.py` | `obstacle_avoidance/single_robot_cbf.py` | Done |
| `dynamic_obstacle_cbf.py` | `obstacle_avoidance/dynamic_obstacle_cbf.py` | Done |
| `augmented_dynamic_obstacle_cbf.py` | `obstacle_avoidance/augmented_dynamic_obstacle_cbf.py` | Done |
| `barrier_activated_cbf.py` | `obstacle_avoidance/barrier_activated_cbf.py` | Done |
| `human_aware_mppi_cbf.py` | `human_aware_navigation/mppi_cbf.py` | Done |
| `multi_scenario_human_aware.py` | `human_aware_navigation/multi_scenario_comparison.py` | Done |

#### Pedestrian (`examples/pedestrian/`) — Reorganized

| Original | Final | Status |
|:---|:---|:---|
| `crowded_demo.py` | `navigate_among_pedestrians/crowded.py` | Done |
| `head_on_demo.py` | `navigate_among_pedestrians/head_on.py` | Done |
| `overtaking_demo.py` | `navigate_among_pedestrians/overtaking.py` | Done |
| `star_burst_demo.py` | `navigate_among_pedestrians/star_burst.py` | Done |
| `pedestrian_manager_demo.py` | `navigate_among_pedestrians/crossing.py` | Done |

#### Adaptive CVaR CBF (`examples/adaptive_cvar_cbf/`)

| Original | Final | Status |
|:---|:---|:---|
| `adaptive_cvar_cbf_cbfkit_demo.py` | `adaptive_cvar_cbf.py` | Done |

### Tutorials

| Original | Final | Status |
|:---|:---|:---|
| `simulate_new_control_system.ipynb` | `code_generation_tutorial.ipynb` | Done |
| `multi_robot_example.ipynb` | `multi_robot_coordination.ipynb` | Done |
| `single_integrator_reach_avoid_dyn_obs.py` | `single_integrator_dynamic_obstacles.py` | Done |
| `simulate_mppi_cbf.py` | `mppi_cbf_reach_avoid.py` | Done |
| `simulate_mppi_cbf_ellipsoidal_stochastic_cbf.py` | Deleted (duplicate of `mppi_stochastic_cbf_reach_avoid.py`) | Done |
| `simulate_mppi_stl.py` | `mppi_stl_reach_avoid.py` | Done |
| `test.py` | `multi_robot_coordination_test.py` | Done |
