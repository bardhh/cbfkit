# Renaming Plan for Examples and Tutorials

## Objectives
1.  **Clarity:** Use descriptive names for files and directories (e.g., `reach_goal` instead of `start_to_goal` or `ra_fxt_clf`).
2.  **Consistency:** Standardize naming conventions (e.g., `*_control.py` for executable scripts).
3.  **Structure:** Group examples by "System" -> "Task/Scenario" -> "Implementation Variant".

## Examples (`examples/`)

### Fixed Wing (`examples/fixed_wing/`)
| Current Path | New Path | Rationale |
| :--- | :--- | :--- |
| `reach_drop_point/ekf_estimation.py` | `reach_drop_point/ekf_control.py` | Script runs a controller using EKF, not just estimation. |

### Single Integrator (`examples/single_integrator/`)
| Current Path | New Path | Rationale |
| :--- | :--- | :--- |
| `ra_fxt_clf/` | `reach_goal/` | "Risk-Aware Fixed-Time CLF" is the method, "Reach Goal" is the task. |
| `ra_fxt_clf/perfect_measurements.py` | `reach_goal/perfect_sensing.py` | Consistent naming. |
| `ra_fxt_clf/ekf_estimation.py` | `reach_goal/ekf_control.py` | Focus on control execution. |
| `ra_fxt_clf/ukf_estimation.py` | `reach_goal/ukf_control.py` | Focus on control execution. |

### Unicycle (`examples/unicycle/`)
| Current Path | New Path | Rationale |
| :--- | :--- | :--- |
| `start_to_goal/` | `reach_goal/` | More standard terminology. |
| `risk_aware_cbf/` | `reach_goal/` | Merge into main task folder (Risk Aware is a variant). |
| `start_to_goal/perfect_measurements.py` | `reach_goal/perfect_sensing.py` | |
| `start_to_goal/ekf_estimation.py` | `reach_goal/ekf_control.py` | |
| `start_to_goal/ukf_estimation.py` | `reach_goal/ukf_control.py` | |
| `start_to_goal/ekf_ukf_hybrid_estimation.py` | `reach_goal/hybrid_estimation_control.py` | |
| `start_to_goal/mppi_cbf_start_to_goal_main.py` | `reach_goal/mppi_cbf_control.py` | Remove redundant `start_to_goal` and `_main`. |
| `start_to_goal/stochastic_cbf_start_to_goal_main.py` | `reach_goal/stochastic_cbf_control.py` | |
| `start_to_goal/vanilla_cbf_start_to_goal_main.py` | `reach_goal/vanilla_cbf_control.py` | |
| `risk_aware_cbf/start_to_goal_montecarlo.py` | `reach_goal/risk_aware_cbf_monte_carlo.py` | |
| `risk_aware_cbf/start_to_goal.py` | `reach_goal/risk_aware_cbf_control.py` | |

### Van der Pol (`examples/van_der_pol/`)
| Current Path | New Path | Rationale |
| :--- | :--- | :--- |
| `ra_fxt_clbf/` | `regulation/` | Task is stabilization/regulation to origin. |
| `ra_fxt_clbf/perfect_measurements.py` | `regulation/perfect_sensing.py` | |
| `ra_fxt_clbf/ukf_estimation.py` | `regulation/ukf_control.py` | |

## Tutorials (`tutorials/`)

| Current Path | New Path | Rationale |
| :--- | :--- | :--- |
| `simulate_new_control_system.ipynb` | `code_generation_tutorial.ipynb` | Describes the content better. |
| `multi_robot_example.ipynb` | `multi_robot_coordination.ipynb` | |
| `single_integrator_reach_avoid_dyn_obs.py` | `single_integrator_dynamic_obstacles.py` | Shorter, cleaner. |
| `simulate_mppi_cbf.py` | `mppi_cbf_reach_avoid.py` | Describes the controller and task. |
| `simulate_mppi_cbf_ellipsoidal_stochastic_cbf.py` | `mppi_stochastic_cbf_reach_avoid.py` | |
| `simulate_mppi_stl.py` | `mppi_stl_reach_avoid.py` | |
| `test.py` | `multi_robot_test.py` | Avoid generic `test.py` name. |
