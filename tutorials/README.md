# Tutorials

Step-by-step guides that teach you how to **define and generate new dynamical systems** using CBFKit's code generation (`cbfkit.codegen`).

## Prerequisites

```bash
pip install "cbfkit[codegen]"
```

## Recommended Order

| # | File | Description |
|---|---|---|
| 1 | `code_generation_tutorial.ipynb` | Generate a Van der Pol oscillator — dynamics, controllers, barrier/Lyapunov functions, and ROS2 nodes |
| 2 | `multi_robot_coordination.ipynb` | Multi-robot collision avoidance with CBF constraints (notebook) |
| 3 | `mppi_cbf_reach_avoid.py` | MPPI planner with CBF safety filter using codegen |
| 4 | `mppi_stl_reach_avoid.py` | MPPI with Signal Temporal Logic specifications |
| 5 | `single_integrator_dynamic_obstacles.py` | Single integrator with dynamic obstacle avoidance |
| 6 | `multi_robot_coordination_codegen.py` | Scripted multi-robot coordination with codegen |
| 7 | `multi_robot_3d_reachavoid.py` | 3D multi-robot reach-avoid with MPPI and CBF |

## Generated Artifacts

Tutorials generate system directories (e.g., `van_der_pol_oscillator/`, `multi_augmented_single_integrators/`) at runtime. These are gitignored and not committed to the repository.

## Looking for ready-to-run examples?

See the [`examples/`](../examples/README.md) directory for scripts that use pre-built library systems with no code generation.
