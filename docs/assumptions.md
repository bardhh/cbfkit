# Assumptions and Design Decisions

This document outlines key assumptions made in the design of `cbfkit` and its usage.

## Coordinate Frames
*   **Global Frame:** All dynamics, barrier functions, and controller inputs are assumed to be expressed in a consistent **Global Coordinate Frame**. The library does not automatically handle frame transformations. It is the user's responsibility to ensure that obstacle positions, robot states, and goal states are in the same frame.

## Stochastic Control
*   **Noise Matching:** Stochastic Control Barrier Functions (SCBFs) and controllers assume that the noise covariance matrix `sigma` provided to the controller exactly matches the true process noise of the system dynamics (and the estimator, if applicable). Mismatches may invalidate the high-probability safety guarantees.

## Control Limits & QP Feasibility
*   **Actuation Limits:** The QP solvers respect the provided `control_limits`. If safety constraints (CBFs) conflict with actuation limits, the behavior depends on the controller configuration:
    *   **Strict CBF (Default):** The QP may become infeasible. The current fallback behavior is to return a zero control input (`u=0`).
    *   **Relaxable CBF (`relaxable_cbf=True`):** The solver introduces a slack variable to relax the safety constraint, prioritizing the minimization of this violation while respecting actuation limits. This provides a "best effort" safety behavior.

## JIT Compilation (JAX)
*   **Static Shapes:** When using `use_jit=True`, all data structures passed to the simulation loop (`ControllerData`, `PlannerData`) must have static shapes and types. The library now uses `NamedTuple` to enforce this.
*   **Recompilation:** Changing the structure of these data objects (e.g., adding a new field at runtime) will trigger a full recompilation of the simulation loop, which can be computationally expensive.
