# Anisotropic (rotating-ellipse) footprint CBF — Implementation Plan

**Goal:** Let the safety layer exploit the G1's body shape: an orientation-dependent
elliptical footprint in the certificate, heading as a control channel, so the QP itself
discovers "turn sideways and sidestep" through gaps a disc-CBF must refuse (1.30 m disc
minimum vs ~0.9-1.0 m ellipse minimum).

**Why this succeeds where the reflex failed (measured 2026-08-20):** sidestepping is an
in-distribution locomotion mode (AMO vy, heading decoupled from travel), not a waist
twist against the gait; and the ellipse is in the *certificate*, so narrow passages are
certified rather than styled.

## Phase A — measurements (constants with provenance)
- `examples/mujoco/g1_footprint_measure.py`: AMO stand / walk vx 0.4 / sidestep vy ±0.3
  (fixed heading) in MJX; FK per step; body-frame extents |body point − CoM| split into
  longitudinal (facing) and lateral (shoulder) axes, for the upper body (z > 0.6) and
  the full body. The barrier uses the UPPER-body ellipse; legs exceed it longitudinally
  during stride (documented choice: pedestrian discs describe whole people too, and feet
  interleave when humans squeeze).
- Per-axis tracking residuals (body frame) for vx-only and vy-only commands → δ_x, δ_y
  quantiles (lateral gait expected worse).

## Phase B — model + barrier + corridor (make-or-break)
- `reduced_order.embedded_heading_double_integrator(state_dim, indices, n_agents)`:
  state `[x | v(2) | θ | ω | agents]`, controls `[ax, ay, αθ]` (heading second-order →
  every barrier uniformly rel-deg 2, existing rectifier applies).
- `reduced_order.com_agent_ellipse_hocbfs(plant, n_agents, axes, ...)`:
  `h_i = ‖diag(1/a_lon, 1/a_lat) R(θ)ᵀ (com − p_i)‖ − 1` (distance shaping).
- `reduced_order.safe_locomotion_controller_hdi(...)`: DI wrapper + heading channel;
  nominal αθ = P-law facing the travel direction; integrates (v, θ, ω); hands
  `[v_safe, θ_cmd]` to the tracking layer.
- `amo_policy.as_controller`: accept `u_nom[2]` as an absolute target-yaw override.
- `examples/mujoco/g1_corridor.py`: two static pedestrians, gap sweep. Assert: disc-CBF
  stalls at 1.0 m gap, ellipse-CBF rotates + sidesteps through, h ≥ 0 (HARD constraints
  — this is a certificate demo, static scene, feasible). Proxy first, then G1 + AMO.
  Known wrinkle: at a perfectly symmetric approach ∂h/∂θ = 0 (saddle) — corridor laid
  out slightly asymmetric; if the pure QP does not rotate, add an orientation suggestion
  to the NOMINAL (certificate untouched) and report which was needed.

## Phase C — scramble (only if B wins)
- `g1_scramble --footprint ellipse` (heading-DI stack, robust margins per-axis from A);
  honest table vs disc.
