# G1 walk — milestone 3 lab log (2026-08-18)

Goal: G1 tracks a planar velocity command (vx, vy) via `SamplingMpc` through `execute(plant=...)`,
without falling. Success = upright ≥ 0.9 throughout, mean pelvis speed ≥ ~60 % of the command,
and net displacement in the commanded direction; then a lateral/diagonal command to prove steerability.
Harness: `examples/mujoco/g1_walk_trial.py` (one JSON line per trial). Compile ≈ 100 s per config on CPU.

| # | config (deltas from harness defaults) | mean v (last 2 s) | dist (x, y) | min upright | fell | note |
|---|---|---|---|---|---|---|
| t01 | defaults (128 samples, no DR, zero spline 4 knots, H=0.6, vx=0.5) | (-0.04, -0.09); whole run (0.22, -0.06) | (0.87, -0.24) | 0.57 | **yes** | lurches 0.87 m forward in ~2 s, then falls; yaw-rate rms 1.06. 49 s wall (compile cache warm) |
| t02 | linear spline, 6 knots, H=0.8 | (0.07, 0.02) | (0.61, 0.40) | 0.77 | **yes** | collapses (min height 0.34) rather than tips; yaw rms 1.42 |
| t03 | w_vel 20, w_posture 0.02, noise 0.4 | (-0.03, -0.05) | (-0.01, -0.21) | 0.84 | **yes** | doesn't even move; collapses (0.31). Less posture prior + more noise = worse |
| t04 | + w_balance 5, w_yaw 5, w_height 10 | (-0.01, -0.06) | (0.15, -0.43) | 0.55 | **yes** | collapses at 0.86 s |
| t05 | t04 @ vx 0.3 | (0.36, 0.20) | (0.89, 0.13) | 0.73 | **yes** | collapses at 0.6 s, STANDS BACK UP at 2 s, moves 0.4-0.7 m/s for 1 s, collapses again |
| t06 | t04 + noise 0.15, 256 samples | (0.01, -0.01) | (0.76, 0.11) | 0.65 | **yes** | stands 1 s, accelerates to 0.74 m/s, collapses at 1.5 s |

**Diagnosis after wave 2:** every run over-accelerates past the command (0.6-0.9 m/s) then collapses onto the knees with torso upright (h≈0.35, up≈1.0) — a lunge, not a gait. Cost makes "collapsed at speed" (~3/step) about as cheap as "standing still" (~2.5/step). Wave 3: hinge on height (w_fall) + lower w_velocity + damped arm noise; horizon 1.0 vs 0.6; sim=planner diagnostic.
| t07 | w_fall 100 (h_min 0.75), w_vel 5, w_height 10, w_balance 5, w_yaw 5, arm noise ×0.3, H=1.0, 5 knots | (0.21, 0.02) | (0.50, 0.12) | 0.30 | **yes** | no kneeling now — TIPS OVER instead |
| t08 | t07 @ H=0.6, 4 knots | (0.17, 0.01) | (0.78, -0.37) | 0.44 | **yes** | same, tips |
| t09 | t07 with sim = planner model (no mismatch) | (-0.10, -0.10) | (-0.25, -0.09) | 0.20 | **yes** | falls too → model mismatch is NOT the primary cause |

**After wave 3:** the cost shaping moved the failure from kneeling to tipping; the search itself is the problem (128 samples, 1 iteration/step, long horizon). DIAL-MPC (the published MJX sampling-MPC legged result) uses H≈0.32 s, 4 nodes, 2 iterations/step, 2048 samples. Wave 4: MPPI-CMA; DIAL-shaped (short horizon, 2 iterations, 256 samples); slow+balanced.
| t10 | t08 cost + MPPI-CMA (min noise 0.2) | (0.24, -0.10) | (1.44, 0.27) | 0.38 | **yes** | fast lunge to 0.75 m/s at 1.5 s, falls, partial recovery |
| t11 | **DIAL-shaped**: H=0.32, 4 knots, 2 iterations/step, 256 samples (t08 cost) | (0.48, -0.20) | **(1.78, 0.22)** | 0.73 | stumble (h 0.46 @3.5 s) | **first run that doesn't fall**: h 0.66-0.94, alternating feet — but kicks to 0.9 m and lurches laterally. 82 s wall |
| t12 | t08 cost, vx 0.25, w_balance 20 | (0.34, -0.10) | (0.83, -0.70) | 0.64 | **yes** | slow start, then lurch/stumble |

**After wave 4:** short horizon + 2 iterations is the key (matches DIAL-MPC). Wave 5 on the t11 base: sustain over 8 s; foot-clearance hinge (z>0.15) + w_vel 10; 512 samples.
| t13 | t11 for 8 s | (0.39, -0.01) | (2.77, 0.34) | 0.37 | 3 stumbles, all recovered | sustained but scrappy; feet still to 0.9 m |
| t14 | t11 + w_feet 20 (z>0.15), w_vel 10 | (0.64, -0.10) | (1.75, -0.11) | 0.78 | 1 stumble @2.5 s (h 0.34), recovered | **first 2 s look like walking** (h 0.83-0.98, feet ≤0.57); over-speed spike 0.97 m/s → stumble → recovers to 0.87 m/s |
| t15 | t11 + 512 samples | (0.57, -0.24) | (2.12, -1.0) | 0.58 | **yes** | more samples is not the lever |

**After wave 5:** stumbles are preceded by over-speed spikes. Wave 6: w_vel 20, fall hinge 300 @0.78, foot cap 0.12 @50, w_qvel 0.02; iterations 3; a diagonal command to test steerability.
| t16 | w_vel 20, fall 300@0.78, feet cap 0.12@50, w_qvel 0.02 | (0.84, -0.35) | (3.17, -1.16) | 0.86 | 2 stumbles | vx spikes to 1.48 — the spikes are the pelvis toppling, i.e. a symptom of bad steps, not the cause |
| t17 | t16 + 3 iterations | (-0.02, 0.64) | (0.79, 1.23) | -0.77 | **flat on its back** | |
| t18 | t16 @ (0.3, 0.3) diagonal | (-0.10, -0.13) | (0.76, 0.27) | 0.12 | **yes** | |

**After wave 6:** cost tweaks on the free-form gait have plateaued. DIAL-MPC's published H1/Go2 walking uses a phase-based gait reward (periodic foot-height reference), not a free-form search. Wave 7 adds that prior (w_gait, freq, swing height, duty); rollout root now carries absolute time.
| t19 | gait prior w 20, 1.5 Hz, swing 0.08 (w_vel 10, fall 100, qvel 0.01) | (0.41, -0.04) | (1.11, -0.42) | 0.75 | crouch, stumble @3.5 s | feet now ≤0.4 (prior works); shuffles at h≈0.65 |
| t20 | gait w 50 | (0.77, 0.21) | (2.69, 0.23) | 0.86 | crouching 0.4-0.6 | moves, low and unsteady |
| t21 | gait w 30, 1.2 Hz, swing 0.06, duty 0.6 | (0.12, -0.06) | (1.21, -0.09) | 0.63 | collapses @1.5 s | |

**After wave 7:** gait prior fixes the kicking; failure moves to crouching/collapse. Wave 8: prior + stronger height/fall; 2 Hz cadence; friction DR ×4.
| t22 | gait 50 + w_height 30, fall 300@0.8 | (0.31, -0.14) | (1.44, 0.67) | -0.05 | **yes** | |
| t23 | gait 30 @2 Hz swing 0.06 + height 30, fall 300 | (0.60, -0.39) | (2.23, -0.12) | -0.02 | deep tip @1.5 s, recovers | moves at 0.5-0.9 m/s but leans to horizontal and back |
| t24 | t22 + friction DR ×4 (128 samples) | (0.52, 0.01) | (2.14, 0.33) | 0.07 | deep tips, recovers | 159 s wall |

**Checkpoint after 24 trials (2026-08-18):** forward locomotion at ~0.5 m/s is reproducible; a *clean, sustained, steerable* gait is not, at 128-256 samples / 2 iterations on CPU. Best runs: t14 (cleanest first 2 s), t11/t13 (sustained with stumbles), t20/t23 (fastest). Recommendation recorded in the session: move the search to GPU (2048 samples, DIAL-MPC regime) or accept the current gait for the CBF demo. Everything needed is in the harness; `--update cma`, `--iterations`, per-actuator noise, gait prior all available.

**End-to-end (g1_navigate.py, 8 s, CBF + t14/t20-class gait, 2026-08-18):** CBF active on 83 % of steps,
CoM path deflects left around the obstacle; but h(x) dips to -0.50 at t≈6 s (CoM 0.50 m from the obstacle
centre vs 0.70 m keep-out; physical clearance ≈0.15 m) — the gait's tracking error is far larger than any
small robust bound. Torso height min 0.33 (stumble); ends 2.5 m short of the goal. Qualitatively right,
quantitatively violated: a better gait (GPU-scale sampling or a trained policy) is what turns this into a demo.
