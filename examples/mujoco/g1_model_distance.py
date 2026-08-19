"""How far is the double-integrator reduced model from the G1 + walking policy?

A battery of open-loop command runs (no CBF) that measures the distance between the
reduced model the CBF certifies -- ``d/dt com = v_cmd`` (commands of the DI class: smooth,
|a| <= A_MAX) or the SI class (piecewise-constant velocity jumps) -- and what the MJX G1
under Unitree's walking policy actually does. Three notions, all from the same data:

1. **Disturbance bound** ``d(t) = v_com(t) - v_cmd(t)`` -- the ``||d||`` the robust CBF-QP
   consumes (``--robust``). Reported as pooled quantiles (a conformal-style
   ``(1 - alpha)(1 + 1/n)`` quantile; samples are temporally correlated within runs, so read
   it as descriptive marginal coverage, not a guarantee), per-run maxima, and as a table
   over command speed and command turn rate -- the bound is not a constant.
2. **Approximate-simulation distance** ``eps(H)`` -- re-anchor the reduced model to the
   measured CoM at time t, integrate the command for H seconds, and measure
   ``||com_G1(t+H) - com_model(t+H)||`` (sup and p95 over all windows and runs). ``eps(dt)``
   is what the CBF relies on step to step; ``eps(2 s)`` is what a planner could rely on.
3. **Identified lag + nu-gap** -- fit ``v_com = k e^{-sL} / (tau s + 1) v_cmd`` (gain, delay,
   first-order lag) by grid search on the battery, report ``(k, tau, L)`` and the
   Vinnicombe nu-gap between ``P_DI = 1/s`` and ``P_id = G_id / s`` (numerical sup of the
   chordal distance on a frequency grid; the SISO winding-number condition is checked on
   the same grid). The gap is the classical one-number "distance between the two plants";
   ``|G_id(jw) - 1|`` over frequency says where the DI assumption breaks.

Caveat: all of this is a property of (reduced model, policy, MJX) jointly -- it measures the
gait's tracking, not humanoid physics. The battery includes stops, restarts and sharp turns
so the avoidance regime (where the CBF needs the bound) is covered.

    python examples/mujoco/g1_model_distance.py [--runs 6] [--duration 20] [--seed 0]

Outputs ``examples/mujoco/results/g1_model_distance.{md,json,png}``.
"""

import argparse
import json
import os
import sys
import time

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import jax
import jax.numpy as jnp
import numpy as np

import cbfkit.simulation.simulator as sim
from cbfkit.systems.mujoco.unitree_policy import (
    UnitreeG1WalkPolicy,
    make_g1_12dof_plant,
    x0_standing,
)
from cbfkit.utils.user_types import PlannerData

TEST_MODE = bool(os.getenv("CBFKIT_TEST_MODE"))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

V_MAX = 0.5  # same operating envelope as g1_navigate / g1_plaza
A_MAX = 1.0
HORIZONS = [0.02, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
ALPHAS = [0.1, 0.05, 0.01]


# --------------------------------------------------------------------------- command batteries
def di_command(rng, n, dt):
    """DI-class command: piecewise-constant random accelerations (|a| <= A_MAX, held 0.5-2 s),
    integrated and clipped to |v| <= V_MAX; one in four segments brakes to a stop."""
    v = np.zeros(2)
    out = np.zeros((n, 2))
    k = 0
    while k < n:
        hold = int(rng.uniform(0.5, 2.0) / dt)
        if rng.uniform() < 0.25:  # brake to a stop at the max rate
            a_seg = None
        else:
            ang = rng.uniform(0, 2 * np.pi)
            a_seg = rng.uniform(0.3, 1.0) * A_MAX * np.array([np.cos(ang), np.sin(ang)])
        for _ in range(hold):
            if k >= n:
                break
            a = a_seg if a_seg is not None else -v / 0.5
            na = np.linalg.norm(a)
            if na > A_MAX:
                a = a * (A_MAX / na)
            v = v + a * dt
            sp = np.linalg.norm(v)
            if sp > V_MAX:
                v = v * (V_MAX / sp)
            out[k] = v
            k += 1
    return out


def si_command(rng, n, dt):
    """SI-class command: piecewise-constant random velocities (speed U[0, V_MAX], 20 % zero,
    held 1.5-4 s) with jumps between segments; 1 s of rest first."""
    out = np.zeros((n, 2))
    k = int(1.0 / dt)
    while k < n:
        hold = int(rng.uniform(1.5, 4.0) / dt)
        if rng.uniform() < 0.2:
            v = np.zeros(2)
        else:
            ang = rng.uniform(0, 2 * np.pi)
            v = rng.uniform(0.1, V_MAX) * np.array([np.cos(ang), np.sin(ang)])
        out[k : k + hold] = v
        k += hold
    return out


def command_planner(dt):
    """Planner that replays ``data.u_traj`` (2 x T): the command enters the JIT as data."""

    def planner(t, x, u_nom, key, data):
        idx = jnp.clip(jnp.round(t / dt).astype(int), 0, data.u_traj.shape[1] - 1)
        return data.u_traj[:, idx], data

    return planner


def run_battery(plant, loco, planner, commands, seed):
    """Execute every command signal; return (com, v_cmd) arrays per run."""
    x0 = x0_standing(plant)
    ci = plant.com_indices
    out = []
    for i, cmd in enumerate(commands):
        t0 = time.time()
        res = sim.execute(
            x0=x0,
            dt=plant.dt,
            num_steps=cmd.shape[0],
            plant=plant,
            planner=planner,
            planner_data=PlannerData(u_traj=jnp.asarray(cmd.T)),
            controller=loco,
            key=jax.random.PRNGKey(seed + i),
            use_jit=True,
            verbose=False,
        )
        S = np.asarray(res["states"])
        com = S[:, ci[0] : ci[0] + 2]
        up = 1 - 2 * (S[:, 4] ** 2 + S[:, 5] ** 2)
        print(
            f"  run {i+1}/{len(commands)}: {cmd.shape[0]} steps in {time.time()-t0:.0f}s, "
            f"pelvis z min {S[:, 2].min():.2f}, upright min {up.min():.2f}"
        )
        out.append((com, cmd[: com.shape[0]]))
    return out


# --------------------------------------------------------------------------- metrics
def residuals(runs, dt):
    """Per-run ``d = v_com - v_cmd`` (finite-difference CoM velocity), plus features."""
    res = []
    for com, cmd in runs:
        v_com = np.gradient(com, dt, axis=0)
        d = v_com - cmd
        speed = np.linalg.norm(cmd, axis=1)
        heading = np.unwrap(np.arctan2(cmd[:, 1], cmd[:, 0] + 1e-12))
        turn = np.abs(np.gradient(heading, dt))
        turn[speed < 0.05] = 0.0  # heading is undefined at rest
        res.append(dict(d=d, norm=np.linalg.norm(d, axis=1), speed=speed, turn=turn))
    return res


def quantile_bound(norms, alpha):
    """Conformal-style (1 - alpha)(1 + 1/n) empirical quantile of pooled residual norms."""
    n = norms.size
    q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(norms, q))


def binned_table(res, key, edges):
    pooled = np.concatenate([r["norm"] for r in res])
    feat = np.concatenate([r[key] for r in res])
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (feat >= lo) & (feat < hi)
        if m.sum() < 20:
            rows.append((lo, hi, int(m.sum()), None, None, None))
            continue
        rows.append(
            (
                lo,
                hi,
                int(m.sum()),
                float(pooled[m].mean()),
                float(np.quantile(pooled[m], 0.95)),
                float(pooled[m].max()),
            )
        )
    return rows


def simulation_distance(runs, dt, horizons):
    """eps(H): re-anchor the model at t, integrate the command, compare to the G1 at t+H."""
    out = {}
    for H in horizons:
        h = max(1, int(round(H / dt)))
        errs = []
        for com, cmd in runs:
            n = com.shape[0]
            if n <= h:
                continue
            disp = np.cumsum(cmd, axis=0) * dt  # integral of the command
            pred = com[:-h] + (disp[h:] - disp[:-h])  # model CoM at t+H, anchored at t
            errs.append(np.linalg.norm(com[h:] - pred, axis=1))
        e = np.concatenate(errs) if errs else np.zeros(1)
        out[H] = dict(
            p50=float(np.quantile(e, 0.5)), p95=float(np.quantile(e, 0.95)), sup=float(e.max())
        )
    return out


def _lag_filter(cmd, dt, k, tau, L):
    """Simulate v = k e^{-sL}/(tau s + 1) cmd (exact ZOH discretisation)."""
    delay = int(round(L / dt))
    u = np.vstack([np.zeros((delay, 2)), cmd[: cmd.shape[0] - delay]]) if delay else cmd
    a = np.exp(-dt / tau) if tau > 1e-6 else 0.0
    y = np.zeros_like(u)
    for i in range(1, u.shape[0]):
        y[i] = a * y[i - 1] + (1 - a) * k * u[i]
    return y


def identify_lag(runs, dt):
    """Grid search (k, tau, L) minimising the pooled squared velocity error."""
    best = None
    ks = np.linspace(0.7, 1.1, 9)
    taus = np.concatenate([[0.0], np.geomspace(0.05, 2.0, 25)])
    Ls = np.arange(0.0, 0.5001, dt * 2)
    vs = [np.gradient(com, dt, axis=0) for com, _ in runs]
    for tau in taus:
        for L in Ls:
            # k enters linearly: solve it in closed form for this (tau, L), then snap to the grid
            num = den = 0.0
            ys = [_lag_filter(cmd, dt, 1.0, tau, L) for _, cmd in runs]
            for y, v in zip(ys, vs):
                num += float((y * v).sum())
                den += float((y * y).sum())
            k = num / den if den > 0 else 1.0
            k = float(ks[np.argmin(np.abs(ks - k))])
            cost = sum(float(((k * y - v) ** 2).sum()) for y, v in zip(ys, vs))
            if best is None or cost < best[0]:
                best = (cost, k, float(tau), float(L))
    cost, k, tau, L = best
    n = sum(v.shape[0] for v in vs)
    baseline = sum(float(((cmd - v) ** 2).sum()) for (_, cmd), v in zip(runs, vs))
    return dict(
        k=k,
        tau=tau,
        delay=L,
        rms=float(np.sqrt(cost / n / 2)),
        rms_di=float(np.sqrt(baseline / n / 2)),
    )


def nu_gap(k, tau, L, wmin=1e-2, wmax=50.0, n=4000):
    """nu-gap between P1 = 1/s and P2 = k e^{-sL}/((tau s + 1) s), numerically.

    Chordal distance kappa(w) = |P1 - P2| / sqrt((1+|P1|^2)(1+|P2|^2)) on a log grid; the
    SISO winding condition is checked as the net phase change of 1 + conj(P2) P1 along the
    grid (both plants share the integrator, so the condition is expected to hold).
    """
    w = np.geomspace(wmin, wmax, n)
    s = 1j * w
    P1 = 1 / s
    P2 = k * np.exp(-s * L) / ((tau * s + 1) * s)
    kappa = np.abs(P1 - P2) / np.sqrt((1 + np.abs(P1) ** 2) * (1 + np.abs(P2) ** 2))
    f = 1 + np.conj(P2) * P1
    wno = float(np.sum(np.diff(np.unwrap(np.angle(f)))) / (2 * np.pi))
    G = k * np.exp(-s * L) / (tau * s + 1)
    return dict(
        nu_gap=float(kappa.max()),
        at_omega=float(w[np.argmax(kappa)]),
        winding=wno,
        w=w.tolist(),
        G_minus_1=np.abs(G - 1).tolist(),
        bandwidth_G_minus_1_le_0p3=float(w[np.argmax(np.abs(G - 1) > 0.3)])
        if np.any(np.abs(G - 1) > 0.3)
        else float("inf"),
    )


# --------------------------------------------------------------------------- main
def main(runs=6, duration=20.0, seed=0):
    plant = make_g1_12dof_plant()
    loco = UnitreeG1WalkPolicy().as_controller()
    planner = command_planner(plant.dt)
    n = 5 if TEST_MODE else int(round(duration / plant.dt))
    runs = 1 if TEST_MODE else runs
    rng = np.random.default_rng(seed)
    batteries = {
        "di": [di_command(rng, n, plant.dt) for _ in range(runs)],
        "si": [si_command(rng, n, plant.dt) for _ in range(runs)],
    }
    report = {"runs": runs, "duration": duration, "dt": plant.dt, "seed": seed}
    data = {}
    for name, cmds in batteries.items():
        print(f"battery {name}: {len(cmds)} runs x {n} steps")
        data[name] = run_battery(plant, loco, planner, cmds, seed)
    lines = ["# Distance between the reduced models and the G1 + walking policy", ""]
    lines.append(
        f"{runs} runs x {duration:.0f} s per command class, dt = {plant.dt} s, seed {seed}."
    )
    lines.append("")
    for name in batteries:
        res = residuals(data[name], plant.dt)
        pooled = np.concatenate([r["norm"] for r in res])
        per_run_max = np.array([r["norm"].max() for r in res])
        per_run_p95 = np.array([np.quantile(r["norm"], 0.95) for r in res])
        q = {a: quantile_bound(pooled, a) for a in ALPHAS}
        eps = simulation_distance(data[name], plant.dt, HORIZONS)
        speed_tab = binned_table(res, "speed", [0, 0.05, 0.2, 0.35, 0.51])
        turn_tab = binned_table(res, "turn", [0, 0.2, 0.6, 1.5, 1e9])
        report[name] = dict(
            residual=dict(
                mean=float(pooled.mean()),
                quantiles={str(a): q[a] for a in ALPHAS},
                per_run_max=per_run_max.tolist(),
                per_run_p95=per_run_p95.tolist(),
                samples=int(pooled.size),
            ),
            eps=eps,
            by_speed=speed_tab,
            by_turn=turn_tab,
        )
        lines += [f"## {name.upper()}-class commands", ""]
        lines.append(
            f"1. Disturbance bound ||v_com - v_cmd|| (m/s): mean {pooled.mean():.3f}; pooled quantile bound "
            + ", ".join(f"delta_{a:g} = {q[a]:.3f}" for a in ALPHAS)
            + f"; per-run max median {np.median(per_run_max):.3f} / worst {per_run_max.max():.3f} "
            f"(per-run p95 median {np.median(per_run_p95):.3f}); {pooled.size} samples."
        )
        lines += ["", "   | |v_cmd| bin | n | mean | p95 | max |", "   |---|---|---|---|---|"]
        for lo, hi, cnt, mu, p95, mx in speed_tab:
            lines.append(
                f"   | [{lo:.2f}, {hi:.2f}) | {cnt} | "
                + (
                    " | ".join(f"{v:.3f}" for v in (mu, p95, mx))
                    if mu is not None
                    else "-- | -- | --"
                )
                + " |"
            )
        lines += [
            "",
            "   | command turn rate bin [rad/s] | n | mean | p95 | max |",
            "   |---|---|---|---|---|",
        ]
        for lo, hi, cnt, mu, p95, mx in turn_tab:
            hi_s = "inf" if hi > 100 else f"{hi:.1f}"
            lines.append(
                f"   | [{lo:.1f}, {hi_s}) | {cnt} | "
                + (
                    " | ".join(f"{v:.3f}" for v in (mu, p95, mx))
                    if mu is not None
                    else "-- | -- | --"
                )
                + " |"
            )
        lines += [
            "",
            "2. Approximate-simulation distance eps(H) [m] (model re-anchored to the measured CoM at t):",
            "",
            "   | H [s] | p50 | p95 | sup |",
            "   |---|---|---|---|",
        ]
        for H in HORIZONS:
            e = eps[H]
            lines.append(f"   | {H:g} | {e['p50']:.3f} | {e['p95']:.3f} | {e['sup']:.3f} |")
        lines.append("")
    # 3. identification on the smooth (DI) battery -- the class the CBF commands in
    ident = (
        identify_lag(data["di"], plant.dt)
        if not TEST_MODE
        else dict(k=1.0, tau=0.0, delay=0.0, rms=0.0, rms_di=0.0)
    )
    gap = nu_gap(ident["k"], ident["tau"], ident["delay"])
    report["identified"] = ident
    report["nu_gap"] = {k: v for k, v in gap.items() if k not in ("w", "G_minus_1")}
    lines += ["## Identified CoM velocity response (DI battery)", ""]
    lines.append(
        f"v_com = k e^(-sL) / (tau s + 1) v_cmd with k = {ident['k']:.2f}, tau = {ident['tau']:.2f} s, "
        f"L = {ident['delay']:.2f} s; velocity RMS error {ident['rms']:.3f} m/s vs {ident['rms_di']:.3f} for the DI "
        f"assumption v_com = v_cmd."
    )
    lines.append(
        f"nu-gap(1/s, G_id/s) = {gap['nu_gap']:.3f} (at omega = {gap['at_omega']:.2f} rad/s; winding number "
        f"{gap['winding']:+.2f}); |G_id(jw) - 1| exceeds 0.3 above omega = {gap['bandwidth_G_minus_1_le_0p3']:.2f} rad/s."
    )
    text = "\n".join(lines)
    print("\n" + text)
    if TEST_MODE:
        return report
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "g1_model_distance.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(RESULTS_DIR, "g1_model_distance.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    _plot(data, report, gap, plant.dt)
    return report


def _plot(data, report, gap, dt):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    ax = axes[0]
    for name, color in (("di", "tab:blue"), ("si", "tab:orange")):
        res = residuals(data[name], dt)
        pooled = np.concatenate([r["norm"] for r in res])
        ax.hist(
            pooled,
            bins=60,
            range=(0, 0.8),
            density=True,
            alpha=0.5,
            color=color,
            label=f"{name} commands",
        )
        for a, ls in ((0.05, "--"), (0.01, ":")):
            ax.axvline(report[name]["residual"]["quantiles"][str(a)], color=color, ls=ls, lw=1)
    ax.set_xlabel("||v_com - v_cmd|| [m/s]")
    ax.set_title("1. disturbance residual (lines: delta_0.05, delta_0.01)")
    ax.legend()
    ax = axes[1]
    for name, color in (("di", "tab:blue"), ("si", "tab:orange")):
        rows = report[name]["by_speed"]
        xs = [0.5 * (lo + hi) for lo, hi, *_ in rows]
        ax.plot(
            xs,
            [r[4] if r[4] is not None else np.nan for r in rows],
            "o-",
            color=color,
            label=f"{name}: p95",
        )
        ax.plot(
            xs,
            [r[5] if r[5] is not None else np.nan for r in rows],
            "^--",
            color=color,
            alpha=0.6,
            label=f"{name}: max",
        )
    ax.set_xlabel("|v_cmd| [m/s]")
    ax.set_ylabel("||d|| [m/s]")
    ax.set_title("residual vs command speed")
    ax.legend(fontsize=8)
    ax = axes[2]
    for name, color in (("di", "tab:blue"), ("si", "tab:orange")):
        eps = report[name]["eps"]
        Hs = sorted(eps)
        ax.plot(Hs, [eps[H]["p95"] for H in Hs], "o-", color=color, label=f"{name}: p95")
        ax.plot(
            Hs, [eps[H]["sup"] for H in Hs], "^--", color=color, alpha=0.6, label=f"{name}: sup"
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("horizon H [s]")
    ax.set_ylabel("eps(H) [m]")
    ax.set_title("2. approximate-simulation distance")
    ax.legend(fontsize=8)
    ax = axes[3]
    w = np.asarray(gap["w"])
    ax.semilogx(w, gap["G_minus_1"], label="|G_id(jw) - 1|")
    ax.axhline(gap["nu_gap"], color="k", ls=":", label=f"nu-gap = {gap['nu_gap']:.2f}")
    ax.set_xlabel("omega [rad/s]")
    ax.set_title("3. identified lag vs the DI assumption")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "g1_model_distance.png")
    fig.savefig(path, dpi=140)
    print(f"saved {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runs", type=int, default=6, help="runs per command class")
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.runs, a.duration, a.seed)
