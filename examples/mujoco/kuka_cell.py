"""Showcase driver for the KUKA iiwa14 palletising cell: simulate once, render many times.

``simulate`` runs the cell (:mod:`cbfkit.systems.mujoco.kuka_cell`) with the CBF-QP on the
commanded joint velocities, or with ``--unfiltered`` the identical stack with an **empty
certificate collection** (the QP reduces to the projection onto the velocity box, so the
arm follows its pick-and-place cycle straight through the worker), and dumps every
per-step array a renderer needs into ``results/showcase/kuka_cell[_unfiltered].npz``:

    python examples/mujoco/kuka_cell.py simulate [--unfiltered] [--duration S] [--out DIR]
           [--seed N] [--worker-enter S]

``render`` hands everything after it to ``kuka_cell_render.py`` (``render --help`` prints
that module's options), filling in the npz paths from ``results/showcase/``:

    python examples/mujoco/kuka_cell.py render [--side-by-side] ...

The npz keeps the showcase core keys (``states, dt, plant_kind, nq, nv, com, n_live, h,
h_names, example, unfiltered, ...``; ``com`` is the tool xy so the G1 render helpers that
frame on it keep working) plus the arm-specific arrays ``qdot_nom``, ``qdot_safe``,
``worker`` ``(T, 4)``, ``tool`` ``(T, 3)`` and ``contact`` ``(T,)`` -- the last recomputed
offline with ``mj_forward`` on the CPU model: true on the steps where an arm geom
penetrates a worker geom. ``h`` is the barrier value the controller logged
(``sub_data_h``), which for this relative-degree-1 filter *is* the certificate value.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import jax  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import cbfkit.simulation.simulator as sim  # noqa: E402
from cbfkit.systems.mujoco import kuka_cell as kc  # noqa: E402

TEST_MODE = bool(os.getenv("CBFKIT_TEST_MODE"))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SHOWCASE_DIR = os.path.join(RESULTS_DIR, "showcase")
JAX_CACHE_DIR = os.path.expanduser("~/.cache/cbfkit/jax")
EXAMPLE = "kuka_cell"
DURATION = 10.0
UNFILTERED_TOL = 1e-3


def npz_path(out_dir: str, unfiltered: bool = False) -> str:
    """Where ``simulate`` writes (and ``render`` looks for) one run."""
    return os.path.join(out_dir, f"{EXAMPLE}{'_unfiltered' if unfiltered else ''}.npz")


def _controller_arrays(res: Any) -> Dict[str, np.ndarray]:
    """Every logged controller array under its bare name (``sub_data_`` prefix stripped)."""
    out: Dict[str, np.ndarray] = {}
    for key, val in res.controller_data.items():
        name = key[len("sub_data_") :] if key.startswith("sub_data_") else key
        arr = np.asarray(val)
        if arr.dtype == object:
            continue
        out[name] = arr
    return out


def _passthrough_dev(cd: Dict[str, Any], enforce: bool) -> float:
    """``max |qdot_safe - clip(qdot_nom)|`` over the non-error steps; raises when ``enforce``."""
    if enforce and "sub_data_bfs" in cd:
        raise RuntimeError("unfiltered run still logged sub_data_bfs: barriers were not removed")
    nom = np.asarray(cd["sub_data_qdot_nom"], dtype=float)
    safe = np.asarray(cd["sub_data_qdot_safe"], dtype=float)
    live = ~np.asarray(cd["error"], dtype=bool)
    if not live.any():
        raise RuntimeError("the controller errored on every step")
    lim = np.asarray(kc.QD_LIMITS)
    dev = float(np.abs(safe[live] - np.clip(nom[live], -lim, lim)).max())
    if enforce:
        print(f"unfiltered check: max |qdot_safe - clip(qdot_nom)| = {dev:.2e}")
        if not np.isfinite(dev) or dev > UNFILTERED_TOL:
            raise RuntimeError(
                f"unfiltered run is still filtering: max |qdot_safe - clip(qdot_nom)| = "
                f"{dev:.3e} > {UNFILTERED_TOL:.0e} over {int(live.sum())} non-error steps"
            )
    return dev


def contact_flags(mj_model: mujoco.MjModel, states: np.ndarray, worker: np.ndarray) -> np.ndarray:
    """Per logged step: does any arm geom penetrate any worker geom? (CPU ``mj_forward``.)"""
    data = mujoco.MjData(mj_model)
    worker_body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "worker")
    is_worker = mj_model.geom_bodyid == worker_body
    base_body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
    # arm bodies are the base and everything below it in the tree
    is_arm = np.zeros(mj_model.ngeom, dtype=bool)
    for g in range(mj_model.ngeom):
        b = mj_model.geom_bodyid[g]
        while b > 0:
            if b == base_body:
                is_arm[g] = True
                break
            b = mj_model.body_parentid[b]
    out = np.zeros(len(states), dtype=bool)
    for k in range(len(states)):
        data.qpos[:] = states[k, : mj_model.nq]
        w = worker[k]
        data.mocap_pos[0] = [w[0], w[1], 0.0]
        yaw = float(np.asarray(kc.worker_yaw(w[2:])))
        data.mocap_quat[0] = np.asarray(kc.worker_quat(yaw))
        mujoco.mj_forward(mj_model, data)
        for c in data.contact[: data.ncon]:
            if c.dist < 0 and (
                (is_arm[c.geom1] and is_worker[c.geom2]) or (is_arm[c.geom2] and is_worker[c.geom1])
            ):
                out[k] = True
                break
    return out


def simulate(
    out_dir: str, seed: int, duration: float, unfiltered: bool, worker_enter: Optional[float] = None
) -> str:
    """Run the cell and write the npz. Returns its path."""
    jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
    if worker_enter is not None:
        kc.WORKER_T_ENTER = float(worker_enter)
    plant, x0, controller, nominal = kc.build(unfiltered=unfiltered)
    steps = 5 if TEST_MODE else int(round(duration / plant.dt))
    t0 = time.time()
    res = sim.execute(
        x0=x0,
        dt=plant.dt,
        num_steps=steps,
        plant=plant,
        nominal_controller=nominal,
        controller=controller,
        key=jax.random.PRNGKey(seed),
        use_jit=True,
        verbose=not TEST_MODE,
    )
    wall = time.time() - t0
    cd = res.controller_data
    dev = _passthrough_dev(cd, enforce=unfiltered)
    states = np.asarray(res["states"])
    out = _controller_arrays(res)
    err = np.flatnonzero(np.asarray(cd["error"]))
    n_live = int(err[0]) if err.size else len(states)
    worker = np.asarray(out["worker"], dtype=float)
    contact = contact_flags(plant.mj_model, states, worker)
    out.update(
        states=states,
        com=np.asarray(out["tool"], dtype=float)[:, :2],
        dt=np.float64(plant.dt),
        plant_kind=np.array("iiwa14"),
        nq=np.int64(plant.nq),
        nv=np.int64(plant.nv),
        com_indices=np.asarray(plant.com_indices, dtype=np.int64),
        n_live=np.int64(n_live),
        h=np.asarray(out["h"], dtype=float),
        h_names=np.asarray(kc.H_NAMES),
        example=np.array(EXAMPLE),
        unfiltered=np.bool_(unfiltered),
        unfiltered_mode=np.array("nominal" if unfiltered else ""),
        seed=np.int64(seed),
        duration=np.float64(duration),
        wall_s=np.float64(wall),
        n_u=np.int64(kc.N_ARM),
        relax=np.bool_(not unfiltered),
        contact=contact,
        qp_dev_max=np.float64(dev),
        worker_capsules_r=np.array([r for _, _, r in kc.WORKER_CAPSULES]),
        arm_sphere_r=np.asarray(plant.arm_radii, dtype=float),
        arm_geoms=np.asarray(plant.arm_geoms, dtype=np.int64),
        d0=np.float64(kc.D0),
        tau=np.float64(kc.TAU),
        qd_limits=np.asarray(kc.QD_LIMITS, dtype=float),
        class_k=np.float64(kc.CLASS_K),
        worker_enter=np.float64(kc.WORKER_T_ENTER),
        config=np.array(
            f"kuka_cell.py simulate{' --unfiltered' if unfiltered else ''} "
            f"--worker-enter {kc.WORKER_T_ENTER}"
        ),
    )
    os.makedirs(out_dir, exist_ok=True)
    path = npz_path(out_dir, unfiltered)
    np.savez_compressed(path, **out)
    _summary(path, out)
    return path


def _summary(path: str, p: Dict[str, Any]) -> None:
    dt = float(p["dt"])
    n_live = int(p["n_live"])
    h = p["h"][:n_live]
    contact = p["contact"][:n_live]
    dev = np.abs(p["qdot_safe"][:n_live] - p["qdot_nom"][:n_live]).max(axis=1)
    active = float(np.mean(dev > 1e-3)) if n_live else 0.0
    first = np.flatnonzero(contact)
    label = f"{p['example']}{' (unfiltered)' if bool(p['unfiltered']) else ''}"
    print(
        f"{label}: T={len(p['states'])} steps ({len(p['states']) * dt:.1f} s), "
        f"n_live={n_live} ({n_live * dt:.1f} s), h_min(live)={float(h.min()) if h.size else float('nan'):+.3f}, "
        f"contact steps={int(contact.sum())}"
        + (f" (first at {first[0] * dt:.2f} s)" if first.size else "")
        + f", intervention {active * 100:.0f}% of live steps, wall {float(p['wall_s']):.1f} s -> {path}"
    )


def render(argv: List[str]):
    """Delegate to ``kuka_cell_render.main`` with the npz paths ``simulate`` wrote."""
    from examples.mujoco import kuka_cell_render as renderer

    argv = list(argv)
    if "--npz" not in argv:
        argv += ["--npz", npz_path(SHOWCASE_DIR)]
    if "--side-by-side" in argv and "--unfiltered-npz" not in argv:
        argv += ["--unfiltered-npz", npz_path(SHOWCASE_DIR, True)]
    return renderer.main(argv)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("simulate", help="run the cell and write the npz")
    s.add_argument("--unfiltered", action="store_true", help="empty certificate collection")
    s.add_argument("--out", default=SHOWCASE_DIR, help=f"output directory (default {SHOWCASE_DIR})")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--duration", type=float, default=DURATION, help="simulated seconds")
    s.add_argument("--worker-enter", type=float, default=None, help="override WORKER_T_ENTER (s)")
    sub.add_parser(
        "render", help="render (everything after is passed to kuka_cell_render)", add_help=False
    )
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "render":
        return render(argv[1:])
    args = parser.parse_args(argv)
    return simulate(args.out, args.seed, args.duration, args.unfiltered, args.worker_enter)


if __name__ == "__main__":
    main()
