"""Smoke test for ``examples/mujoco/kuka_cell_render.py`` on synthetic npz files.

No simulation: each npz has the key layout ``kuka_cell.py simulate`` writes (the home pose
held for a few steps, the scripted worker walking, ``h`` from ``barrier_values``), then the
driver renders it -- model build, overlays, camera, HUD and both encoders. Skipped when the
iiwa14 asset cache is absent (downloads are forbidden here) or when a Linux host has no
offscreen GL backend configured.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
import jax.numpy as jnp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cbfkit.systems.mujoco import kuka_cell as kc  # noqa: E402
from examples.mujoco import kuka_cell_render as renderer  # noqa: E402

pytestmark = pytest.mark.slow

N_STEPS = 30
DT = 0.02


@pytest.fixture(scope="module", autouse=True)
def _offline():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CBFKIT_ASSETS_OFFLINE", "1")
        mp.setenv("MPLBACKEND", "Agg")
        if sys.platform.startswith("linux") and not os.environ.get("MUJOCO_GL"):
            pytest.skip("no offscreen GL backend configured (set MUJOCO_GL=egl or osmesa)")
        try:
            kc.load_cell(offline=True)
        except RuntimeError as exc:
            pytest.skip(f"iiwa14 assets unavailable: {exc}")
        yield


def _write_fake(path: Path, unfiltered: bool) -> None:
    plant = kc.make_plant(offline=True)
    x = np.asarray(plant.to_state(plant.make_data().replace(qpos=jnp.asarray(kc.Q_HOME))))
    states = np.repeat(x[None], N_STEPS, axis=0)
    t = np.arange(N_STEPS) * DT + 2.5  # the worker is already walking
    worker = np.stack([np.asarray(kc.worker_state(ti)) for ti in t])
    h = np.stack([np.asarray(kc.barrier_values(plant, x[:7], w)) for w in worker])
    tool = np.repeat(np.asarray(kc.tool_pose(plant, x[:7])[0])[None], N_STEPS, axis=0)
    qdot = np.zeros((N_STEPS, 7))
    contact = np.zeros(N_STEPS, dtype=bool)
    contact[N_STEPS // 2] = True  # exercise the contact flash and pill
    np.savez_compressed(
        path,
        states=states,
        com=tool[:, :2],
        dt=np.float64(DT),
        plant_kind=np.array("iiwa14"),
        nq=np.int64(7),
        nv=np.int64(7),
        n_live=np.int64(N_STEPS - 5 if unfiltered else N_STEPS),  # one frozen tail
        h=h,
        h_names=np.asarray(kc.H_NAMES),
        example=np.array("kuka_cell"),
        unfiltered=np.bool_(unfiltered),
        unfiltered_mode=np.array("nominal" if unfiltered else ""),
        n_u=np.int64(7),
        relax=np.bool_(False),
        error=np.zeros(N_STEPS, dtype=bool),
        qdot_nom=qdot + 0.2,
        qdot_safe=qdot,
        worker=worker,
        tool=tool,
        contact=contact,
        d0=np.float64(kc.D0),
    )


def test_render_single_and_side_by_side(tmp_path):
    cbf = tmp_path / "kuka_cell.npz"
    unf = tmp_path / "kuka_cell_unfiltered.npz"
    _write_fake(cbf, unfiltered=False)
    _write_fake(unf, unfiltered=True)
    out = renderer.main(
        [
            "--npz",
            str(cbf),
            "--unfiltered-npz",
            str(unf),
            "--out",
            str(tmp_path),
            "--side-by-side",
            "--gif-width",
            "240",
        ]
    )
    names = sorted(Path(p).name for p in out)
    assert names == [
        "kuka_cell.gif",
        "kuka_cell.mp4",
        "kuka_cell_side_by_side.gif",
        "kuka_cell_side_by_side.mp4",
    ]
    assert all(Path(p).stat().st_size > 0 for p in out)


def test_stills_and_bad_example(tmp_path):
    cbf = tmp_path / "kuka_cell.npz"
    _write_fake(cbf, unfiltered=False)
    out = renderer.main(["--npz", str(cbf), "--out", str(tmp_path), "--stills"])
    assert len(out) == 5 and all(Path(p).suffix == ".png" for p in out)
    with pytest.raises(ValueError, match="side_by_side"):
        renderer.render(str(cbf), side_by_side=True)
