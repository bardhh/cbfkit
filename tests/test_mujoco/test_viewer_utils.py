"""Replay / marker / GIF helpers on the cart-pole model (no downloaded assets needed)."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from cbfkit.systems.mujoco import MujocoPlant, load_model
from cbfkit.systems.mujoco.viewer_utils import add_marker, render_gif, replay_states


@pytest.fixture(scope="module")
def cart_pole():
    plant = MujocoPlant(load_model("cart_pole"))
    states = np.zeros((6, plant.state_dim))
    states[:, 0] = np.linspace(0.0, 0.5, 6)  # cart slides
    return plant, states


def test_replay_states_yields_forwarded_data_per_row(cart_pole):
    plant, states = cart_pole
    out = list(replay_states(plant, states))
    assert [k for _, k in out] == list(range(6))
    assert out[-1][0].qpos[0] == pytest.approx(0.5)


def test_add_marker_appends_visual_geom_until_scene_is_full():
    m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='.1'/></worldbody></mujoco>")
    scn = mujoco.MjvScene(m, maxgeom=2)
    ok = [
        add_marker(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.1, 0, 0], [0, 0, 0], [1, 0, 0, 1])
        for _ in range(3)
    ]
    assert ok == [True, True, False] and scn.ngeom == 2


def test_render_gif_writes_a_file_or_reports_no_offscreen(cart_pole, tmp_path):
    plant, states = cart_pole
    seen = []
    path = render_gif(
        plant,
        states,
        str(tmp_path / "x.gif"),
        track_body=0,
        markers=lambda scn, k, t: seen.append((k, t)),
        fps=50,
    )
    if path is None:
        pytest.skip("offscreen rendering unavailable on this machine")
    assert (tmp_path / "x.gif").stat().st_size > 0
    assert seen and seen[0] == (0, 0.0)
