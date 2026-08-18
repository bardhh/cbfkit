"""Menagerie asset provisioning: manifest, cache, checksum, offline behaviour."""

import json
import urllib.error

import mujoco
import pytest

from cbfkit.systems.mujoco import assets
from cbfkit.systems.mujoco.assets import (
    MENAGERIE_COMMIT,
    ensure_menagerie_assets,
    g1_model_dir,
)

MANIFEST = assets.G1_MANIFEST


def test_manifest_pins_commit_and_lists_51_meshes():
    man = json.loads(MANIFEST.read_text())
    assert man["commit"] == MENAGERIE_COMMIT
    assert man["robot"] == "unitree_g1"
    assert len(man["files"]) == 51
    assert all(len(h) == 64 for h in man["files"].values())


def test_offline_with_empty_cache_raises_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CBFKIT_ASSET_DIR", str(tmp_path))
    with pytest.raises(RuntimeError) as exc:
        ensure_menagerie_assets("unitree_g1", manifest=MANIFEST, offline=True)
    msg = str(exc.value)
    assert "CBFKIT_ASSET_DIR" in msg and str(tmp_path) in msg


def test_checksum_mismatch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("CBFKIT_ASSET_DIR", str(tmp_path))
    # Serve wrong bytes for every file: the fetcher must refuse them.
    monkeypatch.setattr(assets, "_download", lambda url, dest: dest.write_bytes(b"not a mesh"))
    with pytest.raises(RuntimeError, match="checksum"):
        ensure_menagerie_assets("unitree_g1", manifest=MANIFEST)


def test_g1_model_dir_loads_in_mujoco():
    try:
        d = g1_model_dir()
    except (RuntimeError, urllib.error.URLError) as exc:  # no cache and no network
        pytest.skip(f"G1 assets unavailable: {exc}")
    assert (d / "scene.xml").exists() and (d / "g1.xml").exists() and (d / "assets").is_dir()
    m = mujoco.MjModel.from_xml_path(str(d / "scene.xml"))
    assert (m.nq, m.nv, m.nu) == (36, 35, 29)
