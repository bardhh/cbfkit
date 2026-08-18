# Unitree G1 (MJCF) — provenance and license

`g1.xml` and `scene.xml` are the [hydrax](https://github.com/vincekurtz/hydrax)
copies of the Unitree G1 description, which are adopted from the
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
`unitree_g1` model with hydrax's modifications (added `framequat` /
`framelinvel` IMU sensors and foot sites, a `stand` keyframe, and a
position-servo actuator set). They are redistributed here under the
BSD-3-Clause license in `LICENSE` (Copyright (c) 2016-2023 HangZhou YuShu
TECHNOLOGY CO., LTD. "Unitree Robotics").

The 51 mesh files (`assets/*.STL`, ~34 MB) are **not** vendored. They are
downloaded on first use from the Menagerie repository at the commit pinned in
`assets_manifest.json` and verified against the SHA-256 digests recorded there
(`cbfkit.systems.mujoco.assets.ensure_menagerie_assets`). Cache location:
`~/.cache/cbfkit/menagerie/<commit>/unitree_g1/`, or `$CBFKIT_ASSET_DIR/...`.
