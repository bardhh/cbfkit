"""Mesh provisioning for vendored MJCF models whose assets are too large to ship.

CBFKit vendors the small text parts of a robot description (``*.xml``,
``LICENSE``) under ``systems/mujoco/models/<robot>/`` together with an
``assets_manifest.json`` that pins a MuJoCo Menagerie commit and the SHA-256 of
every mesh. The meshes themselves are downloaded on first use into a cache
directory and verified against the manifest, so a layout change upstream fails
loudly instead of silently loading a different robot.

Cache layout::

    <root>/menagerie/<commit>/<robot>/assets/*.STL      (fetched, verified)
    <root>/menagerie/<commit>/<robot>/{scene.xml,g1.xml,LICENSE}   (copied from the package)

``<root>`` is ``~/.cache/cbfkit`` or ``$CBFKIT_ASSET_DIR``. Set
``CBFKIT_ASSETS_OFFLINE=1`` (or pass ``offline=True``) to forbid network access.
"""

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict

MENAGERIE_COMMIT = "da76818e269b82289eba39808e2fb91d679d6994"
_RAW = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{commit}/{robot}/assets/{name}"

MODELS_DIR = Path(__file__).parent / "models"
G1_MANIFEST = MODELS_DIR / "g1" / "assets_manifest.json"
UNITREE_RL_GYM_MANIFEST = MODELS_DIR / "g1" / "unitree_rl_gym_manifest.json"
AMO_MANIFEST = MODELS_DIR / "g1" / "amo_manifest.json"
GROOT_MANIFEST = MODELS_DIR / "g1" / "groot_manifest.json"
_RAW_GH = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
_MEDIA_GH = "https://media.githubusercontent.com/media/{repo}/{commit}/{path}"  # git-lfs


def asset_cache_root() -> Path:
    override = os.environ.get("CBFKIT_ASSET_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "cbfkit"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Fetch ``url`` to ``dest`` (module-level so tests can monkeypatch it)."""
    with urllib.request.urlopen(url, timeout=60) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def _load_manifest(manifest: Path) -> Dict:
    man = json.loads(Path(manifest).read_text())
    for key in ("commit", "robot", "files"):
        if key not in man:
            raise ValueError(f"{manifest}: manifest missing {key!r}")
    return man


def ensure_menagerie_assets(
    robot: str = "unitree_g1", *, manifest: Path, offline: bool = False
) -> Path:
    """Return the verified ``assets/`` directory for ``robot``, downloading what is missing.

    Raises ``RuntimeError`` when a file is missing and downloads are forbidden
    (``offline`` or ``CBFKIT_ASSETS_OFFLINE=1``), or when a downloaded file does
    not match the manifest checksum.
    """
    man = _load_manifest(manifest)
    if man["robot"] != robot:
        raise ValueError(f"manifest is for {man['robot']!r}, not {robot!r}")
    commit = man["commit"]
    offline = offline or bool(os.environ.get("CBFKIT_ASSETS_OFFLINE"))
    assets_dir = asset_cache_root() / "menagerie" / commit / robot / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for name, digest in man["files"].items():
        dest = assets_dir / name
        if dest.exists() and _sha256(dest) == digest:
            continue
        if offline:
            missing.append(name)
            continue
        url = _RAW.format(commit=commit, robot=robot, name=name)
        with tempfile.NamedTemporaryFile(dir=assets_dir, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            _download(url, tmp_path)
            got = _sha256(tmp_path)
            if got != digest:
                raise RuntimeError(
                    f"checksum mismatch for {name} from {url}: expected {digest[:12]}..., got {got[:12]}..."
                )
            os.replace(tmp_path, dest)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    if missing:
        raise RuntimeError(
            f"{len(missing)} {robot} mesh file(s) are missing from {assets_dir} and downloads are "
            f"disabled. Populate that directory (menagerie commit {commit[:12]}, "
            f"{robot}/assets/) or point CBFKIT_ASSET_DIR at a cache that has them; unset "
            f"CBFKIT_ASSETS_OFFLINE to allow downloading."
        )
    return assets_dir


def _materialise_model_dir(robot_pkg_dir: Path, assets_dir: Path) -> Path:
    """Copy the vendored text files next to ``assets/`` so ``from_xml_path`` resolves ``meshdir``."""
    model_dir = assets_dir.parent
    for src in robot_pkg_dir.iterdir():
        if src.suffix in (".xml",) or src.name in ("LICENSE", "README.md"):
            dst = model_dir / src.name
            if not dst.exists() or dst.read_bytes() != src.read_bytes():
                shutil.copyfile(src, dst)
    return model_dir


def g1_model_dir(offline: bool = False) -> Path:
    """Directory containing ``scene.xml``, ``g1.xml`` and verified ``assets/`` for the G1."""
    assets_dir = ensure_menagerie_assets("unitree_g1", manifest=G1_MANIFEST, offline=offline)
    return _materialise_model_dir(MODELS_DIR / "g1", assets_dir)


def ensure_repo_files(manifest: Path, *, offline: bool = False) -> Path:
    """Fetch (once) and verify every file listed in a GitHub-repo manifest; return the cache root.

    Manifest: ``{"repo": "owner/name", "commit": "<sha>", "files": {"<relpath>": "<sha256>", ...}}``.
    A file entry may also be ``{"sha256": "...", "lfs": true}`` -- LFS-tracked files are
    fetched from ``media.githubusercontent.com`` (the raw endpoint serves only the pointer).
    Files land at ``<cache>/<name>/<commit>/<relpath>``. Same offline/checksum
    rules as :func:`ensure_menagerie_assets`.
    """
    from urllib.parse import quote

    man = json.loads(Path(manifest).read_text())
    repo, commit = man["repo"], man["commit"]
    offline = offline or bool(os.environ.get("CBFKIT_ASSETS_OFFLINE"))
    root = asset_cache_root() / repo.split("/")[-1] / commit
    missing = []
    for rel, entry in man["files"].items():
        digest = entry["sha256"] if isinstance(entry, dict) else entry
        lfs = bool(entry.get("lfs")) if isinstance(entry, dict) else False
        dest = root / rel
        if dest.exists() and _sha256(dest) == digest:
            continue
        if offline:
            missing.append(rel)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        base = _MEDIA_GH if lfs else _RAW_GH
        url = base.format(repo=repo, commit=commit, path=quote(rel))
        with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            _download(url, tmp_path)
            got = _sha256(tmp_path)
            if got != digest:
                raise RuntimeError(
                    f"checksum mismatch for {rel} from {url}: expected {digest[:12]}..., got {got[:12]}..."
                )
            os.replace(tmp_path, dest)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    if missing:
        raise RuntimeError(
            f"{len(missing)} file(s) from {repo}@{commit[:12]} are missing under {root} and downloads "
            f"are disabled. Populate that directory or point CBFKIT_ASSET_DIR at a cache that has them; "
            f"unset CBFKIT_ASSETS_OFFLINE to allow downloading."
        )
    return root


def unitree_rl_gym_dir(offline: bool = False) -> Path:
    """Root of the cached ``unitree_rl_gym`` files (12-DoF G1 MJCF + meshes, deploy config, policy)."""
    return ensure_repo_files(UNITREE_RL_GYM_MANIFEST, offline=offline)


def amo_dir(offline: bool = False) -> Path:
    """Root of the cached AMO files (23-DoF G1 MJCF + meshes, whole-body policy + adapter weights).

    UCSD's AMO (Li, Cheng, Huang, Yang, Qiu, Wang -- RSS 2025, Apache-2.0), pinned by commit
    with per-file SHA-256; see ``models/g1/amo_manifest.json``.
    """
    return ensure_repo_files(AMO_MANIFEST, offline=offline)


def groot_dir(offline: bool = False) -> Path:
    """Root of the cached GR00T-WholeBodyControl files (29-DoF G1 MJCF + meshes + the
    released GEAR-WBC Balance/Walk ONNX policies).

    NVIDIA GR00T-WholeBodyControl (NVlabs, dual license: Apache-2.0 code + NVIDIA Open
    Model License for the checkpoints), pinned by commit with per-file SHA-256; LFS files
    fetched from the media endpoint. See ``models/g1/groot_manifest.json``.
    """
    return ensure_repo_files(GROOT_MANIFEST, offline=offline)
