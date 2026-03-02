"""Guardrails for example script import path bootstrapping."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"


def _python_file_texts():
    for path in EXAMPLES_ROOT.rglob("*.py"):
        yield path, path.read_text(encoding="utf-8")


def test_examples_do_not_use_sys_path_append() -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path, text in _python_file_texts()
        if "sys.path.append(" in text
    ]
    assert not offenders, (
        "Use sys.path.insert(0, ...) for example bootstrap paths; found append in: "
        + ", ".join(sorted(offenders))
    )


def test_examples_do_not_bootstrap_with_cwd() -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path, text in _python_file_texts()
        if "os.getcwd()" in text and "sys.path" in text
    ]
    assert not offenders, (
        "Use __file__-based project root resolution, not os.getcwd(); found in: "
        + ", ".join(sorted(offenders))
    )
