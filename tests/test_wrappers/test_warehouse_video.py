"""Validate recorded trajectories with a partial final logging interval."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

load_run = runpy.run_path(str(Path(__file__).parents[2] / "examples/isaac_lab/warehouse_video.py"))[
    "load_run"
]


@pytest.mark.parametrize("steps", [5, 7, 10])
def test_partial_final_interval_and_missing_interior_row(tmp_path, steps):
    count = steps // 5 + 1
    (tmp_path / "result.json").write_text(
        json.dumps(
            dict(
                steps=steps,
                rendered=True,
                frame_count=count,
                height=8,
                width=8,
            )
        )
    )
    expected = list(range(5, steps + 1, 5))
    if steps % 5:
        expected.append(steps)
    trajectory = tmp_path / "trajectory.jsonl"
    trajectory.write_text("\n".join(json.dumps({"step": s}) for s in expected))
    (tmp_path / "frames").mkdir()
    for i in range(count):
        rgb = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3) + i
        Image.fromarray(rgb).save(tmp_path / "frames" / f"env-00-{i:04d}.png")
    _, rows = load_run(tmp_path)
    assert [r["step"] for r in rows] == expected
    trajectory.write_text("\n".join(json.dumps({"step": s}) for s in expected[1:]))
    with pytest.raises(ValueError, match="Incomplete trajectory"):
        load_run(tmp_path)
