#!/usr/bin/env bash
# Full experiment + README media. Use a fresh absolute output directory.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${CBFKIT_OUTPUT_DIR:?Set CBFKIT_OUTPUT_DIR to a fresh absolute output directory}"
[[ "$CBFKIT_OUTPUT_DIR" = /* ]] || { echo 'Output directory must be absolute' >&2; exit 1; }
mkdir -p "$CBFKIT_OUTPUT_DIR"
[[ -z "$(ls -A "$CBFKIT_OUTPUT_DIR")" ]] || { echo 'Use a fresh output directory' >&2; exit 1; }
compose=(docker compose -f "$script_dir/compose.sim6.yaml")
"${compose[@]}" build
"${compose[@]}" run --rm lab -m pytest /work/cbfkit/tests/test_wrappers/test_warehouse_model.py -q -o cache_dir=/tmp/pytest-cache
"${compose[@]}" run --rm lab /work/cbfkit/examples/isaac_lab/warehouse.py \
  --headless --enable_cameras --kit_args=--allow-root --device cuda:0 \
  --modes nominal,stop,filtered --seeds 42 --num_envs 4 --steps 750 \
  --frames --output /outputs/hero
"${compose[@]}" run --rm lab /work/cbfkit/examples/isaac_lab/warehouse_report.py \
  /outputs/hero --output /outputs/hero/aggregate.json
"${compose[@]}" run --rm lab -c 'import json; r=json.load(open("/outputs/hero/seed-42-filtered/result.json")); assert all(r["successful"]), "Filtered hero trials contain a contact, fall, or missed goal"'
"${compose[@]}" run --rm lab /work/cbfkit/examples/isaac_lab/warehouse.py \
  --headless --kit_args=--allow-root --device cuda:0 \
  --modes nominal,stop,filtered --seeds 7,123,2026 --num_envs 8 --steps 750 \
  --output /outputs/benchmark
"${compose[@]}" run --rm lab /work/cbfkit/examples/isaac_lab/warehouse_report.py \
  /outputs/benchmark --output /outputs/benchmark/aggregate.json
"${compose[@]}" run --rm lab /work/cbfkit/examples/isaac_lab/warehouse.py \
  --headless --enable_cameras --kit_args=--allow-root --device cuda:0 \
  --modes filtered --seeds 42 --num_envs 16 --steps 750 \
  --frames --all_frames --width 400 --height 225 --output /outputs/grid
"${compose[@]}" run --rm lab /work/cbfkit/examples/isaac_lab/warehouse_report.py \
  /outputs/grid --output /outputs/grid/aggregate.json
"${compose[@]}" run --rm lab /work/cbfkit/examples/isaac_lab/warehouse_video.py \
  /outputs/hero --grid /outputs/grid/seed-42-filtered \
  --output /outputs/warehouse-safety.mp4 --gif /outputs/warehouse-safety.gif
echo "WAREHOUSE_SHOWCASE_COMPLETE"
