#!/usr/bin/env bash
# Rerender the same showcase at native camera detail for a 3840x1800 MP4.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${CBFKIT_OUTPUT_DIR:?Set CBFKIT_OUTPUT_DIR to a fresh absolute output directory}"
[[ "$CBFKIT_OUTPUT_DIR" = /* ]] || { echo 'Output directory must be absolute' >&2; exit 1; }
mkdir -p "$CBFKIT_OUTPUT_DIR"
[[ -z "$(ls -A "$CBFKIT_OUTPUT_DIR")" ]] || { echo 'Use a fresh output directory' >&2; exit 1; }
compose=(docker compose -f "$script_dir/compose.sim6.yaml")
"${compose[@]}" build
"${compose[@]}" run --rm lab /work/cbfkit/examples/isaac_lab/warehouse.py \
  --headless --enable_cameras --kit_args=--allow-root --device cuda:0 \
  --modes nominal,filtered --seeds 42 --num_envs 4 --steps 750 \
  --frames --width 1920 --height 1080 --output /outputs/hero
"${compose[@]}" run --rm lab /work/cbfkit/examples/isaac_lab/warehouse_report.py \
  /outputs/hero --output /outputs/hero/aggregate.json
"${compose[@]}" run --rm lab /work/cbfkit/examples/isaac_lab/warehouse.py \
  --headless --enable_cameras --kit_args=--allow-root --device cuda:0 \
  --modes filtered --seeds 42 --num_envs 16 --steps 750 \
  --frames --all_frames --width 800 --height 450 --output /outputs/grid
"${compose[@]}" run --rm lab /work/cbfkit/examples/isaac_lab/warehouse_report.py \
  /outputs/grid --output /outputs/grid/aggregate.json
"${compose[@]}" run --rm lab /work/cbfkit/examples/isaac_lab/warehouse_video.py \
  /outputs/hero --grid /outputs/grid/seed-42-filtered --scale 3 \
  --output /outputs/warehouse-safety-4k.mp4
echo "WAREHOUSE_HD_COMPLETE"
