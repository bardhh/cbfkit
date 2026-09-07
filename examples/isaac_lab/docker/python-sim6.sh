#!/usr/bin/env bash
set -euo pipefail
# Prioritize the CUDA libraries installed with Lab's PyTorch over any older
# simulator prebundle. Keep the simulator launcher responsible for Kit paths.
cuda_library_dirs=""
for cuda_library_dir in /isaac-sim/kit/python/lib/python3.12/site-packages/nvidia/*/lib /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/*/lib; do
    if [[ -d "$cuda_library_dir" ]]; then
        cuda_library_dirs="${cuda_library_dirs:+$cuda_library_dirs:}$cuda_library_dir"
    fi
done
export LD_LIBRARY_PATH="${cuda_library_dirs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec /isaac-sim/python.sh "$@"
