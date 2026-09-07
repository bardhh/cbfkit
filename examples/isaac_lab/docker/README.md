# Native NVIDIA Docker setup

The supported example pairs Isaac Sim 6.0.1 with Isaac Lab
`v3.0.0-beta2.patch1`, commit `ffff603eafc6b74264a5261cc0183d6a65390d78`.
The simulator image is pinned by digest in `Dockerfile.sim6`; the runtime uses
PyTorch 2.10.0 / CUDA 12.8 and JAX 0.6.2. Tested hardware: native Ubuntu 24.04,
RTX 3090 (24 GB), NVIDIA driver 595.84. Lab 3 is a beta release.

Install Docker and NVIDIA Container Toolkit on the host. Use the build's
PyTorch/JAX pairing: substituting independently installed CUDA libraries can
break cuDNN or cuSolver loading. The launcher exposes Lab's bundled libraries.
The internal `isaaclab` package version is 6.1.14, distinct from its release tag.

From the repository root:

```bash
export CBFKIT_OUTPUT_DIR="$PWD/warehouse-output"
bash examples/isaac_lab/docker/warehouse.sh
```

Choose a fresh absolute output directory. Startup, shader compilation, and
asset downloads can take several minutes. The workflow records paired
comparisons, evaluates more seeds, checks contacts/falls/constraint residuals,
records 16 cameras, and assembles a GIF and MP4. See [the demo guide](../WAREHOUSE.md).
For a native 3840×1800 rerender, run
`bash examples/isaac_lab/docker/warehouse-hd.sh` instead.

The source library is installed in the image; scripts and tests are mounted
read-only. Rebuild when library code changes for example execution. Pytest's
root conftest instead prioritizes the mounted source; rebuild and verify the
installed import separately when validating the image itself. Only outputs and simulator cache
volumes are writable; the container has no Docker socket or host-network mount.
These containers run as root internally, so generated host files may be root-owned.

## Diagnose rendering

Isaac Sim 5.1 crashed during RTX startup on the tested driver, including in an
unmodified NVIDIA image. Sim 6 rendered successfully on the unchanged driver.
Only the supported Sim 6 setup is included here.

To isolate a rendering problem from the robot and JAX integration:

```bash
export CBFKIT_OUTPUT_DIR="$PWD/render-check"
mkdir -p "$CBFKIT_OUTPUT_DIR"
docker compose -f examples/isaac_lab/docker/compose.sim6.yaml run --rm render-smoke \
  /work/cbfkit/examples/isaac_lab/render_smoke.py --output /outputs/render-smoke.png
```

The robot example uses registered Lab camera sensors to synchronize renderer
readback with physics. Each capture checks for nonblank output and verifies
that rendering did not advance the robot base. The compositor rejects missing
or frozen recordings. Exceptions preserve a nonzero exit code through Sim 6's
default fast shutdown; full extension teardown previously crashed after recording.
