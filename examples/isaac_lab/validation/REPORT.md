# Batched policy integration: validation record

The candidate was evaluated on current main (`2c2bfab`) with only the changes
in this feature. It does not depend on the older development branch's solver
recovery changes. The tested source snapshot is `b1c8676`; documentation and CI
were finalized afterward. The runtime and
test files match the SHA-256 values in [source-sha256.json](source-sha256.json).

## Environment and checks

Measured on 2026-09-07: native Ubuntu 24.04, RTX 3090 (24 GB), driver 595.84,
Python 3.12.13, PyTorch 2.10.0+cu128, JAX 0.6.2, and the default matrix precision
setting (no global override). Isaac Sim 6.0.1 and Isaac Lab
`v3.0.0-beta2.patch1` use the pins in the [Docker setup](../docker/README.md).
The evaluated derived image ID was
`sha256:35db1f5d5d085c0ad8d62d7623a7c86baefe67a6edb19da191feab4a5bc331cc`.

- CPU: 85 targeted tests passed; two CUDA-only tests skipped.
- RTX 3090: all 87 targeted tests passed, including CUDA ownership and
  warehouse precision regressions. See [checks.txt](checks.txt).
- Scope: wrapper tests, PDIPM QP tests, solver-failure tests, and controller
  failure-output tests. This was not a full repository test run.
- Default imports did not load PyTorch, Gymnasium, or Isaac Lab. CI now checks
  this and runs the optional bridge on CPU; that GitHub job has not run yet.
- JAX 0.4.23 / jaxlib 0.4.23 on Python 3.10, NumPy 1.26.4, SciPy 1.12.0:
  57 tests passed, with 12 Gymnasium-only tests skipped because the extra was
  absent. CI now includes this floor. The complete dependency matrix was not tested.
- The rebuilt image imported `/opt/cbfkit/src/cbfkit`, separately from pytest
  (whose conftest prefers mounted source). A real seven-step camera run passed
  the startup quaternion check and loaded trajectory rows at steps 5 and 7.

## Warehouse outcomes

Three seeds (7, 123, 2026), eight parallel robots per seed and controller,
750 control steps (15 seconds), without cameras: 72 robot trials total.
All variants use the same configured route follower, pretrained walking policy,
obstacle motion, and scenario parameters. They run sequentially in one
environment instance, with resets between modes and seeds. Matching those
settings and initial base positions does not establish equivalent hidden policy
or physics state, or reproducible trajectories across modes. The counts below
are descriptive outcomes, not a controlled estimate of controller effects.
See [method and limits](../WAREHOUSE.md).

| Controller | Trials | With obstacle contact | With a fall | Reached goal | Clean delivery |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nominal | 24 | 23 | 19 | 5 | 0 |
| Distance stop rule | 24 | 0 | 3 | 21 | 21 |
| CBF-QP | 24 | 0 | 2 | 22 | 22 |

A clean delivery requires goal entry and no detected contact or fall throughout
the run. Contact and fall counts can overlap. The stop baseline also avoided
every obstacle contact; its three unsuccessful trials were falls. Two startup
falls occurred in both the stop and filtered runs: seed 123/environment 2 and
seed 2026/environment 7, first logged at 1.1 and 1.2 seconds respectively.
They occurred without obstacle contact, at logged cart distances of 2.26–2.34 m,
outside the stop rule's 1.9 m threshold. The stop commands equal that run's own
nominal requests at every recorded startup sample. The corresponding nominal
mode fell later, at 3.5 and 4.2 seconds, with obstacle contact.

These are startup falls shared with the stop baseline; the experiment does
not attribute them to the filter or establish a filter regression. It also
does not establish that the filter outperforms stopping. The histories are
not literally identical: base positions and applied commands already differ
across modes at the first 0.1-second sample, and the filter's intervention flag
is set before both startup falls. Logging every five control steps cannot
establish equality of unlogged commands. See [startup-audit.json](startup-audit.json)
for the measurements and source trajectory hashes.

This rerun replaces the earlier 24/24 filtered result. Outcomes also changed in
the unmodified nominal and stop variants, so this comparison does not isolate
the precision change as the cause of the different outcomes. No extra tuning
or selection of a better rerun was performed.

Filtered runs had no fallback or command-residual violations at the configured
tolerance; the minimum residual was −1.78e−7. Median and p95 filter latency
were 2.77 and 4.93 ms for the entire eight-robot batch, including the bridge
and synchronization but excluding policy inference, physics, and rendering.
Median goal times among successful trials were 11.05 s (CBF) and 11.42 s
(stop); these condition on different success sets and are not a paired speedup.

This is one small scenario family, using exact simulator state and empirical
tracking margins. It does not establish real-world safety, perception
robustness, training benefits, or a general advantage over stop controllers.

[warehouse-summary.json](warehouse-summary.json) retains the generated aggregate.
[warehouse-outcomes.json](warehouse-outcomes.json) retains per-trial outcomes,
parameters, sensor coverage, and timing summaries. Merge `common` with each
entry in `trials` to reconstruct its fields. Per-step timing arrays are reduced
to count, mean, p50, and p95 (NumPy linear interpolation); trajectories and
camera frames are regenerated by the workflow rather than committed.

## Fresh-process repeatability check

The stop mode was run twice in independent Docker containers and Python
processes, each constructing a fresh environment: seed 123, eight robots,
250 control steps (five seconds), without cameras. Both used the same tested
runtime and image listed above. The 50 logged samples per run are byte-for-byte
identical, including robot/cart positions, requested/applied commands, and
contact/fall flags. All eight robots remained upright and without obstacle
contact in both runs. See [repeatability.json](repeatability.json) for comparisons
and hashes of both raw outputs.

This establishes repeatability at the logged resolution for this one mode,
seed, and five-second window. It does not demonstrate GPU nondeterminism.
It also does not establish equivalence across modes in the original sequential
benchmark or identify the cause of its shared startup falls. The original
process initialized with seed 7 and reused its environment for later seeds;
these fresh processes initialized with seed 123. The shorter test also sets
the episode horizon to six seconds instead of sixteen. Consequently this
check does not separate state carryover from initialization differences.
The original results retain their counts but support no causal comparison
between controllers. No runtime change or replacement of those trials was made.

To repeat the check after building the image, use a fresh output directory:

```bash
export CBFKIT_OUTPUT_DIR="$PWD/repeatability-output"
mkdir -p "$CBFKIT_OUTPUT_DIR"
for repetition in 1 2; do
  docker compose -f examples/isaac_lab/docker/compose.sim6.yaml run --rm lab \
    /work/cbfkit/examples/isaac_lab/warehouse.py --headless --kit_args=--allow-root \
    --device cuda:0 --modes stop --seeds 123 --num_envs 8 --steps 250 \
    --output "/outputs/repeat-$repetition"
done
cmp "$CBFKIT_OUTPUT_DIR/repeat-1/seed-123-stop/trajectory.jsonl" \
    "$CBFKIT_OUTPUT_DIR/repeat-2/seed-123-stop/trajectory.jsonl"
```

`cmp` exits zero for identical files. Wall-clock timings in `result.json` are
not expected to match and are excluded from the trajectory comparison.

## Matched bridge measurements

Both paths use float32 inputs, the same QP and solver settings, the same GPU,
and float64 controller inputs after normalization. Each measurement has two
warmup calls and 100 synchronized calls per batch size. Three repetitions alternate process order:
JAX/Torch, Torch/JAX, JAX/Torch. All 24 result rows report zero failed controls
and zero constraint violations. See [bridge-timings.jsonl](bridge-timings.jsonl).

| Batch size | JAX mean (ms) | Torch bridge mean (ms) | Added time (ms) |
| --- | ---: | ---: | ---: |
| 1 | 3.008 | 3.341 | 0.333 |
| 64 | 2.192 | 2.605 | 0.413 |
| 1,024 | 3.464 | 3.764 | 0.301 |
| 4,096 | 7.846 | 7.574 | −0.273 |

Values average the three run means. At batch 64 the measured increase is about
19%, so the adapter is not free. At batch 4,096 the bridged runs were faster,
with paired differences from −0.465 to −0.144 ms. This is an end-to-end timing
difference across separate processes, not a measurement of negative copy cost.
Three repetitions do not establish a general speedup or a universal overhead.
Use the measurements for this workload and remeasure the intended application.

Timing includes returned diagnostics and synchronization but excludes input
preparation and compilation. The bridge adds ownership copies on the current
device and output conversion; all controller inputs are normalized to float64
and the Torch result returns to the input action dtype. The direct JAX result
remains float64. Each result
row labels input, filter, and output dtype explicitly. The p95 uses NumPy
linear interpolation, matching the warehouse aggregate. This comparison
measures the incremental API cost, not an isolated memory-copy operation.
Peak memory was not measured. The adapter does not preserve policy gradients.

## Reproduce the measurements

From the repository root, with the [Docker prerequisites](../docker/README.md):

```bash
export CBFKIT_OUTPUT_DIR="$PWD/validation-output"
mkdir -p "$CBFKIT_OUTPUT_DIR"
compose=(docker compose -f examples/isaac_lab/docker/compose.sim6.yaml)
"${compose[@]}" build
"${compose[@]}" run --rm lab -m pytest /work/cbfkit/tests/test_wrappers /work/cbfkit/tests/test_optimization/test_pdipm_qp.py /work/cbfkit/tests/test_optimization/test_solver_failure.py /work/cbfkit/tests/test_controllers/test_cbf_clf_controllers/test_solver_failure_safety.py -q -o cache_dir=/tmp/pytest-cache
for repetition in 1 2 3; do
  order=(jax torch)
  if [[ "$repetition" = 2 ]]; then order=(torch jax); fi
  for mode in "${order[@]}"; do
    flags=()
    if [[ "$mode" = torch ]]; then flags=(--torch); fi
    "${compose[@]}" run --rm lab /work/cbfkit/benchmarks/batched_safety_filter.py --dtype float32 --steps 100 "${flags[@]}" > "$CBFKIT_OUTPUT_DIR/bridge-$repetition-$mode.jsonl"
  done
done
"${compose[@]}" run --rm lab /work/cbfkit/examples/isaac_lab/warehouse.py --headless --kit_args=--allow-root --device cuda:0 --modes nominal,stop,filtered --seeds 7,123,2026 --num_envs 8 --steps 750 --output /outputs/benchmark
"${compose[@]}" run --rm lab /work/cbfkit/examples/isaac_lab/warehouse_report.py /outputs/benchmark --output /outputs/benchmark/aggregate.json
```

The shell example uses Bash arrays. Hardware/runtime changes and GPU physics
can change timings and trajectories. The complete camera workflow is separate:
see [warehouse.sh and the HD rerender instructions](../WAREHOUSE.md).

## Showcase provenance

The README GIF is a retained recording from the development branch, with
its source commit and file hash in
[media-provenance.json](media-provenance.json). It illustrates the same
warehouse scenario, but that branch included additional solver changes and
used mixed precision. It was not rerendered from this focused candidate.
The large MP4 is excluded from the source repository; the HD script generates
it locally. The fresh evaluation above supplies evidence for the candidate's
runtime code; do not attribute the
GIF's individual trajectories to that evaluation. The separate short camera
check above verifies the revised recording path, not a new full showcase.
