# Warehouse safety-filter showcase

A frozen ANYmal-C walking policy follows a delivery route while a cart crosses
the aisle. Compare the same route follower and walking policy with no filter,
a distance-based stop rule, and CBFKit's batched CBF-QP. Camera footage comes
from the actual PhysX simulation; the arrows show logged velocity commands.

The variants match configured scenarios but execute sequentially with resets
in one environment instance. Equivalent hidden policy/physics state and
repeatable trajectories across modes have not been established. The recorded
outcomes describe these runs; they do not isolate controller effects or prove
an advantage over the stop rule. See the [repeatability evidence](validation/REPORT.md).

## Reproduce

Use the [validated native Linux / RTX Docker setup](docker/README.md).
From the repository root, choose a fresh output directory:

```bash
export CBFKIT_OUTPUT_DIR="$PWD/warehouse-output"
bash examples/isaac_lab/docker/warehouse.sh
```

The script builds the pinned image, runs the moving-obstacle unit tests on the
GPU machine, records the paired hero runs, evaluates three further seeds, and
records 16 independent environments together. It validates the results before
assembling `warehouse-safety.gif` and `warehouse-safety.mp4`. The MP4 ends with
a grid of the separately simulated 16-environment batch. The GIF contains the
shorter comparison for the README. No new policy training is required.

The output contains per-trial `result.json` and `trajectory.jsonl`, aggregate
metrics, camera PNGs, and the finished media. Startup can take several minutes.
For a short development run, invoke `warehouse.py` with `--steps 300 --num_envs 2` and a
fresh output path inside `/outputs`.

### Higher-resolution video

To rerender the same scenario and layout at **3840×1800**, preserving the
original wide aspect ratio:

```bash
export CBFKIT_OUTPUT_DIR="$PWD/warehouse-4k-output"
bash examples/isaac_lab/docker/warehouse-hd.sh
```

This records new 1920×1080 hero camera frames and 800×450 frames for each of
the 16 grid cameras. The compositor draws text and annotations at the final
resolution and refuses camera inputs smaller than their output panels. The
22.6-second MP4 keeps the original 10 fps and ends with the same grid segment.
It repeats the two displayed controller variants and the grid; the separate
three-seed benchmark and stop baseline are not rerun for a resolution change.
As with other GPU physics reruns, individual trajectories may vary slightly.

## What the controllers do

All variants use the same goal-seeking route follower, with a maximum command
speed of 0.8 m/s and a lateral limit of 0.35 m/s. A frozen pretrained walking
policy converts the resulting body-frame commands to joint-position actions.
The CBF operates on world-frame planar velocity commands **before** the walking
policy. It does not filter the policy's joint actions.

The controller receives simulator state. Cameras are for visualization; there
is no camera-based perception or learned high-level navigation in this demo.
The cart has prescribed kinematic motion, identical between paired variants,
updated at each 5 ms physics step. It crosses through openings in the shelving.

For relative position `d = robot_xy - cart_xy`, the moving-obstacle barrier is
`h = dᵀd - 1.6²`. Its constraint is
`2 dᵀ(u - cart_velocity) + 1.25 h >= 0`. Two additional barriers constrain
the robot's base to the aisle. The QP minimizes command changes with objective
weights `[1, 8]`, favoring longitudinal yielding over sustained sidestepping.
The filter normalizes controller inputs to float64, matching the scalar API.
This avoids TF32 rounding of float32 barrier/drift products without changing
process-global matrix precision. A recorded state is covered by a regression
test that deliberately enables TF32, and post-conversion residuals are checked
in float64.

The stop baseline commands zero when cart-center distance is below 1.9 m and
resumes the same route follower when the cart clears that threshold.

The 0.78 m ring drawn around the robot is an illustrative body envelope;
the actual barrier uses the 1.6 m robot/cart center separation above. It is not a contact
sensor, a perception result, or a certified reachable set.

## Measurements and limits

- Five one-to-many contact sensors cover the cart and four rack sections
  against all 17 robot links, including feet, shanks, thighs, hips, and base.
  The script checks link coverage and refuses missing contact channels.
- Contact history updates every physics step. A detected obstacle contact is
  a normal force above 1 N in any channel; intended floor contacts are excluded.
- Falls are detected from base height below 0.25 m or tilted projected gravity.
  Fallen robots are retained instead of being reset out of the results.
- Goal entry is within 0.35 m of the delivery target. A successful trial must
  reach the goal without a detected obstacle contact or fall over the full run.
- Filter timing includes GPU synchronization and the Torch/JAX bridge. It
  excludes simulation, rendering, metric collection, and the first two calls.
  Rendering throughput is not inferred from that filter timing.

The inflated geometry is an empirical margin. The single-integrator CBF does
not certify the articulated robot, tracking error, sampled implementation, or
perception. Actual contacts and falls are evaluated separately. The hero case
is seed 42, environment zero, fixed in the scenario generator. The benchmark
uses seeds 7, 123, and 2026; it is a small scenario study, not a general safety
guarantee. The original equal-weight prototype included a fall without obstacle
contact, motivating the revised motion cost and lateral-speed limits.

See [recorded validation and limits](validation/REPORT.md). The HD workflow
produces an MP4 locally; large videos are excluded from the source repository.
