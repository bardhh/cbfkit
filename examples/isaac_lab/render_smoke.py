"""Verify the RTX camera path without Isaac Lab, CBFKit, or JAX imports.

Run with Isaac Sim's python.sh and --output /outputs/render-smoke.png.
A passing process must produce a nonblank RGB image, not just start the app.
"""

import argparse
import inspect
import json
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "renderer": "RaytracedLighting",
            "width": 640,
            "height": 360,
            "extra_args": ["--allow-root"],
        }
    )
    exit_code = 0
    try:
        import numpy as np
        import omni.replicator.core as rep
        from PIL import Image

        rep.create.cube(position=(0, 0, 0.5), scale=1)
        rep.create.light(light_type="Dome", intensity=1500)
        camera = rep.create.camera(position=(3, -3, 2), look_at=(0, 0, 0.5))
        product = rep.create.render_product(camera, (640, 360))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([product])
        for _ in range(4):
            rep.orchestrator.step(rt_subframes=4)
        rgb = np.asarray(annotator.get_data())[:, :, :3]
        if rgb.shape != (360, 640, 3) or float(rgb.std()) <= 5:
            raise RuntimeError(f"Invalid or blank camera output: {rgb.shape}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(args.output)
        print(
            "CBFKIT_RENDER_SMOKE "
            + json.dumps(
                {"shape": list(rgb.shape), "std": float(rgb.std()), "output": str(args.output)}
            ),
            flush=True,
        )
    except BaseException:
        exit_code = 1
        traceback.print_exc()
        raise
    finally:
        kwargs = (
            {"exit_code": exit_code}
            if "exit_code" in inspect.signature(app.close).parameters
            else {}
        )
        app.close(**kwargs)


if __name__ == "__main__":
    main()
