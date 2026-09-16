"""Smoke-test a built distribution from a directory outside the source checkout."""

import argparse
import importlib
import json
import sys
import tempfile
from importlib.resources import files
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mujoco", action="store_true")
    args = parser.parse_args()

    import cbfkit
    import jax.numpy as jnp

    root = files("cbfkit")
    checkout = Path(__file__).resolve().parents[1]
    assert not Path(cbfkit.__file__).resolve().is_relative_to(checkout), cbfkit.__file__
    assert cbfkit.__version__ == root.joinpath("VERSION").read_text().strip()
    assert root.joinpath("py.typed").is_file()
    for template in ("plant", "controller", "barrier", "lyapunov", "cost", "init", "ros2_controller_node"):
        assert root.joinpath(f"codegen/templates/{template}.py.j2").is_file()
    for name in ("assets_manifest", "unitree_rl_gym_manifest", "amo_manifest", "groot_manifest"):
        json.loads(root.joinpath(f"systems/mujoco/models/g1/{name}.json").read_text())
    for asset in ("g1/g1.xml", "g1/scene.xml", "g1/LICENSE", "cart_pole/cart_pole.xml", "cart_pole/NOTICE"):
        assert root.joinpath(f"systems/mujoco/models/{asset}").is_file()

    from cbfkit.codegen.create_new_system.generate_model import generate_model

    with tempfile.TemporaryDirectory() as tmp:
        generate_model(tmp, "smoke_model", "[0.0]", "[[1.0]]")
        sys.path.insert(0, tmp)
        try:
            model = importlib.import_module("smoke_model.plant")
            f, g = model.plant()(jnp.zeros(1))
            assert f.shape == (1,) and g.shape == (1, 1)
        finally:
            sys.path.remove(tmp)

    if args.mujoco:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(root.joinpath("systems/mujoco/models/cart_pole/scene.xml")))
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        assert model.nu > 0
    print(f"Installed distribution verified: {cbfkit.__version__} ({cbfkit.__file__})")


if __name__ == "__main__":
    main()
