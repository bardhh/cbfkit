"""GR00T GEAR-WBC policy: onnx-free loading, JAX-vs-onnxruntime parity, plant, controller.

The parity oracle (onnxruntime) runs in a SUBPROCESS, keeping the pytest process free of
extra native runtimes (same policy as the torch oracle for AMO)."""

import importlib.util
import json
import subprocess
import sys
import urllib.error

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cbfkit.utils.user_types import ControllerData

groot = pytest.importorskip("cbfkit.systems.mujoco.groot_policy")

_ORACLE = r"""
import json, sys
import numpy as np
import onnxruntime as ort

d = sys.argv[1]
rng = np.random.default_rng(0)
x = rng.standard_normal((1, 516)).astype(np.float32)
out = {}
for name in ("Balance", "Walk"):
    s = ort.InferenceSession(d + f"/GR00T-WholeBodyControl-{name}.onnx")
    out[name] = s.run(None, {s.get_inputs()[0].name: x})[0].squeeze().tolist()
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def assets():
    from cbfkit.systems.mujoco.assets import groot_dir

    try:
        return groot_dir()
    except (RuntimeError, urllib.error.URLError) as exc:
        pytest.skip(f"GR00T assets unavailable: {exc}")


@pytest.fixture(scope="module")
def params(assets):
    return groot.load_groot_params()


def test_onnx_reader_recovers_all_tensors(params):
    for p in (params.balance, params.walk):
        assert p["estimator.0.weight"].shape == (256, 516)
        assert p["actor.6.weight"].shape == (15, 256)
    # the two checkpoints are genuinely different networks
    assert not jnp.allclose(params.balance["actor.0.weight"], params.walk["actor.0.weight"])


def test_policy_matches_onnxruntime(assets, params):
    if importlib.util.find_spec("onnxruntime") is None:
        pytest.skip("parity test needs onnxruntime as the oracle")
    proc = subprocess.run(
        [sys.executable, "-c", _ORACLE, str(assets / groot._REL_POLICY)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    golden = json.loads(proc.stdout.strip().splitlines()[-1])
    rng = np.random.default_rng(0)
    hist = rng.standard_normal((1, 516)).astype(np.float32).reshape(6, 86)
    for name, p in (("Balance", params.balance), ("Walk", params.walk)):
        got = np.asarray(groot.policy_forward(p, jnp.asarray(hist)))
        want = np.asarray(golden[name])
        assert np.allclose(got, want, atol=1e-4), (name, np.abs(got - want).max())


def test_plant_dims_and_pd(assets):
    plant = groot.make_g1_29dof_plant()
    assert (plant.nq, plant.nv, plant.nu) == (36, 35, 15)
    assert plant.dt == pytest.approx(0.02)
    x0 = groot.x0_standing(plant)
    assert x0.shape == (36 + 35 + 3,)
    assert float(x0[10]) == pytest.approx(0.3)  # left knee at the default angle


def test_controller_runs_carries_history_and_switches_policies(assets):
    plant = groot.make_g1_29dof_plant()
    policy = groot.GrootGearWbcPolicy()
    ctrl = policy.as_controller(torso_command=(0.1, 0.0, 0.0))
    x0 = groot.x0_standing(plant)
    u1, d1 = ctrl(0.0, x0, jnp.array([0.3, 0.0]), jax.random.PRNGKey(0), ControllerData())
    assert u1.shape == (15,) and bool(jnp.isfinite(u1).all())
    assert float(d1.sub_data["groot_cmd"][4]) == pytest.approx(0.1)  # torso roll
    assert float(d1.sub_data["groot_cmd"][3]) == pytest.approx(0.74)  # height
    u2, d2 = ctrl(0.02, x0, jnp.array([0.3, 0.0]), jax.random.PRNGKey(0), d1)
    st1, st2 = d1.sub_data["_groot"], d2.sub_data["_groot"]
    assert jnp.allclose(st2.hist[-2], st1.hist[-1])  # history rolled
    # zero command -> Balance policy branch: different action than Walk on the same state
    u_bal, _ = ctrl(0.04, x0, jnp.zeros(2), jax.random.PRNGKey(0), d2)
    assert not jnp.allclose(u_bal, u2)


@pytest.mark.slow
@pytest.mark.g1_mjx
def test_groot_stands_in_mjx(assets):
    """Acceptance: the Balance policy holds the 29-DoF G1 upright for 5 s in MJX."""
    plant = groot.make_g1_29dof_plant()
    policy = groot.GrootGearWbcPolicy()
    ctrl = jax.jit(policy.as_controller())
    data = plant.from_state(groot.x0_standing(plant))
    d = ControllerData()
    x = plant.to_state(data)
    for k in range(250):
        u, d = ctrl(k * plant.dt, x, jnp.zeros(2), jax.random.PRNGKey(0), d)
        data = plant.step(data, u)
        x = plant.to_state(data)
    assert float(x[2]) > 0.5, f"pelvis height {float(x[2]):.2f}"
    quat = np.asarray(x[3:7])
    assert 1 - 2 * (quat[1] ** 2 + quat[2] ** 2) > 0.9
