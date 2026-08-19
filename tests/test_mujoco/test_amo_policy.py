"""AMO whole-body policy: torch-free loading, JAX-vs-torch parity, plant, and controller."""

import urllib.error

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cbfkit.utils.user_types import ControllerData

amo = pytest.importorskip("cbfkit.systems.mujoco.amo_policy")
torch = pytest.importorskip("torch", reason="parity tests need torch as the oracle")


@pytest.fixture(scope="module")
def assets():
    from cbfkit.systems.mujoco.assets import amo_dir

    try:
        return amo_dir()
    except (RuntimeError, urllib.error.URLError) as exc:
        pytest.skip(f"AMO assets unavailable: {exc}")


@pytest.fixture(scope="module")
def params(assets):
    return amo.load_amo_params()


def test_torch_free_readers_recover_all_tensors(assets, params):
    assert params.policy["student_actor_backbone.0.weight"].shape == (1024, 2474)
    assert params.adapter["model.1.running_mean"].shape == (512,)
    assert params.input_mean.shape == (12,) and params.output_std.shape == (15,)


def test_adapter_matches_torch(assets, params):
    ad = torch.jit.load(str(assets / "adapter_jit.pt"), map_location="cpu")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 12)).astype(np.float32)
    ns = torch.load(str(assets / "adapter_norm_stats.pt"), map_location="cpu", weights_only=False)
    xin = (torch.tensor(x) - torch.tensor(ns["input_mean"], dtype=torch.float32)) / (
        torch.tensor(ns["input_std"], dtype=torch.float32) + 1e-8
    )
    want = ad(xin).detach().numpy() * ns["output_std"] + ns["output_mean"]
    got = np.stack([np.asarray(amo.adapter_forward(params, jnp.asarray(r))) for r in x])
    assert np.allclose(got, want, atol=1e-4), np.abs(got - want).max()


def test_policy_matches_torch(assets, params):
    """Drive the torch sub-modules exactly as the TorchScript top-level graph does
    (its own forward has a baked cuda zeros and cannot run on CPU) and compare."""
    pol = torch.jit.load(str(assets / "amo_jit.pt"), map_location="cpu")
    oa = pol._orig_actor
    elu = torch.nn.ELU()
    rng = np.random.default_rng(1)
    prop = rng.standard_normal(93).astype(np.float32)
    demo = rng.standard_normal(17).astype(np.float32)
    hist = rng.standard_normal((10, 93)).astype(np.float32)
    extra = rng.standard_normal((25, 93)).astype(np.float32)
    with torch.no_grad():
        feat = oa.text_feat_merger(
            elu, oa.text_feat_encoder(elu, torch.tensor(hist[-4:])).view(1, -1)
        )
        hist_enc = oa.history_encoder(elu, torch.tensor(hist).unsqueeze(0))
        inp = torch.cat(
            [
                torch.tensor(extra.reshape(1, -1)),
                feat,
                torch.tensor(prop[None]),
                torch.tensor(demo[None]),
                torch.zeros(1, 3),
                hist_enc,
            ],
            1,
        )
        want = pol.student_actor_backbone(inp).numpy().squeeze()
    got = np.asarray(
        amo.policy_forward(
            params, jnp.asarray(prop), jnp.asarray(demo), jnp.asarray(hist), jnp.asarray(extra)
        )
    )
    assert np.allclose(got, want, atol=1e-4), np.abs(got - want).max()


def test_plant_dims_keyframe_and_pd(assets):
    plant = amo.make_g1_23dof_plant()
    assert (plant.nq, plant.nv, plant.nu) == (30, 29, 23)
    assert plant.dt == pytest.approx(0.02)
    x0 = amo.x0_standing(plant)
    assert x0.shape == (30 + 29 + 3,)
    assert float(x0[2]) == pytest.approx(1.0)  # keyframe pelvis height
    assert float(x0[25]) == pytest.approx(1.2)  # keyframe left elbow


def test_controller_runs_and_carries_state(assets):
    plant = amo.make_g1_23dof_plant()
    policy = amo.AmoWholeBodyPolicy()
    ctrl = policy.as_controller(torso_command=(0.0, 0.5, 0.0, 0.0))
    x0 = amo.x0_standing(plant)
    u1, d1 = ctrl(0.0, x0, jnp.array([0.3, 0.0]), jax.random.PRNGKey(0), ControllerData())
    assert u1.shape == (23,) and bool(jnp.isfinite(u1).all())
    assert jnp.allclose(u1[15:], policy.q_default[15:])  # arms PD-held at default
    assert float(d1.sub_data["amo_cmd"][4]) == pytest.approx(0.5)  # torso yaw command
    assert float(d1.sub_data["amo_cmd"][1]) == pytest.approx(0.0)  # heading = command direction
    u2, d2 = ctrl(0.02, x0, jnp.array([0.3, 0.0]), jax.random.PRNGKey(0), d1)
    st1, st2 = d1.sub_data["_amo"], d2.sub_data["_amo"]
    assert not jnp.allclose(st2.hist, st1.hist)  # history advanced
    assert jnp.allclose(st2.hist[-1], st2.extra_hist[-1])  # both buffers end with this step's frame
    assert jnp.allclose(st2.hist[-2], st1.hist[-1])  # ... and the previous frame shifted down
    # standing command holds the last heading instead of snapping to atan2(0, 0)
    _, d3 = ctrl(0.04, x0, jnp.array([0.0, 0.0]), jax.random.PRNGKey(0), d2)
    assert float(d3.sub_data["amo_cmd"][1]) == pytest.approx(0.0)
    assert bool(d3.sub_data["_amo"].stand)


@pytest.mark.slow
def test_amo_stands_in_mjx(assets):
    """Acceptance: the AMO policy holds the G1 upright for 5 s in MJX at zero command."""
    plant = amo.make_g1_23dof_plant()
    policy = amo.AmoWholeBodyPolicy()
    ctrl = jax.jit(policy.as_controller())
    data = plant.from_state(amo.x0_standing(plant))
    d = ControllerData()
    x = plant.to_state(data)
    for k in range(250):  # 5 s at 50 Hz
        u, d = ctrl(k * plant.dt, x, jnp.zeros(2), jax.random.PRNGKey(0), d)
        data = plant.step(data, u)
        x = plant.to_state(data)
    assert float(x[2]) > 0.55, f"pelvis height {float(x[2]):.2f}"
    quat = np.asarray(x[3:7])
    upright = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)
    assert upright > 0.9
