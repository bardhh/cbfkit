import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cbfkit.utils.user_types import ControllerData
from cbfkit.wrappers import BatchedSafetyFilter
from cbfkit.wrappers.torch import TorchSafetyFilter


def controller(t, x, u, key, data):
    prev = 0 if data.sub_data is None else data.sub_data["prev"]
    return u + prev, ControllerData(sub_data={"prev": u})


def test_torch_dtype_ownership_and_selective_reset():
    sf = TorchSafetyFilter(BatchedSafetyFilter(controller, num_envs=2))
    x = torch.zeros((2, 1), dtype=torch.float32)
    action = torch.ones_like(x, requires_grad=True)
    first, info = sf.filter(x, action)
    assert first.dtype == action.dtype and first.device == action.device
    assert not first.requires_grad
    assert info["fallback_used"].dtype == torch.bool
    first.fill_(100)
    with torch.no_grad():
        action.fill_(2)
    second, _ = sf.filter(x, action)
    np.testing.assert_array_equal(second, [[3], [3]])
    sf.reset(torch.tensor([True, False]))
    np.testing.assert_array_equal(sf.filter(x, action)[0], [[2], [4]])


def test_noncontiguous_inputs_and_invalid_dtype():
    sf = TorchSafetyFilter(BatchedSafetyFilter(controller, num_envs=2))
    x = torch.zeros((1, 2)).T
    u = torch.arange(4.0, dtype=torch.float64).reshape(2, 2)[:, ::2]
    np.testing.assert_array_equal(sf.filter(x, u)[0], u)
    with pytest.raises(ValueError):
        sf.filter(x, u.long())


def test_nested_diagnostics_and_applied_action_alias():
    def nested_controller(t, x, u, key, data):
        return u, ControllerData(sub_data={"solver_status": (x, [u]), "bfs": {"edge": x}})

    sf = TorchSafetyFilter(BatchedSafetyFilter(nested_controller, num_envs=2))
    x = torch.ones((2, 1))
    applied, info = sf.filter(x, x)
    assert info["u_applied"] is applied
    torch.testing.assert_close(info["solver_status"][0], x.double())
    torch.testing.assert_close(info["solver_status"][1][0], x.double())
    torch.testing.assert_close(info["barrier_values"]["edge"], x.double())


def test_missing_torch_import_has_install_guidance():
    import subprocess
    import sys

    script = """
import sys
sys.modules["torch"] = None
try:
    import cbfkit.wrappers.torch
except ImportError as exc:
    assert "cbfkit[torch]" in str(exc), str(exc)
else:
    raise AssertionError("Missing optional runtime was not rejected")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA PyTorch is required")
def test_cuda_device_and_reset():
    import jax

    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("Run with CUDA-enabled JAX and JAX_PLATFORMS=cuda")
    sf = TorchSafetyFilter(BatchedSafetyFilter(controller, num_envs=2))
    x = torch.zeros((2, 1), device="cuda")
    u = torch.ones_like(x)
    result, info = sf.filter(x, u)
    torch.testing.assert_close(result, u)
    assert info["fallback_used"].device == u.device
    sf.reset(torch.tensor([True, False], device="cuda"))
    torch.testing.assert_close(sf.filter(x, u)[0], torch.tensor([[1.0], [2.0]], device="cuda"))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_qp_float32_controls_and_reset(device):
    import jax
    import jax.numpy as jnp

    from cbfkit.certificates import generate_certificate
    from cbfkit.certificates.conditions.barrier_conditions.zeroing_barriers import linear_class_k
    from cbfkit.optimization.quadratic_program import get_solver
    from cbfkit.systems.single_integrator.dynamics import two_dimensional_single_integrator

    if device == "cuda" and (
        not torch.cuda.is_available() or not any(d.platform == "gpu" for d in jax.devices())
    ):
        pytest.skip("Both CUDA PyTorch and CUDA JAX are required")
    sf = TorchSafetyFilter(
        BatchedSafetyFilter.from_cbf_qp(
            num_envs=4,
            dynamics=two_dimensional_single_integrator(),
            barriers=generate_certificate(lambda x: x[0], linear_class_k(1.0), input_style="state"),
            control_limits=jnp.ones(2),
            solver=get_solver("fast", tol=1e-10, max_iter=40),
        )
    )
    x = torch.tensor([[0.05, 0.0], [0.1, 0.0], [1.0, 0.0], [2.0, 0.0]], device=device)
    nominal = torch.tensor([[-1.0, 0.0]], device=device).repeat(4, 1)
    for step in range(4):
        if step == 2:
            sf.reset(torch.tensor([False, True, False, True], device=device))
        u, info = sf.filter(x, nominal)
        assert u.dtype == torch.float32 and u.device == x.device
        assert not torch.any(info["fallback_used"])
        assert torch.all(u[:, 0] + x[:, 0] >= -1e-5)
        torch.testing.assert_close(
            u[:, 0], torch.maximum(-x[:, 0], nominal[:, 0]), atol=1e-4, rtol=1e-4
        )
