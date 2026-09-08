"""Optional PyTorch boundary for batched JAX filters (no Isaac dependency)."""

import jax.dlpack

try:
    import torch
except ImportError as exc:
    raise ImportError("TorchSafetyFilter requires PyTorch; install cbfkit[torch].") from exc

from .batched import BatchedSafetyFilter


class TorchSafetyFilter:
    """DLPack bridge for CPU/CUDA policy tensors; inference only.

    Inputs are copied on their current device to establish ownership: simulator
    buffers may be overwritten in-place after this call. Outputs are cloned
    for the same reason. There is no NumPy or CPU staging, but this is not a
    zero-copy API. CUDA requires a CUDA-enabled JAX installation on the same
    device. Action/state mappings belong to the application.
    """

    def __init__(self, safety_filter: BatchedSafetyFilter):
        self.safety_filter = safety_filter

    @staticmethod
    def _to_jax(value):
        if not isinstance(value, torch.Tensor):
            raise TypeError("Expected a torch.Tensor")
        if value.device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU and CUDA tensors are supported")
        return jax.dlpack.from_dlpack(value.detach().contiguous(), copy=True)

    def filter(self, states, actions):
        """Return filtered tensors and diagnostics on the input device.

        Float32 and float64 inputs are supported. Controls are cast back to
        the action dtype, so applications must allow for rounding tolerance.
        This does not provide gradients through the safety filter.
        """
        if not isinstance(states, torch.Tensor) or not isinstance(actions, torch.Tensor):
            raise TypeError("states and actions must be torch tensors")
        if states.device != actions.device:
            raise ValueError("states and actions must be on the same device")
        if states.dtype not in (torch.float32, torch.float64) or actions.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError("states and actions must be float32 or float64")
        applied, info = self.safety_filter.filter(self._to_jax(states), self._to_jax(actions))

        def convert(value):
            if value is None:
                return None
            result = torch.utils.dlpack.from_dlpack(value).clone()
            if result.device != actions.device:
                raise RuntimeError(
                    "JAX returned a different device; no implicit transfer is allowed"
                )
            return result

        result = convert(applied).to(dtype=actions.dtype)
        diagnostics = jax.tree_util.tree_map(
            convert, {k: v for k, v in info.items() if k != "u_applied"}
        )
        diagnostics["u_applied"] = result
        return result, diagnostics

    def reset(self, mask=None, *, seed=None):
        """Reset from a boolean tensor, e.g. ``terminated | truncated``.

        Call after an auto-resetting Isaac Lab environment's step, before
        filtering its next action. Explicit environment resets must reset the
        filter too.
        """
        self.safety_filter.reset(None if mask is None else self._to_jax(mask), seed=seed)
