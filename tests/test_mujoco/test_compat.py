"""Milestone 0: MJX imports and steps under CBFKit's pinned JAX/x64 configuration."""

import importlib.metadata

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

import cbfkit  # noqa: F401  -- forces jax_enable_x64
from cbfkit.systems.mujoco import MODELS_DIR, load_model


def test_mujoco_and_mjx_versions_identical():
    assert mujoco.__version__ == importlib.metadata.version("mujoco-mjx")


def test_x64_is_enabled_by_cbfkit():
    assert jax.config.jax_enable_x64 is True


def test_vendored_cart_pole_loads_and_steps():
    assert (MODELS_DIR / "cart_pole" / "scene.xml").exists()
    m = load_model("cart_pole")
    assert (m.nq, m.nv, m.nu) == (2, 2, 1)
    mm = mjx.put_model(m)
    d = mjx.make_data(m)
    d = jax.jit(mjx.step)(mm, d)
    assert jnp.all(jnp.isfinite(d.qpos))
