"""Moving obstacles must contribute relative velocity to the real CBF-QP."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

spec = importlib.util.spec_from_file_location(
    "warehouse_model", Path(__file__).parents[2] / "examples/isaac_lab/warehouse_model.py"
)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def test_cart_velocity_changes_constraint():
    z = jnp.array([0.0, 0.0, 1.8, 0.0, -0.5, 0.0])
    receding = z.at[4].set(0.5)
    u = jnp.array([0.6, 0.0])
    assert model.residuals(z, u)[0] < model.residuals(receding, u)[0]
    grad = jax.grad(model.cart_barrier)(z)
    f, g = model.dynamics(z)
    np.testing.assert_allclose(
        grad @ (f + g @ u) + model.ALPHA * model.cart_barrier(z),
        model.residuals(z, u)[0],
        atol=1e-6,
    )


def test_actual_qp_respects_moving_cart_and_aisle():
    z = jnp.array(
        [[0.0, 0.0, 1.9, -0.3, 0.0, 0.65], [0.0, 1.4, 4.0, 0.0, 0.0, 0.0]], dtype=jnp.float32
    )
    u = jnp.array([[0.8, 0.0], [0.6, 0.6]], dtype=jnp.float32)
    applied, info = model.make_filter(2, 0.02).filter(z, u)
    assert not np.asarray(info["fallback_used"]).any()
    assert np.asarray(jax.vmap(model.residuals)(z, applied)).min() >= -1e-5
    assert not np.allclose(applied, u)


def test_scenarios_are_paired_and_seeded():
    np.testing.assert_array_equal(
        model.scenario_parameters(42, 8), model.scenario_parameters(42, 8)
    )
    assert not np.array_equal(model.scenario_parameters(42, 8), model.scenario_parameters(7, 8))


def test_gpu_batch_constraint_precision():
    """Recorded RTX 3090 batch: TF32 perturbed an active constraint by ~6e-4."""
    z = jnp.array(
        [
            [
                0.5243759155273438,
                -0.13539695739746094,
                2.775057315826416,
                -1.5523653030395508,
                0.0,
                0.7491000294685364,
            ],
            [
                0.5232982635498047,
                0.02297365851700306,
                2.938328266143799,
                -1.757500171661377,
                0.0,
                0.7085323929786682,
            ],
            [
                0.5273551940917969,
                0.0068569183349609375,
                2.8654115200042725,
                -1.8740479946136475,
                0.0,
                0.6744358539581299,
            ],
            [
                0.5261281132698059,
                0.15704917907714844,
                2.5351243019104004,
                -1.812995195388794,
                0.0,
                0.7477920055389404,
            ],
            [
                0.525557279586792,
                0.06104549020528793,
                2.5800998210906982,
                -1.9795036315917969,
                0.0,
                0.5930617451667786,
            ],
            [
                0.5245810151100159,
                0.022603988647460938,
                2.9241321086883545,
                -1.8954193592071533,
                0.0,
                0.582042396068573,
            ],
            [
                0.5267505645751953,
                0.01584625244140625,
                2.4031591415405273,
                -1.7752177715301514,
                0.0,
                0.6725079417228699,
            ],
            [
                0.5246753692626953,
                -0.061331890523433685,
                2.8927371501922607,
                -1.864462971687317,
                0.0,
                0.5587884187698364,
            ],
        ],
        dtype=jnp.float32,
    )
    nominal = jnp.array(
        [
            [0.7997555136680603, 0.01977572962641716],
            [0.7999929785728455, -0.0033558092545717955],
            [0.7999994158744812, -0.0010023546637967229],
            [0.799670934677124, -0.02294311486184597],
            [0.7999502420425415, -0.008920243009924889],
            [0.7999931573867798, -0.003302584867924452],
            [0.7999967336654663, -0.0023161652497947216],
            [0.7999498844146729, 0.008960644714534283],
        ],
        dtype=jnp.float32,
    )
    # Deliberately allow TF32: float64 normalization must protect the solve
    # without changing process-global precision settings in an application.
    with jax.default_matmul_precision("tensorfloat32"):
        applied, info = model.make_filter(8, 0.02).filter(z, nominal)
    assert applied.dtype == jnp.float64
    assert not np.asarray(info["fallback_used"]).any()
    # Independent float64 NumPy arithmetic on the exact float32 input values.
    state, control = np.asarray(z, dtype=np.float64), np.asarray(applied)
    d = state[:, :2] - state[:, 2:4]
    r = 2 * np.sum(d * (control - state[:, 4:6]), axis=1)
    r += model.ALPHA * (np.sum(d * d, axis=1) - model.SAFE_RADIUS**2)
    assert r.min() >= -1e-5
