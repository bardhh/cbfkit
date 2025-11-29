import jax.numpy as jnp

from cbfkit.utils import block_diag_matrix, hat
from cbfkit.utils import matrix_vector_operations as mvo
from cbfkit.utils import miscellaneous as misc
from cbfkit.utils import normalize, tanh_sigmoid_func, vee


def test_helpers_are_reexported():
    """Test that helpers are correctly re-exported."""
    assert normalize is mvo.normalize is misc.normalize
    assert hat is mvo.hat is misc.hat
    assert vee is mvo.vee is misc.vee
    assert block_diag_matrix is mvo.block_diag_matrix is misc.block_diag_matrix


def test_normalize_behaviour():
    """Test normalization behavior."""
    vec = jnp.array([3.0, 4.0, 0.0])
    normalized = normalize(vec)
    assert bool(jnp.allclose(normalized, jnp.array([0.6, 0.8, 0.0])))


def test_normalize_handles_zero_vector():
    """Test normalization of zero vector."""
    vec = jnp.zeros(3)
    normalized = normalize(vec)
    assert bool(jnp.allclose(normalized, jnp.zeros(3)))


def test_hat_and_vee_are_inverses():
    """Test that hat and vee operations are inverses."""
    vec = jnp.array([1.2, -0.5, 0.3])
    reconstructed = vee(hat(vec))
    assert bool(jnp.allclose(reconstructed, vec))


def test_block_diag_matrix_constructs_expected_layout():
    """Test block diagonal matrix construction."""
    mat1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    mat2 = jnp.array([[5.0]])
    block = block_diag_matrix(mat1, mat2)
    expected = jnp.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    assert bool(jnp.allclose(block, expected))


def test_tanh_sigmoid_limits():
    """Test tanh sigmoid limits."""
    upper = 2.0
    below = tanh_sigmoid_func(0.0, upper)
    above = tanh_sigmoid_func(upper, upper)
    assert below >= 0.0
    assert jnp.isclose(above, upper, rtol=1e-5, atol=1e-5)
