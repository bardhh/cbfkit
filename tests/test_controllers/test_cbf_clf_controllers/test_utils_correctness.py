import unittest
import jax.numpy as jnp
import numpy as np
from cbfkit.controllers.cbf_clf.utils.utils import block_diag_matrix_from_vec

class TestUtilsCorrectness(unittest.TestCase):
    """
    Test suite for utility functions in cbfkit.controllers.cbf_clf.utils.utils.
    """

    def test_block_diag_matrix_from_vec_correctness(self):
        """
        Verify that block_diag_matrix_from_vec produces the correct block diagonal structure.
        For n=2, we expect a (4, 2) matrix (transposed from (2, 4)).
        The diagonal blocks (column-wise) should be [1, -1]^T.

        Expected matrix for n=2:
        [[ 1,  0],
         [-1,  0],
         [ 0,  1],
         [ 0, -1]]
        """
        n_blocks = 2
        # Act
        res = block_diag_matrix_from_vec(n_blocks)

        # Assert structure
        expected_shape = (2 * n_blocks, n_blocks)
        self.assertEqual(res.shape, expected_shape, f"Expected shape {expected_shape}, got {res.shape}")

        expected_matrix = jnp.array([
            [ 1.0,  0.0],
            [-1.0,  0.0],
            [ 0.0,  1.0],
            [ 0.0, -1.0]
        ])

        # Verify values
        # We assume float comparison, but block_diag might return ints currently?
        # jax.numpy usually handles type promotion for comparison.
        np.testing.assert_allclose(res, expected_matrix, rtol=1e-6, err_msg="Matrix content mismatch")

        # Verify type is JAX array
        self.assertIsInstance(res, jnp.ndarray, "Result should be a JAX array")

if __name__ == '__main__':
    unittest.main()
