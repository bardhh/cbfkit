
import jax.numpy as jnp
import jax.random as random
from unittest.mock import patch
import pytest
from cbfkit.certificates import rectifiers

class TestRectifiersReproducibility:
    """Test suite for reproducibility and determinism in rectifiers module."""

    @pytest.fixture
    def dynamics_and_constraint(self):
        """Define a simple chain of integrators: x1 -> x2 -> x3 -> u
        Relative degree of x1 w.r.t u is 3.
        """
        def dynamics(x):
            # x is size 3: [x1, x2, x3]
            return jnp.array([x[1], x[2], 0.0]), jnp.array([[0.0], [0.0], [1.0]])

        def constraint(x):
            # x is size 4 (including time/extra)
            return x[0]

        return dynamics, constraint

    def test_unique_keys_recursion(self, dynamics_and_constraint):
        """Test that random keys used in recursion are unique."""
        dynamics, constraint = dynamics_and_constraint

        keys_seen = []
        original_normal = rectifiers.random.normal

        def mock_normal(key, shape):
            keys_seen.append(key)
            return original_normal(key, shape)

        with patch('cbfkit.certificates.rectifiers.random.normal', side_effect=mock_normal):
            # Pass a fixed seed to ensure deterministic starting point
            rectifiers.rectify_relative_degree(
                constraint,
                dynamics,
                state_dim=3,
                rng=42
            )

        # We expect at least 3 levels for relative degree 3 system
        assert len(keys_seen) >= 3

        # Check for duplicates
        for i in range(len(keys_seen)):
            for j in range(i + 1, len(keys_seen)):
                assert not jnp.array_equal(keys_seen[i], keys_seen[j]), \
                    f"Keys at step {i} and {j} are identical: {keys_seen[i]}"

    def test_determinism_with_seed(self, dynamics_and_constraint):
        """Test that same seed produces identical sequence of keys."""
        dynamics, constraint = dynamics_and_constraint

        def get_keys_for_seed(seed):
            keys_seen = []
            original_normal = rectifiers.random.normal

            def mock_normal(key, shape):
                keys_seen.append(key)
                return original_normal(key, shape)

            with patch('cbfkit.certificates.rectifiers.random.normal', side_effect=mock_normal):
                rectifiers.rectify_relative_degree(
                    constraint,
                    dynamics,
                    state_dim=3,
                    rng=seed
                )
            return keys_seen

        keys1 = get_keys_for_seed(123)
        keys2 = get_keys_for_seed(123)

        assert len(keys1) == len(keys2)
        for k1, k2 in zip(keys1, keys2):
            assert jnp.array_equal(k1, k2), "Keys differ for same seed"

    def test_different_seeds(self, dynamics_and_constraint):
        """Test that different seeds produce different keys."""
        dynamics, constraint = dynamics_and_constraint

        def get_first_key(seed):
            keys_seen = []
            original_normal = rectifiers.random.normal

            def mock_normal(key, shape):
                keys_seen.append(key)
                return original_normal(key, shape)

            with patch('cbfkit.certificates.rectifiers.random.normal', side_effect=mock_normal):
                rectifiers.rectify_relative_degree(
                    constraint,
                    dynamics,
                    state_dim=3,
                    rng=seed
                )
            return keys_seen[0]

        key1 = get_first_key(123)
        key2 = get_first_key(456)

        assert not jnp.array_equal(key1, key2), "Different seeds produced same first key"
