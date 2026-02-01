
import unittest
import jax.numpy as jnp
from cbfkit.certificates import certificate_package
from cbfkit.utils.user_types import CertificateInputStyle

class TestInputStyleValidation(unittest.TestCase):
    def test_invalid_string_input(self):
        """Test that passing an invalid string raises ValueError."""
        def dummy_factory(**kwargs):
            def func(x):
                return x[0]
            return func

        with self.assertRaises(ValueError) as cm:
            certificate_package(dummy_factory, n=1, input_style="invalid_style")

        self.assertIn("Invalid input_style 'invalid_style'", str(cm.exception))
        self.assertIn("concatenated", str(cm.exception))

    def test_valid_string_input(self):
        """Test that passing valid strings works correctly."""
        def dummy_factory(**kwargs):
            def func(x):
                return x[0]
            return func

        # Should not raise
        certificate_package(dummy_factory, n=1, input_style="concatenated")
        certificate_package(dummy_factory, n=1, input_style="separated")
        certificate_package(dummy_factory, n=1, input_style="state")

    def test_enum_input(self):
        """Test that passing Enum members works correctly."""
        def dummy_factory(**kwargs):
            def func(x):
                return x[0]
            return func

        # Should not raise
        certificate_package(dummy_factory, n=1, input_style=CertificateInputStyle.CONCATENATED)
        certificate_package(dummy_factory, n=1, input_style=CertificateInputStyle.SEPARATED)
        certificate_package(dummy_factory, n=1, input_style=CertificateInputStyle.STATE)

if __name__ == '__main__':
    unittest.main()
