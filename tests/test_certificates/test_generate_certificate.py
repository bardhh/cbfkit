
import jax.numpy as jnp
from cbfkit.certificates.packager import generate_certificate
from cbfkit.certificates.conditions.barrier_conditions import zeroing_barriers
from cbfkit.utils.user_types import CertificateCollection

def test_generate_certificate_state():
    # h(x) = x[0] - 1
    def h(x):
        return x[0] - 1.0

    # Conditions
    conditions = zeroing_barriers.linear_class_k(1.0)

    # Create certificate
    coll = generate_certificate(h, conditions, input_style="state")

    assert isinstance(coll, CertificateCollection)

    # Test values
    t = 0.0
    x = jnp.array([2.0, 0.0])

    # h(x) = 2 - 1 = 1
    assert jnp.isclose(coll.functions[0](t, x), 1.0)

    # grad(h) = [1, 0]
    assert jnp.allclose(coll.jacobians[0](t, x), jnp.array([1.0, 0.0]))

    # partial_t = 0
    assert jnp.isclose(coll.partials[0](t, x), 0.0)

def test_generate_certificate_separated():
    # h(t, x) = x[0] - t
    def h(t, x):
        return x[0] - t

    conditions = zeroing_barriers.linear_class_k(1.0)

    coll = generate_certificate(h, conditions, input_style="separated")

    t = 1.0
    x = jnp.array([2.0, 0.0])

    # h = 2 - 1 = 1
    assert jnp.isclose(coll.functions[0](t, x), 1.0)

    # grad_x = [1, 0]
    assert jnp.allclose(coll.jacobians[0](t, x), jnp.array([1.0, 0.0]))

    # partial_t = -1
    assert jnp.isclose(coll.partials[0](t, x), -1.0)

if __name__ == "__main__":
    try:
        test_generate_certificate_state()
        print("test_generate_certificate_state passed")
        test_generate_certificate_separated()
        print("test_generate_certificate_separated passed")
    except ImportError:
        print("generate_certificate not yet implemented")
    except Exception as e:
        print(f"Tests failed: {e}")
        raise
