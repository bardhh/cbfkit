
import jax.numpy as jnp
import pytest
from cbfkit.certificates.rectifiers import rectify_relative_degree

def f(x):
    # Double integrator: x0_dot = x1, x1_dot = 0
    return jnp.array([x[1], 0.0])

def g(x):
    # x0_dot += 0, x1_dot += u
    return jnp.array([[0.0], [1.0]])

def dynamics(x):
    return f(x), g(x)

def h(x):
    return x[0]

def test_rectifier_roots_length_and_coeffs():
    """
    Test that rectify_relative_degree correctly interprets 'roots' and constructs the ECBF.

    For a relative degree 2 system (n_fl=2), we expect to provide 1 root (rel_degree - 1).
    If we provide root [-1.0], the polynomial is (s + 1) -> coeffs [1, 1].
    The resulting h_new should be 1*h_dot + 1*h.

    If coefficients were mapped incorrectly (reversed), we might get 1*h + 1*h_dot (same for [1,1]).
    So let's use a root that produces distinct coeffs.
    Root at -10.0. Poly (s + 10). Coeffs [1, 10].

    Correct mapping (s^1 <-> h_dot, s^0 <-> h):
    h_new = 1 * h_dot + 10 * h.

    Incorrect mapping (forward zip):
    h <-> 1, h_dot <-> 10.
    h_new = 1 * h + 10 * h_dot.

    Let's test with root -10.0.
    At x = [1, 1] (h=1, h_dot=1):
    Correct: 1*1 + 10*1 = 11.
    Incorrect: 1*1 + 10*1 = 11.
    Wait, if coeffs are [1, 10].
    Correct: s^1 (1) * h_dot + s^0 (10) * h = 1*h_dot + 10*h.
    Incorrect: zip([h, h_dot], [1, 10]) -> 1*h + 10*h_dot.

    These are different!
    Correct: h_dot + 10h = 1 + 10 = 11.
    Incorrect: h + 10h_dot = 1 + 10 = 11.
    Wait. 1*h + 10*h_dot = 1*1 + 10*1 = 11.
    1*h_dot + 10*h = 1*1 + 10*1 = 11.
    They are the same if h = h_dot.

    Let's choose x such that h != h_dot.
    x = [1.0, 0.0]. h=1, h_dot=0.
    Correct (h_dot + 10h): 0 + 10*1 = 10.
    Incorrect (h + 10h_dot): 1 + 10*0 = 1.

    So x=[1, 0] with root -10 distinguishes them.
    """

    roots = jnp.array([-10.0])

    factory = rectify_relative_degree(
        h,
        dynamics,
        state_dim=2,
        roots=roots,
        form="exponential"
    )

    # Dummy condition
    dummy_condition = lambda x: x
    pkg = factory(certificate_conditions=dummy_condition)
    cbf_func = pkg.functions[0]

    # Test point: x=[1.0, 0.0]
    x_test = jnp.array([1.0, 0.0])

    val = cbf_func(0.0, x_test)
    print(f"CBF Value: {val}")

    # We expect 10.0
    assert jnp.isclose(val, 10.0), f"Expected 10.0 (h_dot + 10h), got {val}"

if __name__ == "__main__":
    try:
        test_rectifier_roots_length_and_coeffs()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
