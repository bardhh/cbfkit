import jax.numpy as jnp
import cbfkit.systems.unicycle.models.accel_unicycle as unicycle

def test_dynamics_attributes():
    # Initialize dynamics with custom parameters
    a_max = 5.0
    omega_max = 2.0
    custom_param = "test_value"

    dyn = unicycle.plant(
        a_max=a_max,
        omega_max=omega_max,
        custom_param=custom_param
    )

    # Check if attributes are set correctly
    print(f"Checking attributes on dynamics object...")
    assert hasattr(dyn, "a_max"), "a_max not found"
    assert dyn.a_max == a_max, f"a_max mismatch: {dyn.a_max} != {a_max}"

    assert hasattr(dyn, "omega_max"), "omega_max not found"
    assert dyn.omega_max == omega_max, f"omega_max mismatch: {dyn.omega_max} != {omega_max}"

    assert hasattr(dyn, "custom_param"), "custom_param not found"
    assert dyn.custom_param == custom_param, f"custom_param mismatch: {dyn.custom_param} != {custom_param}"

    print("Attributes verification successful!")

    # Check if dynamics function still works
    print("Checking dynamics execution...")
    x = jnp.array([0.0, 0.0, 1.0, 0.0]) # x, y, v, theta
    f, g = dyn(x)

    assert f.shape == (4,), f"f shape mismatch: {f.shape}"
    assert g.shape == (4, 2), f"g shape mismatch: {g.shape}"

    print("Dynamics execution successful!")

if __name__ == "__main__":
    test_dynamics_attributes()
