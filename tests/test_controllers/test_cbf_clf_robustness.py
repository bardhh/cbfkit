"""
Tests for CBF-CLF-QP controller robustness.
Verifies behavior under infeasible constraints and degenerate inputs.
"""
import jax.numpy as jnp
import pytest
from cbfkit.controllers.cbf_clf.cbf_clf_qp_generator import cbf_clf_qp_generator
from cbfkit.controllers.cbf_clf.generate_constraints.zeroing_cbfs import generate_compute_zeroing_cbf_constraints
from cbfkit.controllers.cbf_clf.generate_constraints.vanilla_clfs import generate_compute_vanilla_clf_constraints
from cbfkit.utils.user_types import ControllerData

def setup_controller(relaxable_cbf=False):
    # 1D system: x_dot = u
    # State: x (scalar)
    # Control: u (scalar)

    def dynamics(x):
        return jnp.zeros((1,)), jnp.ones((1, 1))

    # Barriers:
    # h1(x) = x - 1 >= 0  (forces x >= 1)
    # h2(x) = -x - 1 >= 0 (forces x <= -1)
    # These are mutually exclusive.
    barriers = (
        [lambda t, x: x[0] - 1.0, lambda t, x: -x[0] - 1.0], # functions
        [lambda t, x: jnp.array([1.0]), lambda t, x: jnp.array([-1.0])], # gradients
        [lambda t, x: jnp.zeros((1,1))]*2, # hessians
        [lambda t, x: 0.0]*2, # partials
        [lambda h: h]*2 # conditions (identity)
    )

    # Lyapunov: V(x) = x^2
    lyapunovs = (
        [lambda t, x: x[0]**2],
        [lambda t, x: 2*x],
        [lambda t, x: 2*jnp.eye(1)],
        [lambda t, x: 0.0],
        [lambda V: 0.1*V]
    )

    controller_gen = cbf_clf_qp_generator(
        generate_compute_zeroing_cbf_constraints,
        generate_compute_vanilla_clf_constraints
    )

    controller = controller_gen(
        control_limits=jnp.array([10.0]),
        dynamics_func=dynamics,
        barriers=barriers,
        lyapunovs=lyapunovs,
        relaxable_cbf=relaxable_cbf,
        relaxable_clf=True
    )

    return controller

def test_infeasible_cbf_handling():
    """
    Verifies that the controller returns NaN and reports failure status when
    CBF constraints are mutually exclusive and not relaxable.
    """
    controller = setup_controller(relaxable_cbf=False)

    t = 0.0
    x = jnp.array([0.0]) # Violates both constraints
    u_nom = jnp.array([0.0])
    key = None
    data = ControllerData(error=jnp.array(False), error_data=0)

    u, new_data = controller(t, x, u_nom, key, data)

    # 1. Check control is NaN (safe fallback)
    assert jnp.any(jnp.isnan(u)), f"Control should be NaN for infeasible QP, got {u}"

    # 2. Check status is not success (1)
    status = new_data.error_data
    # 3: Primal Infeasible, 5: Max Iter Unsolved, 0: Unsolved, 2: Max Iter Reached
    assert status != 1, f"Solver status should not be success (1), got {status}"
    assert status in [0, 2, 3, 4, 5], f"Expected failure status (0, 2, 3, 4, 5), got {status}"

def test_nan_input_handling():
    """
    Verifies that the controller detects NaN inputs and reports error -2.
    """
    controller = setup_controller(relaxable_cbf=False)

    t = 0.0
    x = jnp.array([0.0])
    u_nom = jnp.array([jnp.nan])
    key = None
    data = ControllerData(error=jnp.array(False), error_data=0)

    u, new_data = controller(t, x, u_nom, key, data)

    # 1. Check control is NaN
    assert jnp.any(jnp.isnan(u)), "Control should be NaN for NaN input"

    # 2. Check status is -2 (NAN_INPUT_DETECTED)
    status = new_data.error_data
    assert status == -2, f"Expected status -2 for NaN input, got {status}"

if __name__ == "__main__":
    test_infeasible_cbf_handling()
    test_nan_input_handling()
    print("Tests passed!")
