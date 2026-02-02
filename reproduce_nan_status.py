import jax
import jax.numpy as jnp
from cbfkit.controllers.cbf_clf.vanilla_cbf_clf_qp_control_laws import vanilla_cbf_clf_qp_controller
from cbfkit.utils.user_types import ControllerData

# Dummy dynamics: 2 states, 2 controls
def dynamics(x):
    return jnp.zeros(2), jnp.eye(2)

# Dummy controller
# p_mat is identity
controller_factory = vanilla_cbf_clf_qp_controller(
    control_limits=jnp.array([1.0, 1.0]),
    dynamics_func=dynamics,
    barriers=[],
    lyapunovs=[],
    p_mat=jnp.eye(2),
    relaxable_clf=False,
    relaxable_cbf=False
)

# Initialize data
data = ControllerData(
    error=jnp.array(False),
    error_data=None,
    complete=False,
    sol=jnp.zeros(2),
    u=jnp.zeros(2),
    u_nom=jnp.zeros(2),
    sub_data={}
)

# Run with NaN u_nom
# This should trigger nan_in_inputs -> status -2
u_nom = jnp.array([jnp.nan, jnp.nan])
x = jnp.zeros(2)
t = 0.0
key = jax.random.PRNGKey(0)

print("Running controller with NaN input...")
# We use jit to ensure lax.cond/switch are compiled and executed as expected
jitted_controller = jax.jit(controller_factory)

try:
    u, new_data = jitted_controller(t, x, u_nom, key, data)
    # Block until ready to ensure print happens
    u.block_until_ready()
    print(f"Status in data: {new_data.error_data}")
except Exception as e:
    print(f"Exception: {e}")
