import jax
import jax.numpy as jnp
from cbfkit.controllers.cbf_clf.vanilla_cbf_clf_qp_control_laws import vanilla_cbf_clf_qp_controller
from cbfkit.utils.user_types import ControllerData, CertificateCollection

# Dynamics that return NaNs in f
def dynamics(x):
    return jnp.full(2, jnp.nan), jnp.eye(2)

# Barrier
def h(t, x): return x[0]
def grad(t, x): return jnp.array([1.0, 0.0])
def hess(t, x): return jnp.zeros((2, 2))
def partial_t(t, x): return 0.0
def condition(val): return 1.0 * val # Class K

# Use tuple for barriers which is accepted and converted
barriers = ([h], [grad], [hess], [partial_t], [condition])

# Dummy controller
controller_factory = vanilla_cbf_clf_qp_controller(
    control_limits=jnp.array([1.0, 1.0]),
    dynamics_func=dynamics,
    barriers=barriers,
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

# Run with valid u_nom, but dynamics will produce NaNs -> inputs to QP will be NaN
u_nom = jnp.array([0.0, 0.0])
x = jnp.zeros(2)
t = 0.0
key = jax.random.PRNGKey(0)

print("Running controller with NaN dynamics...")
jitted_controller = jax.jit(controller_factory)

try:
    u, new_data = jitted_controller(t, x, u_nom, key, data)
    u.block_until_ready()
    print(f"Status in data: {new_data.error_data}")
except Exception as e:
    print(f"Exception: {e}")
