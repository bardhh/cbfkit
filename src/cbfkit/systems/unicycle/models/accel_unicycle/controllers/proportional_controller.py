# controllers/proportional_controller.py
import jax
import jax.numpy as jnp
from jax import jit, lax


def proportional_controller(dynamics, Kp_pos, Kp_theta, desired_state):
    # ── read limits from the plant (fallback if absent) ─────────────
    a_max = float(getattr(dynamics, "a_max", 1.0))  # m s⁻²
    omega_max = float(getattr(dynamics, "omega_max", 4.0))  # rad s⁻¹
    v_max = float(getattr(dynamics, "v_max", 2.0))  # m s⁻¹
    eps_pos = float(getattr(dynamics, "goal_tol", 0.25))  # m

    @jit
    def controller(_t, state):
        x, y, v, theta = state
        xd, yd, _, thetad = desired_state

        dx, dy = xd - x, yd - y
        dist = jnp.hypot(dx, dy)

        theta_goal = jnp.where(dist > eps_pos, jnp.arctan2(dy, dx), thetad)
        theta_err = jnp.arctan2(jnp.sin(theta_goal - theta), jnp.cos(theta_goal - theta))

        v_prop = Kp_pos * dist
        v_brake = jnp.sqrt(2.0 * a_max * dist + 1e-9)
        v_des = jnp.minimum(jnp.minimum(v_prop, v_brake), v_max)

        alpha_cmd = Kp_pos * (v_des - v)  # linear accel
        omega_cmd = jnp.clip(Kp_theta * theta_err, -omega_max, omega_max)

        # Check if controller is within goal tolerance
        data = {"complete": dist <= eps_pos}

        return jnp.array([alpha_cmd, omega_cmd]), data

    return controller
