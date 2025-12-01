from typing import Any, Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array, jit, random

from cbfkit.utils.user_types import (
    Control,
    ControllerCallable,
    ControllerData,
    DynamicsCallable,
    Key,
    State,
    Time,
)

from .models.single_integrator import plant as pedestrian_dynamics


class CrowdManager:
    """
    Manages a collection of pedestrians and integrates them with a robot's dynamics.
    """

    def __init__(self):
        self.pedestrians: List[Dict[str, Any]] = []
        self.robot_dim = 0  # Will be set when build_system is called
        self.pedestrian_dim = 4  # Fixed for single integrator model

    def add_pedestrian(
        self,
        init_state: Array,
        behavior: Callable[[float, Array, Dict[str, Array]], Array],
        id: Optional[str] = None,
    ):
        """
        Adds a pedestrian to the crowd.

        Args:
            init_state (Array): Initial state [px, py, vx, vy].
            behavior (Callable): Policy function returning acceleration.
            id (str, optional): Identifier for the pedestrian.
        """
        if id is None:
            id = f"ped_{len(self.pedestrians)}"

        self.pedestrians.append(
            {"id": id, "init_state": jnp.array(init_state), "behavior": behavior}
        )

    def get_augmented_dynamics(self, robot_dynamics: DynamicsCallable) -> DynamicsCallable:
        """
        Constructs the combined dynamics function for the robot and all pedestrians.

        State Z = [x_robot, x_ped_1, x_ped_2, ..., x_ped_N]

        Args:
            robot_dynamics (DynamicsCallable): Function returning f_robot(x), g_robot(x).

        Returns:
            DynamicsCallable: Function returning F(z), G(z) for the augmented system.
        """
        num_peds = len(self.pedestrians)
        # We assume standard single integrator model for all pedestrians for now
        ped_dyn = pedestrian_dynamics()

        def augmented_dynamics(z: Array) -> Tuple[Array, Array]:
            # 1. Robot Part (Assume robot state is at the beginning)
            # We don't know robot dim a priori inside the function without inspecting z or being told.
            # However, since we build this closure, we can rely on self.robot_dim being set correctly
            # BEFORE this function is used, or we infer it if we assume standard layout.
            # Better: We determine robot_dim when this method is called or passed.
            # Limitation: We need to know robot state dimension to slice z correctly.
            # Solution: We will infer it from the first call or user must set it?
            # Actually, usually 'robot_dynamics' doesn't tell us the dim directly.
            # Let's assume the user provided robot_dim is handled or we infer from z.

            # HACK: We'll assume robot_dim is len(z) - 4 * num_peds
            total_dim = z.shape[0]
            robot_dim = total_dim - num_peds * 4

            x_robot = z[:robot_dim]
            f_r, g_r = robot_dynamics(x_robot)

            f_list = [f_r]
            # g_rows = [g_r]  # Unused in this logic branch for augmented dynamics building below

            # 2. Pedestrian Part
            # Pedestrians are "autonomous" in the sense that the robot's control 'u' doesn't affect them directly.
            # Their motion comes from their own internal 'u_ped' (acceleration) which acts as drift term
            # in the context of the robot's controller, OR we model them as fully actuated but controlled by 'behavior'.
            # IN CBFKIT: If we put them in dynamics f(z), g(z), usually 'g' corresponds to the CONTROLLED inputs (u).
            # If the robot cannot control the pedestrians, the rows of G corresponding to pedestrians must be zero.
            # The pedestrian's motion (including their acceleration from policy) effectively becomes part of the DRIFT f(z).
            # HOWEVER, the policy depends on time and state. Standard dynamics f(x) usually doesn't take 't'.
            # If policies are state-dependent only, we can bake them into f(z).
            # If policies depend on 't', we have a problem with standard f(x).

            # COMPROMISE: For now, we assume policies are f(t, x, env).
            # But standard `sim.execute` expects `dynamics(x) -> f, g`.
            # To support time-varying behavior in dynamics, we might need to augment state with time
            # or use a closure that captures 't' (but 't' varies during integration).

            # ALTERNATIVE: The "Nominal Controller" approach.
            # The simulator allows `nominal_controller` to return `u`.
            # But `u` typically matches the columns of `g`.
            # If we want to simulate pedestrians using the standard loop, we can:
            # 1. Treat pedestrians as "controlled" systems in the dynamics (G has identity blocks for them).
            # 2. The "Controller" we pass to `sim.execute` computes u_robot AND u_peds.
            # This allows u_peds to be time/state dependent (Policies).

            current_idx = robot_dim
            for _ in range(num_peds):
                # Pedestrian state extraction not needed for f/g structure, just for f calculation if needed.
                # Standard Single Integrator: f = [vx, vy, 0, 0], g = [0 0; 0 0; 1 0; 0 1]
                x_p = z[current_idx : current_idx + 4]
                f_p, g_p = ped_dyn(x_p)

                f_list.append(f_p)

                # Append zeros to G because Robot U doesn't affect Pedestrians
                # AND Pedestrian U (acceleration) is separate.
                # Wait, if we want to simulate them, we need their G terms to be active
                # so that the 'controller' can drive them.

                # So G matrix will be block diagonal:
                # [ G_robot  0       0      ]
                # [ 0        G_ped1  0      ]
                # [ 0        0       G_ped2 ]

                # But `cbfkit` usually assumes `u` is the ROBOT's control.
                # If we expand `u` to be [u_robot, u_ped1, ...], then the MPPI/CBF planners
                # might try to optimize u_ped1 to help the robot! This is "Cooperative" planning.
                # We likely want non-cooperative.

                # SOLUTION for Non-Cooperative simulation in `cbfkit`:
                # The "Dynamics" seen by the Planner/Controller should likely treat pedestrians as drift (g_rows=0).
                # But the "Dynamics" used by the Simulator needs to simulate them.

                # We will implement the "Fully Actuated System" approach here.
                # The user must be careful to mask out pedestrian controls when designing the Robot's controller
                # (e.g. by setting control limits to 0 for those indices in the optimizer, or by just ignoring them).
                # OR, simpler: `get_nominal_controller` will output the combined [u_robot, u_ped1...] vector.
                # The MPPI/CBF should ideally only be planning for u_robot.

                # For this implementation: We construct the full block-diagonal G.
                # The robot controller wrapper will handle assembling the full u vector.

                # Pad g_p to align with previous columns (robot + prev peds)
                # and pad previous rows to align with new columns.

                # To do this cleanly in a loop with lists is tricky.
                # Let's assume we are building a Block Diagonal G.
                pass  # Logic handled below outside loop for G construction
                current_idx += 4

            # Concatenate F
            F = jnp.concatenate(f_list)

            # Block Diagonal G Construction
            # G_robot is (dim_x_r, dim_u_r)
            # G_peds are (4, 2)

            # Dimensions
            n_r, m_r = g_r.shape

            # Total rows = n_r + 4 * num_peds
            # Total cols = m_r + 2 * num_peds

            # We can use jax.scipy.linalg.block_diag but it might not work with arbitrary arrays easily in loop.
            # We'll listify.
            g_blocks = [g_r] + [
                ped_dyn(z[robot_dim + i * 4 : robot_dim + (i + 1) * 4])[1] for i in range(num_peds)
            ]

            # Manual block diag construction or use helper if available.
            # JAX doesn't have a direct `block_diag` for arrays in numpy style in all versions,
            # but `jax.scipy.linalg.block_diag` exists.
            from jax.scipy.linalg import block_diag

            G = block_diag(*g_blocks)

            return F, G

        return augmented_dynamics

    def get_closed_loop_dynamics(self, robot_dynamics: DynamicsCallable) -> DynamicsCallable:
        """
        Returns dynamics for planning where pedestrians are treated as autonomous (drift-only).
        The pedestrian policy controls are baked into the drift term f(z).
        The control matrix G will only have columns for the robot.

        Args:
            robot_dynamics (Callable): Robot dynamics function.

        Returns:
            DynamicsCallable: f(z), g(z) for planning.
        """
        num_peds = len(self.pedestrians)
        ped_dyn = pedestrian_dynamics()

        def closed_loop_dynamics(z: Array) -> Tuple[Array, Array]:
            total_dim = z.shape[0]
            robot_dim = total_dim - num_peds * 4

            # 1. Robot Dynamics
            x_robot = z[:robot_dim]
            f_r, g_r = robot_dynamics(x_robot)

            f_list = [f_r]

            # 2. Pedestrian Dynamics (with Policy baked in)
            # We need to evaluate policies here to get u_ped
            # This replicates logic in get_nominal_controller but inside dynamics

            # Prepare environment state
            robot_state_std = jnp.array(
                [x_robot[0], x_robot[1], 0.0, 0.0]
            )  # Assume SI/Unicycle pos/vel
            ped_states = [z[robot_dim + i * 4 : robot_dim + (i + 1) * 4] for i in range(num_peds)]
            all_states = jnp.stack([robot_state_std] + ped_states)

            for i, ped in enumerate(self.pedestrians):
                p_state = ped_states[i]
                policy = ped["behavior"]

                # Evaluate Policy
                # Note: 't' is not available in standard dynamics signature f(x).
                # We assume t=0 or policies are time-invariant for planning purposes.
                # If policies strongly depend on t, this is an approximation.
                env_state = {"others_states": all_states}
                u_ped = policy(0.0, p_state, env_state, key=None)

                # Get open loop dynamics
                f_p, g_p = ped_dyn(p_state)

                # Bake control into drift: f_closed = f + g @ u
                f_p_closed = f_p + g_p @ u_ped
                f_list.append(f_p_closed)

            F = jnp.concatenate(f_list)

            # G matrix only for robot
            # Pad g_r with zeros for all pedestrian states rows?
            # No, G is (dim_z, dim_u_robot).
            # Rows for pedestrians should be zero (they are not controlled by robot u).

            g_zeros = jnp.zeros((num_peds * 4, g_r.shape[1]))
            G = jnp.concatenate([g_r, g_zeros], axis=0)

            return F, G

        return closed_loop_dynamics

    def get_nominal_controller(
        self,
        robot_nominal_controller: ControllerCallable,
        use_augmented_state: bool = False,
    ) -> ControllerCallable:
        """
        Wraps the robot's nominal controller to include pedestrian behaviors.

        Args:
            robot_nominal_controller: Controller for the robot.
            use_augmented_state: If True, passes full state `z` to robot controller.
        """
        num_peds = len(self.pedestrians)

        def combined_controller(
            t: Time,
            z: State,
            u_prev: Optional[Control] = None,
            key: Optional[Key] = None,
            data: Optional[ControllerData] = None,
        ) -> Tuple[Array, ControllerData]:
            # 1. Extract State
            total_dim = z.shape[0]
            robot_dim = total_dim - num_peds * 4
            x_robot = z[:robot_dim]

            # Default key if None
            if key is None:
                key = random.PRNGKey(0)

            # Default data if None
            if data is None:
                data = ControllerData()

            # Split keys for stochastic policies
            # key is now Key (Array)
            keys = random.split(key, num_peds + 1)
            robot_key = keys[0]
            ped_keys = keys[1:]

            # 2. Get Robot Control
            u_robot_prev = None
            if u_prev is not None:
                ped_control_dim = num_peds * 2
                robot_control_dim = u_prev.shape[0] - ped_control_dim
                if robot_control_dim > 0:
                    u_robot_prev = u_prev[:robot_control_dim]

            # Pass z or x_robot based on flag
            state_to_pass = z if use_augmented_state else x_robot

            # Unpack inner data if present (e.g., PlannerData for MPPI)
            # We assume data is ControllerData.
            inner_data = None
            if hasattr(data, "sub_data"):
                inner_data = data.sub_data.get("inner_controller_data", None)

            # If inner_data is None, fall back to data (if it wasn't ControllerData, unlikely here)
            # or let controller handle None.
            if inner_data is None:
                inner_data = data  # Fallback to passing the wrapper data

            # robot_nominal_controller is ControllerCallable: (t, x, u_nom, key, data)
            # We pass u_robot_prev as u_nom.
            u_robot, robot_data = robot_nominal_controller(
                t, state_to_pass, u_robot_prev, robot_key, inner_data
            )

            # 3. Calculate Pedestrian Controls (Policies)
            u_peds = []

            # Prepare environment
            robot_state_std = jnp.array([x_robot[0], x_robot[1], 0.0, 0.0])
            ped_states = [z[robot_dim + i * 4 : robot_dim + (i + 1) * 4] for i in range(num_peds)]
            all_states = jnp.stack([robot_state_std] + ped_states)

            for i, ped in enumerate(self.pedestrians):
                p_state = ped_states[i]
                policy = ped["behavior"]
                env_state = {"others_states": all_states}
                # Pass key to policy if it accepts it (we assume updated policy will)
                # For now, pass as kwarg 'key'.
                acc = policy(t, p_state, env_state, key=ped_keys[i])
                u_peds.append(acc)

            # 4. Combine
            u_combined = jnp.concatenate([u_robot] + u_peds)

            # Pack result into ControllerData
            # If we started with ControllerData, preserve other fields
            if hasattr(data, "sub_data"):
                new_sub_data = data.sub_data.copy()
                new_sub_data["inner_controller_data"] = robot_data
                # Create new ControllerData preserving original structure (if NamedTuple, use _replace)
                # JAX jit compatible way: _replace
                return u_combined, data._replace(u=u_combined, sub_data=new_sub_data)
            else:
                # If data was None or something else, wrap it fresh
                return u_combined, ControllerData(
                    u=u_combined, sub_data={"inner_controller_data": robot_data}
                )

        return combined_controller

    def get_initial_state(self, robot_init: Array) -> Array:
        """
        Returns the combined initial state vector z0.
        """
        ped_inits = [p["init_state"] for p in self.pedestrians]
        if not ped_inits:
            return robot_init
        return jnp.concatenate([robot_init] + ped_inits)
