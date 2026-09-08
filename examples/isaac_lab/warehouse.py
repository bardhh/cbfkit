"""Real ANYmal warehouse delivery: paired route follower, stop rule, and CBF.

Requires docker/compose.sim6.yaml. Ground-truth obstacle state, frozen walking
policy, prescribed kinematic cart motion. Contact measurements exclude ground.
"""

import argparse
import importlib.metadata
import inspect
import json
import time
import traceback
from pathlib import Path

ROBOT_LINKS = ["base"] + [
    f"{leg}_{link}"
    for leg in ("LF", "LH", "RF", "RH")
    for link in ("HIP", "THIGH", "SHANK", "FOOT")
]
OBSTACLES = ("Cart", "LeftRackA", "LeftRackB", "RightRackA", "RightRackB")


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="nominal,stop,filtered")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", action="store_true")
    parser.add_argument("--all_frames", action="store_true")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=450)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.modes = args.modes.split(",")
    args.seeds = [int(s) for s in args.seeds.split(",")]
    if any(m not in ("nominal", "stop", "filtered") for m in args.modes):
        parser.error("modes must be nominal,stop,filtered")
    if args.frames and not args.enable_cameras:
        parser.error("--frames requires --enable_cameras")
    if args.steps < 1 or args.num_envs < 1:
        parser.error("steps and num_envs must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("use a fresh output directory")
    launcher = AppLauncher(args)
    exit_code = 0
    try:
        run(args)
    except BaseException:
        exit_code = 1
        traceback.print_exc()
        raise
    finally:
        kwargs = (
            {"exit_code": exit_code}
            if "exit_code" in inspect.signature(launcher.app.close).parameters
            else {}
        )
        launcher.app.close(**kwargs)


def configure(args):
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab_physx.sensors import ContactSensorCfg
    from isaaclab_tasks.manager_based.navigation.config.anymal_c.navigation_env_cfg import (
        NavigationEnvCfg,
    )
    from isaaclab_tasks.utils.hydra import resolve_presets

    cfg = resolve_presets(NavigationEnvCfg())
    cfg.seed = args.seeds[0]
    cfg.sim.device = args.device
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 22.0
    cfg.scene.lazy_sensor_update = False  # contact history at every 5 ms physics step
    cfg.decimation = cfg.actions.pre_trained_policy_action.low_level_decimation
    cfg.sim.render_interval = cfg.decimation * 5
    cfg.episode_length_s = (args.steps + 50) * cfg.sim.dt * cfg.decimation
    cfg.actions.pre_trained_policy_action.debug_vis = False
    cfg.actions.pre_trained_policy_action.low_level_observations.enable_corruption = False
    cfg.commands.pose_command.debug_vis = False
    cfg.commands.pose_command.resampling_time_range = (1000.0, 1000.0)
    cfg.observations.policy.enable_corruption = False
    cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    cfg.terminations.base_contact = None

    def material(color, metallic=0.0):
        return sim.PreviewSurfaceCfg(diffuse_color=color, metallic=metallic, roughness=0.55)

    def box(name, pos, size, color, physical=False):
        spawn = sim.CuboidCfg(size=size, visual_material=material(color))
        cls = AssetBaseCfg
        if physical:
            cls = RigidObjectCfg
            spawn.rigid_props = sim.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True
            )
            spawn.collision_props = sim.CollisionPropertiesCfg()
            spawn.mass_props = sim.MassPropertiesCfg(mass=80.0)
            spawn.activate_contact_sensors = True
        setattr(
            cfg.scene,
            name,
            cls(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=spawn,
                init_state=cls.InitialStateCfg(pos=pos),
            ),
        )

    # Each collision body gets its own one-to-many sensor against ALL robot links.
    # Ground contacts are intentionally excluded. Three independent sensors avoid
    # PhysX's unsupported many-to-many filtered contact configuration.
    box("Cart", (2.6, -2.4, 0.55), (0.9, 1.1, 0.9), (0.93, 0.46, 0.065), True)
    for wheel_id, (x, y) in enumerate(((-0.45, -0.36), (-0.45, 0.36), (0.45, -0.36), (0.45, 0.36))):
        setattr(
            cfg.scene,
            f"CartWheel{wheel_id}",
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Cart/Wheel" + str(wheel_id),
                spawn=sim.CylinderCfg(
                    radius=0.11,
                    height=0.08,
                    axis="X",
                    visual_material=material((0.035, 0.045, 0.055)),
                    collision_props=sim.CollisionPropertiesCfg(),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, -0.44)),
            ),
        )
    for side, y in (("LeftRack", 2.7), ("RightRack", -2.7)):
        box(side + "A", (0.15, y, 0.8), (2.3, 0.55, 1.6), (0.10, 0.16, 0.20), True)
        box(side + "B", (5.5, y, 0.8), (3.4, 0.55, 1.6), (0.10, 0.16, 0.20), True)
    for name in OBSTACLES:
        setattr(
            cfg.scene,
            name + "Contacts",
            ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                update_period=0.0,
                history_length=cfg.decimation,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/" + link for link in ROBOT_LINKS],
                max_contact_data_count_per_prim=64,
            ),
        )
    # Shelving details and lane markings are decorative; physical rack envelopes
    # are the boxes above, conservatively filling their visible storage volumes.
    for side, y in enumerate((-2.7, 2.7)):
        for i in (0, 1, 5, 6, 7):
            x = -0.5 + i
            box(
                f"RackPost{side}_{i}",
                (x, y - 0.32, 0.88),
                (0.055, 0.07, 1.76),
                (0.035, 0.055, 0.075),
            )
            color = ((0.57, 0.35, 0.18), (0.21, 0.43, 0.49), (0.48, 0.55, 0.57))[i % 3]
            box(f"Cargo{side}_{i}", (x + 0.35, y, 1.83), (0.58, 0.48, 0.46), color)
        for z in (0.3, 0.95, 1.6):
            for section, x, length in (("A", 0.15, 2.3), ("B", 5.5, 3.4)):
                box(
                    f"Rail{side}_{section}_{int(z * 100)}",
                    (x, y - 0.32, z),
                    (length, 0.08, 0.065),
                    (0.85, 0.52, 0.10),
                )
        box(f"AisleStripe{side}", (3.0, y * 0.74, 0.008), (8.5, 0.055, 0.01), (0.94, 0.72, 0.16))
    for i in range(10):
        box(
            f"CrossingStripe{i}",
            (2.6, -1.8 + i * 0.4, 0.009),
            (1.15, 0.16, 0.012),
            (0.48, 0.49, 0.43),
        )
    box("DeliveryPad", (6.0, 0.0, 0.012), (1.0, 1.25, 0.016), (0.10, 0.49, 0.37))
    box("StartPad", (0.0, 0.0, 0.011), (1.0, 1.25, 0.014), (0.18, 0.32, 0.47))
    cfg.viewer.eye = (3.1, -8.3, 8.2)
    cfg.viewer.lookat = (3.0, 0.0, 0.2)
    if args.frames:
        cfg.scene.recording_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/RecordingCamera",
            update_period=0.1,
            width=args.width,
            height=args.height,
            data_types=["rgb"],
            spawn=sim.PinholeCameraCfg(
                focal_length=24.0, horizontal_aperture=25.0, clipping_range=(0.1, 100.0)
            ),
        )
    return cfg


def run(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from isaac_quaternion import IDENTITY_XYZW, assert_xyzw_identity, quaternion_yaw_cos_sin
    from isaaclab.envs import ManagerBasedRLEnv
    from PIL import Image
    from warehouse_model import (
        CONTROL_LIMITS,
        GOAL,
        SAFE_RADIUS,
        make_filter,
        residuals,
        scenario_parameters,
    )

    from cbfkit.wrappers.torch import TorchSafetyFilter

    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("CUDA JAX required")
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = configure(args)
    env = ManagerBasedRLEnv(cfg=cfg)
    robot, cart = env.scene["robot"], env.scene["Cart"]
    for asset in (robot, cart):
        assert_xyzw_identity(asset.data.default_root_state[:, 3:7])
    if set(robot.body_names) != set(ROBOT_LINKS):
        raise RuntimeError(f"Contact coverage does not match robot links: {robot.body_names}")
    origins = env.scene.env_origins
    device = env.device
    contacts = [env.scene[n + "Contacts"] for n in OBSTACLES]
    sf = TorchSafetyFilter(make_filter(args.num_envs, env.step_dt))
    compute_residual = jax.jit(jax.vmap(residuals))
    # The cart follows the same exogenous path in every policy variant. Update
    # its pose/velocity before EACH physics substep, not only on control ticks.
    original_apply = env.action_manager.apply_action
    driver = {"tick": 0, "params": None}

    def cart_state(t):
        p = driver["params"]
        y = torch.minimum(p[:, 1] + t * p[:, 2], torch.full_like(p[:, 1], 3.8))
        vy = torch.where(y < 3.8, p[:, 2], 0.0)
        return torch.stack((p[:, 0], y, torch.zeros_like(y), vy), dim=-1)

    def place_cart(t):
        state = cart_state(t)
        pose = torch.zeros((args.num_envs, 7), device=device)
        pose[:, :3] = origins
        pose[:, :2] += state[:, :2]
        pose[:, 2] += 0.55
        pose[:, 3:7] = torch.tensor(IDENTITY_XYZW, device=device)
        cart.write_root_pose_to_sim(pose)
        velocity = torch.zeros((args.num_envs, 6), device=device)
        velocity[:, :2] = state[:, 2:4]
        cart.write_root_velocity_to_sim(velocity)

    def apply_action():
        driver["tick"] += 1
        place_cart(driver["tick"] * env.physics_dt)
        original_apply()

    env.action_manager.apply_action = apply_action
    all_results = []
    try:
        for seed in args.seeds:
            for mode in args.modes:
                name = f"seed-{seed}-{mode}"
                folder = args.output / name
                folder.mkdir()
                driver["params"] = torch.tensor(
                    scenario_parameters(seed, args.num_envs), device=device
                )
                driver["tick"] = 0
                with torch.inference_mode():
                    env.reset(seed=seed)
                root = robot.data.default_root_state.clone()
                root[:, :3] += origins
                root[:, 1] += driver["params"][:, 3]
                robot.write_root_pose_to_sim(root[:, :7])
                robot.write_root_velocity_to_sim(root[:, 7:])
                place_cart(0.0)
                env.scene.write_data_to_sim()
                env.sim.forward()
                env.scene.update(0.0)
                sf.reset(seed=seed)
                collided = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
                fallen = torch.zeros_like(collided)
                reached = torch.zeros_like(collided)
                first_goal = torch.full((args.num_envs,), -1.0, device=device)
                peak_force = torch.zeros(args.num_envs, device=device)
                min_clearance = torch.full((args.num_envs,), float("inf"), device=device)
                interventions = 0
                filter_failures = residual_violations = 0
                min_command_residual = float("inf")
                residual_examples = []
                filter_ms, step_ms = [], []
                frame_count = 0
                if args.frames:
                    camera = env.scene["recording_camera"]
                    camera.set_world_poses_from_view(
                        origins + torch.tensor(cfg.viewer.eye, device=device),
                        origins + torch.tensor(cfg.viewer.lookat, device=device),
                    )
                    (folder / "frames").mkdir()

                def capture(index):
                    nonlocal frame_count
                    before = robot.data.root_pos_w.clone()
                    env.sim.render()
                    camera.update(0.0, force_recompute=True)
                    pixels = camera.data.output["rgb"]
                    if not torch.allclose(before, robot.data.root_pos_w, atol=1e-7, rtol=0):
                        raise RuntimeError("Camera advanced physics")
                    for camera_id in range(args.num_envs if args.all_frames else 1):
                        rgb = pixels[camera_id].cpu().numpy()[:, :, :3]
                        if rgb.shape != (args.height, args.width, 3) or rgb.std() <= 1:
                            raise RuntimeError("Missing/blank warehouse camera frame")
                        Image.fromarray(rgb).save(
                            folder / "frames" / f"env-{camera_id:02d}-{index:04d}.png"
                        )
                    frame_count += 1

                if args.frames:
                    capture(0)
                initial_xy = (robot.data.root_pos_w - origins)[:, :2].tolist()
                with (folder / "trajectory.jsonl").open("w") as trajectory:
                    for step in range(args.steps):
                        xy = (robot.data.root_pos_w - origins)[:, :2]
                        obstacle = cart_state(driver["tick"] * env.physics_dt)
                        state = torch.cat((xy, obstacle), dim=-1)
                        delta = torch.tensor(GOAL, device=device) - xy
                        nominal = 0.9 * delta
                        nominal *= torch.clamp(
                            CONTROL_LIMITS[0]
                            / torch.linalg.vector_norm(nominal, dim=-1).clamp_min(1e-6),
                            max=1.0,
                        )[:, None]
                        nominal = torch.where(fallen[:, None], 0.0, nominal)
                        nominal[:, 1] = nominal[:, 1].clamp(-CONTROL_LIMITS[1], CONTROL_LIMITS[1])
                        applied = nominal.clone()
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        if mode == "filtered":
                            applied, info = sf.filter(state, nominal)
                            torch.cuda.synchronize()
                            filter_ms.append(1000 * (time.perf_counter() - started))
                            filter_failures += int(info["fallback_used"].sum())
                            if filter_failures or not bool(torch.isfinite(applied).all()):
                                raise RuntimeError("Failed CBF control; aborting")
                            r = compute_residual(
                                jnp.from_dlpack(state).astype(jnp.float64),
                                jnp.from_dlpack(applied).astype(jnp.float64),
                            )
                            residual_violations += int((r < -1e-4).any(axis=-1).sum())
                            min_command_residual = min(min_command_residual, float(r.min()))
                            if float(r.min()) < -1e-4 and len(residual_examples) < 8:
                                sample = dict(
                                    step=step,
                                    state=state.tolist(),
                                    nominal=nominal.tolist(),
                                    applied=applied.tolist(),
                                    residuals=r.tolist(),
                                )
                                residual_examples.append(sample)
                                print("WAREHOUSE_RESIDUAL " + json.dumps(sample), flush=True)
                        elif mode == "stop":
                            distance = torch.linalg.vector_norm(xy - obstacle[:, :2], dim=-1)
                            applied = torch.where(
                                (distance < SAFE_RADIUS + 0.3)[:, None], 0.0, nominal
                            )
                        changed = torch.linalg.vector_norm(applied - nominal, dim=-1) > 1e-3
                        interventions += int(changed.sum())
                        c, s = quaternion_yaw_cos_sin(robot.data.root_quat_w, "xyzw")
                        actions = torch.zeros((args.num_envs, 3), device=device)
                        actions[:, 0] = c * applied[:, 0] + s * applied[:, 1]
                        actions[:, 1] = -s * applied[:, 0] + c * applied[:, 1]
                        actions[:, 2] = torch.clamp(-1.5 * torch.atan2(s, c), -0.6, 0.6)
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        with torch.inference_mode():
                            _, _, terminated, truncated, _ = env.step(actions)
                        torch.cuda.synchronize()
                        step_ms.append(1000 * (time.perf_counter() - started))
                        if bool((terminated | truncated).any()):
                            raise RuntimeError(
                                "Unexpected reset would invalidate paired measurements"
                            )
                        for sensor in contacts:
                            forces = sensor.data.force_matrix_w_history
                            if forces is None or forces.shape[-2] == 0:
                                raise RuntimeError("Missing obstacle-to-robot contact channels")
                            f = (
                                torch.linalg.vector_norm(forces.torch, dim=-1)
                                .flatten(1)
                                .amax(dim=1)
                            )
                            peak_force = torch.maximum(peak_force, f)
                            collided |= f > 1.0
                        xy = (robot.data.root_pos_w - origins)[:, :2]
                        fallen |= (robot.data.root_pos_w[:, 2] - origins[:, 2] < 0.25) | (
                            robot.data.projected_gravity_b[:, 2] > -0.5
                        )
                        distance_goal = torch.linalg.vector_norm(
                            torch.tensor(GOAL, device=device) - xy, dim=-1
                        )
                        now_reached = distance_goal < 0.35
                        first_goal = torch.where(
                            now_reached & ~reached, (step + 1) * env.step_dt, first_goal
                        )
                        reached |= now_reached
                        clearance = (
                            torch.linalg.vector_norm(
                                xy - cart_state(driver["tick"] * env.physics_dt)[:, :2], dim=-1
                            )
                            - SAFE_RADIUS
                        )
                        min_clearance = torch.minimum(min_clearance, clearance)
                        if step % 5 == 4 or step == args.steps - 1:
                            trajectory.write(
                                json.dumps(
                                    {
                                        "step": step + 1,
                                        "time_s": (step + 1) * env.step_dt,
                                        "xy": xy.tolist(),
                                        "cart": cart_state(
                                            driver["tick"] * env.physics_dt
                                        ).tolist(),
                                        "nominal": nominal.tolist(),
                                        "applied": applied.tolist(),
                                        "intervened": changed.tolist(),
                                        "collided": collided.tolist(),
                                        "reached": reached.tolist(),
                                        "fallen": fallen.tolist(),
                                    }
                                )
                                + "\n"
                            )
                        if args.frames and (step + 1) % 5 == 0:
                            capture((step + 1) // 5)
                        if step % 100 == 0:
                            print(
                                f"WAREHOUSE_STEP {name} {step}/{args.steps} contacts={int(collided.sum())} goals={int(reached.sum())}",
                                flush=True,
                            )
                result = dict(
                    name=name,
                    mode=mode,
                    seed=seed,
                    num_envs=args.num_envs,
                    steps=args.steps,
                    dt_s=env.step_dt,
                    physics_dt_s=env.physics_dt,
                    contact_sample_dt_s=env.physics_dt,
                    contact_threshold_n=1.0,
                    contact_sensor_bodies=[s.body_names for s in contacts],
                    robot_link_names=robot.body_names,
                    contact_channel_shapes=[
                        list(s.data.force_matrix_w_history.shape) for s in contacts
                    ],
                    scenario_parameters=driver["params"].tolist(),
                    initial_xy=initial_xy,
                    goal=list(GOAL),
                    safety_radius_m=SAFE_RADIUS,
                    control_limits_m_s=list(CONTROL_LIMITS),
                    cbf_objective_weights=[1.0, 8.0],
                    collided=collided.tolist(),
                    fallen=fallen.tolist(),
                    reached=reached.tolist(),
                    successful=(reached & ~collided & ~fallen).tolist(),
                    first_goal_s=first_goal.tolist(),
                    peak_obstacle_force_n=peak_force.tolist(),
                    min_model_clearance_m=min_clearance.tolist(),
                    intervention_fraction=interventions / (args.steps * args.num_envs),
                    filter_failures=filter_failures,
                    command_residual_violations=residual_violations,
                    min_command_residual=min_command_residual if mode == "filtered" else None,
                    residual_examples=residual_examples,
                    filter_ms_after_warmup=filter_ms[2:],
                    step_ms_after_warmup=step_ms[2:],
                    rendered=args.frames,
                    frame_count=frame_count,
                    width=args.width,
                    height=args.height,
                    camera_eye=list(cfg.viewer.eye),
                    camera_lookat=list(cfg.viewer.lookat),
                    camera_intrinsics=(
                        camera.data.intrinsic_matrices[0].cpu().tolist() if args.frames else None
                    ),
                    policy_path=cfg.actions.pre_trained_policy_action.policy_path,
                    versions={
                        "torch": torch.__version__,
                        "jax": jax.__version__,
                        "isaaclab_core": importlib.metadata.version("isaaclab"),
                    },
                    jax_matmul_precision=jax.config.jax_default_matmul_precision,
                    filter_dtype="float64" if mode == "filtered" else None,
                    limitations="Simulator obstacle state; route follower, not learned navigation. Kinematic cart, geometric CBF margin, no articulated safety proof. Contacts exclude ground. Goal entry within 0.35 m. Timing excludes recording and metrics.",
                )
                (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                all_results.append(
                    {
                        k: result[k]
                        for k in (
                            "name",
                            "collided",
                            "fallen",
                            "reached",
                            "successful",
                            "filter_failures",
                            "command_residual_violations",
                        )
                    }
                )
                print("WAREHOUSE_RESULT " + json.dumps(all_results[-1]), flush=True)
        (args.output / "summary.json").write_text(json.dumps(all_results, indent=2) + "\n")
        print("WAREHOUSE_COMPLETE", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
