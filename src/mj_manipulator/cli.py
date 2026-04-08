# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""CLI entry point for mj_manipulator interactive console.

Launches an IPython console with a single-arm robot (UR5e or Franka),
physics simulation, and viser web viewer with teleop.

Usage::

    python -m mj_manipulator                        # UR5e + viser (default)
    python -m mj_manipulator --robot franka          # Franka + viser
    python -m mj_manipulator --physics               # with physics simulation
    python -m mj_manipulator --objects '{"can": 3}'  # spawn objects
    python -m mj_manipulator --no-viser              # headless, no viewer
"""

from __future__ import annotations

import argparse
import sys

from mj_manipulator.robot import RobotBase


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mj_manipulator",
        description="Interactive manipulation console",
    )
    parser.add_argument(
        "--robot",
        choices=["ur5e", "franka"],
        default="ur5e",
        help="Robot type (default: ur5e)",
    )
    parser.add_argument("--physics", action="store_true", help="Enable physics simulation")
    parser.add_argument("--no-viser", action="store_true", help="Disable viser web viewer")
    parser.add_argument("--objects", type=str, default=None, help="Objects JSON, e.g. '{\"can\": 3}'")
    args = parser.parse_args()

    objects = {}
    if args.objects:
        import json

        objects = json.loads(args.objects)

    print(f"\nLoading {args.robot}...", flush=True)

    robot = _setup_robot(args.robot, objects)

    arm_name = list(robot.arms.keys())[0]

    def commands():
        """Print available commands."""
        print(
            f"""
Quick Reference
===============

Scene:
  robot.find_objects()              — list all graspable objects
  robot.get_object_pose("can_0")    — 4x4 pose matrix
  robot.holding()                   — (arm, name) or None

Actions:
  robot.pickup()                    — pick up nearest reachable object
  robot.pickup("can_0")             — pick up specific object
  robot.place("yellow_tote")        — place in recycling bin
  robot.place("worktop")            — place on table surface
  robot.go_home()                   — return arm to ready

Arm:
  robot.arms["{arm_name}"].get_ee_pose()
  robot.arms["{arm_name}"].get_joint_positions()

Teleop:
  Click "Activate" in the viser viewer

IPython:
  robot.<tab>             — tab completion
  ?robot.pickup           — docstring
  commands()              — this help
"""
        )

    from mj_manipulator.console import start_console

    start_console(
        robot,
        physics=args.physics,
        viser=not args.no_viser,
        robot_name=args.robot.upper(),
        extra_ns={"commands": commands},
    )


def _setup_robot(robot_type: str, objects: dict):
    """Create a single-arm robot with optional objects."""

    from mj_manipulator.menagerie import menagerie_scene

    if robot_type == "ur5e":
        scene_path = menagerie_scene("universal_robots_ur5e")
        if not scene_path.exists():
            print(f"ERROR: UR5e scene not found at {scene_path}")
            print("Run ./setup.sh from the robot-code workspace to clone mujoco_menagerie.")
            sys.exit(1)
        robot = _setup_ur5e(scene_path, objects)
    elif robot_type == "franka":
        scene_path = menagerie_scene("franka_emika_panda")
        if not scene_path.exists():
            print(f"ERROR: Franka scene not found at {scene_path}")
            print("Run ./setup.sh from the robot-code workspace to clone mujoco_menagerie.")
            sys.exit(1)
        robot = _setup_franka(scene_path, objects)
    else:
        print(f"Unknown robot: {robot_type}")
        sys.exit(1)

    return robot


class _SimpleRobot(RobotBase):
    """Minimal robot for the CLI demo. Inherits all convenience methods
    from RobotBase (pickup, place, go_home, find_objects, etc.).

    This is the reference implementation — ~20 LOC on top of RobotBase.
    """

    def __init__(self, env, arm, home_config, has_objects=False):
        from mj_manipulator.grasp_manager import GraspManager as _GM

        gm = arm.grasp_manager or _GM(env.model, env.data)
        super().__init__(
            model=env.model,
            data=env.data,
            arms={arm.config.name: arm},
            grasp_manager=gm,
            named_poses={"ready": {arm.config.name: list(home_config)}},
        )
        self._env = env
        self._has_objects = has_objects

    @property
    def grasp_source(self):
        if self._grasp_source is None:
            if self._has_objects:
                from mj_manipulator.grasp_sources.prl_assets import PrlAssetsGraspSource

                registry = getattr(self._env, "registry", None)
                self._grasp_source = PrlAssetsGraspSource(
                    self.model,
                    self.data,
                    self.grasp_manager,
                    self.arms,
                    registry=registry,
                )
        return super().grasp_source


def _save_robot_xml(spec) -> str:
    """Save assembled robot MjSpec to a temp file for Environment."""
    import tempfile

    xml_str = spec.to_xml()
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
    f.write(xml_str)
    f.close()
    return f.name


def _create_scene_config(objects: dict) -> str:
    """Create a temp scene_config.yaml from objects dict."""
    import tempfile

    import yaml

    config = {"objects": {obj_type: {"count": count} for obj_type, count in objects.items()}}
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(config, f)
    f.close()
    return f.name


def _setup_ur5e(scene_path, objects):
    """Set up UR5e + Robotiq."""
    import geodude_assets
    import mujoco
    from mj_environment import Environment
    from prl_assets import OBJECTS_DIR

    from mj_manipulator.arms.ur5e import UR5E_HOME, UR5E_ROBOTIQ_EE_SITE, create_ur5e_arm
    from mj_manipulator.grasp_manager import GraspManager
    from mj_manipulator.grippers.robotiq import RobotiqGripper

    # Assemble robot: UR5e + Robotiq gripper
    spec = mujoco.MjSpec.from_file(str(scene_path))
    robotiq_path = geodude_assets.MODELS_DIR / "robotiq_2f140" / "2f140.xml"
    robotiq_spec = mujoco.MjSpec.from_file(str(robotiq_path))
    wrist = spec.worldbody.find_child("wrist_3_link")
    frame = wrist.add_frame()
    frame.pos = [0, 0.1, 0]
    frame.quat = [-1, 1, 0, 0]
    frame.attach_body(robotiq_spec.worldbody.first_body(), prefix="gripper/")

    # Save assembled robot XML, then use Environment to add objects
    # (Environment handles body naming, registry, and object lifecycle)
    robot_xml = _save_robot_xml(spec)

    if objects:
        scene_config = _create_scene_config(objects)
        env = Environment(
            base_scene_xml=robot_xml,
            objects_dir=str(OBJECTS_DIR),
            scene_config_yaml=scene_config,
        )
    else:
        env = Environment(base_scene_xml=robot_xml)

    gm = GraspManager(env.model, env.data)
    gripper = RobotiqGripper(env.model, env.data, "ur5e", prefix="gripper/", grasp_manager=gm)
    arm = create_ur5e_arm(env, ee_site=UR5E_ROBOTIQ_EE_SITE, gripper=gripper, grasp_manager=gm)

    for i, idx in enumerate(arm.joint_qpos_indices):
        env.data.qpos[idx] = UR5E_HOME[i]
    mujoco.mj_forward(env.model, env.data)

    return _SimpleRobot(env, arm, UR5E_HOME, has_objects=bool(objects))


def _setup_franka(scene_path, objects):
    """Set up Franka Panda."""
    import mujoco
    from mj_environment import Environment
    from prl_assets import OBJECTS_DIR

    from mj_manipulator.arms.franka import FRANKA_HOME, add_franka_ee_site, create_franka_arm
    from mj_manipulator.grasp_manager import GraspManager
    from mj_manipulator.grippers.franka import FrankaGripper

    # Assemble robot: Franka + EE site
    spec = mujoco.MjSpec.from_file(str(scene_path))
    add_franka_ee_site(spec)

    robot_xml = _save_robot_xml(spec)

    if objects:
        scene_config = _create_scene_config(objects)
        env = Environment(
            base_scene_xml=robot_xml,
            objects_dir=str(OBJECTS_DIR),
            scene_config_yaml=scene_config,
        )
    else:
        env = Environment(base_scene_xml=robot_xml)

    gm = GraspManager(env.model, env.data)
    gripper = FrankaGripper(env.model, env.data, "franka", grasp_manager=gm)
    arm = create_franka_arm(env, gripper=gripper, grasp_manager=gm)

    gripper.kinematic_open()
    if gripper.actuator_id is not None:
        env.data.ctrl[gripper.actuator_id] = gripper.ctrl_open

    for i, idx in enumerate(arm.joint_qpos_indices):
        env.data.qpos[idx] = FRANKA_HOME[i]
    mujoco.mj_forward(env.model, env.data)

    return _SimpleRobot(env, arm, FRANKA_HOME, has_objects=bool(objects))


def _attach_prl_objects(spec, objects: dict):
    """Attach prl_assets objects + table + recycling bin to the scene."""
    import mujoco
    from prl_assets import OBJECTS_DIR

    # Table
    table = spec.worldbody.add_body()
    table.name = "table"
    table.pos = [0.45, -0.20, 0.23]
    g = table.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [0.15, 0.15, 0.23]
    g.rgba = [0.4, 0.3, 0.2, 1.0]

    # Worktop site on the table surface (for surface placement)
    s = table.add_site()
    s.name = "worktop"
    s.pos = [0, 0, 0.23]  # top of the table
    s.size = [0.12, 0.12, 0.001]
    s.type = mujoco.mjtGeom.mjGEOM_BOX
    s.rgba = [0, 0, 0, 0]  # invisible

    # Recycling bin on the floor
    from asset_manager import AssetManager

    assets = AssetManager(str(OBJECTS_DIR))
    try:
        bin_xml = assets.get_path("yellow_tote", "mujoco")
        bin_spec = mujoco.MjSpec.from_file(bin_xml)
        f = spec.worldbody.add_frame()
        f.pos = [0.25, -0.70, 0.0]
        f.attach_body(bin_spec.worldbody.first_body(), prefix="yellow_tote_0/")
    except (KeyError, TypeError):
        pass  # no tote model available

    # Attach objects on the table
    idx = 0
    for obj_type, count in objects.items():
        try:
            xml_path = assets.get_path(obj_type, "mujoco")
        except (KeyError, TypeError):
            continue
        for i in range(count):
            obj_spec = mujoco.MjSpec.from_file(xml_path)
            f = spec.worldbody.add_frame()
            x = 0.40 + (idx % 3) * 0.05
            y = -0.25 + (idx // 3) * 0.05
            f.pos = [x, y, 0.46 + 0.05]
            f.attach_body(obj_spec.worldbody.first_body(), prefix=f"{obj_type}_{i}/")
            idx += 1
