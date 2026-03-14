# mj_manipulator

Generic MuJoCo manipulator control framework. Plan trajectories, execute them, grasp/ungrasp objects, do cartesian teleop, and run joint-based or cartesian policies — for any robot arm.

## Supported Robots

Works with any MuJoCo arm model. Tested with:
- **UR5e** (6-DOF) via [geodude](https://github.com/siddhss5/geodude)
- **Franka Emika Panda** (7-DOF) via [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)

## Installation

```bash
uv add mj-manipulator
```

For development:
```bash
git clone <repo-url> mj_manipulator
cd mj_manipulator
uv sync --extra dev
uv run pytest tests/ -v
```

## Quick Start

```python
from mj_manipulator import Arm, ArmConfig, KinematicLimits, GraspManager

# Define your robot's config
config = ArmConfig(
    name="franka",
    entity_type="arm",
    joint_names=[f"panda/joint{i}" for i in range(1, 8)],
    kinematic_limits=KinematicLimits(
        velocity=np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]),
        acceleration=np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]),
    ),
    ee_site="panda/attachment_site",
    gripper_actuator="panda/fingers_actuator",
    gripper_bodies=["panda/left_finger", "panda/right_finger"],
    hand_type="franka_hand",
)

# Create arm (more components added in later phases)
arm = Arm(env=env, config=config, grasp_manager=grasp_manager,
          gripper=my_gripper, ik_solver=my_ik_solver, name="franka")
```

## Architecture

```
mj_environment  →  mj_manipulator  →  geodude (UR5e + Robotiq)
                                    →  franka_control (future)
                                    →  xarm_control (future)
```

`mj_manipulator` provides the generic manipulation layer. Robot-specific packages provide:
- Gripper implementations (Robotiq, Franka hand, etc.)
- IK solver configuration (EAIK for UR5e, analytical for Franka)
- Robot-specific configs (joint names, body lists, named poses)

## Development Status

This package is being extracted from [geodude](https://github.com/siddhss5/geodude). See the WIP PR for progress tracking.

### Phases

- [x] **Phase 1**: Data types (Trajectory, PlanResult, Config, Protocols)
- [ ] **Phase 2**: Grasp management + collision checking
- [ ] **Phase 3**: Executors + cartesian control
- [ ] **Phase 4**: Arm class
- [ ] **Phase 5**: Physics controller + execution context
- [ ] **Phase 6**: Manipulation primitives (pickup/place)
- [ ] **Phase 7**: Cleanup + geodude integration
- [ ] **Phase 8**: Franka end-to-end validation
