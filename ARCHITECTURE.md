# Plan: Create `mj_manipulator` package

## Context

Geodude (10,508 lines) mixes generic MuJoCo arm control with UR5e/Robotiq-specific code. The lab has multiple robots (Franka, Xarm, UR5e). Extracting the generic manipulation layer into `mj_manipulator` lets any MuJoCo manipulator plan trajectories, execute them, grasp/ungrasp objects (with kinematic weld + collision filtering), do cartesian teleop, and run joint-based or cartesian policies — without reimplementing the simulation loop.

**Geodude becomes a thin robot configuration package** (~2,000 lines) that plugs UR5e + Robotiq + Vention into the generic framework.

## Design Decisions (confirmed with user)

- **Depends on mj_environment** — forking is key for thread-safe planning and already well-implemented
- **Arm is concrete + injection** — one `Arm` class parameterized by config, injected IK solver, injected Gripper. No subclassing.
- **Primitives move to mj_manipulator** — with a `GraspSource` protocol so any robot that can provide grasps gets pickup/place
- **Execution context is generic** — `PhysicsController` and `SimContext` move to mj_manipulator, parameterized by arm/gripper registry instead of hardcoded "left"/"right"

## Package Structure

```
mj_manipulator/
  pyproject.toml
  src/mj_manipulator/
    __init__.py
    protocols.py        # Gripper, GraspSource, IKSolver protocols
    config.py           # ArmConfig, KinematicLimits, PlanningDefaults, PhysicsConfig
    arm.py              # Generic Arm (FK, IK, planning, execution)
    adapters.py         # pycbirrt RobotModel/IKSolver/CollisionChecker adapters
    collision.py        # Unified grasp-aware CollisionChecker
    grasp_manager.py    # GraspManager + detect_grasped_object
    gripper.py          # Gripper base utilities (protocol is in protocols.py)
    trajectory.py       # Trajectory + TOPP-RA + linear trajectory
    cartesian.py        # QP solver, twist control, move_until_touch, execute_twist
    executor.py         # KinematicExecutor, PhysicsExecutor
    controller.py       # PhysicsController (generic multi-arm physics stepping)
    context.py          # SimContext (generic execution context: physics/kinematic modes)
    primitives.py       # pickup/place via GraspSource protocol
    planning.py         # PlanResult dataclass
  tests/
    ...
```

## Core Protocols

### `protocols.py`

```python
class Gripper(Protocol):
    """Any gripper (Robotiq, Franka hand, suction, etc.)."""
    arm_name: str
    gripper_body_names: list[str]
    attachment_body: str          # Body objects weld to during kinematic grasp
    actuator_id: int | None
    ctrl_open: float
    ctrl_closed: float

    def kinematic_close(self, steps: int = 50) -> str | None: ...
    def kinematic_open(self) -> None: ...
    def get_actual_position(self) -> float: ...  # 0=open, 1=closed
    is_holding: bool
    held_object: str | None
    def set_candidate_objects(self, objects: list[str] | None) -> None: ...

class GraspSource(Protocol):
    """Provides grasps/placements for objects. Geodude's AffordanceRegistry implements this."""
    def get_grasps(self, object_name: str, hand_type: str) -> list[TSR]: ...
    def get_placements(self, destination: str, object_name: str) -> list[TSR]: ...
    def get_graspable_objects(self) -> list[str]: ...
    def get_place_destinations(self, object_name: str) -> list[str]: ...

class IKSolver(Protocol):
    """Mirrors pycbirrt's IKSolver protocol."""
    def solve(self, pose, q_init=None) -> list[np.ndarray]: ...
    def solve_valid(self, pose, q_init=None) -> list[np.ndarray]: ...
```

### Key: `Gripper.attachment_body`

This is the body that objects weld to during kinematic grasping. Currently hardcoded as `f"{side}_ur5e/gripper/right_follower"` in geodude's `execution.py`. Each gripper implementation provides its own (Robotiq follower link, Franka finger pad, suction cup tip, etc.).

## What Moves vs Stays

### Moves to mj_manipulator

| geodude module | → mj_manipulator module | Notes |
|---|---|---|
| `trajectory.py` (350) | `trajectory.py` | Verbatim |
| `planning.py` (58) | `planning.py` | Verbatim |
| `grasp_manager.py` (353) | `grasp_manager.py` | Parameterize finger detection |
| `collision.py` (652) | `collision.py` | Unify 3 classes → 1 (per #62) |
| `cartesian.py` (988) | `cartesian.py` | Replace `robot._active_context` with `step_fn` param |
| `arm.py` generic parts (~800) | `arm.py` | FK, joint control, planning, execution |
| `arm.py` adapters (~200) | `adapters.py` | ArmRobotModel, ContextRobotModel, SimpleIKSolver |
| `executor.py` executors (~300) | `executor.py` | KinematicExecutor, PhysicsExecutor |
| `executor.py` controller (~600) | `controller.py` | Generalize RobotPhysicsController (dict of arms, not "left"/"right") |
| `execution.py` context (~400) | `context.py` | Generalize SimContext (arm registry, not hardcoded sides) |
| `primitives.py` (~843) | `primitives.py` | Use GraspSource protocol instead of AffordanceRegistry directly |
| `config.py` generic parts (~400) | `config.py` | KinematicLimits, PlanningDefaults, ArmConfig, PhysicsConfig |
| NEW | `protocols.py` | Gripper, GraspSource, IKSolver protocols |

### Stays in geodude

| Module | Lines (est.) | What remains |
|---|---|---|
| `config.py` | ~280 | GeodudConfig, VentionBaseConfig, VentionKinematicLimits, DebugConfig, UR5e defaults |
| `arm.py` | ~200 | UR5e IK solver factory (EAIK setup, base rotation, EE offset), F/T sensor utility |
| `gripper.py` | ~520 | Robotiq 2F-140 (implements Gripper protocol). Unchanged. |
| `robot.py` | ~500 | Geodude class: constructs arms/grippers/bases, named poses, wires everything |
| `vention_base.py` | ~371 | Vention linear actuator. Unchanged. |
| `affordances.py` | ~392 | AffordanceRegistry (implements GraspSource protocol) |
| `tsr_utils.py` | ~200 | Gripper frame compensation (shrinks after #69 TSR migration) |
| `__init__.py` | ~60 | Re-exports |
| **Total** | **~2,500** | **76% reduction from 10,508** |

### mj_manipulator estimated size: ~3,500 lines

Combined total: ~6,000 lines (vs 10,508 before). ~4,500 lines eliminated through dedup during extraction (collision unification, cartesian dedup, dead code removal, executor cleanup).

## Arm Constructor

```python
# mj_manipulator/arm.py
class Arm:
    def __init__(
        self,
        env: Environment,           # mj_environment — provides model, data, fork, sync
        config: ArmConfig,          # joint names, EE site, limits, etc.
        grasp_manager: GraspManager,
        gripper: Gripper | None = None,
        ik_solver: IKSolver | None = None,
        name: str = "",             # e.g., "left", "right", "franka"
    ): ...
```

Takes `env: Environment` (not raw model/data) since forking is essential for planning.

## How Geodude Uses mj_manipulator

```python
# geodude/robot.py
from mj_manipulator import Arm, GraspManager, ArmConfig
from geodude.gripper import Robotiq2F140
from geodude.ur5e import create_ur5e_ik_solver

class Geodude:
    def __init__(self, config, objects=None):
        self.env = Environment(...)
        self.grasp_manager = GraspManager(self.env.model, self.env.data)

        left_gripper = Robotiq2F140(self.env, "left", config.left_arm, self.grasp_manager)
        right_gripper = Robotiq2F140(self.env, "right", config.right_arm, self.grasp_manager)

        self.left_arm = Arm(
            env=self.env, config=config.left_arm,
            grasp_manager=self.grasp_manager,
            gripper=left_gripper,
            ik_solver=create_ur5e_ik_solver(self.env, config.left_arm),
            name="left",
        )
        # ... right_arm similarly
```

## How Franka Would Use mj_manipulator

```python
# franka_control/robot.py
from mj_manipulator import Arm, GraspManager, ArmConfig, KinematicLimits

franka_config = ArmConfig(
    name="franka", entity_type="arm",
    joint_names=[f"franka/joint{i}" for i in range(1, 8)],
    ee_site="franka/ee_site",
    gripper_actuator="franka/gripper_actuator",
    gripper_bodies=["franka/left_finger", "franka/right_finger"],
    hand_type="franka_hand",
    kinematic_limits=KinematicLimits(
        velocity=np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]),
        acceleration=np.array([15, 7.5, 10, 12.5, 15, 20, 20]),
    ),
)

grasp_manager = GraspManager(env.model, env.data)
gripper = FrankaHand(env, franka_config, grasp_manager)  # implements Gripper protocol
arm = Arm(env=env, config=franka_config, grasp_manager=grasp_manager,
          gripper=gripper, ik_solver=FrankaIK(env), name="franka")

# All of these work immediately:
arm.get_ee_pose()
arm.plan_to_pose(target)
arm.plan_to_tsr(grasp_tsr)

# Cartesian teleop:
from mj_manipulator.cartesian import execute_twist
execute_twist(arm, twist, step_fn=ctx.step_cartesian, control_dt=ctx.control_dt)

# Run a joint policy:
with SimContext(env, arms={"franka": arm}, physics=True) as ctx:
    while not done:
        q_target = policy(arm.get_joint_positions())
        ctx.step({"franka": q_target})

# Pickup/place:
from mj_manipulator.primitives import pickup
pickup(arm, "mug", grasp_source=my_grasp_source, context=ctx)
```

## Kinematic Grasp Pipeline (critical for correctness)

When `pickup()` grasps an object in kinematic mode:

1. `gripper.kinematic_close()` → detects contact, returns object name
2. `grasp_manager.mark_grasped(object_name, arm_name)` → records grasp state
3. `grasp_manager.attach_object(object_name, gripper.attachment_body)` → computes & stores relative transform (weld)
4. All subsequent `collision_checker.is_valid(q)` calls filter out gripper↔object contacts
5. Every simulation step: `grasp_manager.update_attached_poses(data)` → moves object with gripper
6. On release: `grasp_manager.detach_object()` + `mark_released()` → object stays in place, collisions re-enabled

`gripper.attachment_body` is the key generic interface — each gripper implementation specifies which body objects attach to.

## PhysicsController (generalized from RobotPhysicsController)

Current `RobotPhysicsController` hardcodes "left"/"right" arms. Generic version uses a dict:

```python
class PhysicsController:
    def __init__(self, env, arms: dict[str, Arm], grippers: dict[str, Gripper],
                 physics_config: PhysicsConfig, viewer=None): ...

    def step(self, targets: dict[str, np.ndarray] | None = None) -> None:
        """Step all arms toward their targets."""

    def step_cartesian(self, arm_name: str, position: np.ndarray,
                       velocity: np.ndarray | None = None) -> None:
        """Reactive step for cartesian control."""

    def execute(self, arm_name: str, trajectory: Trajectory) -> None: ...
    def close_gripper(self, arm_name: str) -> str | None: ...
    def open_gripper(self, arm_name: str) -> None: ...
```

## SimContext (generalized)

```python
class SimContext:
    def __init__(self, env, arms: dict[str, Arm],
                 physics: bool = False, viewer: bool = True,
                 physics_config: PhysicsConfig | None = None): ...
```

Takes `arms: dict[str, Arm]` instead of assuming `robot.left_arm`/`robot.right_arm`. Camera setup becomes optional/configurable rather than hardcoded Geodude angles.

## Migration Strategy (incremental, never breaks geodude)

### Phase 1: Scaffold + pure data types
- Create `mj_manipulator/` with pyproject.toml
- Move: `trajectory.py`, `planning.py`, `config.py` (generic parts), `protocols.py` (new)
- Geodude re-exports from mj_manipulator (shims)
- **Test**: `uv run pytest tests/ -v` in geodude passes

### Phase 2: Grasp management + collision
- Move: `grasp_manager.py`, unified `collision.py` (does #62 during extraction)
- Update geodude imports
- **Test**: geodude tests pass

### Phase 3: Executors + cartesian control
- Move: `KinematicExecutor`, `PhysicsExecutor`, cartesian functions
- Break `arm.robot._active_context` dependency (step_fn parameter)
- Add thin compat wrappers in geodude during transition
- **Test**: geodude tests pass

### Phase 4: Arm class
- Create `mj_manipulator.Arm` (generic, takes `env: Environment`)
- Geodude's Arm becomes a thin construction wrapper that builds mj_manipulator.Arm with UR5e config + EAIK solver + Robotiq gripper
- **Test**: all planning/execution tests pass

### Phase 5: Controller + context
- Generalize `RobotPhysicsController` → `PhysicsController` (arm dict, not left/right)
- Generalize `SimContext` → parameterized by arm registry
- **Test**: physics-mode tests pass

### Phase 6: Primitives
- Move `primitives.py` with `GraspSource` protocol
- Geodude's `AffordanceRegistry` implements `GraspSource`
- **Test**: pickup/place tests pass

### Phase 7: Cleanup
- Remove geodude shims/re-exports
- Delete moved files from geodude
- Update geodude's `__init__.py`
- Run full test suite

### Phase 8: Validate
- Create minimal Franka example using mj_manipulator + mujoco_menagerie model
- Verify: plan trajectory, execute, cartesian teleop, grasp/release all work

## Cleanup Issues Disposition

| Issue | Action |
|---|---|
| **#61** (dead code) | Do first, before extraction. Pure deletion, reduces noise. |
| **#62** (collision unify) | Fold into Phase 2 — unification happens during extraction |
| **#63** (arm.py decomp) | Fold into Phase 4 — splitting into generic Arm + UR5e parts IS the decomposition |
| **#64** (planning dispatch) | Fold into Phase 4 — simplified during Arm extraction |
| **#65** (cartesian dedup) | Fold into Phase 3 — dedup during extraction |
| **#66** (executor internals) | Fold into Phase 5 — controller generalization |
| **#67** (config/data extern) | Do separately — gripper trajectory extraction is geodude-internal |
| **#68** (small modules) | Mostly absorbed — planning.py merges, parallel.py deleted, primitives move |
| **#69** (TSR migration) | Do separately — orthogonal to extraction |
| **#70** (robot.py pass-throughs) | Absorbed — robot.py shrinks naturally |

## Verification (2-arm validation at each phase)

Each phase must include tests and a working demo with at least **UR5e and Franka** arms:

- **Phase 1**: Unit tests for Trajectory, PlanResult, config serialization. Demo: create ArmConfig for both UR5e and Franka.
- **Phase 2**: Unit tests for CollisionChecker, GraspManager. Demo: check collisions with both arm models.
- **Phase 3**: Unit tests for executors, cartesian QP solver. Demo: cartesian twist step with both arms.
- **Phase 4**: Integration tests for Arm (FK, planning). Demo: plan + execute trajectory with both arms.
- **Phase 5**: Integration tests for PhysicsController, SimContext. Demo: physics-mode stepping with both arms.
- **Phase 6**: Integration tests for pickup/place. Demo: pick object with both arms.
- **Phase 7**: Full regression. All geodude tests pass.
- **Phase 8**: End-to-end Franka example: plan, execute, grasp, cartesian teleop.

After each phase: `cd mj_manipulator && uv run pytest tests/ -v`
After Phase 7+: `cd geodude && uv run pytest tests/ -v`

## Workflow

- This plan lives as a WIP PR in the mj_manipulator repo
- Each phase is a separate commit (or small set of commits)
- PR description tracks progress with checkboxes
- Tests and demos are committed alongside code

## Key Files to Modify

**geodude (read before modifying):**
- `src/geodude/arm.py` (2091 lines) — split into generic + UR5e
- `src/geodude/collision.py` (652 lines) — unify + move
- `src/geodude/cartesian.py` (988 lines) — break context dep + move
- `src/geodude/executor.py` (1066 lines) — split executors vs controller
- `src/geodude/execution.py` (539 lines) — generalize context
- `src/geodude/primitives.py` (843 lines) — add GraspSource protocol + move
- `src/geodude/config.py` (680 lines) — split generic vs geodude-specific
- `src/geodude/grasp_manager.py` (353 lines) — parameterize + move
- `src/geodude/trajectory.py` (350 lines) — move
- `src/geodude/gripper.py` (520 lines) — stays, implements Gripper protocol
- `src/geodude/robot.py` (882 lines) — slim down to construction + wiring
- `src/geodude/affordances.py` (392 lines) — stays, implements GraspSource

**pycbirrt (read-only reference):**
- `src/pycbirrt/interfaces/robot_model.py` — RobotModel protocol
- `src/pycbirrt/interfaces/ik_solver.py` — IKSolver protocol
- `src/pycbirrt/interfaces/collision_checker.py` — CollisionChecker protocol

**mj_environment (read-only reference):**
- `mj_environment/environment.py` — Environment.fork(), model, data
