"""Generic robot arm abstraction for MuJoCo manipulators.

Wraps an Environment + ArmConfig to provide:
- State queries (joint positions, EE pose, joint limits)
- Forward kinematics (non-destructive, for planning)
- Motion planning via pycbirrt (config-to-config, TSR-based, pose-based)

Robot-specific code (IK solvers, grippers) is injected via protocols.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from pycbirrt import CBiRRT, CBiRRTConfig

from mj_manipulator.collision import CollisionChecker
from mj_manipulator.config import ArmConfig
from mj_manipulator.trajectory import Trajectory

if TYPE_CHECKING:
    from mj_environment import Environment

    from mj_manipulator.grasp_manager import GraspManager
    from mj_manipulator.protocols import Gripper, IKSolver

logger = logging.getLogger(__name__)


# =============================================================================
# pycbirrt RobotModel adapters
# =============================================================================


class ArmRobotModel:
    """Adapts Arm for pycbirrt's RobotModel protocol (single-threaded).

    Uses Arm.forward_kinematics() which creates a temporary MjData copy,
    so it's safe for planning but not thread-safe.
    """

    def __init__(self, arm: Arm):
        self._arm = arm

    @property
    def dof(self) -> int:
        return self._arm.dof

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._arm.get_joint_limits()

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        return self._arm.forward_kinematics(q)


class ContextRobotModel:
    """Thread-safe RobotModel adapter using isolated MjData.

    Each instance owns a private MjData copy for FK computation.
    Created by Arm.create_planner() for parallel planning.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_qpos_indices: list[int],
        ee_site_id: int,
        joint_limits: tuple[np.ndarray, np.ndarray],
        tcp_offset: np.ndarray | None = None,
    ):
        self._model = model
        self._data = data
        self._joint_qpos_indices = joint_qpos_indices
        self._ee_site_id = ee_site_id
        self._joint_limits = joint_limits
        self._tcp_offset = tcp_offset

    @property
    def dof(self) -> int:
        return len(self._joint_qpos_indices)

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._joint_limits

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Compute EE pose on private data (thread-safe)."""
        for i, idx in enumerate(self._joint_qpos_indices):
            self._data.qpos[idx] = q[i]
        mujoco.mj_forward(self._model, self._data)
        return _read_site_pose(
            self._data, self._ee_site_id, self._tcp_offset
        )


# =============================================================================
# Helpers
# =============================================================================


def _read_site_pose(
    data: mujoco.MjData,
    site_id: int,
    tcp_offset: np.ndarray | None = None,
) -> np.ndarray:
    """Read a 4x4 pose from a MuJoCo site, optionally applying tcp_offset."""
    pos = data.site_xpos[site_id]
    mat = data.site_xmat[site_id].reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = mat
    T[:3, 3] = pos
    if tcp_offset is not None:
        T = T @ tcp_offset
    return T


# =============================================================================
# Arm
# =============================================================================


class Arm:
    """Generic robot arm abstraction.

    Provides state queries, forward kinematics, and motion planning for
    any MuJoCo robot arm. Robot-specific capabilities (IK, gripper) are
    injected via protocols.

    Args:
        env: MuJoCo environment (provides model and data).
        config: Arm configuration (joint names, limits, ee_site, etc.).
        gripper: Optional gripper implementation.
        grasp_manager: Optional grasp state tracker.
        ik_solver: Optional IK solver for pose-based planning.
    """

    def __init__(
        self,
        env: Environment,
        config: ArmConfig,
        *,
        gripper: Gripper | None = None,
        grasp_manager: GraspManager | None = None,
        ik_solver: IKSolver | None = None,
    ):
        self.env = env
        self.config = config
        self.gripper = gripper
        self.grasp_manager = grasp_manager
        self.ik_solver = ik_solver

        model = env.model

        # Resolve joint IDs and cache indices
        self.joint_ids: list[int] = []
        self.joint_qpos_indices: list[int] = []
        self.joint_qvel_indices: list[int] = []

        for name in config.joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"Joint '{name}' not found in model")
            self.joint_ids.append(jid)
            self.joint_qpos_indices.append(model.jnt_qposadr[jid])
            self.joint_qvel_indices.append(model.jnt_dofadr[jid])

        # Resolve EE site
        if config.ee_site:
            self.ee_site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, config.ee_site
            )
            if self.ee_site_id == -1:
                raise ValueError(
                    f"EE site '{config.ee_site}' not found in model"
                )
        else:
            self.ee_site_id = -1

        # Resolve actuator IDs (actuators whose transmission targets our joints)
        self.actuator_ids: list[int] = []
        joint_id_set = set(self.joint_ids)
        for act_id in range(model.nu):
            if model.actuator_trnid[act_id, 0] in joint_id_set:
                self.actuator_ids.append(act_id)

        # Cache DOF and joint limits
        self.dof = len(config.joint_names)
        self._joint_limits: tuple[np.ndarray, np.ndarray] | None = None

    # -----------------------------------------------------------------
    # State queries
    # -----------------------------------------------------------------

    def get_joint_positions(self) -> np.ndarray:
        """Current joint positions (rad)."""
        return np.array([
            self.env.data.qpos[idx] for idx in self.joint_qpos_indices
        ])

    def get_joint_velocities(self) -> np.ndarray:
        """Current joint velocities (rad/s)."""
        return np.array([
            self.env.data.qvel[idx] for idx in self.joint_qvel_indices
        ])

    def get_ee_pose(self) -> np.ndarray:
        """Current end-effector pose as 4x4 homogeneous transform.

        Calls mj_forward to ensure kinematics are up-to-date, then reads
        the EE site pose. Applies tcp_offset if configured.
        """
        if self.ee_site_id == -1:
            raise RuntimeError("No ee_site configured")
        mujoco.mj_forward(self.env.model, self.env.data)
        return _read_site_pose(
            self.env.data, self.ee_site_id, self.config.tcp_offset
        )

    def get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Joint position limits as (lower, upper) arrays."""
        if self._joint_limits is None:
            model = self.env.model
            lower = np.array([model.jnt_range[jid, 0] for jid in self.joint_ids])
            upper = np.array([model.jnt_range[jid, 1] for jid in self.joint_ids])
            self._joint_limits = (lower, upper)
        return self._joint_limits

    # -----------------------------------------------------------------
    # Forward kinematics (non-destructive, for planning)
    # -----------------------------------------------------------------

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Compute EE pose at configuration q without modifying live state.

        Creates a temporary MjData copy, sets joints to q, runs mj_forward,
        and reads the resulting pose. The live env.data is never touched.
        """
        if self.ee_site_id == -1:
            raise RuntimeError("No ee_site configured")

        tmp_data = mujoco.MjData(self.env.model)
        # Copy current state as baseline
        np.copyto(tmp_data.qpos, self.env.data.qpos)
        # Set arm joints to requested config
        for i, idx in enumerate(self.joint_qpos_indices):
            tmp_data.qpos[idx] = q[i]
        mujoco.mj_forward(self.env.model, tmp_data)
        return _read_site_pose(tmp_data, self.ee_site_id, self.config.tcp_offset)

    # -----------------------------------------------------------------
    # Planning
    # -----------------------------------------------------------------

    def create_planner(
        self,
        config: CBiRRTConfig | None = None,
    ) -> CBiRRT:
        """Create a thread-safe planner with isolated state.

        Each planner has its own MjData copy and adapters, so multiple
        planners can run in parallel threads.

        Args:
            config: Planner configuration. Defaults built from
                    self.config.planning_defaults.

        Returns:
            Configured CBiRRT planner ready to call .plan().
        """
        if config is None:
            defaults = self.config.planning_defaults
            config = CBiRRTConfig(
                timeout=defaults.timeout,
                max_iterations=defaults.max_iterations,
                step_size=defaults.step_size,
                goal_bias=defaults.goal_bias,
                smoothing_iterations=defaults.smoothing_iterations,
            )

        # Fork environment for isolated planning state
        planning_env = self.env.fork()
        model = planning_env.model
        data = planning_env.data

        # Build adapters
        robot_model = ContextRobotModel(
            model=model,
            data=data,
            joint_qpos_indices=self.joint_qpos_indices,
            ee_site_id=self.ee_site_id,
            joint_limits=self.get_joint_limits(),
            tcp_offset=self.config.tcp_offset,
        )

        # Collision checker with snapshot of current grasp state
        if self.grasp_manager is not None:
            grasped_objects = frozenset(self.grasp_manager.grasped.items())
            attachments = dict(self.grasp_manager._attachments)
            collision_checker = CollisionChecker(
                model=model,
                data=data,
                joint_names=self.config.joint_names,
                grasped_objects=grasped_objects,
                attachments=attachments,
            )
        else:
            collision_checker = CollisionChecker(
                model=model,
                data=data,
                joint_names=self.config.joint_names,
            )

        # IK solver — use injected solver or a no-op stub
        ik = self.ik_solver if self.ik_solver is not None else _NoIKSolver()

        return CBiRRT(
            robot=robot_model,
            ik_solver=ik,
            collision_checker=collision_checker,
            config=config,
        )

    def plan_to_configuration(
        self,
        q_goal: np.ndarray,
        timeout: float = 30.0,
        seed: int | None = None,
        planner_config: CBiRRTConfig | None = None,
    ) -> list[np.ndarray] | None:
        """Plan a collision-free path from current config to q_goal.

        Forks the environment to preserve state during planning.

        Args:
            q_goal: Goal joint configuration.
            timeout: Planning timeout in seconds.
            seed: RNG seed for reproducibility.
            planner_config: Override planner configuration.

        Returns:
            List of waypoint configurations, or None if planning failed.
        """
        if planner_config is None:
            defaults = self.config.planning_defaults
            planner_config = CBiRRTConfig(
                timeout=timeout,
                max_iterations=defaults.max_iterations,
                step_size=defaults.step_size,
                goal_bias=defaults.goal_bias,
                smoothing_iterations=defaults.smoothing_iterations,
            )
        else:
            planner_config.timeout = timeout

        q_start = self.get_joint_positions()
        planner = self.create_planner(planner_config)

        path = planner.plan(start=q_start, goal=q_goal, seed=seed)
        return path

    def plan_to_configurations(
        self,
        q_goals: list[np.ndarray],
        timeout: float = 30.0,
        seed: int | None = None,
        planner_config: CBiRRTConfig | None = None,
    ) -> list[np.ndarray] | None:
        """Plan to the nearest reachable goal from a set of configurations.

        The planner explores all goals simultaneously and returns the
        shortest path to any of them.

        Args:
            q_goals: List of candidate goal configurations.
            timeout: Planning timeout in seconds.
            seed: RNG seed for reproducibility.
            planner_config: Override planner configuration.

        Returns:
            Path to nearest reachable goal, or None if all failed.
        """
        if planner_config is None:
            defaults = self.config.planning_defaults
            planner_config = CBiRRTConfig(
                timeout=timeout,
                max_iterations=defaults.max_iterations,
                step_size=defaults.step_size,
                goal_bias=defaults.goal_bias,
                smoothing_iterations=defaults.smoothing_iterations,
            )
        else:
            planner_config.timeout = timeout

        q_start = self.get_joint_positions()
        planner = self.create_planner(planner_config)

        path = planner.plan(start=q_start, goal=q_goals, seed=seed)
        return path

    def plan_to_tsrs(
        self,
        tsrs: list,
        constraint_tsrs: list | None = None,
        timeout: float = 30.0,
        seed: int | None = None,
        planner_config: CBiRRTConfig | None = None,
    ) -> list[np.ndarray] | None:
        """Plan to a TSR-defined goal region.

        Requires an IK solver to be configured (for TSR sampling).

        Args:
            tsrs: Goal TSRs (union — any is acceptable).
            constraint_tsrs: Path constraints (all must be satisfied).
            timeout: Planning timeout in seconds.
            seed: RNG seed for reproducibility.
            planner_config: Override planner configuration.

        Returns:
            Path to a goal satisfying the TSRs, or None if failed.
        """
        if self.ik_solver is None:
            raise RuntimeError(
                "plan_to_tsrs requires an IK solver. "
                "Pass ik_solver= to the Arm constructor."
            )

        if planner_config is None:
            defaults = self.config.planning_defaults
            planner_config = CBiRRTConfig(
                timeout=timeout,
                max_iterations=defaults.max_iterations,
                step_size=defaults.step_size,
                goal_bias=defaults.goal_bias,
                smoothing_iterations=defaults.smoothing_iterations,
            )
        else:
            planner_config.timeout = timeout

        q_start = self.get_joint_positions()
        planner = self.create_planner(planner_config)

        path = planner.plan(
            start=q_start,
            goal_tsrs=tsrs,
            constraint_tsrs=constraint_tsrs,
            seed=seed,
        )
        return path

    def plan_to_pose(
        self,
        pose: np.ndarray,
        timeout: float = 30.0,
        seed: int | None = None,
        planner_config: CBiRRTConfig | None = None,
    ) -> list[np.ndarray] | None:
        """Plan to an end-effector pose via IK + configuration planning.

        Solves IK for the target pose, then plans to the nearest valid
        IK solution.

        Args:
            pose: 4x4 target end-effector pose.
            timeout: Planning timeout in seconds.
            seed: RNG seed for reproducibility.
            planner_config: Override planner configuration.

        Returns:
            Path to pose, or None if IK fails or planning fails.
        """
        if self.ik_solver is None:
            raise RuntimeError(
                "plan_to_pose requires an IK solver. "
                "Pass ik_solver= to the Arm constructor."
            )

        # Apply inverse tcp_offset if configured
        ik_target = pose
        if self.config.tcp_offset is not None:
            ik_target = pose @ np.linalg.inv(self.config.tcp_offset)

        q_init = self.get_joint_positions()
        solutions = self.ik_solver.solve_valid(ik_target, q_init=q_init)

        if not solutions:
            logger.warning("IK found no valid solutions for target pose")
            return None

        return self.plan_to_configurations(
            q_goals=solutions,
            timeout=timeout,
            seed=seed,
            planner_config=planner_config,
        )

    def plan_trajectory(
        self,
        q_goal: np.ndarray,
        timeout: float = 30.0,
        seed: int | None = None,
        control_dt: float = 0.008,
        planner_config: CBiRRTConfig | None = None,
    ) -> Trajectory | None:
        """Plan and retime a trajectory to q_goal.

        Convenience method that combines plan_to_configuration() with
        Trajectory.from_path() for TOPP-RA retiming.

        Args:
            q_goal: Goal joint configuration.
            timeout: Planning timeout in seconds.
            seed: RNG seed for reproducibility.
            control_dt: Control timestep for trajectory sampling.
            planner_config: Override planner configuration.

        Returns:
            Time-optimal Trajectory, or None if planning failed.
        """
        path = self.plan_to_configuration(
            q_goal, timeout=timeout, seed=seed,
            planner_config=planner_config,
        )
        if path is None:
            return None

        limits = self.config.kinematic_limits
        return Trajectory.from_path(
            path=path,
            vel_limits=limits.velocity,
            acc_limits=limits.acceleration,
            control_dt=control_dt,
            entity=self.config.name,
            joint_names=self.config.joint_names,
        )


class _NoIKSolver:
    """Stub IK solver that returns no solutions.

    Used when no IK solver is injected. Config-to-config planning
    still works; only pose/TSR-based planning requires real IK.
    """

    def solve(
        self, pose: np.ndarray, q_init: np.ndarray | None = None
    ) -> list[np.ndarray]:
        return []

    def solve_valid(
        self, pose: np.ndarray, q_init: np.ndarray | None = None
    ) -> list[np.ndarray]:
        return []
