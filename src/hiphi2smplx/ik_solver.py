"""SMPL-X retargeting and frame-wise Mink inverse kinematics."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import (
    align_vectors,
    global_to_local_rotations,
    local_to_global_rotations,
)
from .skeleton import (
    BODY_SEMANTICS,
    SMPLX_BODY_JOINT_NAMES,
    SMPLX_BODY_MAP,
    SMPLX_BODY_PARENTS,
    MotionClip,
    Skeleton,
)

EXTREMITY_JOINT_NAMES = frozenset(
    {
        "left_ankle",
        "right_ankle",
        "left_foot",
        "right_foot",
        "left_wrist",
        "right_wrist",
    }
)
_ARM_RAY_PAIRS = (
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
)
_MAX_LINE_SEARCH_STEPS = 8
_LINE_SEARCH_DECAY = 0.5


@dataclass(frozen=True)
class MinkPositionConfig:
    """Numerical settings for one-frame differential IK.

    Attributes:
        iterations: Maximum differential-IK iterations per frame.
        dt: Integration time step passed to Mink.
        damping: Levenberg-Marquardt damping for Mink tasks.
        posture_cost: Regularization cost for non-root joint posture.
        root_posture_cost: Orientation cost and posture cost for the root.
        position_cost: Base joint-position cost.
        ankle_wrist_toe_weight: Multiplier for extremity position costs.
        orientation_cost: Orientation cost for non-root frame tasks.
        solver: QP solver registered with qpsolvers.
        tolerance: Minimum squared-error improvement before early stopping.
    """

    iterations: int = 10
    dt: float = 0.01
    damping: float = 1e-2
    posture_cost: float = 0.1
    root_posture_cost: float = 0.1
    position_cost: float = 1.0
    ankle_wrist_toe_weight: float = 1.0
    orientation_cost: float = 0.1
    solver: str = "daqp"
    tolerance: float = 1e-5


def prepare_body_inputs(
    source: MotionClip,
    source_mapping: dict[str, str],
    target: Skeleton,
) -> tuple[np.ndarray, np.ndarray]:
    """Build canonical position targets and copied target rotations.

    Args:
        source: Parsed source motion.
        source_mapping: Semantic-to-source-joint mapping validated for the clip.
        target: Target SMPL-X body skeleton.

    Returns:
        Position targets and target local axis-angle rotations, both with shape
        ``(F, 22, 3)``.
    """
    source_positions, source_global = source.world_transforms()
    source_indices = {name: index for index, name in enumerate(source.skeleton.joint_names)}
    indices = np.asarray(
        [source_indices[source_mapping[semantic]] for semantic in BODY_SEMANTICS],
        dtype=np.int64,
    )
    targets = source_positions[:, indices].astype(np.float32)
    target_semantics = {joint_name: semantic for semantic, joint_name in SMPLX_BODY_MAP.items()}
    target_global = np.empty(
        (source.num_frames, target.num_joints, 3, 3),
        dtype=np.float32,
    )
    identity = np.eye(3, dtype=np.float32)
    for joint, parent in enumerate(target.parents):
        semantic = target_semantics.get(target.joint_names[joint])
        if semantic is not None:
            source_joint = source_mapping[semantic]
            target_global[:, joint] = source_global[:, source_indices[source_joint]]
        elif parent >= 0:
            target_global[:, joint] = target_global[:, parent]
        else:
            target_global[:, joint] = identity

    local = global_to_local_rotations(target_global, target.parents)
    reference = (
        Rotation.from_matrix(local.reshape(-1, 3, 3))
        .as_rotvec()
        .reshape(
            source.num_frames,
            target.num_joints,
            3,
        )
        .astype(np.float32)
    )
    return targets, reference


def equivalent_foot_targets(
    targets: np.ndarray,
    rest_joints: np.ndarray,
    reference_pose: np.ndarray,
    sole_rotations: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Replace BVH toe markers with beta-shaped SMPL-X foot targets.

    Args:
        targets: Mapped BVH joint positions with shape ``(F, 22, 3)``.
        rest_joints: Shaped SMPL-X rest joints with shape ``(22, 3)``.
        reference_pose: Copied local rotations with shape ``(F, 22, 3)``.
        sole_rotations: Left and right sole-leveling rotation matrices.

    Returns:
        A copy of ``targets`` whose foot targets use shaped SMPL-X offsets.

    Raises:
        ValueError: If any input has an invalid shape or non-finite values.
    """
    output = np.asarray(targets, dtype=np.float32).copy()
    rest = np.asarray(rest_joints, dtype=np.float32)
    reference = np.asarray(reference_pose, dtype=np.float32)
    expected = (len(output), len(SMPLX_BODY_JOINT_NAMES), 3)
    if output.shape != expected or reference.shape != expected:
        raise ValueError(f"targets and reference_pose must both have shape {expected}")
    if rest.shape != (len(SMPLX_BODY_JOINT_NAMES), 3):
        raise ValueError(f"rest_joints must have shape (22,3), got {rest.shape}")
    if not all(np.isfinite(array).all() for array in (output, rest, reference)):
        raise ValueError("body inputs must contain only finite values")

    local = Rotation.from_rotvec(reference.reshape(-1, 3)).as_matrix()
    local = local.reshape(-1, len(SMPLX_BODY_JOINT_NAMES), 3, 3)
    index = {name: joint for joint, name in enumerate(SMPLX_BODY_JOINT_NAMES)}
    global_rotations = local_to_global_rotations(local, SMPLX_BODY_PARENTS)
    pairs = (("left_ankle", "left_foot"), ("right_ankle", "right_foot"))
    for level, (ankle_name, foot_name) in zip(sole_rotations, pairs, strict=True):
        ankle = index[ankle_name]
        foot = index[foot_name]
        sole_level = np.asarray(level, dtype=np.float32)
        if sole_level.shape != (3, 3) or not np.isfinite(sole_level).all():
            raise ValueError("sole rotations must be finite 3x3 matrices")
        bind_offset = sole_level @ (rest[foot] - rest[ankle])
        output[:, foot] = output[:, ankle] + np.einsum(
            "fij,j->fi",
            global_rotations[:, ankle],
            bind_offset,
        )
    return output


def _format_vector(values: np.ndarray) -> str:
    """Format a numeric vector for an MJCF XML attribute.

    Args:
        values: One-dimensional numeric values.

    Returns:
        A space-separated decimal representation.
    """
    return " ".join(f"{float(value):.9g}" for value in values)


def build_skeleton_mjcf(rest_joints: np.ndarray, parents: np.ndarray) -> str:
    """Build a kinematic MuJoCo model for the 22-joint SMPL-X body.

    Args:
        rest_joints: Absolute rest-joint positions with shape ``(22, 3)``.
        parents: Parent indices with shape ``(22,)``.

    Returns:
        MJCF XML containing a free root, ball joints, and one site per joint.

    Raises:
        ValueError: If the skeleton dimensions or root are invalid.
    """
    joints = np.asarray(rest_joints, dtype=np.float64)
    hierarchy = np.asarray(parents, dtype=np.int64)
    count = len(SMPLX_BODY_JOINT_NAMES)
    if joints.shape != (count, 3) or hierarchy.shape != (count,):
        raise ValueError(
            f"expected rest_joints ({count},3) and parents ({count},), "
            f"got {joints.shape} and {hierarchy.shape}"
        )
    if hierarchy[0] != -1:
        raise ValueError("the pelvis must be the root joint")

    root = ET.Element("mujoco", model="hiphi2smplx_mink")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", gravity="0 0 0", timestep="1")
    worldbody = ET.SubElement(root, "worldbody")

    def add_body(index: int, parent_node: ET.Element | None) -> None:
        """Append one joint body and recursively append its children."""
        parent = int(hierarchy[index])
        offset = np.zeros(3, dtype=np.float64) if parent < 0 else joints[index] - joints[parent]
        body = ET.SubElement(
            worldbody if parent_node is None else parent_node,
            "body",
            name=SMPLX_BODY_JOINT_NAMES[index],
            pos=_format_vector(offset),
        )
        ET.SubElement(
            body,
            "inertial",
            pos="0 0 0",
            mass="0.01",
            diaginertia="0.00001 0.00001 0.00001",
        )
        if parent < 0:
            ET.SubElement(body, "freejoint", name="root")
        else:
            ET.SubElement(
                body,
                "joint",
                name=f"{SMPLX_BODY_JOINT_NAMES[index]}_joint",
                type="ball",
            )
        ET.SubElement(
            body,
            "site",
            name=f"{SMPLX_BODY_JOINT_NAMES[index]}_site",
            pos="0 0 0",
            size="0.005",
        )
        for child in np.flatnonzero(hierarchy == index):
            add_body(int(child), body)

    add_body(0, None)
    return ET.tostring(root, encoding="unicode")


class MinkPositionSolver:
    """Solve serial 22-joint position IK with Mink's differential QP solver."""

    def __init__(
        self,
        rest_joints: np.ndarray,
        parents: np.ndarray,
        config: MinkPositionConfig | None = None,
    ) -> None:
        """Initialize the MuJoCo model, Mink tasks, and solver state.

        Args:
            rest_joints: Absolute SMPL-X rest joints with shape ``(22, 3)``.
            parents: SMPL-X body parent indices with shape ``(22,)``.
            config: Numerical IK settings. Defaults to ``MinkPositionConfig()``.

        Raises:
            ValueError: If a solver setting is invalid.
            ModuleNotFoundError: If Mink or MuJoCo is not installed.
        """
        config = MinkPositionConfig() if config is None else config
        if config.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if config.dt <= 0.0 or config.damping <= 0.0:
            raise ValueError("dt and damping must be positive")
        costs = (
            config.position_cost,
            config.orientation_cost,
            config.posture_cost,
            config.root_posture_cost,
        )
        if any(cost < 0.0 for cost in costs):
            raise ValueError("task costs must be non-negative")
        if config.ankle_wrist_toe_weight <= 0.0:
            raise ValueError("ankle_wrist_toe_weight must be positive")
        if not config.solver:
            raise ValueError("a qpsolvers solver name is required")
        try:
            import mink
            import mujoco as mj
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MinkPositionSolver requires mink, mujoco, and qpsolvers"
            ) from exc

        self.mink = mink
        self.mj = mj
        self.config = config
        self.rest_joints = np.asarray(rest_joints, dtype=np.float64)
        self.parents = np.asarray(parents, dtype=np.int64)
        self.model = mj.MjModel.from_xml_string(build_skeleton_mjcf(self.rest_joints, self.parents))
        self.configuration = mink.Configuration(self.model)
        self.position_costs = np.asarray(
            [
                config.position_cost
                * (config.ankle_wrist_toe_weight if name in EXTREMITY_JOINT_NAMES else 1.0)
                for name in SMPLX_BODY_JOINT_NAMES
            ],
            dtype=np.float64,
        )
        self.tasks = [
            mink.FrameTask(
                frame_name=f"{name}_site",
                frame_type="site",
                position_cost=self.position_costs[index],
                orientation_cost=(
                    config.root_posture_cost if index == 0 else config.orientation_cost
                ),
                lm_damping=config.damping,
            )
            for index, name in enumerate(SMPLX_BODY_JOINT_NAMES)
        ]

        posture_cost = np.zeros(self.model.nv, dtype=np.float64)
        # Root translation stays free while all rotations retain a pose prior.
        posture_cost[3:6] = config.root_posture_cost
        posture_cost[6:] = config.posture_cost
        self.posture_task = mink.PostureTask(
            self.model,
            cost=posture_cost,
            lm_damping=config.damping,
        )
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.site_ids = np.asarray(
            [
                mj.mj_name2id(
                    self.model,
                    mj.mjtObj.mjOBJ_SITE,
                    f"{name}_site",
                )
                for name in SMPLX_BODY_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        if np.any(self.site_ids < 0):
            raise RuntimeError("failed to create one or more SMPL-X joint sites")

        root_joint_id = mj.mj_name2id(
            self.model,
            mj.mjtObj.mjOBJ_JOINT,
            "root",
        )
        self._root_qposadr = int(self.model.jnt_qposadr[root_joint_id])
        self._joint_qposadr = np.asarray(
            [
                int(
                    self.model.jnt_qposadr[
                        mj.mj_name2id(
                            self.model,
                            mj.mjtObj.mjOBJ_JOINT,
                            f"{name}_joint",
                        )
                    ]
                )
                for name in SMPLX_BODY_JOINT_NAMES[1:]
            ],
            dtype=np.int64,
        )

    def _qpos_from_pose(
        self,
        pose: np.ndarray,
        root_position: np.ndarray,
    ) -> np.ndarray:
        """Convert an axis-angle body pose and root position to MuJoCo qpos.

        Args:
            pose: Body rotations with shape ``(24, 3)``.
            root_position: MuJoCo root position with shape ``(3,)``.

        Returns:
            A complete MuJoCo qpos vector.

        Raises:
            ValueError: If either input has an invalid shape.
        """
        pose_values = np.asarray(pose, dtype=np.float64)
        root = np.asarray(root_position, dtype=np.float64)
        if pose_values.shape != (24, 3):
            raise ValueError(f"pose must have shape (24,3), got {pose_values.shape}")
        if root.shape != (3,):
            raise ValueError(f"root_position must have shape (3,), got {root.shape}")

        qpos = self.model.qpos0.copy()
        qpos[self._root_qposadr : self._root_qposadr + 3] = root
        qpos[self._root_qposadr + 3 : self._root_qposadr + 7] = Rotation.from_rotvec(
            pose_values[0]
        ).as_quat(scalar_first=True)
        for pose_index, qpos_index in enumerate(self._joint_qposadr, start=1):
            qpos[qpos_index : qpos_index + 4] = Rotation.from_rotvec(
                pose_values[pose_index]
            ).as_quat(scalar_first=True)
        return qpos

    def _pose_from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Convert MuJoCo qpos rotations to a 24-joint axis-angle pose.

        Args:
            qpos: MuJoCo generalized-position vector.

        Returns:
            Axis-angle rotations with shape ``(24, 3)``. The final two entries
            remain zero because the body-only model contains 22 joints.
        """
        pose = np.zeros((24, 3), dtype=np.float32)
        pose[0] = Rotation.from_quat(
            qpos[self._root_qposadr + 3 : self._root_qposadr + 7],
            scalar_first=True,
        ).as_rotvec()
        for pose_index, qpos_index in enumerate(self._joint_qposadr, start=1):
            pose[pose_index] = Rotation.from_quat(
                qpos[qpos_index : qpos_index + 4],
                scalar_first=True,
            ).as_rotvec()
        return pose

    def _joint_positions(self) -> np.ndarray:
        """Return current world-space positions for all SMPL-X body sites."""
        return self.configuration.data.site_xpos[self.site_ids].copy()

    def _weighted_position_error(
        self,
        joints: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """Return the weighted sum of squared joint-position errors.

        Args:
            joints: Current positions with shape ``(22, 3)``.
            targets: Target positions with shape ``(22, 3)``.

        Returns:
            Scalar weighted squared error.
        """
        weighted = (joints - targets) * self.position_costs[:, None]
        return float(np.sum(weighted * weighted))

    def solve(
        self,
        targets: np.ndarray,
        init_pose: np.ndarray,
        init_translation: np.ndarray | None = None,
        reference_pose: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Solve one motion frame.

        Args:
            targets: World-space body targets with shape ``(22, 3)``.
            init_pose: Initial axis-angle pose with shape ``(24, 3)`` or ``(72,)``.
            init_translation: Previous SMPL-X translation, or ``None`` for the
                first frame.
            reference_pose: Pose used by orientation and posture tasks. Defaults
                to ``init_pose``.

        Returns:
            A tuple containing the solved pose ``(24, 3)``, SMPL-X translation
            ``(3,)``, fitted joints ``(22, 3)``, and mean position error.

        Raises:
            ValueError: If an input has an invalid shape or non-finite values.
        """
        target_positions = np.asarray(targets, dtype=np.float64)
        if target_positions.shape != (22, 3):
            raise ValueError(f"targets must have shape (22,3), got {target_positions.shape}")
        if not np.isfinite(target_positions).all():
            raise ValueError("targets contain non-finite values")

        initial_pose = np.asarray(init_pose, dtype=np.float64)
        if initial_pose.shape == (72,):
            initial_pose = initial_pose.reshape(24, 3)
        if initial_pose.shape != (24, 3):
            raise ValueError(f"init_pose must have shape (24,3) or (72,), got {initial_pose.shape}")
        if reference_pose is None:
            reference_pose = initial_pose
        reference = np.asarray(reference_pose, dtype=np.float64)
        if reference.shape == (72,):
            reference = reference.reshape(24, 3)
        if reference.shape != (24, 3):
            raise ValueError(
                f"reference_pose must have shape (24,3) or (72,), got {reference.shape}"
            )
        if not np.isfinite(initial_pose).all() or not np.isfinite(reference).all():
            raise ValueError("init_pose and reference_pose must contain finite values")

        if init_translation is None:
            # The first target pelvis is a world position, while later frames
            # receive the previous SMPL-X translation.
            root_qpos = target_positions[0]
        else:
            root_qpos = np.asarray(init_translation, dtype=np.float64) + self.rest_joints[0]
        qpos = self._qpos_from_pose(initial_pose, root_qpos)
        self.configuration.update(q=qpos)
        posture_qpos = self._qpos_from_pose(reference, root_qpos)
        self.posture_task.set_target(posture_qpos)

        reference_local = Rotation.from_rotvec(reference[:22]).as_matrix()[None]
        reference_global = local_to_global_rotations(
            reference_local,
            self.parents,
        )[0]
        index = {name: joint for joint, name in enumerate(SMPLX_BODY_JOINT_NAMES)}
        # Positions determine arm swing but not axial twist. Preserve copied
        # twist while aligning each reference bone ray with its target ray.
        for joint_name, child_name in _ARM_RAY_PAIRS:
            joint = index[joint_name]
            child = index[child_name]
            rest_ray = self.rest_joints[child] - self.rest_joints[joint]
            predicted_ray = reference_global[joint] @ rest_ray
            target_ray = target_positions[child] - target_positions[joint]
            reference_global[joint] = (
                align_vectors(predicted_ray, target_ray) @ reference_global[joint]
            )

        for task, target, rotation_matrix in zip(
            self.tasks,
            target_positions,
            reference_global,
            strict=True,
        ):
            rotation = self.mink.SO3.from_matrix(rotation_matrix)
            task.set_target(self.mink.SE3.from_rotation_and_translation(rotation, target))

        previous_error = np.inf
        for _ in range(self.config.iterations):
            self.configuration.update()
            current_joints = self._joint_positions()
            current_error = self._weighted_position_error(
                current_joints,
                target_positions,
            )
            base_qpos = self.configuration.q.copy()
            velocity = self.mink.solve_ik(
                self.configuration,
                [*self.tasks, self.posture_task],
                self.config.dt,
                self.config.solver,
                damping=self.config.damping,
                safety_break=False,
                limits=self.limits,
            )

            accepted = False
            step_scale = 1.0
            # Backtracking prevents a differential-IK step from increasing the
            # weighted position error.
            for _ in range(_MAX_LINE_SEARCH_STEPS):
                self.configuration.update(q=base_qpos)
                self.configuration.integrate_inplace(
                    velocity * step_scale,
                    self.config.dt,
                )
                self.configuration.update()
                joints = self._joint_positions()
                error = self._weighted_position_error(joints, target_positions)
                if error < current_error:
                    accepted = True
                    break
                step_scale *= _LINE_SEARCH_DECAY

            if not accepted:
                self.configuration.update(q=base_qpos)
                error = current_error
                joints = current_joints
            if previous_error - error <= self.config.tolerance**2:
                break
            previous_error = error

        self.configuration.update()
        joints = self._joint_positions()
        pose = self._pose_from_qpos(self.configuration.q)
        translation = (self.configuration.q[:3] - self.rest_joints[0]).astype(np.float32)
        mean_error = float(np.linalg.norm(joints - target_positions, axis=1).mean())
        return pose, translation, joints.astype(np.float32), mean_error
