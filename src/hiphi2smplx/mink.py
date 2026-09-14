"""Mink-based frame-wise full-body position IK for SMPL/SMPL-X."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)
# Indices 10/11 are SMPL foot joints fed by equivalent BVH toe targets.
ANKLE_WRIST_TOE_INDICES = np.asarray((7, 8, 10, 11, 20, 21), dtype=np.int64)


@dataclass(frozen=True)
class MinkPositionConfig:
    """Numerical settings for one-frame differential IK."""

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


def _fmt(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the minimal world-space rotation from source to target."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm < 1e-8 or target_norm < 1e-8:
        return np.eye(3, dtype=np.float64)
    source = source / source_norm
    target = target / target_norm
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1e-8:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        basis = np.eye(3)[int(np.argmin(np.abs(source)))]
        axis = np.cross(source, basis)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    axis = cross / sine
    return Rotation.from_rotvec(np.arctan2(sine, cosine) * axis).as_matrix()


def build_skeleton_mjcf(rest_joints: np.ndarray, parents: np.ndarray) -> str:
    """Build a pure-kinematic MuJoCo model with a free root and ball joints."""

    import xml.etree.ElementTree as ET

    joints = np.asarray(rest_joints, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int64)
    if joints.shape != (22, 3) or parents.shape != (22,):
        raise ValueError(f"expected rest_joints (22,3), parents (22,), got {joints.shape}, {parents.shape}")
    if parents[0] != -1:
        raise ValueError("the pelvis must be the root joint")

    root = ET.Element("mujoco", model="joint2smplx_mink")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", gravity="0 0 0", timestep="1")
    worldbody = ET.SubElement(root, "worldbody")

    def add_body(index: int, parent_node: ET.Element | None) -> None:
        parent = int(parents[index])
        offset = np.zeros(3, dtype=np.float64) if parent < 0 else joints[index] - joints[parent]
        body = ET.SubElement(
            worldbody if parent_node is None else parent_node,
            "body", name=JOINT_NAMES[index], pos=_fmt(offset),
        )
        ET.SubElement(body, "inertial", pos="0 0 0", mass="0.01",
                      diaginertia="0.00001 0.00001 0.00001")
        if parent < 0:
            ET.SubElement(body, "freejoint", name="root")
        else:
            ET.SubElement(body, "joint", name=f"{JOINT_NAMES[index]}_joint", type="ball")
        ET.SubElement(body, "site", name=f"{JOINT_NAMES[index]}_site",
                      pos="0 0 0", size="0.005")
        for child in np.flatnonzero(parents == index):
            add_body(int(child), body)

    add_body(0, None)
    return ET.tostring(root, encoding="unicode")


class MinkPositionSolver:
    """Serial 22-joint position IK using Mink's differential QP solver."""

    def __init__(
        self,
        rest_joints: np.ndarray,
        parents: np.ndarray,
        config: MinkPositionConfig = MinkPositionConfig(),
    ) -> None:
        if config.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if config.dt <= 0.0 or config.damping <= 0.0:
            raise ValueError("dt and damping must be positive")
        costs = (
            config.position_cost, config.orientation_cost,
            config.posture_cost, config.root_posture_cost,
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
                "solver='mink' requires the 'mink', 'mujoco', and 'qpsolvers' packages"
            ) from exc

        self.mink = mink
        self.mj = mj
        self.config = config
        self.rest_joints = np.asarray(rest_joints, dtype=np.float64)
        self.parents = np.asarray(parents, dtype=np.int64)
        self.model = mj.MjModel.from_xml_string(
            build_skeleton_mjcf(self.rest_joints, self.parents)
        )
        self.configuration = mink.Configuration(self.model)
        self.position_costs = np.full(
            len(JOINT_NAMES), config.position_cost, dtype=np.float64,
        )
        self.position_costs[ANKLE_WRIST_TOE_INDICES] *= config.ankle_wrist_toe_weight
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
            for index, name in enumerate(JOINT_NAMES)
        ]
        posture_cost = np.zeros(self.model.nv, dtype=np.float64)
        # Keep root rotation regularized while leaving root translation free.
        posture_cost[3:6] = config.root_posture_cost
        posture_cost[6:] = config.posture_cost
        self.posture_task = mink.PostureTask(
            self.model, cost=posture_cost, lm_damping=config.damping,
        )
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.site_ids = np.asarray([
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, f"{name}_site")
            for index, name in enumerate(JOINT_NAMES)
        ], dtype=np.int64)
        if np.any(self.site_ids < 0):
            raise RuntimeError("failed to create one or more SMPL joint sites")
        root_joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "root")
        self._root_qposadr = int(self.model.jnt_qposadr[root_joint_id])
        self._joint_qposadr = np.asarray([
            int(self.model.jnt_qposadr[mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_JOINT, f"{name}_joint"
            )])
            for name in JOINT_NAMES[1:]
        ], dtype=np.int64)

    def _qpos_from_pose(
        self, pose: np.ndarray, root_position: np.ndarray,
    ) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float64)
        root_position = np.asarray(root_position, dtype=np.float64)
        if pose.shape != (24, 3):
            raise ValueError(f"pose must have shape (24,3), got {pose.shape}")
        if root_position.shape != (3,):
            raise ValueError(f"root_position must have shape (3,), got {root_position.shape}")
        qpos = self.model.qpos0.copy()
        qpos[self._root_qposadr:self._root_qposadr + 3] = root_position
        qpos[self._root_qposadr + 3:self._root_qposadr + 7] = (
            Rotation.from_rotvec(pose[0]).as_quat(scalar_first=True)
        )
        for pose_index, qpos_index in enumerate(self._joint_qposadr, start=1):
            qpos[qpos_index:qpos_index + 4] = Rotation.from_rotvec(
                pose[pose_index]
            ).as_quat(scalar_first=True)
        return qpos

    def _pose_from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        pose = np.zeros((24, 3), dtype=np.float32)
        pose[0] = Rotation.from_quat(
            qpos[self._root_qposadr + 3:self._root_qposadr + 7],
            scalar_first=True,
        ).as_rotvec()
        for pose_index, qpos_index in enumerate(self._joint_qposadr, start=1):
            pose[pose_index] = Rotation.from_quat(
                qpos[qpos_index:qpos_index + 4], scalar_first=True
            ).as_rotvec()
        return pose

    def _joint_positions(self) -> np.ndarray:
        return self.configuration.data.site_xpos[self.site_ids].copy()

    def _weighted_position_error(
        self, joints: np.ndarray, targets: np.ndarray,
    ) -> float:
        weighted = (joints - targets) * self.position_costs[:, None]
        return float(np.sum(weighted * weighted))

    def solve(
        self,
        targets: np.ndarray,
        init_pose: np.ndarray,
        init_translation: np.ndarray | None = None,
        reference_pose: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Solve one frame and return pose, translation, joints, and mean error."""

        targets = np.asarray(targets, dtype=np.float64)
        if targets.shape != (22, 3):
            raise ValueError(f"targets must have shape (22,3), got {targets.shape}")
        if not np.isfinite(targets).all():
            raise ValueError("targets contain non-finite values")
        init_pose = np.asarray(init_pose, dtype=np.float64)
        if init_pose.shape == (72,):
            init_pose = init_pose.reshape(24, 3)
        if init_pose.shape != (24, 3):
            raise ValueError(f"init_pose must have shape (24,3) or (72,), got {init_pose.shape}")
        if reference_pose is None:
            reference_pose = init_pose
        reference_pose = np.asarray(reference_pose, dtype=np.float64)
        if reference_pose.shape == (72,):
            reference_pose = reference_pose.reshape(24, 3)
        if reference_pose.shape != (24, 3):
            raise ValueError(
                f"reference_pose must have shape (24,3) or (72,), got {reference_pose.shape}"
            )
        if not np.isfinite(init_pose).all() or not np.isfinite(reference_pose).all():
            raise ValueError("init_pose and reference_pose must contain only finite values")
        if init_translation is None:
            # First frame: target pelvis is a world position; convert it to
            # MuJoCo root qpos. Subsequent frames receive SMPL translation.
            root_qpos = targets[0]
        else:
            root_qpos = np.asarray(init_translation, dtype=np.float64) + self.rest_joints[0]
        qpos = self._qpos_from_pose(init_pose, root_qpos)
        self.configuration.update(q=qpos)
        posture_qpos = self._qpos_from_pose(reference_pose, root_qpos)
        self.posture_task.set_target(posture_qpos)
        reference_local = Rotation.from_rotvec(reference_pose[:22]).as_matrix()
        reference_global = np.empty_like(reference_local)
        reference_global[0] = reference_local[0]
        for joint in range(1, 22):
            reference_global[joint] = (
                reference_global[self.parents[joint]] @ reference_local[joint]
            )
        # Position determines arm-joint swing through the child-bone ray; the
        # calibrated reference supplies the remaining upper-arm/forearm twist.
        for joint, child in ((16, 18), (17, 19), (18, 20), (19, 21)):
            rest_ray = self.rest_joints[child] - self.rest_joints[joint]
            predicted_ray = reference_global[joint] @ rest_ray
            target_ray = targets[child] - targets[joint]
            reference_global[joint] = (
                _align_vectors(predicted_ray, target_ray) @ reference_global[joint]
            )
        for index, (task, target) in enumerate(zip(self.tasks, targets)):
            rotation = self.mink.SO3.from_matrix(reference_global[index])
            task.set_target(
                self.mink.SE3.from_rotation_and_translation(rotation, target)
            )

        previous_error = np.inf
        for _ in range(self.config.iterations):
            self.configuration.update()
            current_joints = self._joint_positions()
            current_error = self._weighted_position_error(current_joints, targets)
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
            for _ in range(8):
                self.configuration.update(q=base_qpos)
                self.configuration.integrate_inplace(
                    velocity * step_scale, self.config.dt,
                )
                self.configuration.update()
                joints = self._joint_positions()
                error = self._weighted_position_error(joints, targets)
                if error < current_error:
                    accepted = True
                    break
                step_scale *= 0.5
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
        error = float(np.linalg.norm(joints - targets, axis=1).mean())
        return pose, translation, joints.astype(np.float32), error
