"""Skeleton structures, joint standards, and named source mappings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import forward_kinematics

BODY_SEMANTICS = (
    "hips",
    "left_upper_leg",
    "right_upper_leg",
    "spine",
    "left_lower_leg",
    "right_lower_leg",
    "spine2",
    "left_foot",
    "right_foot",
    "chest",
    "left_toe",
    "right_toe",
    "neck",
    "left_clavicle",
    "right_clavicle",
    "head",
    "left_upper_arm",
    "right_upper_arm",
    "left_lower_arm",
    "right_lower_arm",
    "left_hand",
    "right_hand",
)

HIPHI_MAP = {
    "hips": "Hips",
    "left_upper_leg": "LeftUpLeg",
    "right_upper_leg": "RightUpLeg",
    "spine": "Spine",
    "left_lower_leg": "LeftLeg",
    "right_lower_leg": "RightLeg",
    "spine2": "Spine2",
    "left_foot": "LeftFoot",
    "right_foot": "RightFoot",
    "chest": "Spine4",
    "left_toe": "LeftToeBase",
    "right_toe": "RightToeBase",
    "neck": "Neck",
    "left_clavicle": "LeftShoulder",
    "right_clavicle": "RightShoulder",
    "head": "Head",
    "left_upper_arm": "LeftArm",
    "right_upper_arm": "RightArm",
    "left_lower_arm": "LeftForeArm",
    "right_lower_arm": "RightForeArm",
    "left_hand": "LeftHand",
    "right_hand": "RightHand",
}

SMPLX_BODY_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

SMPLX_JOINT_NAMES = (
    SMPLX_BODY_JOINT_NAMES
    + (
        "jaw",
        "left_eye_smplhf",
        "right_eye_smplhf",
    )
    + tuple(
        f"{side}_{digit}{segment}"
        for side in ("left", "right")
        for digit in ("index", "middle", "pinky", "ring", "thumb")
        for segment in (1, 2, 3)
    )
)

SMPLX_BODY_PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int64,
)

SMPLX_BODY_MAP = dict(zip(BODY_SEMANTICS, SMPLX_BODY_JOINT_NAMES, strict=True))
SOURCE_SKELETON_MAPS = {"hiphi": HIPHI_MAP}


@dataclass(frozen=True)
class Skeleton:
    """A named joint hierarchy and its parent-relative rest offsets.

    Attributes:
        joint_names: Joint names in hierarchy order.
        parents: Parent indices, using ``-1`` for the single root.
        rest_offsets: Parent-relative rest offsets with shape ``(J, 3)``.
    """

    joint_names: tuple[str, ...]
    parents: np.ndarray
    rest_offsets: np.ndarray

    def __post_init__(self) -> None:
        """Validate and normalize the hierarchy arrays."""
        parents = np.asarray(self.parents, dtype=np.int64)
        offsets = np.asarray(self.rest_offsets, dtype=np.float32)
        count = len(self.joint_names)
        if parents.shape != (count,) or offsets.shape != (count, 3):
            raise ValueError("invalid skeleton dimensions")
        if np.flatnonzero(parents < 0).tolist() != [0]:
            raise ValueError("skeleton must have one root at index zero")
        if any(parent >= joint for joint, parent in enumerate(parents[1:], start=1)):
            raise ValueError("parents must precede their children")
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "rest_offsets", offsets)

    @property
    def num_joints(self) -> int:
        """Return the number of joints in the skeleton."""
        return len(self.joint_names)


@dataclass(frozen=True)
class MotionClip:
    """A local-space skeletal animation.

    Attributes:
        skeleton: Skeleton shared by every frame.
        local_rotations: Axis-angle rotations with shape ``(F, J, 3)``.
        root_translations: World-space root translations with shape ``(F, 3)``.
        frame_time: Duration of one frame in seconds.
        local_translations: Optional animated local offsets with shape
            ``(F, J, 3)``.
    """

    skeleton: Skeleton
    local_rotations: np.ndarray
    root_translations: np.ndarray
    frame_time: float
    local_translations: np.ndarray | None = None

    @property
    def num_frames(self) -> int:
        """Return the number of motion frames."""
        return int(self.local_rotations.shape[0])

    def sliced(self, count: int) -> "MotionClip":
        """Return a clip containing at most the first ``count`` frames.

        Args:
            count: Maximum number of frames to retain.

        Returns:
            A motion clip sharing the same skeleton and frame timing.
        """
        return MotionClip(
            self.skeleton,
            self.local_rotations[:count],
            self.root_translations[:count],
            self.frame_time,
            None if self.local_translations is None else self.local_translations[:count],
        )

    def world_transforms(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute global positions and rotation matrices for every joint.

        Returns:
            A pair containing positions ``(F, J, 3)`` and rotations
            ``(F, J, 3, 3)`` in world space.
        """
        frames, joints = self.local_rotations.shape[:2]
        local_rotations = Rotation.from_rotvec(self.local_rotations.reshape(-1, 3)).as_matrix()
        local_rotations = local_rotations.reshape(frames, joints, 3, 3).astype(np.float32)
        local_translations = self.local_translations
        if local_translations is None:
            local_translations = np.broadcast_to(
                self.skeleton.rest_offsets,
                (frames, joints, 3),
            )
        root_positions = self.root_translations + self.skeleton.rest_offsets[0]
        return forward_kinematics(
            local_rotations,
            local_translations,
            root_positions,
            self.skeleton.parents,
        )


def map_skeleton(skeleton: Skeleton, map_name: str = "hiphi") -> dict[str, str]:
    """Validate and return one exact source-skeleton mapping.

    Args:
        skeleton: Source skeleton whose joint names are validated.
        map_name: Registered source-map name.

    Returns:
        A copy of the semantic-to-source-joint mapping.

    Raises:
        ValueError: If the map is unknown, malformed, or incompatible with the
            source skeleton.
    """
    try:
        expected = SOURCE_SKELETON_MAPS[map_name]
    except KeyError as exc:
        available = ", ".join(sorted(SOURCE_SKELETON_MAPS))
        raise ValueError(f"unknown skeleton map {map_name!r}; choose one of: {available}") from exc

    if set(expected) != set(BODY_SEMANTICS):
        raise ValueError(f"skeleton map {map_name!r} has invalid semantic keys")
    if len(set(expected.values())) != len(expected):
        raise ValueError(f"skeleton map {map_name!r} contains duplicate joint names")

    names = set(skeleton.joint_names)
    missing = [semantic for semantic in BODY_SEMANTICS if expected[semantic] not in names]
    if missing:
        missing_names = [f"{semantic}={expected[semantic]}" for semantic in missing]
        raise ValueError(f"skeleton map {map_name!r} misses joints: {missing_names}")
    return dict(expected)
