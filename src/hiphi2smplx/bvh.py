"""HiPHI BVH input preparation for SMPL-X body fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SMPL22_SOURCE_SEMANTICS = (
    "hips", "left_upper_leg", "right_upper_leg", "spine",
    "left_lower_leg", "right_lower_leg", "spine2", "left_foot",
    "right_foot", "chest", "left_toe", "right_toe", "neck",
    "left_clavicle", "right_clavicle", "head", "left_upper_arm",
    "right_upper_arm", "left_lower_arm", "right_lower_arm",
    "left_hand", "right_hand",
)

HIPHI_LONG_SPINE_MAPPING = {
    "spine": "Spine",
    "spine2": "Spine2",
    "chest": "Spine4",
}

SMPLX_JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee",
    "right_knee", "spine2", "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck", "left_collar", "right_collar",
    "head", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "jaw",
    "left_eye_smplhf", "right_eye_smplhf",
) + tuple(
    f"{side}_{digit}{segment}"
    for side in ("left", "right")
    for digit in ("index", "middle", "pinky", "ring", "thumb")
    for segment in (1, 2, 3)
)

_ALIASES = {
    "hips": ("hips", "pelvis", "root"),
    "spine": ("spine", "spine1"),
    "spine2": ("spine2", "spine1", "spine"),
    "chest": ("chest", "spine3", "spine2", "upperchest"),
    "neck": ("neck", "neck1", "neck2"),
    "head": ("head",),
    "left_clavicle": ("leftcollar", "left_clavicle", "left_collar", "leftshoulder"),
    "right_clavicle": ("rightcollar", "right_clavicle", "right_collar", "rightshoulder"),
    "left_upper_arm": ("leftarm", "leftshoulder", "left_upper_arm", "left_shoulder"),
    "right_upper_arm": ("rightarm", "rightshoulder", "right_upper_arm", "right_shoulder"),
    "left_lower_arm": ("leftforearm", "leftelbow", "left_lower_arm", "left_elbow"),
    "right_lower_arm": ("rightforearm", "rightelbow", "right_lower_arm", "right_elbow"),
    "left_hand": ("lefthand", "leftwrist", "left_hand", "left_wrist"),
    "right_hand": ("righthand", "rightwrist", "right_hand", "right_wrist"),
    "left_upper_leg": ("leftupleg", "leftleg", "left_hip", "lefthip"),
    "right_upper_leg": ("rightupleg", "rightleg", "right_hip", "righthip"),
    "left_lower_leg": ("leftshin", "leftleg", "left_knee", "leftknee"),
    "right_lower_leg": ("rightshin", "rightleg", "right_knee", "rightknee"),
    "left_foot": ("leftankle", "left_ankle", "leftfoot", "left_foot"),
    "right_foot": ("rightankle", "right_ankle", "rightfoot", "right_foot"),
    "left_toe": ("lefttoeend", "lefttoebase", "lefttoe", "left_foot"),
    "right_toe": ("righttoeend", "righttoebase", "righttoe", "right_foot"),
}


@dataclass(frozen=True)
class Skeleton:
    joint_names: tuple[str, ...]
    parents: np.ndarray
    rest_offsets: np.ndarray

    def __post_init__(self) -> None:
        parents = np.asarray(self.parents, dtype=np.int64)
        offsets = np.asarray(self.rest_offsets, dtype=np.float32)
        count = len(self.joint_names)
        if parents.shape != (count,) or offsets.shape != (count, 3):
            raise ValueError("invalid skeleton dimensions")
        if np.flatnonzero(parents < 0).tolist() != [0]:
            raise ValueError("skeleton must have one root at index zero")
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "rest_offsets", offsets)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)


@dataclass(frozen=True)
class MotionClip:
    skeleton: Skeleton
    local_rotations: np.ndarray
    root_translations: np.ndarray
    frame_time: float
    local_translations: np.ndarray | None = None

    @property
    def num_frames(self) -> int:
        return int(self.local_rotations.shape[0])

    def sliced(self, count: int) -> "MotionClip":
        return MotionClip(
            self.skeleton,
            self.local_rotations[:count],
            self.root_translations[:count],
            self.frame_time,
            None if self.local_translations is None else self.local_translations[:count],
        )

    def world_transforms(self) -> tuple[np.ndarray, np.ndarray]:
        frames, joints = self.local_rotations.shape[:2]
        local = Rotation.from_rotvec(self.local_rotations.reshape(-1, 3)).as_matrix()
        local = local.reshape(frames, joints, 3, 3).astype(np.float32)
        translations = self.local_translations
        if translations is None:
            translations = np.broadcast_to(self.skeleton.rest_offsets, (frames, joints, 3))
        positions = np.empty((frames, joints, 3), dtype=np.float32)
        global_rotations = np.empty_like(local)
        for joint, parent in enumerate(self.skeleton.parents):
            if parent < 0:
                global_rotations[:, joint] = local[:, joint]
                positions[:, joint] = self.root_translations + self.skeleton.rest_offsets[joint]
            else:
                global_rotations[:, joint] = global_rotations[:, parent] @ local[:, joint]
                positions[:, joint] = positions[:, parent] + np.einsum(
                    "fij,fj->fi", global_rotations[:, parent], translations[:, joint]
                )
        return positions, global_rotations


@dataclass
class _BvhJoint:
    name: str
    parent: int
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    channels: list[str] = field(default_factory=list)
    channel_indices: list[int] = field(default_factory=list)


def load_bvh(path: str | Path, unit_scale: float = 0.01) -> MotionClip:
    source = Path(path)
    lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
    joints: list[_BvhJoint] = []
    stack: list[int | str] = []
    channel_cursor = 0
    cursor = 0
    while cursor < len(lines):
        line = lines[cursor].strip()
        if line == "MOTION":
            cursor += 1
            break
        if line.startswith(("ROOT ", "JOINT ")):
            parent = next((item for item in reversed(stack) if isinstance(item, int)), -1)
            joints.append(_BvhJoint(line.split(maxsplit=1)[1].strip(), int(parent)))
            stack.append(len(joints) - 1)
        elif line.startswith("End Site"):
            stack.append("END")
        elif line.startswith("OFFSET ") and stack and isinstance(stack[-1], int):
            joints[stack[-1]].offset = np.asarray(line.split()[1:4], dtype=np.float32)
        elif line.startswith("CHANNELS ") and stack and isinstance(stack[-1], int):
            parts = line.split()
            count = int(parts[1])
            joints[stack[-1]].channels = parts[2:2 + count]
            joints[stack[-1]].channel_indices = list(range(channel_cursor, channel_cursor + count))
            channel_cursor += count
        elif line == "}" and stack:
            stack.pop()
        cursor += 1
    if not joints or cursor + 1 >= len(lines):
        raise ValueError(f"{source}: incomplete BVH")
    frames = int(lines[cursor].split(":", 1)[1])
    frame_time = float(lines[cursor + 1].split(":", 1)[1])
    rows = [[float(value) for value in line.split()] for line in lines[cursor + 2:cursor + 2 + frames] if line.strip()]
    values = np.asarray(rows, dtype=np.float32)
    if values.shape != (frames, channel_cursor):
        raise ValueError(f"{source}: motion shape {values.shape}, expected {(frames, channel_cursor)}")

    local_rotations = np.zeros((frames, len(joints), 3), dtype=np.float32)
    local_translations = np.broadcast_to(
        np.stack([joint.offset for joint in joints])[None], (frames, len(joints), 3)
    ).copy()
    root_translations = np.zeros((frames, 3), dtype=np.float32)
    for joint_index, joint in enumerate(joints):
        joint_values = values[:, joint.channel_indices]
        rotation_indices = [i for i, channel in enumerate(joint.channels) if channel.lower().endswith("rotation")]
        order = "".join(joint.channels[i][0].upper() for i in rotation_indices)
        if order:
            local_rotations[:, joint_index] = Rotation.from_euler(
                order, joint_values[:, rotation_indices], degrees=True
            ).as_rotvec().astype(np.float32)
        positions = {channel[0].lower(): i for i, channel in enumerate(joint.channels) if channel.lower().endswith("position")}
        destination = root_translations if joint_index == 0 else local_translations[:, joint_index]
        if joint_index == 0:
            destination[:] = joint.offset
        for axis, component in (("x", 0), ("y", 1), ("z", 2)):
            if axis in positions:
                destination[:, component] = joint_values[:, positions[axis]]

    offsets = np.stack([joint.offset for joint in joints]).astype(np.float32) * unit_scale
    offsets[0] = 0.0
    local_translations *= unit_scale
    local_translations[:, 0] = 0.0
    root_translations *= unit_scale
    return MotionClip(
        Skeleton(tuple(joint.name for joint in joints), np.asarray([joint.parent for joint in joints]), offsets),
        local_rotations,
        root_translations,
        frame_time,
        local_translations,
    )


def _normalized(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def _infer_mapping(skeleton: Skeleton) -> dict[str, str]:
    names = {_normalized(name): name for name in skeleton.joint_names}
    mapping = {}
    for semantic, aliases in _ALIASES.items():
        match = next((names.get(_normalized(alias)) for alias in aliases if _normalized(alias) in names), None)
        if match is not None:
            mapping[semantic] = match
    if all(name in skeleton.joint_names for name in HIPHI_LONG_SPINE_MAPPING.values()):
        mapping.update(HIPHI_LONG_SPINE_MAPPING)
    short_chest = names.get("spine1")
    if (
        "chest" not in mapping
        and short_chest is not None
        and mapping.get("spine2") == short_chest
    ):
        mapping["chest"] = short_chest
    missing = sorted(set(SMPL22_SOURCE_SEMANTICS) - set(mapping))
    if missing:
        raise ValueError(f"BVH characterization misses joints: {missing}")
    return mapping


def _load_smplx_skeleton(model_path: str | Path, betas: np.ndarray) -> Skeleton:
    with np.load(Path(model_path), allow_pickle=True) as model:
        template = np.asarray(model["v_template"], dtype=np.float32)
        shapedirs = np.asarray(model["shapedirs"], dtype=np.float32)
        regressor = model["J_regressor"]
        if hasattr(regressor, "toarray"):
            regressor = regressor.toarray()
        coefficients = np.asarray(betas, dtype=np.float32).reshape(-1)
        count = min(coefficients.size, shapedirs.shape[-1])
        vertices = template + np.einsum("vdn,n->vd", shapedirs[..., :count], coefficients[:count])
        joints = np.asarray(regressor, dtype=np.float32) @ vertices
        parents = np.asarray(model["kintree_table"], dtype=np.int64)[0].copy()
    parents[parents > 1_000_000] = -1
    joint_count = min(len(SMPLX_JOINT_NAMES), len(joints), len(parents))
    offsets = joints[:joint_count].copy()
    for joint in range(1, joint_count):
        offsets[joint] -= joints[int(parents[joint])]
    return Skeleton(SMPLX_JOINT_NAMES[:joint_count], parents[:joint_count], offsets)


def _rotation_copy(source: MotionClip, source_mapping: dict[str, str], target: Skeleton) -> np.ndarray:
    _, source_global = source.world_transforms()
    source_indices = {name: index for index, name in enumerate(source.skeleton.joint_names)}
    target_mapping = _infer_mapping(target)
    target_semantics = {joint: semantic for semantic, joint in target_mapping.items()}
    target_global = np.empty((source.num_frames, target.num_joints, 3, 3), dtype=np.float32)
    identity = np.eye(3, dtype=np.float32)
    for joint, parent in enumerate(target.parents):
        semantic = target_semantics.get(target.joint_names[joint])
        if semantic is not None and semantic in source_mapping:
            target_global[:, joint] = source_global[:, source_indices[source_mapping[semantic]]]
        elif parent >= 0:
            target_global[:, joint] = target_global[:, parent]
        else:
            target_global[:, joint] = identity
    local = target_global.copy()
    for joint, parent in enumerate(target.parents):
        if parent >= 0:
            local[:, joint] = np.swapaxes(target_global[:, parent], -1, -2) @ target_global[:, joint]
    return Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(
        source.num_frames, target.num_joints, 3
    ).astype(np.float32)


def foot_sole_pitch_offsets(betas: np.ndarray, model_path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(model_path), allow_pickle=True) as model:
        template = np.asarray(model["v_template"], dtype=np.float64)
        shapedirs = np.asarray(model["shapedirs"], dtype=np.float64)
        weights = np.asarray(model["weights"], dtype=np.float64)
    coefficients = np.asarray(betas, dtype=np.float64).reshape(-1)
    count = min(coefficients.size, shapedirs.shape[-1])
    vertices = template + np.einsum("vcn,n->vc", shapedirs[..., :count], coefficients[:count])
    output = {}
    for semantic, joints in (("left_foot", (7, 10)), ("right_foot", (8, 11))):
        foot = vertices[np.flatnonzero(weights[:, joints].sum(axis=1) > 0.5)]
        lower_z, upper_z = np.quantile(foot[:, 2], (0.25, 0.65))
        heel, toe = foot[foot[:, 2] <= lower_z], foot[foot[:, 2] >= upper_z]
        heel_floor, toe_floor = np.quantile(heel[:, 1], 0.05), np.quantile(toe[:, 1], 0.05)
        heel_z = np.median(heel[heel[:, 1] <= np.quantile(heel[:, 1], 0.15), 2])
        toe_z = np.median(toe[toe[:, 1] <= np.quantile(toe[:, 1], 0.15), 2])
        angle = float(np.arctan2(toe_floor - heel_floor, toe_z - heel_z))
        cosine, sine = np.cos(angle), np.sin(angle)
        output[semantic] = np.asarray(
            ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)), dtype=np.float32
        )
    return output


def materialize_bvh_inputs(
    bvh_path: str | Path,
    model_path: str | Path,
    betas: np.ndarray,
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return no-scale joint targets and rotation-copy initialization for Mink."""
    source = load_bvh(bvh_path, unit_scale=0.01)
    if max_frames is not None:
        source = source.sliced(min(source.num_frames, max_frames))
    source_mapping = _infer_mapping(source.skeleton)
    target = _load_smplx_skeleton(model_path, betas)
    source_positions, _ = source.world_transforms()
    source_indices = {name: index for index, name in enumerate(source.skeleton.joint_names)}
    indices = np.asarray(
        [source_indices[source_mapping[semantic]] for semantic in SMPL22_SOURCE_SEMANTICS],
        dtype=np.int64,
    )
    targets = source_positions[:, indices].astype(np.float32)
    # Short BVH torsos need a virtual midpoint for SMPL-X spine2.
    if source_mapping["spine2"] == source_mapping["chest"]:
        spine = SMPL22_SOURCE_SEMANTICS.index("spine")
        spine2 = SMPL22_SOURCE_SEMANTICS.index("spine2")
        chest = SMPL22_SOURCE_SEMANTICS.index("chest")
        targets[:, spine2] = 0.5 * (targets[:, spine] + targets[:, chest])
    return (
        targets,
        _rotation_copy(source, source_mapping, target)[:, :22],
        float(source.frame_time),
        1.0,
    )


def load_smplx_rest_joints(model_path: str | Path, betas: np.ndarray) -> np.ndarray:
    skeleton = _load_smplx_skeleton(model_path, betas)
    positions = np.empty_like(skeleton.rest_offsets)
    for joint, parent in enumerate(skeleton.parents):
        positions[joint] = skeleton.rest_offsets[joint]
        if parent >= 0:
            positions[joint] += positions[parent]
    return positions
