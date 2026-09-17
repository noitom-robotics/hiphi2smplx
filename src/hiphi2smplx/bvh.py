"""BVH parsing for HiPHI skeletal motion files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .skeleton import MotionClip, Skeleton


@dataclass
class _BvhJoint:
    """Mutable joint data collected while parsing a BVH hierarchy."""

    name: str
    parent: int
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    channels: list[str] = field(default_factory=list)
    channel_indices: list[int] = field(default_factory=list)


def load_bvh(path: str | Path, unit_scale: float = 0.01) -> MotionClip:
    """Load one BVH file as a local-space motion clip.

    Args:
        path: Path to the source BVH file.
        unit_scale: Scale applied to offsets and position channels. The default
            converts HiPHI centimetres to metres.

    Returns:
        Parsed skeleton, local joint transforms, root motion, and frame timing.

    Raises:
        FileNotFoundError: If the source file does not exist.
        ValueError: If the hierarchy or motion section is incomplete.
    """
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
            parent = next(
                (item for item in reversed(stack) if isinstance(item, int)),
                -1,
            )
            joints.append(_BvhJoint(line.split(maxsplit=1)[1].strip(), int(parent)))
            stack.append(len(joints) - 1)
        elif line.startswith("End Site"):
            stack.append("END")
        elif line.startswith("OFFSET ") and stack and isinstance(stack[-1], int):
            joints[stack[-1]].offset = np.asarray(
                line.split()[1:4],
                dtype=np.float32,
            )
        elif line.startswith("CHANNELS ") and stack and isinstance(stack[-1], int):
            parts = line.split()
            count = int(parts[1])
            joints[stack[-1]].channels = parts[2 : 2 + count]
            joints[stack[-1]].channel_indices = list(range(channel_cursor, channel_cursor + count))
            channel_cursor += count
        elif line == "}" and stack:
            stack.pop()
        cursor += 1

    if not joints or cursor + 1 >= len(lines):
        raise ValueError(f"{source}: incomplete BVH")

    frames = int(lines[cursor].split(":", 1)[1])
    frame_time = float(lines[cursor + 1].split(":", 1)[1])
    motion_lines = lines[cursor + 2 : cursor + 2 + frames]
    rows = [[float(value) for value in line.split()] for line in motion_lines if line.strip()]
    values = np.asarray(rows, dtype=np.float32)
    if values.shape != (frames, channel_cursor):
        raise ValueError(
            f"{source}: motion shape {values.shape}, expected {(frames, channel_cursor)}"
        )

    local_rotations = np.zeros((frames, len(joints), 3), dtype=np.float32)
    local_translations = np.broadcast_to(
        np.stack([joint.offset for joint in joints])[None],
        (frames, len(joints), 3),
    ).copy()
    root_translations = np.zeros((frames, 3), dtype=np.float32)

    for joint_index, joint in enumerate(joints):
        joint_values = values[:, joint.channel_indices]
        rotation_indices = [
            index
            for index, channel in enumerate(joint.channels)
            if channel.lower().endswith("rotation")
        ]
        order = "".join(joint.channels[index][0].upper() for index in rotation_indices)
        if order:
            local_rotations[:, joint_index] = (
                Rotation.from_euler(
                    order,
                    joint_values[:, rotation_indices],
                    degrees=True,
                )
                .as_rotvec()
                .astype(np.float32)
            )

        # HiPHI joints may provide position channels alongside rotations.
        # We prefer the position channels here
        position_indices = {
            channel[0].lower(): index
            for index, channel in enumerate(joint.channels)
            if channel.lower().endswith("position")
        }
        destination = root_translations if joint_index == 0 else local_translations[:, joint_index]
        if joint_index == 0:
            destination[:] = joint.offset
        for axis, component in (("x", 0), ("y", 1), ("z", 2)):
            if axis in position_indices:
                destination[:, component] = joint_values[:, position_indices[axis]]

    offsets = np.stack([joint.offset for joint in joints]).astype(np.float32)
    offsets *= unit_scale
    offsets[0] = 0.0
    local_translations *= unit_scale
    local_translations[:, 0] = 0.0
    root_translations *= unit_scale

    skeleton = Skeleton(
        tuple(joint.name for joint in joints),
        np.asarray([joint.parent for joint in joints], dtype=np.int64),
        offsets,
    )
    return MotionClip(
        skeleton=skeleton,
        local_rotations=local_rotations,
        root_translations=root_translations,
        frame_time=frame_time,
        local_translations=local_translations,
    )
