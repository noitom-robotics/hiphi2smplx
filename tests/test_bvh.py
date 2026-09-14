from __future__ import annotations

import numpy as np

from hiphi2smplx.bvh import Skeleton, _infer_mapping


def test_short_spine_uses_upper_spine_as_chest() -> None:
    names = (
        "Hips", "Spine", "Spine1", "Neck", "Head",
        "LeftShoulder", "RightShoulder", "LeftArm", "RightArm",
        "LeftForeArm", "RightForeArm", "LeftHand", "RightHand",
        "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg",
        "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase",
    )
    parents = np.zeros(len(names), dtype=np.int64)
    parents[0] = -1
    skeleton = Skeleton(names, parents, np.zeros((len(names), 3), dtype=np.float32))

    mapping = _infer_mapping(skeleton)

    assert mapping["spine2"] == "Spine1"
    assert mapping["chest"] == "Spine1"
