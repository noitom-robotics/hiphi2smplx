from __future__ import annotations

import numpy as np

from hiphi2smplx.mink import build_skeleton_mjcf
from hiphi2smplx.pipeline import SMPL22_PARENTS


def test_mjcf_contains_22_named_sites() -> None:
    rest = np.zeros((22, 3), dtype=np.float32)
    rest[:, 1] = np.arange(22) * 0.01

    xml = build_skeleton_mjcf(rest, SMPL22_PARENTS)

    assert 'model="hiphi2smplx_mink"' in xml
    assert xml.count('type="ball"') == 21
    assert xml.count('site name=') == 22
