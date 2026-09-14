from __future__ import annotations

import numpy as np

from hiphi2smplx.pipeline import equivalent_foot_targets


def test_equivalent_foot_targets_follow_ankles() -> None:
    targets = np.zeros((2, 22, 3), dtype=np.float32)
    targets[:, 7] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    targets[:, 8] = [[-1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]]
    rest = np.zeros((22, 3), dtype=np.float32)
    rest[10] = rest[7] + [0.0, -0.05, 0.15]
    rest[11] = rest[8] + [0.0, -0.05, 0.15]
    reference = np.zeros_like(targets)

    result = equivalent_foot_targets(
        targets,
        rest,
        reference,
        (np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)),
    )

    np.testing.assert_allclose(result[:, 10] - result[:, 7], [0.0, -0.05, 0.15])
    np.testing.assert_allclose(result[:, 11] - result[:, 8], [0.0, -0.05, 0.15])
