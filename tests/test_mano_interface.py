from __future__ import annotations

import numpy as np
import pytest

from hiphi2smplx.mano import ManoFittingResult, merge_mano_result


def test_merge_mano_result_uses_smplx_finger_slots() -> None:
    body = np.zeros((2, 55, 3), dtype=np.float32)
    left = np.full((2, 15, 3), 0.1, dtype=np.float32)
    right = np.full((2, 15, 3), -0.2, dtype=np.float32)

    merged, metadata = merge_mano_result(
        body,
        ManoFittingResult(left, right, {"backend": "external"}),
    )

    np.testing.assert_array_equal(merged[:, :25], 0.0)
    np.testing.assert_array_equal(merged[:, 25:40], left)
    np.testing.assert_array_equal(merged[:, 40:55], right)
    assert metadata == {"backend": "external"}
    np.testing.assert_array_equal(body, 0.0)


def test_merge_mano_result_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="must each have shape"):
        merge_mano_result(
            np.zeros((2, 55, 3), dtype=np.float32),
            ManoFittingResult(
                np.zeros((2, 14, 3), dtype=np.float32),
                np.zeros((2, 15, 3), dtype=np.float32),
            ),
        )
