from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from hiphi2smplx.cli import build_parser, load_actor_betas, load_sources


_REQUIRED = [
    "--input-root", "input",
    "--output-root", "output",
    "--model-path", "SMPLX_NEUTRAL.npz",
]


def test_parser_requires_exactly_one_beta_source() -> None:
    parser = build_parser()
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        parser.parse_args(_REQUIRED)
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        parser.parse_args([*_REQUIRED, "--betas", "a.npy", "--actor-betas", "b.npz"])


def test_parser_exposes_mano_only_as_external_plugin() -> None:
    args = build_parser().parse_args(
        [*_REQUIRED, "--betas", "a.npy", "--mano-fitter", "example:Plugin"]
    )
    assert args.mano_fitter == "example:Plugin"
    assert not hasattr(args, "mano_root")


def test_actor_beta_json(tmp_path: Path) -> None:
    path = tmp_path / "betas.json"
    path.write_text(
        json.dumps([{"actor_id": "7", "beta": list(range(10))}]),
        encoding="utf-8",
    )
    result = load_actor_betas(path)
    np.testing.assert_array_equal(result["7"], np.arange(10, dtype=np.float32))


def test_metadata_subset_selection(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata" / "hiphi_metadata.csv"
    metadata.parent.mkdir()
    metadata.write_text(
        "frame,lu,motion_id,is_hoi\n"
        "Frame,action,hoi_clip,true\n"
        "Frame,action,plain_clip,false\n",
        encoding="utf-8",
    )
    hoi = load_sources(tmp_path, None, hoi_only=True)
    plain = load_sources(tmp_path, None, non_hoi_only=True)
    assert hoi[0].parent.name == "hoi_clip"
    assert plain[0].parent.name == "plain_clip"
