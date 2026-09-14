"""Command-line batch conversion for HiPHI release trees."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .mano import ManoFitter
from .pipeline import (
    PIPELINE_VERSION,
    ConversionConfig,
    fit_bvh,
    save_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    beta = parser.add_mutually_exclusive_group(required=True)
    beta.add_argument("--betas", type=Path, help="One NPY/NPZ beta vector for every clip")
    beta.add_argument(
        "--actor-betas",
        type=Path,
        help="NPZ or JSON mapping from HiPHI actor_id to 10-D betas",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-only", action="store_true")
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument("--hoi-only", action="store_true")
    subset.add_argument("--non-hoi-only", action="store_true")
    parser.add_argument("--include", help="Keep relative paths containing this text")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--mink-iters", type=int, default=10)
    parser.add_argument("--mink-dt", type=float, default=0.01)
    parser.add_argument("--mink-damping", type=float, default=1e-2)
    parser.add_argument("--position-cost", type=float, default=1.0)
    parser.add_argument("--ankle-wrist-toe-weight", type=float, default=5.0)
    parser.add_argument("--orientation-cost", type=float, default=0.1)
    parser.add_argument("--posture-cost", type=float, default=0.1)
    parser.add_argument("--root-posture-cost", type=float, default=0.1)
    parser.add_argument("--qp-solver", default="daqp")
    parser.add_argument(
        "--mano-fitter",
        metavar="MODULE:ATTRIBUTE",
        help="Optional external ManoFitter class or instance; implementation is not included",
    )
    parser.add_argument(
        "--copy-assets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy metadata.json, object_tracks, and object_meshes when present",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser


def load_betas(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as data:
            key = "betas" if "betas" in data.files else "beta"
            if key not in data.files:
                raise KeyError(f"{path} must contain betas or beta")
            values = data[key]
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size < 10 or not np.isfinite(values).all():
        raise ValueError(f"{path} must contain at least 10 finite beta coefficients")
    return values[:10].copy()


def load_actor_betas(path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("actors", payload) if isinstance(payload, dict) else payload
        result = {
            str(row["actor_id"]): np.asarray(row["beta"], dtype=np.float32).reshape(-1)[:10]
            for row in rows
        }
    else:
        with np.load(path, allow_pickle=False) as data:
            if "actor_ids" not in data.files or "betas" not in data.files:
                raise KeyError(f"{path} must contain actor_ids and betas")
            actor_ids = np.asarray(data["actor_ids"]).reshape(-1)
            values = np.asarray(data["betas"], dtype=np.float32)
            if values.ndim != 2 or values.shape[0] != len(actor_ids) or values.shape[1] < 10:
                raise ValueError(f"{path} has incompatible actor beta shape {values.shape}")
            result = {
                str(actor_id): values[index, :10].copy()
                for index, actor_id in enumerate(actor_ids)
            }
    if not result or any(
        value.shape != (10,) or not np.isfinite(value).all()
        for value in result.values()
    ):
        raise ValueError(f"{path} contains no valid 10-D actor betas")
    return result


def _metadata_sources(input_root: Path, is_hoi: bool) -> list[Path]:
    metadata_path = input_root / "metadata" / "hiphi_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"subset selection requires {metadata_path}")
    expected = "true" if is_hoi else "false"
    paths: list[Path] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("is_hoi", "").strip().lower() != expected:
                continue
            fields = [row.get(name, "").strip() for name in ("frame", "lu", "motion_id")]
            if not all(fields):
                raise ValueError(f"invalid metadata row in {metadata_path}: {row}")
            paths.append(input_root / "data" / fields[0] / fields[1] / fields[2] / "motion_actor.bvh")
    return sorted(paths)


def load_sources(
    input_root: Path,
    manifest: Path | None,
    *,
    hoi_only: bool = False,
    non_hoi_only: bool = False,
) -> list[Path]:
    if manifest is not None and manifest.is_file():
        relatives = [
            Path(line.strip())
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        invalid = [
            path
            for path in relatives
            if path.is_absolute() or ".." in path.parts or path.name != "motion_actor.bvh"
        ]
        if invalid:
            raise ValueError(f"manifest contains invalid source path: {invalid[0]}")
        sources = [input_root / path for path in relatives]
        if hoi_only or non_hoi_only:
            allowed = set(_metadata_sources(input_root, hoi_only))
            sources = [path for path in sources if path in allowed]
        return sources

    if hoi_only or non_hoi_only:
        sources = _metadata_sources(input_root, hoi_only)
    else:
        sources = sorted(input_root.rglob("motion_actor.bvh"))
    if manifest is not None:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest.with_name(f".{manifest.name}.tmp")
        temporary.write_text(
            "".join(f"{path.relative_to(input_root)}\n" for path in sources),
            encoding="utf-8",
        )
        temporary.replace(manifest)
    return sources


def load_mano_fitter(spec: str | None) -> ManoFitter | None:
    if spec is None:
        return None
    if ":" not in spec:
        raise ValueError("--mano-fitter must use MODULE:ATTRIBUTE syntax")
    module_name, attribute_name = spec.rsplit(":", 1)
    target = getattr(importlib.import_module(module_name), attribute_name)
    fitter = target() if isinstance(target, type) else target
    if not isinstance(fitter, ManoFitter):
        raise TypeError(f"{spec} does not implement ManoFitter.fit")
    return fitter


def _clip_betas(
    bvh_path: Path,
    fixed: np.ndarray | None,
    actors: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, str | None]:
    if fixed is not None:
        return fixed, None
    metadata_path = bvh_path.parent / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"actor beta mode requires {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actor_id = str(metadata.get("actor_id", "")).strip()
    if actors is None or actor_id not in actors:
        raise KeyError(f"no beta found for actor_id={actor_id!r}")
    return actors[actor_id], actor_id


def _copy_assets(source_dir: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    for name in ("metadata.json", "object_tracks", "object_meshes"):
        source = source_dir / name
        destination = output_dir / name
        if not source.exists():
            continue
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
        copied.append(name)
    return copied


def _is_current(path: Path, hands_fitted: bool) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(np.asarray(data["pipeline"]).reshape(())) == PIPELINE_VERSION
                and bool(np.asarray(data["hands_fitted"]).reshape(())) == hands_fitted
            )
    except (OSError, ValueError, KeyError):
        return False


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    model_path = args.model_path.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive")
    if args.max_clips is not None and args.max_clips < 1:
        raise ValueError("--max-clips must be positive")

    manifest = None if args.manifest is None else args.manifest.resolve()
    sources = load_sources(
        input_root,
        manifest,
        hoi_only=args.hoi_only,
        non_hoi_only=args.non_hoi_only,
    )
    if args.manifest_only:
        if manifest is None:
            raise ValueError("--manifest-only requires --manifest")
        print(f"manifest={manifest} clips={len(sources)}")
        return 0
    if args.include:
        sources = [
            path for path in sources if args.include in str(path.relative_to(input_root))
        ]
    sources = sources[args.shard_index :: args.num_shards]
    if args.max_clips is not None:
        sources = sources[: args.max_clips]
    if not sources:
        raise ValueError("no source BVHs selected")

    fixed_betas = None if args.betas is None else load_betas(args.betas.resolve())
    actor_betas = (
        None if args.actor_betas is None else load_actor_betas(args.actor_betas.resolve())
    )
    mano_fitter = load_mano_fitter(args.mano_fitter)
    config = ConversionConfig(
        iterations=args.mink_iters,
        dt=args.mink_dt,
        damping=args.mink_damping,
        posture_cost=args.posture_cost,
        root_posture_cost=args.root_posture_cost,
        position_cost=args.position_cost,
        ankle_wrist_toe_weight=args.ankle_wrist_toe_weight,
        orientation_cost=args.orientation_cost,
        qp_solver=args.qp_solver,
        progress=args.progress,
    )
    report_path = output_root / (
        "pipeline_report.json"
        if args.num_shards == 1
        else f"pipeline_report_{args.shard_index:03d}_of_{args.num_shards:03d}.json"
    )
    records: list[dict[str, Any]] = []
    batch_start = time.perf_counter()

    def flush_report() -> None:
        _write_json(
            report_path,
            {
                "pipeline": PIPELINE_VERSION,
                "input_root": str(input_root),
                "output_root": str(output_root),
                "model_path": str(model_path),
                "mano_fitter": args.mano_fitter,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "elapsed_sec": time.perf_counter() - batch_start,
                "clips": records,
            },
        )

    for index, bvh_path in enumerate(sources, start=1):
        relative = bvh_path.relative_to(input_root)
        output_dir = output_root / relative.parent
        output = output_dir / "motion_actor_smplx.npz"
        if not args.overwrite and _is_current(output, mano_fitter is not None):
            records.append({"source": str(relative), "output": str(output), "status": "skipped"})
            flush_report()
            print(f"[{index}/{len(sources)}] skip {relative}", flush=True)
            continue

        started = time.perf_counter()
        print(f"[{index}/{len(sources)}] fit {relative}", flush=True)
        try:
            beta, actor_id = _clip_betas(bvh_path, fixed_betas, actor_betas)
            result = fit_bvh(
                bvh_path,
                model_path,
                beta,
                config=config,
                max_frames=args.max_frames,
                mano_fitter=mano_fitter,
            )
            save_result(output, result, source_bvh=bvh_path, config=config)
            assets = _copy_assets(bvh_path.parent, output_dir) if args.copy_assets else []
            record = {
                "source": str(relative),
                "output": str(output),
                "status": "complete",
                "actor_id": actor_id,
                "frames": len(result.poses),
                "hands_fitted": result.hand_metadata is not None,
                "copied_assets": assets,
                "mean_residual_cm": float(result.residuals.mean() * 100.0),
                "max_residual_cm": float(result.residuals.max() * 100.0),
                "elapsed_sec": time.perf_counter() - started,
            }
            if result.hand_metadata is not None:
                record["hand_metadata"] = result.hand_metadata
            records.append(record)
        except Exception as exc:
            records.append(
                {
                    "source": str(relative),
                    "output": str(output),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "elapsed_sec": time.perf_counter() - started,
                }
            )
            flush_report()
            if args.fail_fast:
                raise
            print(f"[{index}/{len(sources)}] failed {relative}: {exc}", flush=True)
            continue
        flush_report()
        print(f"[{index}/{len(sources)}] saved {output}", flush=True)

    return 1 if any(record["status"] == "failed" for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
