# hiphi2smplx

Convert HiPHI actor BVH files to SMPL-X body parameters with
[Mink](https://github.com/kevinzakka/mink) differential inverse kinematics.

This repository intentionally contains one fitting path:

- HiPHI BVH parsing and semantic 22-joint mapping
- beta-shaped SMPL-X rest skeletons
- T-pose rotation-copy initialization
- no-scale, frame-wise Mink body IK
- beta-shaped equivalent toe targets and sole calibration
- optional copying of HiPHI metadata and object assets

Finger fitting is not included. Body-only results use the standard SMPL-X
55-joint pose layout, with jaw, eye, and finger rotations set to zero. A typed
MANO plugin interface is available for separately distributed implementations.

## Model files

SMPL-X model files are not included and must not be committed to this
repository. Download them from the official SMPL-X project after accepting its
license. The converter expects the NPZ model file, typically named
SMPLX_NEUTRAL.npz.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The default QP backend is DAQP. Select another qpsolvers backend with
`--qp-solver` if it is installed in the same environment.

## Convert a HiPHI tree

Use one beta vector for all clips:

```bash
hiphi2smplx \
  --input-root /path/to/HiPHI \
  --output-root /path/to/output \
  --model-path /path/to/SMPLX_NEUTRAL.npz \
  --betas /path/to/betas.npy \
  --progress
```

Use per-actor betas:

```bash
hiphi2smplx \
  --input-root /path/to/HiPHI \
  --output-root /path/to/output \
  --model-path /path/to/SMPLX_NEUTRAL.npz \
  --actor-betas /path/to/actor_betas.npz \
  --hoi-only
```

The actor manifest NPZ must contain `actor_ids` and a two-dimensional
`betas` array. A JSON list of objects with `actor_id` and `beta` is also
accepted. In actor mode, each clip directory must contain `metadata.json`
with its `actor_id`.

For parallel batch conversion, assign disjoint shards:

```bash
hiphi2smplx ... --num-shards 8 --shard-index 0
hiphi2smplx ... --num-shards 8 --shard-index 1
```

Each source `motion_actor.bvh` produces `motion_actor_smplx.npz` at the
matching relative output path. The output includes 55-joint axis-angle poses,
translation, betas, frame timing, fitted body joints, residuals, and solver
provenance. `metadata.json`, `object_tracks`, and `object_meshes` are
copied when present; pass `--no-copy-assets` to disable this.

## Skeleton maps

HiPHI input uses the strict `hiphi` map, which matches the fixed HiPHI names
such as `Hips`, `Spine4`, `LeftUpLeg`, and `LeftToeBase`. Source map registry lives
in `hiphi2smplx.bvh.SOURCE_SKELETON_MAPS`. Select it explicitly with
`--skeleton-map hiphi`. A future BVH skeleton should add a new named map to the
registry rather than adding aliases. Missing mapped joints fail immediately.

## Python API

```python
import numpy as np
from hiphi2smplx import ConversionConfig, fit_bvh, save_result

betas = np.load("/path/to/betas.npy")
config = ConversionConfig(iterations=10)
result = fit_bvh(
    "/path/to/motion_actor.bvh",
    "/path/to/SMPLX_NEUTRAL.npz",
    betas,
    config=config,
)
save_result(
    "/path/to/motion_actor_smplx.npz",
    result,
    source_bvh="/path/to/motion_actor.bvh",
    config=config,
)
```

## Optional MANO integration

`hiphi2smplx.mano.ManoFitter` defines the boundary for an external hand
fitter. It receives the body solution and returns left/right SMPL-X finger
rotations with shape `(frames, 15, 3)`. The CLI loads an external class or
instance using `--mano-fitter package.module:Attribute`.

See `examples/mano_plugin_stub.py` for the contract. The example deliberately
does not implement MANO fitting.

## Coordinate and fitting assumptions

- BVH positions and offsets are converted from centimetres to metres.
- The source motion is not scaled to the SMPL-X body.
- SMPL-X beta is fixed during IK; this repository does not fit body shape.
- Joint positions do not fully determine axial twist. Rotation-copy
  initialization supplies the reference orientation.
- Output uses a Y-up coordinate system and axis-angle local rotations.

## Validation

```bash
python -m ruff check .
python -m build
```

A real conversion smoke test requires a valid HiPHI BVH and SMPL-X NPZ.

## Release status

This is a review tree, not yet a public release. Complete
`RELEASE_CHECKLIST.md`, especially license and copyright selection, before
publishing.
