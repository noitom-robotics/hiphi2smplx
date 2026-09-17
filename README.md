# hiphi2smplx

`hiphi2smplx` converts [HiPHI](https://noitom-robotics.github.io/hiphi/)
actor BVH motion into SMPL-X body parameters. It fits the actor's body shape
and uses [Mink](https://github.com/kevinzakka/mink) differential inverse
kinematics to retarget each frame to a shape-aware SMPL-X skeleton.

The current release supports the fixed HiPHI BVH skeleton and the 22 SMPL-X
body joints. Face, eye, and finger poses are not fitted. The codebase is
designed to remain extensible: support for a new BVH skeleton is added as a
separate named semantic map instead of broadening the HiPHI mapping with joint
aliases.

## Examples

Example conversions and qualitative results will be added here.

A planned downstream workflow is:

```text
HiPHI BVH -> hiphi2smplx -> SMPL-X body motion -> UMR humanoid retargeting
```

See [UMR](https://github.com/hanyang9/UMR) for unified motion retargeting from
SMPL-X motion to humanoid robots. An end-to-end HiPHI example will be added
after the input adapter has been validated.

## Installation

Python 3.10 or newer is required.

```bash
cd hiphi2smplx
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The default quadratic-programming backend is DAQP. Another backend supported
by `qpsolvers` can be selected with `--qp-solver` when installed in the same
environment.

### Download model files

Model files and pose-prior data are not distributed with this repository.
Download them separately and follow the terms of their respective licenses.

1. Download the SMPL-X model from the
   [official SMPL-X website](https://smpl-x.is.tue.mpg.de/) after registering
   and accepting its license. This converter expects the NPZ neutral model,
   normally named `SMPLX_NEUTRAL.npz`.
2. Beta fitting additionally requires `neutral_smpl_mean_params.h5` and
   `gmm_08.pkl`. The mean parameters are available through the
   [HMR model setup](https://github.com/akanazawa/hmr/blob/master/doc/train.md),
   and the GMM pose prior is part of the registered
   [SMPLify](https://smplify.is.tue.mpg.de/) resources. The fitting code is
   adapted from [joints2smpl](https://github.com/wangsen1312/joints2smpl).

One possible local layout is:

```text
assets/
  models/
    SMPLX_NEUTRAL.npz
  beta_fit/
    neutral_smpl_mean_params.h5
    gmm_08.pkl
```

## Quick Start

Fit the first-frame body shape and convert one HiPHI BVH file:

```bash
hiphi2smplx \
  --input-bvh /path/to/motion_actor.bvh \
  --output /path/to/motion_actor_smplx.npz \
  --model-path assets/models/SMPLX_NEUTRAL.npz \
  --fit-betas \
  --beta-fit-data assets/beta_fit \
  --beta-device cuda:0 \
  --progress
```

Beta fitting uses CUDA by default. Use `--beta-device cpu` when no CUDA device
is available.

To reuse an existing SMPL-X shape instead of fitting it again, pass an NPY or
NPZ file containing at least 10 finite beta coefficients:

```bash
hiphi2smplx \
  --input-bvh /path/to/motion_actor.bvh \
  --output /path/to/motion_actor_smplx.npz \
  --model-path assets/models/SMPLX_NEUTRAL.npz \
  --betas /path/to/betas.npy \
  --progress
```

The output is an AMASS-style NPZ containing `poses` with shape `(frames, 165)`,
`trans`, `betas`, `gender`, and `mocap_framerate`. The SMPL-X pose layout is
kept intact, while jaw, eye, and finger rotations remain zero.

## Adding another BVH skeleton

HiPHI uses the `hiphi` map in
`hiphi2smplx.skeleton.SOURCE_SKELETON_MAPS`. To support another BVH hierarchy,
add a new semantic-to-joint-name map under a new key. The converter validates every required mapped joint before fitting.

## Citation

## References

Parts of the beta-fitting implementation are adapted from the following
projects:

- [joints2smpl](https://github.com/wangsen1312/joints2smpl)
- [VIBE](https://github.com/mkocabas/VIBE)
