# Release checklist

The code is separated from the original mixed solver repository and is ready
for technical review. Before publishing:

- [ ] Add the final non-commercial research LICENSE approved by the copyright holder.
- [ ] Confirm copyright ownership and add NOTICE or attribution if required.
- [ ] Confirm that extracted code may be redistributed under the selected license.
- [ ] Obtain or verify redistribution permission for code adapted from
      `wangsen1312/joints2smpl`; its repository currently has no root LICENSE.
- [ ] Obtain explicit redistribution permission for the VIBE/SMPLify-derived
      pose-prior implementation and any bundled `gmm_08.pkl` or mean-pose data;
      a non-commercial-use grant alone does not necessarily allow redistribution.
- [ ] Add the final public repository URL to pyproject.toml project metadata.
- [ ] Run tests, Ruff, and wheel/sdist builds in a clean Python 3.10+ environment.
- [ ] Run one end-to-end conversion using a locally licensed SMPL-X model.
- [ ] Confirm no unapproved SMPL-X, MANO, GMM, mean-pose, HiPHI dataset, beta,
      output, or credential files are tracked.
- [ ] Review the README naming and public contact information.
