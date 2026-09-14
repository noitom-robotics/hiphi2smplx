# Release checklist

The code is separated from the original mixed solver repository and is ready
for technical review. Before publishing:

- [ ] Select and add an open-source LICENSE approved by the copyright holder.
- [ ] Confirm copyright ownership and add NOTICE or attribution if required.
- [ ] Confirm that extracted code may be redistributed under the selected license.
- [ ] Add the final public repository URL to pyproject.toml project metadata.
- [ ] Run tests, Ruff, and wheel/sdist builds in a clean Python 3.10+ environment.
- [ ] Run one end-to-end conversion using a locally licensed SMPL-X model.
- [ ] Confirm no SMPL-X, MANO, HiPHI dataset, beta, output, or credential files are tracked.
- [ ] Review the README naming and public contact information.
