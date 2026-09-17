# Beta-fitting resources

The integrated beta fitter looks for these files in this directory:

- `neutral_smpl_mean_params.h5`
- `gmm_08.pkl`

Do not add either file to a public release until its redistribution terms have
been reviewed and approved. During development, pass an external directory with
`--beta-fit-data` or `BetaFitConfig(data_dir=...)`.
