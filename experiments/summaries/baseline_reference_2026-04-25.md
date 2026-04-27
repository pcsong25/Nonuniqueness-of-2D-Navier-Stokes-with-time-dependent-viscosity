# Historical Baseline Reference Run - 2026-04-25

Note: this run remains useful as historical validation data, but it is not the
current active `src/baseline/driver.py` input. As of the c3-only M3 validation on
2026-04-26, the authoritative driver uses `c_3=60+0j` and `c_4=0+0j`,
with `h_min=0.05` and `h_max=0.10`.

## Command

```bash
env MPLCONFIGDIR=/tmp/mplconfig-euler conda run -n femx_spc python src/baseline/driver.py
```

## Inputs Used For This Historical Run

- `mu = 0.8`
- `R = 150.0`
- `Rdom = 10.0`
- `M = 48`
- `sigma_grid = np.linspace(0.0, 1.0, 21)`
- `h_min = 0.06`
- `h_max = 0.12`
- active Fourier coefficient: `c_4 = 50 + 0j`
- `k = 6`
- eigen shift target: `sigma = -1.0`

## Baseline Outputs

- profile fields: `experiments/raw/baseline/results.xdmf`
- profile HDF5 data: `experiments/raw/baseline/results.h5`
- exported operator blocks: `experiments/raw/baseline/linop_blocks.mat`
- debug plots: `src/baseline/mesh.png`, `src/baseline/streamlines.png`,
  `src/baseline/vorticity.png`, `src/baseline/eig_plots/`
- timing log: `experiments/raw/baseline/run_time.txt`

## Numerical Summary

- mesh vertices: `25600`
- cells: `50674`
- DoFs: `Omega = 25600`, `psi = 25600`
- total wall time: `243.419 s`
- `eps = 4.444e-03`
- raw weak residual `||R_Omega||_2 = 6.070e-04`
- raw weak residual `||R_Psi||_2 = 1.797e-13`

## Eigenvalues

```text
#1 lambda = 4.537468912529909e-01 -1.176662542133605e+00j
#2 lambda = 4.537468912529909e-01 +1.176662542133605e+00j
#3 lambda = 4.677034814983481e-01 +0.000000000000000e+00j
#4 lambda = 4.783460496236707e-01 -1.276583293491166e+00j
#5 lambda = 4.791564448291539e-01 -1.267528281259206e+00j
#6 lambda = 4.791564448291539e-01 +1.267528281259206e+00j
```

## Notes

- The shell default `python` does not have `gmsh`; the working FEniCSx
  environment is `conda run -n femx_spc`.
- OpenMPI printed a TCP bind warning under the sandbox but the serial run
  completed successfully.
- Residuals are baseline raw PETSc vector norms and are not normalized.
