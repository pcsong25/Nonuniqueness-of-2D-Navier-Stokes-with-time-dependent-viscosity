# Progress log

## Current objective

Build and validate a new sparse-matrix finite-difference implementation for the self-similar
2D Euler time-dependent-viscosity eigenvalue workflow, while preserving the logic and outputs
of the verified baseline implementation.

## Current status

Baseline reference run completed on 2026-04-25 using `conda run -n femx_spc`.
The default shell Python still lacks `gmsh`, but the `femx_spc` environment has
the required FEniCSx/gmsh/PETSc/SLEPc stack.

The `src/nextgen/` foundation now includes configuration, Fourier boundary
data, square-grid sparse operators, diagnostics, a sparse Neumann Poisson
solve with gauge handling, the original square M3 sparse Newton profile
solver, and a general full-disk polar profile solver for disk validation. No
nextgen eigensolver has been implemented yet.

The current authoritative `src/baseline/driver.py` input now uses
`c_3=60+0j` and `c_4=0+0j`, with `h_min=0.05` and `h_max=0.10`.
Older m4 or mixed m3/m4 baseline rows and summaries are historical references,
not the current active driver state.

## 2026-04-27 M3 general full-disk polar validation

Correction to the previous c3-sector experiment: the production M3 solver
should not use a mesh customized to a particular symmetry. Symmetry remains a
debugging diagnostic only. The current validation path uses `symmetry=1`, a
full polar disk, and arbitrary Fourier boundary data.

### Added / changed

- `src/nextgen/polar_profile.py` now uses a full-disk mean-gauge Poisson block
  through a bordered sparse system instead of replacing one angular row.
- `experiments/run_m3_c3_profile_visual_check.py` now uses a full disk
  `nr=54`, `ntheta=162` for the active `c_3=60`, `c_4=0` visual check.
- `experiments/run_m3_polar_validation_suite.py` validates three additional
  reduced boundary conditions without symmetry-sector meshes:
  `c_4=5`, mixed complex modes `c_2=3+i`, `c_5=-2+0.5i`, and mixed
  `c_3=6`, `c_4=5`.
- `experiments/summaries/m3_c3_60_c4_0_full_disk_visual_check_2026-04-27.json`
- `experiments/summaries/m3_polar_full_disk_validation_suite_2026-04-27.json`
- plots under
  `experiments/raw/nextgen/m3_c3_60_c4_0_full_disk_visual/`

Norm definitions used in the validation:

- Native nextgen residual: the assembled polar residual weighted by polar cell
  areas, reported as
  `sqrt(||R_Omega||_L2^2 + ||R_Poisson||_L2^2) /
  max(||Omega||_L2, ||Delta psi||_L2)`.
- Common sampled strong residual: both nextgen and baseline are interpolated to
  the same Cartesian sample grid. Centered finite differences are applied to
  compute
  `R_Omega = v . grad(Omega) - Omega - eps Delta(Omega)` and
  `R_Psi = Delta(psi) - Omega`; the reported relative norm is
  `sqrt(||R_Omega||_L2^2 + ||R_Psi||_L2^2) /
  max(||Omega||_L2, ||Delta psi||_L2)`.
- Probe RMS: `sqrt(mean((Omega_nextgen - Omega_baseline)^2))` on common polar
  probe points with radius in `[0.5, 8]`.
- Probe Linf: `max(abs(nextgen - baseline))` on the same valid probes.
- Streamfunction probe errors remove the mean difference first, because psi is
  defined only up to a gauge constant.
- Baseline raw residuals from `residuals_quick.py` are PETSc vector 2-norms of
  the baseline weak residuals. They are logged, but the common sampled strong
  residual is the fairer cross-code residual comparison.

Full-amplitude active c3 visual run:

```bash
conda run -n femx_spc python experiments/run_m3_c3_profile_visual_check.py
```

Results:

- nextgen full disk: `nr=54`, `ntheta=162`, runtime `49.409 s`,
  native relative residual `2.282e-14`
- baseline: `h_min=0.12`, `h_max=0.18`, runtime `10.050 s`,
  mesh vertices `11513`, cells `22674`
- common sampled strong residual: nextgen `5.893e-01`, baseline `1.118e+00`
- probe discrepancies on radius `<=8`: Omega RMS `3.020e-01`, Omega Linf
  `4.543`, psi RMS after mean-offset removal `3.814e-02`, psi Linf
  `1.092e-01`
- c3 symmetry diagnostic on this c3-only test: nextgen Omega relative RMS
  `1.157e-14`, psi relative RMS `5.620e-15`; this is a diagnostic result, not
  a design assumption.

General reduced-boundary validation suite:

```bash
conda run -n femx_spc python experiments/run_m3_polar_validation_suite.py
```

All three cases completed with the full-disk grid `nr=32`, `ntheta=96` and
baseline mesh `h_min=0.20`, `h_max=0.30`.

| case | nextgen runtime | baseline runtime | nextgen common strong rel | baseline common strong rel | Omega probe RMS | psi probe RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `c4_reduced` | `3.982 s` | `1.772 s` | `7.054e-01` | `2.938e+00` | `1.081e-01` | `2.091e-02` |
| `mixed_2_5_reduced` | `5.666 s` | `1.907 s` | `6.913e-01` | `2.093e+00` | `1.990e-01` | `1.456e-02` |
| `c3_c4_reduced` | `5.078 s` | `1.987 s` | `7.401e-01` | `3.629e+00` | `1.994e-01` | `5.395e-02` |

Interpretation:

- The current polar solver is now general with respect to Fourier boundary
  data and no longer relies on an n-fold sector mesh.
- On the reduced validation suite, the nextgen common sampled strong residuals
  are consistently lower than the baseline sampled strong residuals on the
  chosen grids, and the field probe errors are moderate.
- Runtime is still slower than baseline on all tested cases. The full-amplitude
  c3 case especially requires `ntheta=162` to match the angular resolution of
  the earlier sector experiment, giving `49.409 s` versus baseline `10.050 s`.

Recommended next step:

- Stay in M3. Improve full-disk polar robustness/performance before M4,
  especially around the active full-amplitude c3 resolution threshold.

### Full-disk polar refinement study

Command:

```bash
conda run -n femx_spc python experiments/run_m3_polar_refinement_study.py
```

Summary:

- `experiments/summaries/m3_polar_full_disk_refinement_study_2026-04-27.json`
- status: `partial`, because the coarser active-c3 nextgen grid stalled before
  reaching sigma `1`.
- no eigensolver or M4 code was used.

The refinement study used a full polar disk for nextgen and disk meshes for
baseline. Comparisons used the same common sampled strong residual and probe
RMS/Linf definitions described above. Baseline weak residuals are also logged,
but they are not used as the cross-code residual metric.

Active full-amplitude `c_3=60`, `c_4=0`:

| run | runtime | status | common strong rel | Omega probe RMS | psi probe RMS |
| --- | ---: | --- | ---: | ---: | ---: |
| baseline h `0.20/0.30` | `17.609 s` | completed but unstable | `4.111e+03` | mesh-diff `3.307e+04` | mesh-diff `3.135e+02` |
| baseline h `0.12/0.18` | `18.859 s` | completed | `8.353e-01` | reference | reference |
| nextgen `nr=48`, `ntheta=144` | `50.186 s` | nonconverged at sigma `0.7` | n/a | n/a | n/a |
| nextgen `nr=54`, `ntheta=162` | `52.408 s` | completed | `6.496e-01` | `3.024e-01` | `3.842e-02` |

Mixed non-symmetric reduced case
`c_2=3+i`, `c_5=-2+0.5i`:

| run | runtime | status | common strong rel | Omega probe RMS | psi probe RMS |
| --- | ---: | --- | ---: | ---: | ---: |
| baseline h `0.30/0.45` | `2.567 s` | completed | `7.631e+00` | mesh-diff `1.117e-01` | mesh-diff `8.351e-02` |
| baseline h `0.20/0.30` | `3.914 s` | completed | `2.093e+00` | reference | reference |
| nextgen `nr=30`, `ntheta=90` | `13.617 s` | completed | `7.592e-01` | `2.622e-01` | `1.670e-02` |
| nextgen `nr=32`, `ntheta=96` | `12.988 s` | completed | `6.913e-01` | `1.990e-01` | `1.456e-02` |
| nextgen `nr=36`, `ntheta=108` | `15.775 s` | completed | `6.535e-01` | `1.424e-01` | `1.252e-02` |

Interpretation:

- The mixed non-symmetric reduced case gives the cleanest refinement evidence:
  nextgen common strong residual, Omega probe RMS, and offset-removed psi probe
  RMS all improve as the full-disk polar grid is refined.
- The active full-amplitude c3 case has a practical resolution threshold.
  `nr=48`, `ntheta=144` stalls in continuation, while `nr=54`,
  `ntheta=162` completes and has a lower common sampled strong residual than
  the fine baseline mesh.
- Baseline mesh refinement matters strongly for the active c3 case: the coarse
  baseline mesh produces a very large common strong residual and field blow-up,
  so comparisons should use the `h_min=0.12`, `h_max=0.18` or finer baseline
  reference.
- The nextgen profile solver is not yet an efficiency improvement. It is more
  accurate than the tested baseline meshes under the common strong residual,
  but direct sparse Newton solves are slower on these grid sizes.

## 2026-04-26 M3 circular c3-sector profile validation

Superseded diagnostic note: this section used a c3-sector mesh and should not
be treated as the production validation path. It is retained only as historical
debugging evidence that the square-domain issue was symmetry related.

### Added

- `src/nextgen/polar_profile.py`
- polar profile tests in `tests/test_nextgen_profile.py`
- updated `experiments/run_m3_c3_profile_visual_check.py` to use the polar
  c3-sector solver for the active `c_3=60`, `c_4=0` visual comparison
- `experiments/summaries/m3_c3_60_c4_0_polar_visual_check_2026-04-26.json`
- simple plots under
  `experiments/raw/nextgen/m3_c3_60_c4_0_polar_visual/`

Implementation notes:

- The new disk path uses a cell-centered polar finite-difference grid with a
  `2*pi/3` periodic sector for c3 data. Sampling maps the sector back to the
  full disk, so threefold symmetry is enforced by the discrete domain.
- The Poisson block uses the same gauged Neumann convention as the square
  nextgen solver.
- The Omega equation uses first-order upwind advection in polar coordinates
  and centered diffusion. Centered polar advection preserved symmetry but
  produced the wrong bounded radial profile for this inward-advection case.
- The polar velocity convention is now
  `v_r = -psi_theta/r - mu*r`, `v_theta/r = psi_r/r`, matching
  `(-psi_y, psi_x) - mu*(x,y)`.

Validation run:

```bash
conda run -n femx_spc python experiments/run_m3_c3_profile_visual_check.py
```

Unit checks:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests
```

Result: `19` tests passed.

Inputs:

- `c_3=60+0j`, `c_4=0+0j`
- `mu=0.8`, `R=150`, `Rdom=10`, `M=48`
- nextgen polar sector grid: `nr=54`, `ntheta=54`, `symmetry=3`,
  `dr=0.185185`
- nextgen sigma continuation: `0:0.1:1`
- baseline profile-only disk mesh: `h_min=0.12`, `h_max=0.18`,
  baseline sigma grid `(0, 0.25, 0.5, 0.75, 1)`
- no eigensolver

Results:

- nextgen polar-upwind: completed, runtime `16.288 s`, `71` recorded Newton
  history entries, native relative residual `2.245e-14`, native absolute
  residual `2.607e-13`, physical Poisson residual norm `8.889e-12`
- baseline profile-only: completed, runtime `10.255 s`, mesh vertices `11513`,
  cells `22674`, raw weak residuals `||R_Omega||_2=1.541e-01`,
  `||R_Psi||_2=3.551e-12`
- common sampled strong residual on `[-7,7]^2`:
  nextgen relative `5.893e-01`, baseline relative `1.118e+00`
- probe comparison on `10092` shared points with radius `<=8`:
  Omega RMS discrepancy `3.020e-01`, Omega Linf discrepancy `4.543`,
  psi RMS discrepancy after mean-offset removal `3.814e-02`,
  psi Linf discrepancy after mean-offset removal `1.092e-01`
- threefold symmetry defect:
  nextgen Omega relative RMS `4.842e-15`, nextgen psi relative RMS
  `1.419e-15`; baseline sampled defects were `2.347e-01` and `4.512e-02`,
  mostly reflecting unstructured-mesh interpolation/noise on the probe set.

Interpretation:

- The circular c3-sector solver fixes the fundamental square-domain symmetry
  failure. The nextgen visual field now matches the baseline c3 structure
  qualitatively and has a lower common sampled strong residual than the
  baseline on this grid/probe comparison.
- The nextgen run was slower than the baseline in this validation
  (`16.288 s` vs `10.255 s`) because it needed finer sigma continuation and
  repeated direct Newton solves. It is not yet an efficiency improvement for
  this full-amplitude c3 profile case.
- The remaining Omega Linf discrepancy is concentrated near the central
  high-gradient vortex arms; radial/angular refinement and convergence checks
  are still required before treating M3 as fully validated for production
  profile data.

Recommended next step:

- Stay in M3 and run a polar-grid refinement/convergence study for
  `c_3=60`, `c_4=0` before starting M4. Keep using the common sampled strong
  residual, probe RMS/Linf discrepancies, symmetry defect, and runtime.

## 2026-04-26 M3 c3-only visual/residual check

### Added

- `experiments/run_m3_c3_profile_visual_check.py`
- `experiments/summaries/m3_c3_60_c4_0_visual_check_2026-04-26.json`
- simple plots under `experiments/raw/nextgen/m3_c3_60_c4_0_visual/`
- common sampled strong residual metrics in `src/nextgen/diagnostics.py`
- radially scaled square-boundary vorticity Dirichlet data in
  `src/nextgen/boundary.py`

Input:

- `c_3=60+0j`, `c_4=0+0j`
- `mu=0.8`, `R=150`, `Rdom=10`, `M=48`
- `sigma_grid=(0, 0.25, 0.5, 0.75, 1)`
- nextgen square grid: half-width `10`, `n=129`, `h=0.15625`
- baseline profile-only disk mesh: `h_min=0.12`, `h_max=0.18`
- no eigensolver

Outputs:

- nextgen vorticity heat map:
  `experiments/raw/nextgen/m3_c3_60_c4_0_visual/nextgen_vorticity_heatmap.png`
- nextgen streamfunction contours:
  `experiments/raw/nextgen/m3_c3_60_c4_0_visual/nextgen_streamlines.png`
- baseline vorticity heat map:
  `experiments/raw/nextgen/m3_c3_60_c4_0_visual/baseline_vorticity_heatmap.png`
- baseline streamfunction contours:
  `experiments/raw/nextgen/m3_c3_60_c4_0_visual/baseline_streamlines.png`

Latest results after the square-boundary vorticity scaling fix:

- nextgen: converged, runtime `43.373 s`, `25` recorded Newton history entries,
  final relative native algebraic residual `7.944e-14`, absolute algebraic
  residual `2.302e-12`, Omega boundary mismatch `0`, compatibility defect
  `7.494e-15`
- baseline profile-only: completed, runtime `10.000 s`, mesh vertices `11513`,
  cells `22674`, raw weak residuals `||R_Omega||_2=1.541e-01`,
  `||R_Psi||_2=3.551e-12`
- probe comparison on `8245` shared points with radius `<=8`:
  Omega RMS discrepancy `8.017e-01`, Omega Linf discrepancy `9.504`,
  psi RMS discrepancy after mean-offset removal `1.436`,
  psi Linf discrepancy after mean-offset removal `3.159`
- common sampled strong residual on a uniform square `[-7,7]^2`:
  nextgen relative `1.638e+01`, baseline relative `1.118e+00`
- threefold symmetry defect on polar samples with radius `<=7`:
  nextgen Omega relative RMS `1.416`, baseline Omega relative RMS `0.235`;
  nextgen psi relative RMS `1.730`, baseline psi relative RMS `0.045`

Additional square-domain symmetry probe:

- On a small `n=65` square nextgen run, `c_3` data produced Omega threefold
  relative RMS defect `1.417`, while `c_4` data produced fourfold relative RMS
  defect `2.05e-15`.
- This confirms that the current square-domain truncation preserves the square
  grid's fourfold symmetry, not the threefold symmetry required by c3 boundary
  data.

Interpretation:

- The visual comparison shows a substantial qualitative mismatch between the
  square-grid nextgen profile and the disk baseline profile for this c3-only
  case.
- The nextgen native algebraic residual confirms it solves its assembled
  square-grid equations accurately, but the common sampled strong residual and
  symmetry defect show that this is not yet a reliable approximation of the
  intended whole-space/disk profile.
- The c3/c4 symmetry probe indicates that the square boundary/domain shape is a
  fundamental issue for odd non-square-compatible modes.
- At comparable practical resolution in this test, nextgen is slower
  (`43.373 s`) than the baseline profile-only solve (`10.000 s`).

Recommended next step:

- Stay in M3 validation/fix mode. Before M4, move the nextgen profile solve to
  a disk/rotationally compatible discretization or otherwise enforce a circular
  truncation; the current square domain is not suitable for validating c3
  profiles.

## 2026-04-26 M3 nextgen profile solver

### Added

- `src/nextgen/profile.py`
- `tests/test_nextgen_profile.py`
- `experiments/run_m3_profile_comparison.py`
- `experiments/summaries/m3_profile_reduced_comparison_2026-04-26.json`
- preliminary debug plots under `experiments/raw/nextgen/m3_profile_reduced/`

Implemented:

- strong-form interior profile residual
  `v . grad(Omega) - Omega - eps * Delta(Omega)`
- monolithic residual on `(Omega_interior, psi_full)`
- analytic sparse Jacobian blocks for Omega/Omega, Omega/psi, Poisson/Omega,
  and Poisson/psi
- sigma-continuation with sparse direct Newton solves and residual-decreasing
  backtracking
- profile diagnostics for residual components, Omega boundary mismatch,
  Poisson compatibility, gauge value, runtime, and Newton history
- preliminary Omega, psi, and Newton residual-history debug plots

### Tests

Checks run:

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache-euler python -m py_compile src/nextgen/profile.py experiments/run_m3_profile_comparison.py tests/test_nextgen_profile.py
python -m unittest discover -s tests
```

Result: `14` tests passed.

Added tests cover:

- full-grid psi gradient matrix on a linear field
- finite-difference check of the analytic profile Jacobian
- zero-boundary profile solve
- small nonzero continuation solve with preserved Omega boundary data

### Reduced baseline comparison

Command:

```bash
conda run -n femx_spc python experiments/run_m3_profile_comparison.py
```

Input:

- historical active boundary shape scaled by `0.1`: `c_3=6+0j`, `c_4=5+0j`
- `mu=0.8`, `R=150`, `Rdom=10`, `M=48`
- `sigma_grid=(0, 0.25, 0.5, 0.75, 1)`
- nextgen square grid: half-width `10`, `n=65`, `h=0.3125`
- baseline profile-only disk mesh: `h_min=0.30`, `h_max=0.45`
- no eigensolver

Results:

- nextgen: converged, runtime `7.409 s`, `16` recorded Newton history entries,
  final relative residual `1.505e-13`, Omega boundary mismatch `0`
- baseline profile-only: completed, runtime `1.280 s`, mesh vertices `1901`,
  cells `3660`, raw weak residuals `||R_Omega||_2=1.166`,
  `||R_Psi||_2=4.510e-13`
- probe comparison on `2061` shared points with radius `<=8`:
  Omega RMS discrepancy `1.218e-01`, Omega Linf discrepancy `1.469`,
  psi RMS discrepancy after mean-offset removal `2.346e-01`,
  psi Linf discrepancy after mean-offset removal `7.013e-01`

Remaining uncertainties:

- The square-vs-disk comparison is a coarse reduced validation; boundary-shape
  and discretization differences likely dominate the probe discrepancies.
- The square Neumann boundary closure is first-order and shows a large
  compatibility defect (`8.945e-01`) on the reduced run.
- The centered-advection Newton solve worked on the reduced case; full active
  amplitude and refinement should be tested before relying on high-accuracy
  profile outputs.

Recommended next step:

- Superseded by the later c3-only visual/residual check above: stay in M3
  validation/fix mode before starting M4.

## 2026-04-26 square-domain review and Neumann Poisson scaffold

### Review result

- The square-domain far-field Neumann data in `src/nextgen/boundary.py` is kept
  as the normal projection of the analytic scaled far-field streamfunction
  gradient:
  `Psi_far(z) = (Rdom/R)^(1/mu) |z|^(2-1/mu) Psi_bar(theta)`.
- On a circular boundary of radius `Rdom` with radial normals, this reduces to
  the baseline formula
  `d_n psi = alpha * Rdom * R^(-1/mu) * Psi_bar(theta)`.
- Square-grid corner normals are currently normalized diagonal normals. The new
  Poisson boundary closure matches that convention by using the diagonal inward
  neighbor and distance `sqrt(2) * h` at corners.

### Added

- `src/nextgen/poisson.py`

Implemented:

- full-grid sparse system for `Delta psi = omega`
- interior five-point Laplacian rows
- first-order Neumann boundary rows
- diagonal corner Neumann rows consistent with normalized diagonal corner normals
- configurable gauge row, defaulting to the center grid point
- direct sparse solve through `scipy.sparse.linalg.spsolve`
- assembled-system residual helper

### Tests

Updated `tests/test_nextgen_foundations.py` with:

- explicit comparison of `psi_neumann_circle` against the baseline Fourier
  coefficient formula
- zero Neumann/zero source Poisson solve with gauge
- manufactured linear Neumann solution, including corner boundary rows

Checks run:

```bash
python -m unittest discover -s tests
PYTHONPYCACHEPREFIX=/tmp/pycache-euler python -m py_compile src/nextgen/__init__.py src/nextgen/config.py src/nextgen/boundary.py src/nextgen/grid.py src/nextgen/operators.py src/nextgen/diagnostics.py src/nextgen/poisson.py tests/test_nextgen_foundations.py
```

Result: `10` tests passed.

Remaining uncertainties:

- The Poisson boundary closure is first-order at the boundary. It is suitable
  for the scaffold but should be benchmarked against refinement and, if needed,
  upgraded before high-accuracy profile/eigen comparisons.
- Gauge row replacement drops one discrete equation. This is comparable in
  spirit to baseline gauge pinning, but later diagnostics should track the
  omitted row and compatibility defect.
- The corner convention is now explicit and test-covered, but it is still a
  modeling choice for the square domain.

## 2026-04-25 baseline reference run and nextgen scaffold

### Baseline run

Command:

```bash
env MPLCONFIGDIR=/tmp/mplconfig-euler conda run -n femx_spc python src/baseline/driver.py
```

Historical active input for this 2026-04-25 baseline reference run:

- `mu=0.8`, `R=150.0`, `Rdom=10.0`, `M=48`
- `sigma_grid=np.linspace(0.0, 1.0, 21)`
- `h_min=0.06`, `h_max=0.12`
- active Fourier coefficient at the time: `c_4=50+0j`
- `k=6`, eigensolver shift `sigma=-1.0`

Outputs:

- `experiments/raw/baseline/results.xdmf`
- `experiments/raw/baseline/results.h5`
- `experiments/raw/baseline/linop_blocks.mat`
- debug plots in `src/baseline/`
- summary in `experiments/summaries/baseline_reference_2026-04-25.md`

Numerical results:

- vertices: `25600`
- cells: `50674`
- DoFs: `Omega=25600`, `psi=25600`
- total wall time: `243.419 s`
- raw weak residuals: `||R_Omega||_2=6.070e-04`,
  `||R_Psi||_2=1.797e-13`
- eigenvalues:
  `0.4537468912529909-1.176662542133605j`,
  `0.4537468912529909+1.176662542133605j`,
  `0.4677034814983481+0j`,
  `0.4783460496236707-1.276583293491166j`,
  `0.4791564448291539-1.267528281259206j`,
  `0.4791564448291539+1.267528281259206j`

### Nextgen scaffold

Added:

- `src/nextgen/__init__.py`
- `src/nextgen/config.py`
- `src/nextgen/boundary.py`
- `src/nextgen/grid.py`
- `src/nextgen/operators.py`
- `src/nextgen/diagnostics.py`
- `tests/test_nextgen_foundations.py`

Implemented only foundational pieces:

- current-baseline config dataclasses
- Fourier mirroring and `Omega_bar`, `Psi_bar`, angular derivative evaluation
- circular and arbitrary-normal Neumann data for the scaled far-field
  streamfunction
- square grid coordinates, masks, indexing, interior flattening, and outward
  boundary normals
- sparse interior five-point Laplacian, centered gradients, Dirichlet boundary
  contributions, and centered advection matrix assembly
- basic diagnostics for operator residuals, grid norms, boundary mismatch,
  domain/boundary integrals, and Poisson compatibility

Checks run:

```bash
python -m unittest discover -s tests
PYTHONPYCACHEPREFIX=/tmp/pycache-euler python -m py_compile src/nextgen/__init__.py src/nextgen/config.py src/nextgen/boundary.py src/nextgen/grid.py src/nextgen/operators.py src/nextgen/diagnostics.py tests/test_nextgen_foundations.py
```

Result: `7` tests passed.

Remaining uncertainties:

- Square-domain `psi` Neumann data is implemented as the normal projection of
  the analytic far-field gradient. This should be reviewed before solver work.
- Corner treatment currently uses normalized diagonal normals for diagnostics;
  later finite-difference boundary closures may need a solver-specific corner
  convention.
- The active baseline input has changed again since the earlier intake notes:
  the current driver uses `m=3` coefficient `60` and `m=4` coefficient `0`.
  Future comparisons should use the current driver state unless a specific
  historical case is requested.

## 2026-04-25 intake report

### Sources used

- Followed `docs/AGENTS.md`.
- Used `src/baseline/driver.py` as the authoritative workflow.
- Used the active imported/called baseline files to determine implementation details.
- Used `docs/project_notes.tex` as the primary mathematical reference.
- Did not need `docs/project_notes.pdf`; the LaTeX source was sufficient for this pass.

### Minimal baseline compatibility fix

- Changed `src/baseline/driver.py` to write active outputs under
  `experiments/raw/baseline/` instead of old absolute locations outside the repository.
- Affected outputs: `results.xdmf`, `results.h5`, `mesh.png`, `streamlines.png`,
  `vorticity.png`, `eig_plots/`, `linop_blocks.mat`, and `run_time.txt`.
- No conceptual baseline refactor was made.

### Active baseline computational workflow

1. Set parameters in `driver.py`: `mu=0.8`, `R=150`, `Rdom=10`, `M=48`,
   `sigma_grid=linspace(0,1,21)`, `h_min=0.05`, `h_max=0.10`,
   `tol_newton=1e-9`, `max_newton=48`.
2. Build a gmsh disk mesh on `B_Rdom(0)` and create scalar CG1 spaces for
   `Omega` and `psi`.
3. Build Fourier boundary data from the hard-coded coefficients. In the current
   active input, only mode `m=3` is nonzero with coefficient `60`, mirrored to
   negative modes for real-valued data.
4. Construct the Neumann Poisson solver for `Delta psi = Omega` with far-field
   Neumann data from the Fourier coefficients and one pinned gauge dof.
5. Construct the Newton block assembler for the coupled profile residual.
6. Run sigma-continuation. For each positive sigma, impose the `Omega` boundary
   trace, solve Poisson for the correct `psi` flux, then Newton update the
   coupled `(Omega, psi)` system on the Poisson manifold.
7. Write the converged profile, plots, raw weak residual diagnostics, and timing.
8. Assemble the reduced linearized eigenproblem and compute `k=6` modes with a
   manual shift-invert SLEPc workflow targeting `sigma=-1`.
9. Print eigenvalues, save real-part vorticity/streamfunction mode plots, and
   export sparse blocks to MATLAB format.

### Mathematical objects computed

The profile solve follows the scaled finite-domain system from the notes:

```text
(grad_perp psi - mu*x) . grad Omega = Omega + eps*Delta Omega,
Delta psi = Omega,
eps = Rdom^2 / R^2.
```

Boundary data:

```text
Omega|boundary = sigma * R^(-1/mu) * Omega_bar(theta)
d_n psi|boundary = sigma * alpha * Rdom * R^(-1/mu) * Psi_bar(theta),
alpha = 2 - 1/mu,
Psi_bar_m = Omega_bar_m / (alpha^2 - m^2).
```

The eigen solve uses homogeneous perturbation boundary data:

```text
delta Omega|boundary = 0,
Delta delta psi = delta Omega,
d_n delta psi|boundary = 0,
one gauge dof pinned for delta psi.
```

Implementation note: the active eigen code assembles the residual-Jacobian
convention used by the Newton profile solve. Before labeling eigenvalues as
unstable modes of the LaTeX operator, the sign convention should be checked
against the notes and the saved baseline output.

### Active baseline dependency graph

Actually called by the active `driver.py` workflow:

- `driver.py`
- `config.py`: `SimulationParams`, `eps_from_R`
- `mesh.py`: gmsh disk mesh construction
- `spaces.py`: scalar CG1 function spaces
- `boundary_data_fourier.py`: Fourier boundary values and Neumann modes
- `poisson_neumann.py`: Poisson solve for `psi`
- `newton_block.py`: coupled Newton block assembly
- `continuation.py`: sigma-continuation and Newton iterations
- `io_utils.py`: XDMF/HDF5 profile output
- `viz.py`: mesh/profile plots
- `residuals_quick.py`: raw weak residual diagnostics
- `eigs_linearized_new.py`: reduced eigenproblem, mode reconstruction, plots, MATLAB export

Imported by `driver.py` but not called in the active path:

- `newton_direct.py`

Present but not used by the active main workflow:

- `velocity.py`
- `diagnostics.py`
- `warmstart_picard.py`
- `eigs_linearized.py`
- `run_eigs_from_saved.py`
- `test.py`
- pre-existing output artifacts under `src/baseline/`

Main external runtime stack:

- `numpy`, `gmsh`, `mpi4py`, `dolfinx`, `ufl`, `petsc4py`, `slepc4py`,
  `matplotlib`, `scipy`

### Likely performance bottlenecks

- Repeated sparse assembly through dolfinx/UFL inside Newton iterations.
- Direct LU solves for Poisson and monolithic Newton systems.
- Shift-invert eigensolve: each Krylov application requires applying a MatShell
  with an embedded Poisson solve, plus an inner shifted GMRES solve.
- Large default eigensolver subspace (`ncv` can reach hundreds) relative to
  `k=6`.
- Python-level PETSc vector insertion/scattering in the reduced eigen operator.
- Rank-0 sparse matrix gathering during MATLAB export.

### Likely numerical fragilities

- The active case is advection dominated because `eps = Rdom^2/R^2` is small.
- SUPG stabilization exists in the assembler but is off in the active workflow.
- Newton convergence is based on relative `Omega` step size; the reported weak
  residuals are raw vector norms and are not normalized.
- Boundary Neumann data is applied by nodal sampling with uniform arc weights,
  not facet quadrature; this can be sensitive to boundary mesh nonuniformity and
  higher Fourier modes.
- Fourier boundary construction has a resonance guard when `alpha^2 = m^2`.
- The Neumann gauge is a single pinned boundary dof; this should be preserved
  for comparability but checked in diagnostics.
- The eigen solve depends on sign conventions, shift target, inner KSP accuracy,
  and the assumption that `Vomega` and `Vpsi` share the same CG1 layout.

### Must preserve in the new implementation

- Parameter meanings: `mu`, `R`, `Rdom`, `M`, `eps=Rdom^2/R^2`, sigma grid,
  Fourier coefficient convention, and default `k=6`.
- Far-field boundary scaling for both `Omega` and `psi`.
- The profile system and the perturbation/eigenproblem boundary conditions.
- The Poisson gauge convention or an explicitly equivalent gauge.
- Main outputs: profile fields, eigenvalues, first `k` eigenfunctions, residual
  and boundary diagnostics, timing, and debug plots.
- Ability to run the same Fourier input case as the baseline.
- Enough metadata to compare mesh/grid size, solver tolerances, domain size,
  runtime, residuals, and eigenvalue discrepancies.

### Proposed `src/nextgen/` architecture

Use direct sparse finite differences on a uniform square domain first, with
boundary data evaluated from the same Fourier far-field formulas. If the square
boundary is used, compute the Neumann condition for `psi` as the normal
projection of the analytic exterior far-field gradient, not as the circular
radial derivative.

Proposed files:

- `src/nextgen/__init__.py`
- `src/nextgen/config.py`: dataclasses for profile, grid, solver, eigen, and IO parameters
- `src/nextgen/boundary.py`: Fourier coefficients, `Omega_bar`, `Psi_bar`,
  Dirichlet `Omega`, and normal Neumann data for arbitrary boundary points
- `src/nextgen/grid.py`: square grid, masks, indexing, boundary normals, metadata
- `src/nextgen/operators.py`: sparse Laplacian, gradients, advection, restriction,
  boundary-condition utilities
- `src/nextgen/poisson.py`: sparse Neumann Poisson solve with gauge handling
- `src/nextgen/profile.py`: residual, Jacobian, continuation Newton solver
- `src/nextgen/eigen.py`: linearized sparse operator, shift-invert eigensolve,
  mode reconstruction
- `src/nextgen/diagnostics.py`: PDE residuals, boundary mismatch, compatibility,
  eigenpair residuals, convergence metrics
- `src/nextgen/io.py`: save/load fields, eigenpairs, metadata, and summaries
- `src/nextgen/plots.py`: preliminary profile and eigenmode visualizations
- `src/nextgen/driver.py`: runnable entry point for the default baseline case
- `src/nextgen/benchmark.py`: baseline-vs-nextgen timing and result comparison helpers

### Validation and benchmarking plan

1. Baseline reference run: run the patched baseline with the active input and
   record profile output, residuals, eigenvalues, domain, mesh, tolerances,
   runtime, and memory if available.
2. Same-input nextgen run: use identical `mu`, `R`, `Rdom`, Fourier coefficients,
   sigma grid, and `k`. If the first nextgen domain is square, compare fields on
   the common interior disk or on interpolation probes, and record the domain
   shape difference explicitly.
3. Known/reference checks: include the zero-boundary case and compare the base
   operator spectrum against the analytic values from the notes for `mu=4/5`,
   with sign convention stated.
4. Refinement checks: run multiple grid sizes and track profile differences,
   eigenvalue drift, and normalized residuals.
5. Domain/truncation checks: vary square half-width or disk radius and Fourier
   truncation `M`; track interior profile and eigenvalue sensitivity.
6. Boundary checks: measure `Omega` trace mismatch, `psi` Neumann mismatch, and
   Poisson compatibility `integral Omega` vs boundary flux.
7. Eigen diagnostics: compute normalized eigenpair residuals, mode
   reconstruction errors for `Delta psi_mode = omega_mode`, and sensitivity to
   shift target and solver tolerance.
8. Runtime comparison: split timings into grid/mesh setup, profile solve,
   eigen assembly, eigensolve, and IO/plotting. Compare accuracy for equal
   runtime and runtime for comparable accuracy.

Residuals alone are not sufficient; every validation claim should be paired
with refinement, truncation/domain sensitivity, same-input comparison, or a
known-case check.

## Open questions

- Whether the nextgen square-domain implementation should be the default, or
  whether a disk/masked-grid mode is required for closer baseline comparison.
- Whether eigenvalues should be reported in the baseline residual-Jacobian sign
  convention or converted to the LaTeX linearized-operator convention.
- Exact performance target for "outperform".
- Whether `k=6` should remain the default for all runs.
- Allowed extra dependencies for nextgen sparse solvers and IO.
- Baseline authoritative run command/environment, including FEniCSx/gmsh versions.

## Latest completed step

- Intake of `src/baseline/driver.py` and active baseline dependencies completed.
- Project notes reviewed from LaTeX source.
- Repo-local baseline output paths patched in `src/baseline/driver.py`.
- Syntax compilation of active baseline files passed with cache redirected to `/tmp`.

## Next requested step

- Confirm the baseline runtime environment or install/use an environment with
  `gmsh`, `dolfinx`, `petsc4py`, and `slepc4py`.
- Run the patched baseline once and capture the reference outputs.
- Start a minimal `src/nextgen/` scaffold and implement the boundary/grid/operator
  layer before writing the full nonlinear/eigen solvers.

## Recommended next prompt

Run the patched baseline in the correct FEniCSx/gmsh environment, capture the
reference outputs, then scaffold `src/nextgen/` with config, Fourier boundary
data, square grid indexing, sparse finite-difference operators, and diagnostics
tests only. Do not implement the full Newton/eigen solvers until those
foundational pieces are reviewed.
