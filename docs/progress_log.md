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
- status: `completed` for the optimized full-disk comparison.
- no eigensolver or M4 code was used.

The refinement study used a full polar disk for nextgen and disk meshes for
baseline. Comparisons used the same common sampled strong residual and probe
RMS/Linf definitions described above. Baseline weak residuals are also logged,
but they are not used as the cross-code residual metric.

Active full-amplitude `c_3=60`, `c_4=0`:

| run | runtime | status | common strong rel | Omega probe RMS | psi probe RMS |
| --- | ---: | --- | ---: | ---: | ---: |
| baseline h `0.20/0.30` | `17.488 s` | completed but unstable | `4.111e+03` | mesh-diff n/a | mesh-diff n/a |
| baseline h `0.12/0.18` | `18.945 s` | completed | `8.353e-01` | mesh-diff `2.671e-01` | mesh-diff `2.726e-02` |
| baseline h `0.08/0.12` | `48.056 s` | completed | `6.369e-01` | reference | reference |
| nextgen `nr=54`, `ntheta=162` | `17.585 s` | completed | `6.496e-01` | `2.304e-01` | `2.636e-02` |

Additional active-grid robustness probe:

- nextgen `nr=48`, `ntheta=144` with adaptive sigma retries improved from the
  previous fixed-continuation stall at sigma `0.7` to final accepted sigma
  `0.98125`, with native relative residual `1.821e-14` at that accepted sigma.
  It still failed before sigma `1` and took `227.059 s`, so it is retained only
  as robustness/stability evidence, not as the performance path.

Mixed non-symmetric reduced case
`c_2=3+i`, `c_5=-2+0.5i`:

| run | runtime | status | common strong rel | Omega probe RMS | psi probe RMS |
| --- | ---: | --- | ---: | ---: | ---: |
| baseline h `0.30/0.45` | `2.617 s` | completed | `7.631e+00` | mesh-diff `1.117e-01` | mesh-diff `8.351e-02` |
| baseline h `0.20/0.30` | `3.875 s` | completed | `2.093e+00` | reference | reference |
| nextgen `nr=30`, `ntheta=90` | `5.064 s` | completed | `7.592e-01` | `2.622e-01` | `1.670e-02` |
| nextgen `nr=32`, `ntheta=96` | `4.601 s` | completed | `6.913e-01` | `1.990e-01` | `1.456e-02` |
| nextgen `nr=36`, `ntheta=108` | `5.536 s` | completed | `6.535e-01` | `1.424e-01` | `1.252e-02` |

Interpretation:

- The mixed non-symmetric reduced case gives the cleanest refinement evidence:
  nextgen common strong residual, Omega probe RMS, and offset-removed psi probe
  RMS all improve as the full-disk polar grid is refined.
- The active full-amplitude c3 case still has a practical resolution threshold,
  but adaptive continuation now gets the coarser `48/144` grid much closer to
  sigma `1`. The accepted production path remains `54/162`.
- Baseline mesh refinement matters strongly for the active c3 case: the coarse
  baseline mesh produces a very large common strong residual and field blow-up,
  and the finer `0.08/0.12` baseline improves the common strong residual from
  `0.835` to `0.637`.
- The optimized nextgen `54/162` run is now a runtime improvement over the
  finer active baseline (`17.585 s` versus `48.056 s`) at nearly the same
  common strong residual (`0.650` versus `0.637`). It is also faster than the
  `0.12/0.18` baseline (`18.945 s`) while giving a lower common strong
  residual.

### General boundary visual validation

Command:

```bash
conda run -n femx_spc python experiments/run_m3_general_boundary_visual_suite.py
```

Summary:

- `experiments/summaries/m3_general_boundary_visual_suite_2026-04-28.json`
- plots under `experiments/raw/nextgen/m3_general_boundary_visual_suite/`
- status: `completed` for four non-c3/general completed cases.
- no eigensolver or M4 code was used.

Validation design:

- nextgen medium grid: `nr=40`, `ntheta=120`
- nextgen fine grid: `nr=54`, `ntheta=162`
- baseline mesh: `h_min=0.12`, `h_max=0.18`
- metrics: common sampled strong residual, probe RMS/Linf versus baseline,
  fine-minus-medium nextgen self-convergence on common probes, and visual
  vorticity/streamfunction/difference plots.

Completed cases:

| case | nextgen fine runtime | baseline runtime | nextgen fine common rel | baseline common rel | Omega probe RMS | nextgen self Omega RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `single_c4_20` | `10.842 s` | `13.388 s` | `5.167e-01` | `8.858e-01` | `1.674e-01` | `5.016e-02` |
| `phase_shifted_c2_20` | `14.748 s` | `16.392 s` | `4.819e-01` | `1.465e+00` | `2.030e-01` | `1.320e-01` |
| `mixed_1_4_7_reduced` | `20.040 s` | `15.975 s` | `8.710e-01` | `7.314e-01` | `1.088e-01` | `6.419e-02` |
| `mixed_2_5_8_reduced` | `11.865 s` | `12.763 s` | `5.368e-01` | `6.731e-01` | `5.458e-02` | `9.424e-02` |

Plot files per case:

- `nextgen_omega.png`
- `nextgen_psi_contours.png`
- `baseline_omega.png`
- `baseline_psi_contours.png`
- `omega_nextgen_minus_baseline.png`
- `psi_nextgen_minus_baseline_offset_removed.png`

Stress cases that did not validate:

- Moderate `mixed_1_4_7` data
  `{c_1=4-2i, c_4=8, c_7=-3+2i}`: baseline completed, but nextgen medium
  stalled at accepted sigma `0.2125` and nextgen fine stalled at accepted
  sigma `0.5703125`.
- Moderate `mixed_2_5_8` data
  `{c_2=6+3i, c_5=-5+i, c_8=2-1.5i}`: baseline completed, nextgen medium
  completed, but nextgen fine stalled at accepted sigma `0.65625`.
- A tighter continuation probe for moderate `mixed_2_5_8` with sigma spacing
  `0.05`, `min_sigma_step=5e-4`, `max_iter=60`, and `max_backtracks=28`
  still stalled at accepted sigma `0.657665` after `247.168 s`.

Interpretation:

- The optimized full-disk polar solver now handles several non-c3 and
  nonsymmetric boundary inputs on fine grids and produces useful debug plots.
- It is not yet a general solver for arbitrary moderate-amplitude mixed
  boundary data. The failure mode is continuation/Newton robustness, not the
  final algebraic residual at the last accepted sigma.
- M3 should not be marked complete under the stricter goal "any given boundary
  values with high accuracy." The next M3 work should focus specifically on
  robust nonlinear continuation for mixed-mode data.

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
## 2026-04-28 — M3 mixed-mode continuation robustness pass

Scope stayed in M3; no eigensolver/M4 work was started.

Changes made:
- Added a baseline-compatible pinned Poisson gauge option for diagnostics, while retaining the
  mean-gauge bordered Neumann solve as the main physical compatibility path.
- Added secant continuation prediction, optional residual-only acceptance, anchored/diffusion
  fallback hooks, and a sparse LSMR least-squares Newton fallback.
- Added affine Fourier boundary data plus coefficient-homotopy support so future experiments can
  continue through coefficient space instead of scaling all modes together.
- Added selectable polar advection schemes: `upwind`, `centered`, and a small blended mode.

Targeted robustness results on `{c_2=6+3i,c_5=-5+i,c_8=2-1.5i}`, `Rdom=10`, `nr=54`,
`ntheta=162`:
- Upwind with tighter adaptive continuation still stalled at accepted sigma `0.657702`
  after `125.996 s`; reducing `min_sigma_step` to `1e-5` did not cross the threshold.
- Pinned-gauge Poisson stalled earlier and produced a large physical Poisson defect, so it is
  diagnostic only, not the M3 default.
- Pure centered advection reached sigma `1` in `11.298 s` with native relative residual
  `4.76e-14`, but the vorticity heat map showed radial ringing and the common sampled strong
  residual was large.
- A `1%` centered/upwind blend reached sigma `1` in `8.988 s` with native relative residual
  `3.14e-14` and physical Poisson residual `3.91e-7`, but still showed visible radial ringing
  and common sampled strong residual near `20.36`.
- Heavier `5%` blending stalled at sigma `0.234`, so the blend is not a robust accuracy fix.

Validation run:
- `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests` passed: `21` tests.
- `conda run -n femx_spc python experiments/run_m3_general_boundary_visual_suite.py` completed
  with baseline comparisons and plots for five cases:
  `c_4=20`, `c_2=15+8i`, small mixed `c_1/c_4/c_7`, reduced mixed `c_2/c_5/c_8`, and the
  target moderate mixed `c_2/c_5/c_8`.
- Plots are under `experiments/raw/nextgen/m3_general_boundary_visual_suite/`; summary is
  `experiments/summaries/m3_general_boundary_visual_suite_2026-04-28.json`.

Conclusion:
M3 should remain open. The new continuation options are useful diagnostics and centered/blended
schemes can reach sigma `1` for the target mixed case, but the resulting fields are not accurate
enough because of radial ringing and poor common strong residuals. The upwind discretization is
visually smoother and historically much better under the common residual, but still has a genuine
fine-grid continuation barrier for the target mixed case.

## 2026-04-28 — M3 target mixed case resolved by finer upwind polar grid

Scope stayed in M3; no eigensolver/M4 work was started.

Numerical conclusion:
- The target mixed case `{c_2=6+3i,c_5=-5+i,c_8=2-1.5i}` was not fixed by
  centered/blended advection, smooth-upwind stencil transitions, filtered centered predictors, or
  square FD fallback. Those paths either stalled or converged to visibly oscillatory low-amplitude
  branches.
- The smooth branch is recovered by resolving the core more finely with the original upwind polar
  discretization. On `Rdom=10`, `nr=72`, `ntheta=216`, the upwind polar solve reaches sigma `1`.

Validation:
- `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests` passed: `21` tests.
- `conda run -n femx_spc python experiments/run_m3_target_mixed_method_probe.py` completed.
- Target mixed results:
  - polar upwind `72/216`: runtime `13.227 s`, native relative residual `1.25e-10`,
    common sampled strong residual `0.421`, Omega probe RMS vs baseline `0.1065`,
    offset-removed psi probe RMS `0.0101`.
  - baseline `h_min=0.12`, `h_max=0.18`: runtime `14.964 s`, common sampled strong residual
    `0.964`.
  - polar blended `54/162`: common sampled strong residual `20.36`, rejected as oscillatory.
  - square FD `n=97`: common sampled strong residual `8.03`, rejected as oscillatory/domain
    inappropriate for this validation.
- Plots are in `experiments/raw/nextgen/m3_target_mixed_method_probe/`.
- Summary is `experiments/summaries/m3_target_mixed_method_probe_2026-04-28.json`.

Updated interpretation:
- The previous `54/162` upwind stall was primarily an underresolution/continuation-threshold
  issue for the localized mixed-mode core, not a need to switch away from upwind polar FD.
- For moderate/large mixed Fourier data, the M3 validation path should use a finer polar grid,
  starting at least around `nr=72`, `ntheta=216`, and judge quality by plots plus common sampled
  residual/probe errors, not by native algebraic residual alone.

## 2026-04-28 — M3 fine-grid upwind acceptance suite

Scope stayed in M3; no eigensolver/M4 work was started.

Validation run:
- `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests` passed: `21` tests.
- `conda run -n femx_spc python experiments/run_m3_fine_upwind_acceptance_suite.py` completed
  all requested fine-grid cases with nextgen `nr=72`, `ntheta=216`, `Rdom=10`.
- The common sampled strong residual is measured after sampling both codes on the same
  Cartesian grid over `[-7,7]^2`: `sqrt(||R_Omega||_L2^2 + ||R_Psi||_L2^2) /
  max(||Omega||_L2, ||Delta psi||_L2)`. Probe RMS errors are over common polar probe points
  `r in [0.5,8]`; psi errors subtract the mean probe offset to remove gauge differences.

Results:

| case | nextgen runtime | baseline runtime | nextgen native rel | nextgen common strong rel | baseline common strong rel | Omega probe RMS | psi probe RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `c3_60_c4_0` | `33.274 s` | `53.414 s` | `2.18e-11` | `5.441e-01` | `5.371e-01` | `1.926e-01` | `1.951e-02` |
| `target_mixed_2_5_8` | `13.491 s` | `15.032 s` | `1.25e-10` | `4.207e-01` | `9.644e-01` | `1.065e-01` | `1.007e-02` |
| `larger_mixed_2_5_8_1p5x` | `16.855 s` | `15.756 s` | `5.06e-10` | `4.620e-01` | `1.104e+00` | `2.039e-01` | `2.108e-02` |

Visual check:
- Omega heat maps and psi stream-contour plots are under
  `experiments/raw/nextgen/m3_fine_upwind_acceptance_suite/`.
- The active c3 case preserves the expected three-arm structure.
- The two mixed cases are non-symmetric, smooth, and free of the radial ringing seen in the
  centered/blended `54/162` diagnostics.

Interpretation:
- The fine-grid full-disk upwind polar solver passes the current M3 acceptance suite. It is faster
  than the finer baseline on active c3, slightly faster on the target mixed case, and comparable
  on the `1.5x` mixed case while giving lower common sampled strong residuals on the mixed cases.
- This validates the M3 profile path for the tested envelope, but not yet for arbitrary
  high-amplitude mixed Fourier data at `60`- to `100+`-scale coefficients.

## 2026-04-28 — M3 high-amplitude mixed stress suite

Scope stayed in M3; no eigensolver/M4 work was started.

Stress input:
- `{c_2=60,c_5=-5+i,c_8=2-1.5i}`, `Rdom=10`, `mu=0.8`, `R=150`.
- Requested nextgen refinement path: `nr=72,ntheta=216`, then `nr=96,ntheta=288` if feasible.
- Baseline comparison meshes: `h_min=0.12,h_max=0.18` and `h_min=0.08,h_max=0.12`.

Results:

| run | status | runtime | final sigma | common sampled strong residual |
| --- | --- | ---: | ---: | ---: |
| nextgen `72/216` | nonconverged | `248.238 s` | `0.45` | `4.681e-01` for accepted partial state only |
| nextgen `96/288` | skipped | `0.000 s` | n/a | n/a |
| baseline `0.12/0.18` | bad branch | `186.127 s` | `1.0` | `5.939e+03` |
| baseline `0.08/0.12` | completed | `105.068 s` | `1.0` | `6.509e-01` |

Notes:
- The nextgen partial common residual is reported only for the accepted sigma `0.45` state; it is
  not a valid sigma-1 profile comparison and no probe errors are reported.
- The coarse baseline run printed large Newton relative residuals near sigma `0.75` and `1.0`,
  and its Omega plot is visibly noisy. The fine baseline is the usable reference for this stress
  case.
- Plots are under `experiments/raw/nextgen/m3_high_amplitude_stress_suite/`.
- Summary is `experiments/summaries/m3_high_amplitude_stress_suite_2026-04-28.json`.

Conclusion:
- The moderate-amplitude M3 acceptance suite remains valid, but the current full-disk polar
  upwind Newton solver is not yet an acceptable profile generator for the larger mixed boundary
  values needed before eigenvalue searches.
- M3 should stay open for one more numerical-method pass focused on high-amplitude continuation
  and discretization robustness, rather than moving to M4 immediately.

## 2026-04-28 — M3 pseudo-transient robustness pass

Scope stayed in M3; no eigensolver/M4 work was started.

Change made:
- Added a pseudo-transient shifted-Newton fallback to `src/nextgen/polar_profile.py`.
- When the exact Newton correction fails the residual line search, the solver now tries shifted
  Omega-block systems `(J + beta M_Omega) delta = -F` with configurable shifts before declaring
  the sigma step failed.
- Updated `experiments/run_m3_high_amplitude_stress_suite.py` to run the high-amplitude stress
  case with this fallback enabled explicitly.

Validation:
- `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests` passed: `21` tests.
- `conda run --no-capture-output -n femx_spc python experiments/run_m3_high_amplitude_stress_suite.py`
  reran the stress suite against the fine baseline.

Results:

| run | status | runtime | final sigma | common sampled strong residual |
| --- | --- | ---: | ---: | ---: |
| nextgen pseudo-transient `72/216` | nonconverged | `472.999 s` | `0.45` | `4.681e-01` for accepted partial state only |
| nextgen pseudo-transient `96/288` | skipped | `0.000 s` | n/a | n/a |
| baseline `0.08/0.12` | completed | `106.815 s` | `1.0` | `6.509e-01` |

Interpretation:
- The pseudo-transient fallback did not solve the high-amplitude mixed case. It kept the same
  sigma threshold as the previous direct upwind run and increased runtime substantially.
- Diagnostic probes also confirmed that a centered `72/216` solve can algebraically reach
  sigma `1`, but it lands on the rejected low-amplitude/ringing branch with common sampled
  strong residual near `19.5`, so centered convergence is not an acceptable fix.
- The high-amplitude blocker is therefore likely in the discrete nonlinear continuation path or
  transport discretization, not just in Newton damping.

## 2026-04-28 — M3 conservative FV transport pass

Scope stayed in M3; no eigensolver/M4 work was started.

Change made:
- Added `advection_scheme="fv_upwind"` to `src/nextgen/polar_profile.py`.
- The new transport residual discretizes the same continuous term
  `v . grad(Omega)` as the conservative identity
  `div(v Omega) - Omega div(v)`.
- Radial and angular face fluxes use monotone upwinding; Dirichlet Omega data is used only on
  incoming outer radial faces.
- Added a small mixed-boundary FV unit test in `tests/test_nextgen_profile.py`.

Validation:
- `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests` passed: `22` tests.
- `conda run --no-capture-output -n femx_spc python experiments/run_m3_high_amplitude_stress_suite.py`
  reran the high-amplitude stress suite using `fv_upwind` against the fine baseline.

Results:

| run | status | runtime | final sigma | common sampled strong residual |
| --- | --- | ---: | ---: | ---: |
| nextgen FV `30/90` | completed | `68.953 s` | `1.0` | `7.376e-01` |
| nextgen FV `36/108` | nonconverged | `354.832 s` | `0.933203125` | `6.677e-01` for accepted partial state only |
| nextgen FV `72/216` | skipped | `0.000 s` | n/a | n/a |
| baseline `0.08/0.12` | completed | `108.811 s` | `1.0` | `6.509e-01` |

Additional comparison for the completed `30/90` FV run:
- Omega probe RMS versus fine baseline: `0.602`.
- Offset-removed psi probe RMS versus fine baseline: `0.0397`.
- Plots are under `experiments/raw/nextgen/m3_high_amplitude_stress_suite/`.

Interpretation:
- Conservative FV transport is a substantial improvement over the old upwind path: it reaches
  sigma `1` on `30/90` and pushes the refined `36/108` case to sigma `0.933`, compared with the
  old `72/216` upwind barrier at sigma `0.45`.
- It is not yet sufficient for M3 high-amplitude acceptance because the refined grid still stalls
  before sigma `1`, and `72/216` was skipped after that refinement failure.
- The most likely next fix is improving the FV Jacobian/globalization. The current FV block uses
  a conservative Omega transport Jacobian, but the psi-coupling block is still approximate.

## 2026-04-29 — M3 FV psi-Jacobian high-amplitude fix

Scope stayed in M3; no M4 operator assembly or eigensolver work was started.

Change made:
- Replaced the `fv_upwind` omega-psi Jacobian block in `src/nextgen/polar_profile.py`.
- The old `fv_upwind` Jacobian reused the centered/blended strong-form psi derivative even though
  the residual is a conservative face-flux discretization.
- The new block differentiates the active upwind finite-volume radial and angular face fluxes
  with respect to `psi`, including the `div(v Omega) - Omega div(v)` correction and incoming
  outer Dirichlet Omega faces.
- Added a finite-difference test for the `fv_upwind` Jacobian path in `tests/test_nextgen_profile.py`.

Validation:
- `python -m pytest tests/test_nextgen_profile.py -q` passed: `11` tests.
- Standalone finite-difference check on the `fv_upwind` omega block improved from about
  `4.48e-02` relative error before the fix to about `1.18e-08` after the fix.
- Reran the high-amplitude target `{c_2=60,c_5=-5+i,c_8=2-1.5i}` with the existing
  stress-suite controls.

Results:

| run | status | runtime | final sigma | native relative residual | common sampled strong residual |
| --- | --- | ---: | ---: | ---: | ---: |
| nextgen FV `36/108` | completed | `27.169 s` | `1.0` | `2.114e-12` | n/a for direct gate run |
| nextgen FV `72/216` | completed | `203.917 s` | `1.0` | `8.428e-10` | `4.818e-01` |
| recorded fine baseline `h_min=0.08,h_max=0.12` | completed | `108.811 s` | `1.0` | n/a | `6.509e-01` |

Notes:
- The previous `36/108` FV run stalled at accepted sigma `0.933203125`; after the psi-Jacobian
  fix it reached sigma `1` with no adaptive retries.
- The requested `72/216` run also reached sigma `1` with no adaptive retries. Its common sampled
  strong residual is lower than the recorded fine baseline reference for the same stress case.
- A live baseline rerun in the current shell was blocked by `ModuleNotFoundError: No module named
  'gmsh'`; therefore the baseline comparison above uses the already recorded fine baseline from
  `experiments/summaries/m3_high_amplitude_stress_suite_2026-04-28.json`.
- Probe errors versus the fine baseline were not recomputed in this environment because the live
  baseline field could not be regenerated without `gmsh`.

## 2026-04-29 — M3 high-amplitude FV acceptance closure

Scope stayed in M3; no M4 operator assembly or eigensolver work was started.

Assessment before closure:
- M0-M2 are complete.
- M3 had one remaining blocker: the corrected high-amplitude `fv_upwind` path still needed a
  canonical stress-suite rerun with a live fine baseline, common sampled residuals, probe errors,
  plots, and registry rows.
- The shell-default Python lacks `gmsh`, but the intended `femx_spc` environment provides it
  (`gmsh 4.13.1`), so the live baseline comparison is feasible there.

Validation:
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_nextgen_profile.py -q` passed:
  `11` tests.
- `conda run --no-capture-output -n femx_spc python experiments/run_m3_high_amplitude_stress_suite.py`
  completed the full high-amplitude stress suite with the corrected `fv_upwind` psi Jacobian.
- A non-fatal OpenMPI TCP bind warning appeared during the serial baseline run, but gmsh meshing,
  continuation, residual sampling, probe comparison, plot generation, and JSON/registry logging
  all completed.

High-amplitude stress results for `{c_2=60,c_5=-5+i,c_8=2-1.5i}`:

| run | status | runtime | final sigma | native relative residual | common sampled strong residual | Omega probe RMS | psi probe RMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nextgen FV `30/90` | completed | `20.297 s` | `1.0` | `1.22e-10` | `7.376e-01` | `6.024e-01` | `3.966e-02` |
| nextgen FV `36/108` | completed | `29.772 s` | `1.0` | `2.11e-12` | `6.789e-01` | `5.949e-01` | `3.469e-02` |
| nextgen FV `72/216` | completed | `136.261 s` | `1.0` | `8.43e-10` | `4.818e-01` | `5.585e-01` | `2.016e-02` |
| baseline `h_min=0.08,h_max=0.12` | completed | `107.966 s` | `1.0` | n/a | `6.509e-01` | reference | reference |

Artifacts:
- Full summary: `experiments/summaries/m3_high_amplitude_stress_suite_2026-04-29.json`.
- Plots: `experiments/raw/nextgen/m3_high_amplitude_stress_suite/large_leading_mixed_2_5_8/`.
- Registry rows appended for all three nextgen grids.

Conclusion:
- The prior high-amplitude blocker is closed for the documented M3 profile envelope. The accepted
  production profile path is full-disk polar `fv_upwind` with the exact active-upwind psi
  Jacobian for high-amplitude mixed cases, and full-disk polar upwind remains validated for the
  earlier moderate acceptance suite.
- M3 is complete enough to move to M4. Remaining caveats are not M3 blockers: arbitrary
  `100+`-scale Fourier combinations, memory/runtime of direct sparse factorization on finer
  grids, and the fact that M4 must reuse/document the same discrete profile conventions rather
  than infer a different linearization.

## 2026-04-29 — M3 c4/c7 high-amplitude spot check

Scope stayed in M3; no M4 operator assembly or eigensolver work was started.

Request:
- Solve the profile with boundary data `{c_4=100,c_7=40}` and all other
  Fourier coefficients zero.
- Use fine enough resolution for the large boundary values, report runtime,
  plots, residuals, and compare against the baseline.

Resolution choice:
- Baseline meshes: `h_min=0.08,h_max=0.12` and refined
  `h_min=0.05,h_max=0.08`.
- Nextgen completed grids: `64/256` and `72/288` with `fv_upwind`.
- Larger nextgen runtime probes `80/320` and `96/384` were stopped after
  extended runtime without completion output. This confirms that direct sparse
  factorization remains the limiting cost for pushing this stress case finer.

Results:

| run | status | runtime | final sigma | native residual | common sampled strong residual | Omega RMS vs fine baseline | psi RMS vs fine baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nextgen FV `64/256` | completed | `226.561 s` | `1.0` | `1.785e-10` | `8.199e-01` | `1.228` | `0.128` |
| nextgen FV `72/288` | completed | `251.747 s` | `1.0` | `8.221e-09` | `8.811e-01` | `1.273` | `0.141` |
| baseline `h_min=0.08,h_max=0.12` | completed | `178.638 s` | `1.0` | n/a | `7.580e-01` | compared to fine: `0.218` | compared to fine: `0.0175` |
| baseline `h_min=0.05,h_max=0.08` | completed | `503.579 s` | `1.0` | n/a | `4.564e-01` | reference | reference |

Artifacts:
- Summary: `experiments/summaries/m3_c4_c7_high_amplitude_validation_2026-04-29.json`.
- Plots: `experiments/raw/nextgen/m3_c4_c7_high_amplitude_validation/c4_100_c7_40/`.
- Registry rows appended for both completed nextgen grids.

Assessment:
- The corrected `fv_upwind` path can continue this large c4/c7 profile to
  sigma `1` with tight native nonlinear residuals.
- Accuracy against the fine baseline is not at the level of the accepted M3
  high-amplitude envelope: the nextgen/fine-baseline Omega probe RMS is about
  `1.27`, much larger than the baseline mesh-refinement Omega RMS `0.218`.
- This test should therefore be treated as a cautionary stress check, not as
  evidence that M3 supports arbitrary `100`-scale Fourier combinations. If
  `{c_4=100,c_7=40}` is required as an M4 input, another M3 pass should target
  finer-grid performance or a closer baseline-matching discretization before
  moving on.

## 2026-04-29 — M3 c4/c7 reduced-Newton and FV-blend probe

Scope stayed in M3; no M4 operator assembly or eigensolver work was started.

Changes made:
- Added conservative FV variants `fv_centered` and `fv_blended` to support
  baseline-closer fluxes without deleting the robust `fv_upwind` path.
- Added finite-difference Jacobian coverage for `fv_centered` and
  `fv_blended`.
- Added `reduced_poisson_newton=True`, which eliminates the linear Poisson
  constraint from the Newton solve and applies the reduced Omega Schur
  complement by GMRES, preconditioned by an LU factorization of the Omega
  block.
- Fixed a face-orientation bug introduced during the FV blend refactor:
  inward radial and left angular faces must use the neighboring cell as the
  upwind value for positive face velocity. The bug only affected the new
  generalized helper; the finite-difference tests now cover the corrected path.

Validation and probes:
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_nextgen_profile.py -q`
  passed: `13` tests.
- Pure centered and very lightly blended FV were not efficient on the c4/c7
  stress case; they were stopped during screening before completion.
- A buggy low-branch reduced run briefly made `96/384` and `128/512` appear
  practical, but those results were invalidated by the face-orientation fix and
  removed from the registry as acceptance evidence.
- Corrected reduced Newton on the high-amplitude branch at `72/288` completed
  with the same field as monolithic Newton: native residual `8.22e-09`, common
  sampled strong residual `0.881`, Omega L2 `17.754`, and final sigma `1`.
  Runtime was `201.333 s`, with `83.082 s` in reduced linear solves. A
  corrected monolithic check on the same grid took `395.715 s`, with
  `279.212 s` in linear solves.
- Corrected reduced `96/384` high-branch probes, including a looser GMRES
  tolerance `1e-6`, were stopped after extended runtime without completion.

Assessment:
- The reduced Poisson Newton path is a real performance improvement at the
  validated `72/288` c4/c7 grid, but the current Omega-block Schur
  preconditioner is not strong enough to make the high-branch `96/384` solve
  operational.
- The c4/c7 discrepancy against the fine baseline is therefore reduced in
  linear-solve cost, not yet in field accuracy. The fine-baseline comparison
  remains the earlier high-branch result: nextgen `72/288` common sampled
  residual `0.881` versus baseline `0.456`, Omega probe RMS `1.273`, and
  offset-removed psi RMS `0.141`.
- M3 should remain open for general large boundary data. The next useful work
  is a stronger reduced-system preconditioner or a discretization that matches
  the baseline branch more closely without losing high-amplitude robustness.

## 2026-04-29 — M3 nested reduced c4/c7 fine-grid validation

Scope stayed in M3; no M4 operator assembly or eigensolver work was started.

Change made:
- Updated the c4/c7 reduced fine validation to use nested high-branch
  refinement:
  `72/288` monolithic `fv_upwind` continuation to sigma `1`, then
  prolongation to `96/384` and `128/512` followed by
  `reduced_poisson_newton=True` corrections at sigma `1`.
- This changes the operational path from trying to continue the reduced
  high-branch solve from zero on the fine grid to using the coarser converged
  high-branch profile as the fine-grid initial state.

Validation:
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_nextgen_profile.py -q`
  passed: `13` tests.
- `conda run --no-capture-output -n femx_spc python experiments/run_m3_c4_c7_reduced_fine_validation.py`
  completed nextgen `72/288`, nested reduced `96/384`, nested reduced
  `128/512`, and a fresh fine baseline `h_min=0.05,h_max=0.08`.
- A non-fatal OpenMPI TCP bind warning appeared during the serial baseline
  run, but gmsh meshing, Newton continuation, residual sampling, probe
  comparison, plot generation, JSON, and registry logging completed.

Results for `{c_4=100,c_7=40}`:

| run | status | runtime | final sigma | native residual | common sampled strong residual | Omega probe RMS | psi probe RMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nextgen seed `72/288` | completed | `285.952 s` | `1.0` | `8.221e-09` | `0.881` | `1.273` | `0.141` |
| nested reduced `96/384` | completed | `32.771 s` | `1.0` | `8.130e-10` | `0.935` | `1.585` | `0.281` |
| nested reduced `128/512` | completed | `92.078 s` | `1.0` | `1.436e-11` | `0.616` | `1.612` | `0.291` |
| baseline `h_min=0.05,h_max=0.08` | completed | `505.636 s` | `1.0` | n/a | `0.456` | reference | reference |

Artifacts:
- Summary: `experiments/summaries/m3_c4_c7_reduced_fine_validation_2026-04-29.json`.
- Plots: `experiments/raw/nextgen/m3_c4_c7_reduced_fine_validation/c4_100_c7_40/`.
- Registry rows appended for `72/288`, `96/384`, and `128/512`.

Assessment:
- The corrected nested reduced path now makes fine high-branch c4/c7 grids
  operational: `96/384` and `128/512` reach sigma `1` quickly after the
  `72/288` seed.
- The common sampled strong residual improves substantially at `128/512`
  (`0.616` versus `0.881` at `72/288`) and is closer to the fine baseline
  (`0.456`), but the probe field discrepancy remains large. In particular,
  Omega probe RMS versus the fine baseline is `1.612`, not improved over the
  `72/288` value `1.273`.
- Seeded less-diffusive `fv_blended` corrections at `96/384` were also tested
  (`upwind_blend=0.5`, `0.25`, and `0.10`) and gave worse common residuals
  (`1.08`, `1.15`, and `1.20`), so simply reducing upwind dissipation is not
  the next accuracy fix.
- M3 large-boundary performance has improved, but general large-boundary
  accuracy is not closed. The next issue is the remaining branch/discretization
  mismatch versus the baseline, not just the ability to reach fine polar grids.

## 2026-04-29 — M3 c4/c7 branch/discretization diagnostic

Scope stayed in M3; no M4 operator assembly or eigensolver work was started.

Diagnostic added:
- Added `experiments/run_m3_c4_c7_structure_diagnostics.py` to compare the
  `{c_4=100,c_7=40}` nextgen profile structure against the fresh
  `h_min=0.05,h_max=0.08` baseline.
- The diagnostic checks Fourier boundary conventions, baseline boundary dofs,
  Poisson-only behavior, nextgen circulation/compatibility, radial field
  errors, Fourier-mode amplitude plots, and a `128/512` reduced Newton solve
  seeded directly from the fine baseline field.

Validation:
- `python -m py_compile experiments/run_m3_c4_c7_structure_diagnostics.py`
  passed.
- `python -m pytest tests/test_nextgen_profile.py -q` passed: `13` tests.
- `conda run --no-capture-output -n femx_spc python experiments/run_m3_c4_c7_structure_diagnostics.py`
  completed the nested nextgen `72/288 -> 96/384 -> 128/512` path, a fresh
  fine baseline, and the baseline-seeded `128/512` nextgen correction. The
  usual non-fatal OpenMPI TCP bind warning appeared during the serial baseline
  run, but meshing, continuation, sampling, plotting, JSON, and registry
  logging completed.

Results:

| run | status | runtime | common sampled strong residual | Omega RMS vs baseline | psi RMS vs baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| nested nextgen `72/288` | completed | `286.919 s` | `0.881` | `1.169` | `0.141` |
| nested reduced nextgen `96/384` | completed | `32.125 s` | `0.935` | `1.454` | `0.279` |
| nested reduced nextgen `128/512` | completed | `92.461 s` | `0.616` | `1.476` | `0.289` |
| baseline-seeded reduced nextgen `128/512` | completed | `56.291 s` | `0.433` | `0.651` | `0.022` |
| baseline `h_min=0.05,h_max=0.08` | completed | `505.265 s` | `0.456` | reference | reference |

Convention checks:
- Analytic nextgen and baseline Fourier boundary conventions match to
  roundoff: Omega Dirichlet max error `2.22e-16`, psi Neumann max error
  `5.55e-17`, and trapezoidal Neumann flux `4.36e-16`.
- The baseline applied Omega boundary dofs match the Fourier data to
  roundoff: max error `5.55e-16` across `786` boundary dofs.
- Solving nextgen Poisson on the `128/512` grid using the fine-baseline Omega
  gives offset-removed psi RMS `2.21e-03` and Linf `1.67e-02` against the
  baseline psi. The Poisson convention is therefore not the dominant source of
  the field mismatch.

Structure assessment:
- The nested high-branch `128/512` solution has an Omega integral defect
  `2.477` and Poisson multiplier `387.6`; the baseline-seeded nextgen branch
  reduces these to `-0.154` and `-24.0`.
- The discrepancy is concentrated in the inner core. For nested `128/512`,
  Omega radial RMS peaks at `r=0.5` with `3.84`, while it is only `0.00198` at
  `r=8` and `0.00146` at `r=9.5`.
- The baseline-seeded `128/512` correction stays near the baseline rather than
  returning to the nested high branch: its common residual is `0.433` versus
  the baseline `0.456`, and its offset-removed psi RMS is `0.022`. This is
  direct evidence that the remaining issue is branch selection/continuation,
  not a Fourier boundary scaling error or a Poisson sign/gauge convention.

Artifacts:
- Summary:
  `experiments/summaries/m3_c4_c7_structure_diagnostics_2026-04-29.json`.
- Plots:
  `experiments/raw/nextgen/m3_c4_c7_structure_diagnostics/c4_100_c7_40/`.
- Registry row appended:
  `m3-c4-c7-structure-diagnostics`.

Assessment:
- M3 can now solve the large c4/c7 profile on `128/512` in more than one
  branch. The branch reached by zero-amplitude continuation and nested
  refinement is not the fine-baseline branch, even though it has a small native
  residual.
- The next M3 work should implement a baseline-free branch-selection strategy:
  for example an anchored/arclength continuation path, a circulation/Poisson
  compatibility constraint, or a coarse Galerkin/less-dissipative predictor
  that lands on the baseline-close branch before the reduced FV correction.

## 2026-04-29 — M3 c3=140 symmetry-breaking probe

Scope stayed in M3; no M4 operator assembly or eigensolver work was started.

Experiment added:
- Added `experiments/run_m3_c3_140_symmetry_probe.py` for the large single-mode
  boundary data `{c_3=140}`.
- The script first constructs the threefold-symmetric branch by full-disk
  `fv_upwind` continuation on `72/216`, then nested reduced corrections on
  `96/288` and `128/384`.
- At fixed sigma `1`, it seeds reduced Newton from non-threefold interior
  perturbations of the `128/384` symmetric state and classifies each converged
  root by weighted threefold defect, distance to the symmetric state,
  independent sampled strong residual, and Poisson compatibility.

Validation:
- `PYTHONDONTWRITEBYTECODE=1 python -m py_compile experiments/run_m3_c3_140_symmetry_probe.py`
  passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_nextgen_profile.py -q`
  passed: `13` tests.
- `PYTHONDONTWRITEBYTECODE=1 python experiments/run_m3_c3_140_symmetry_probe.py`
  completed the nested symmetric branch and five fixed-sigma perturbed starts.
- `PYTHONDONTWRITEBYTECODE=1 python -m json.tool experiments/summaries/m3_c3_140_symmetry_probe_2026-04-29.json`
  passed.

Results:

| run | class | runtime | native residual | common sampled residual | Omega non-3 fraction | Omega RMS to symmetric |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| nested `72/216` | symmetric | `465.654 s` | `4.42e-14` | `0.695` | `6.02e-14` | n/a |
| nested `96/288` | symmetric | `7.311 s` | `3.19e-13` | `0.627` | `1.28e-13` | n/a |
| nested `128/384` | symmetric | `13.267 s` | `2.16e-13` | `0.598` | `5.35e-14` | reference |
| m1 perturb amp `0.5` | symmetric | `27.885 s` | `2.26e-13` | `0.598` | `5.82e-14` | `1.91e-14` |
| m1 perturb amp `1.5` | symmetry-breaking candidate | `42.330 s` | `3.43e-09` | `0.586` | `1.68e-01` | `2.93e-01` |
| m2 perturb amp `1.5` | symmetry-breaking candidate | `52.198 s` | `2.19e-13` | `0.602` | `1.69e-01` | `2.96e-01` |
| m4 perturb amp `1.5` | symmetric | `35.556 s` | `2.16e-13` | `0.598` | `5.09e-14` | `1.59e-14` |
| mixed non-3 perturb amp `1.0` | symmetry-breaking candidate | `45.219 s` | `3.24e-10` | `0.586` | `1.68e-01` | `2.93e-01` |

Assessment:
- This is a much cleaner clue of discrete nonuniqueness than the c4/c7 branch
  mismatch. For `{c_3=140}`, both the symmetric branch and the
  symmetry-breaking candidates satisfy the discrete Poisson compatibility to
  roundoff: Omega integral and Neumann flux mismatch are around `1e-14` to
  `1e-12`, and physical Poisson residual norms are around `5e-12`.
- The symmetry-breaking candidates preserve the same boundary data but have a
  measurable non-threefold interior component: Omega non-3 fraction about
  `0.168`, Omega RMS separation from the symmetric branch about `0.293`, and
  local Omega Linf separation about `8.3`.
- The small m1 perturbation and the m4 perturbation return to the symmetric
  branch, while larger m1, m2, and mixed non-3 perturbations converge to
  broken-symmetry roots. This suggests distinct basins of attraction at fixed
  boundary data.
- A quick orbit check showed the m1 and mixed perturbations converge to the
  same broken branch to numerical precision, while the m2 result is not the
  same under the tested C3 rotations/reflections. This needs a formal orbit and
  refinement check before claiming more than "symmetry-breaking candidates."

Artifacts:
- Summary:
  `experiments/summaries/m3_c3_140_symmetry_probe_2026-04-29.json`.
- State snapshots:
  `experiments/raw/nextgen/m3_c3_140_symmetry_probe/states/`.
- Plots:
  `experiments/raw/nextgen/m3_c3_140_symmetry_probe/plots/`.
- Registry rows appended for all eight c3=140 runs.

Next numerical step:
- Treat `{c_3=140}` as the primary M3 nonuniqueness target. Refine the
  symmetric and broken branches to `160/480` or finer, run a basin sweep between
  the symmetric and broken states, and trace both branches in amplitude with
  pseudo-arclength or seeded continuation. Only after persistence under
  refinement should this be called evidence for continuum nonuniqueness.

## 2026-04-29 — M3 c3=140 nonuniqueness validation at 160/480

Scope stayed in M3; no M4 operator assembly or eigensolver work was started.

Experiment added:
- Added `experiments/run_m3_c3_140_nonuniqueness_validation.py`.
- The script loads the saved `128/384` symmetric and broken candidates,
  prolongs them to `160/480`, solves the reduced `fv_upwind` equations at
  sigma `1`, maps a fixed-sigma Newton basin between the symmetric and m1
  broken branches, traces both branches downward in amplitude, runs orbit
  comparisons under C3 rotations/reflections, and attempts FEM baseline
  cross-checks at `h_min=0.08,h_max=0.12`.

Validation methods adopted:
- Branch refinement: `128/384 -> 160/480` prolongation followed by reduced
  Poisson Newton on the same FV equations.
- Basin mapping: convex combinations of the symmetric and broken Omega fields,
  with psi recomputed by Poisson projection before Newton.
- Amplitude tracing: seeded continuation on each branch for sigma
  `1.00, 0.95, ..., 0.60`.
- Symmetry/orbit diagnostics: weighted threefold defect plus best C3/reflection
  orbit distance between branches.
- Independent residual checks: sampled strong residuals on a common Cartesian
  grid and Poisson compatibility diagnostics including Omega integral,
  Neumann flux, Poisson multiplier, and physical Poisson residual.
- FEM/baseline cross-checks: zero-continuation baseline run and a seeded
  fixed-sigma FEM Newton attempt from the broken nextgen state.

Validation commands:
- `PYTHONDONTWRITEBYTECODE=1 python -m py_compile experiments/run_m3_c3_140_nonuniqueness_validation.py`
  passed.
- `PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n femx_spc python experiments/run_m3_c3_140_nonuniqueness_validation.py`
  completed.
- `PYTHONDONTWRITEBYTECODE=1 python -m json.tool experiments/summaries/m3_c3_140_nonuniqueness_validation_2026-04-29.json`
  passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_nextgen_profile.py -q`
  passed: `13` tests.
- `git diff --check` passed.

Refined sigma `1` results:

| run | class | runtime | native residual | common sampled residual | Omega non-3 fraction | physical Poisson residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| symmetric `160/480` | symmetric | `22.798 s` | `3.46e-13` | `0.602` | `6.88e-14` | `8.39e-12` |
| m1 broken `160/480` | broken m1 | `47.573 s` | `3.55e-13` | `0.588` | `2.52e-01` | `8.40e-12` |
| m2 broken `160/480` | broken-like | `41.800 s` | `3.74e-09` | `0.606` | `2.53e-01` | `8.39e-12` |

Basin map:
- Interpolants with `t=0.0` through `0.4` converged back to the symmetric
  branch; interpolants with `t=0.5` through `1.0` converged to the m1 broken
  branch.
- The fixed-sigma basin boundary is therefore bracketed between `t=0.4` and
  `t=0.5`. Both endpoints have small native residuals and the same boundary
  data, so this is not just a transient from the initial guess.

Amplitude trace:
- The symmetric branch remains threefold to roundoff throughout sigma
  `1.00` down to `0.60`, with Omega non-3 fractions between about `5e-14` and
  `5e-13`.
- The broken branch persists with clear non-threefold content from sigma
  `1.00` down to at least `0.70`: Omega non-3 fractions decrease from `0.252`
  at sigma `1.00` to `0.0509` at sigma `0.70`.
- The broken trace collapses back to an essentially symmetric solution between
  sigma `0.70` and `0.65`: Omega non-3 falls to `1.42e-09` at sigma `0.65`
  and `9.20e-14` at sigma `0.60`. This is consistent with a symmetry-breaking
  branch that merges into the symmetric branch near that amplitude range.

Orbit and branch separation:
- The best C3/reflection distance between the symmetric and m1 broken branches
  at `160/480` is weighted Omega RMS `0.435` and psi RMS `0.0135`.
- The best tested distance between m1 and m2 broken branches is weighted Omega
  RMS `0.436` and psi RMS `0.0142`, so m2 is not the same branch under the
  tested C3/reflection orbit. This may indicate multiple broken branches, but
  it still needs a more systematic orbit/finer-grid check.

FEM/baseline cross-check:
- The baseline zero-continuation run at `h_min=0.08,h_max=0.12` took
  `1317.256 s`, but the result is not a credible validation solution: common
  sampled residual was `6.27e4`, and the baseline Newton printout reached
  high-amplitude steps with relative updates near `1`.
- The baseline run seeded from the nextgen broken branch took `36.348 s` and
  ended nonconverged with last relative update `1.033` and common sampled
  residual `8.38e3`.
- These FEM checks are therefore inconclusive stress tests, not evidence
  against the nextgen nonuniqueness. A more robust FEM continuation or seeded
  line-search path is still needed for independent discretization validation.

Assessment:
- The `160/480` data gives strong numerical evidence for discrete
  nonuniqueness and symmetry breaking for `{c_3=140}`. The symmetric and
  broken branches solve the same nextgen FV problem with the same boundary
  data, native residuals near roundoff, Poisson compatibility near roundoff,
  and order-one separation in the interior symmetry diagnostic.
- The basin split and amplitude trace strengthen the interpretation as a
  branch phenomenon rather than a single isolated bad Newton endpoint.
- The continuum claim remains open until the branches are pushed to another
  refinement level, for example `192/576` or finer, and a robust independent
  FEM/baseline check is made to converge to at least one of the branches.

Artifacts:
- Summary:
  `experiments/summaries/m3_c3_140_nonuniqueness_validation_2026-04-29.json`.
- State snapshots:
  `experiments/raw/nextgen/m3_c3_140_nonuniqueness_validation/states/`.
- Plots:
  `experiments/raw/nextgen/m3_c3_140_nonuniqueness_validation/plots/`.
- Registry rows appended for the branch refinements, basin sweep, amplitude
  traces, and baseline attempts.

## 2026-04-30 — M3 c3=140 refinement to 192/576 and h=0.06 FEM cross-check

Scope stayed in M3; no M4 operator assembly or eigensolver work was started.

Experiment added:
- Added `experiments/run_m3_c3_140_refine192_fem_crosscheck.py`.
- The script refines the saved `160/480` symmetric and m1 broken branches to
  `192/576`, traces both branches through sigma `0.65` to `0.70` in steps of
  `0.005`, and runs a fine FEM/baseline cross-check on a uniform
  `h_min=h_max=0.06` disk mesh.
- The FEM cross-check is a residual-line-search Newton solve seeded from the
  refined nextgen branch fields. It is an independent-discretization
  confirmation of the two branches, not an independent branch-discovery
  continuation from zero data.

Validation commands:
- `PYTHONDONTWRITEBYTECODE=1 python -m py_compile experiments/run_m3_c3_140_refine192_fem_crosscheck.py`
  passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_nextgen_profile.py -q`
  passed before the long run: `13` tests.
- `PYTHONDONTWRITEBYTECODE=1 python experiments/run_m3_c3_140_refine192_fem_crosscheck.py --stage nextgen`
  completed the `192/576` branch refinement and amplitude trace.
- `PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n femx_spc python experiments/run_m3_c3_140_refine192_fem_crosscheck.py --stage baseline --h-min 0.06 --h-max 0.06`
  completed the fine FEM cross-checks. The usual non-fatal OpenMPI TCP bind
  warning appeared during the serial baseline run.
- `PYTHONDONTWRITEBYTECODE=1 python -m json.tool experiments/summaries/m3_c3_140_refine192_fem_crosscheck_2026-04-30.json`
  passed.

Nextgen refinement results at sigma `1`:

| run | runtime | native residual | common sampled residual | Omega non-3 fraction | physical Poisson residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| symmetric `192/576` | `26.545 s` | `3.30e-09` | `0.606` | `1.36e-13` | `1.17e-11` |
| m1 broken `192/576` | `55.539 s` | `4.92e-13` | `0.585` | `3.07e-01` | `1.17e-11` |

Bifurcation-window trace on `192/576`:
- The symmetric branch stayed threefold to roundoff throughout the trace. In
  the dense window from sigma `0.700` to `0.650`, Omega non-3 remained around
  `1e-13`.
- The broken branch did not collapse in the old `160/480` window. It retained
  Omega non-3 `0.123` at sigma `0.700`, `0.0994` at `0.680`, `0.0848` at
  `0.670`, `0.0656` at `0.655`, and `0.0597` at `0.650`.
- Therefore the apparent merge/bifurcation point is below sigma `0.650` on the
  `192/576` grid. The previous `160/480` collapse near `0.65` was not stable
  under refinement.

FEM/baseline cross-check:
- The `h_min=h_max=0.06` disk mesh had `202248` cells and `101649` vertices.
- The symmetric-seeded FEM Newton solve completed in `50.594 s`. The line
  search reduced the assembled residual norm from `4.94e-01` to `7.32e-15`
  in `7` Newton iterations. The common sampled residual was `0.725`, quick
  weak residual norms were `Omega_l2=3.44e-04`, `Psi_l2=6.26e-05`, and Omega
  non-3 was `0.0151`.
- The broken-seeded FEM Newton solve completed in `206.741 s`. It needed
  damped steps initially, then full Newton steps, reducing the assembled
  residual norm from `4.95e-01` to `8.48e-15` in `24` iterations. The common
  sampled residual was `0.670`, quick weak residual norms were
  `Omega_l2=3.44e-04`, `Psi_l2=1.16e-04`, and Omega non-3 was `0.525`.
- The two FEM solutions are clearly distinct on a common polar probe:
  Omega RMS separation `1.276`, Omega Linf separation `38.408`, and
  offset-removed psi RMS separation `0.0462`.

Assessment:
- The `192/576` refinement strengthens the nonuniqueness evidence. The broken
  branch survives refinement with larger non-threefold content than at
  `160/480`, clean Poisson compatibility, and small native residual.
- The refined amplitude trace shows the broken branch persists at least to
  sigma `0.650`; the bifurcation window must be moved lower.
- The fine FEM line-search Newton runs are the first credible independent
  discretization cross-check for the c3=140 nonuniqueness. They converge to
  two distinct FEM roots with small assembled residuals and distinct symmetry
  diagnostics.
- Remaining caveat: the FEM runs were seeded from nextgen branch fields. This
  validates that the FEM discretization has nearby symmetric and broken roots,
  but it does not yet show that baseline zero-continuation or pseudo-arclength
  continuation can discover both branches without nextgen branch seeds.

Artifacts:
- Summary:
  `experiments/summaries/m3_c3_140_refine192_fem_crosscheck_2026-04-30.json`.
- State snapshots:
  `experiments/raw/nextgen/m3_c3_140_refine192_fem_crosscheck/states/`.
- Plots:
  `experiments/raw/nextgen/m3_c3_140_refine192_fem_crosscheck/plots/`.
- Registry rows appended for the `192/576` refinements, amplitude traces, and
  the two `h=0.06` baseline/FEM cross-checks.

## 2026-04-30 — M3 c3=140 residual-audit table and branch-comparison plots

Scope stayed in M3; no M4 operator assembly or eigensolver work was started.

Experiment added:
- Added `experiments/run_m3_c3_140_residual_audit.py`.
- The script audits the existing `{c_3=140}` symmetric and m1 broken branch
  data at nextgen `160/480`, nextgen `192/576`, and baseline/FEM
  `h_min=h_max=0.06`.
- It regenerates and caches the FEM branch fields because the earlier FEM
  summary intentionally sanitized non-serializable `dolfinx` Function objects.
  A repo-local FEniCS cache is used under
  `experiments/raw/nextgen/m3_c3_140_residual_audit/fenics_cache` to keep JIT
  files inside the writable project tree.

Validation commands:
- `PYTHONDONTWRITEBYTECODE=1 python -m py_compile experiments/run_m3_c3_140_residual_audit.py experiments/run_m3_c3_140_refine192_fem_crosscheck.py`
  passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_nextgen_profile.py -q`
  passed: `13` tests.
- `PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n femx_spc python experiments/run_m3_c3_140_residual_audit.py --refresh-fem`
  completed. The usual non-fatal OpenMPI TCP bind warning appeared; conda
  also printed duplicate-rpath linker warnings while compiling FEniCS forms.
- `PYTHONDONTWRITEBYTECODE=1 python -m json.tool experiments/summaries/m3_c3_140_residual_audit_2026-04-30.json`
  passed.

Audit results:

| run | native/weak residual | common sampled residual | Omega non-3 | compatibility defect | branch runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| nextgen symmetric `160/480` | native `3.46e-13` | `0.602` | `6.88e-14` | `-9.23e-14` | loaded |
| nextgen broken `160/480` | native `3.55e-13` | `0.588` | `0.252` | `-5.79e-14` | loaded |
| nextgen symmetric `192/576` | native `3.30e-09` | `0.606` | `1.36e-13` | `5.48e-14` | loaded |
| nextgen broken `192/576` | native `4.92e-13` | `0.585` | `0.307` | `6.29e-14` | loaded |
| FEM symmetric `h=0.06` | weak `3.44e-04`, `6.26e-05` | `0.725` | `0.0151` | `-6.26e-05` | `50.843 s` |
| FEM broken `h=0.06` | weak `3.44e-04`, `1.16e-04` | `0.670` | `0.525` | `-1.16e-04` | `209.945 s` |

Branch-separation audit:
- Nextgen `160/480` symmetric-vs-broken separation on the standard polar
  probe: Omega RMS `0.478`, Omega Linf `12.331`, offset-removed psi RMS
  `0.0149`.
- Nextgen `192/576` symmetric-vs-broken separation: Omega RMS `0.607`, Omega
  Linf `15.297`, offset-removed psi RMS `0.0190`.
- FEM `h=0.06` symmetric-vs-broken separation on the common Cartesian audit
  grid: Omega RMS `1.390`, Omega Linf `37.917`, offset-removed psi RMS
  `0.0507`.
- Nextgen-vs-FEM branch agreement is qualitative but branch-consistent:
  symmetric Omega RMS difference `0.481`, broken Omega RMS difference `1.177`.

Definitions recorded by the audit:
- Nextgen native residual is the weighted discrete FV residual
  `sqrt(||R_omega||_w^2 + ||R_poisson||_w^2) / max(||Omega||_w, ||Lap psi||_w)`.
- FEM weak residuals are raw PETSc vector norms of the assembled Galerkin
  residuals for the Omega and psi equations.
- Common sampled strong residual samples both codes onto the same Cartesian
  grid and evaluates centered-difference strong residuals
  `v.grad(Omega) - Omega - eps Lap Omega` and `Lap psi - Omega`.
- Symmetry breaking is the weighted relative norm of
  `f - (f + rotate_120 f + rotate_240 f)/3`.
- Compatibility defect is `integral_disk Omega` minus the discrete outer
  Neumann flux for psi.
- Branch separation is RMS/Linf difference on common probes; psi differences
  have their mean offset removed because psi has a Neumann gauge.

Artifacts:
- Script: `experiments/run_m3_c3_140_residual_audit.py`.
- Summary:
  `experiments/summaries/m3_c3_140_residual_audit_2026-04-30.json`.
- CSV table:
  `experiments/raw/nextgen/m3_c3_140_residual_audit/tables/c3_140_residual_audit_table.csv`.
- FEM cache:
  `experiments/raw/nextgen/m3_c3_140_residual_audit/fem_cache/`.
- Plots:
  `experiments/raw/nextgen/m3_c3_140_residual_audit/plots/omega_nextgen_fem_branch_comparison.png`,
  `experiments/raw/nextgen/m3_c3_140_residual_audit/plots/omega_branch_difference_common_scale.png`.
- Registry row appended:
  `m3-c3-140-residual-audit-validation-table`.

Assessment:
- The residual audit packages the current c3=140 nonuniqueness evidence into a
  reproducible validation table. The two nextgen branches are discrete roots
  with small native residuals, near-roundoff compatibility defects, and stable
  order-one branch separation under refinement.
- The FEM cross-check again finds two distinct roots with small assembled
  weak residuals. FEM symmetry metrics differ in magnitude from nextgen, but
  the branch ordering and broken-vs-symmetric distinction persist.
- The common sampled strong residual remains a cross-code diagnostic, not the
  native solver residual. It is useful for comparing both discretizations on a
  shared grid, but the native FV and FEM weak residuals are the primary
  discrete accuracy measures.

## 2026-04-30 — M3 c3=140 symmetric nextgen/FEM discrepancy audit

Scope stayed in M3; no M4 operator assembly or eigensolver work was started.

Motivation:
- Visual inspection of the residual-audit plot showed a large
  `FEM symmetric - nextgen symmetric` difference, concentrated near the three
  sharp symmetric cores. This needed to be checked before treating the FEM
  cross-check as quantitative validation.

Experiment added:
- Added `experiments/run_m3_c3_140_symmetric_discrepancy_audit.py`.
- The script compares the nextgen symmetric `160/480` and `192/576` states,
  reruns the symmetric FEM solve, samples FEM both through the old generic
  `LinearNDInterpolator` and through direct DOLFINx cell lookup, checks
  nextgen-vs-baseline boundary conventions, tests small rotation/scale fits,
  and writes close-up plots.

Validation commands:
- `PYTHONDONTWRITEBYTECODE=1 python -m py_compile experiments/run_m3_c3_140_symmetric_discrepancy_audit.py`
  passed.
- `PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n femx_spc python experiments/run_m3_c3_140_symmetric_discrepancy_audit.py`
  completed the direct `h_min=h_max=0.06` FEM sampling audit. The usual
  non-fatal OpenMPI TCP bind warning appeared.
- `PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n femx_spc python experiments/run_m3_c3_140_symmetric_discrepancy_audit.py --h-min 0.05 --h-max 0.05`
  completed a finer symmetric FEM check on a `290895`-cell mesh.
- A quick residual check showed that the nextgen `fv_upwind` symmetric
  `192/576` state has native relative residual `3.30e-09` in `fv_upwind`, but
  relative residual `0.371` if evaluated with the `fv_centered` transport
  discretization.

Results:
- Boundary conventions are not the source of the discrepancy:
  nextgen and baseline/FEM Fourier boundary values agree to roundoff
  (`Omega` max difference `2.78e-16`, psi Neumann max difference `1.11e-16`).
- FEM boundary enforcement is also not the source: the symmetric FEM rerun had
  `Omega` Dirichlet max/RMS defect `0`.
- The old plot interpolation is not the source: direct DOLFINx cell sampling
  and the old `LinearNDInterpolator` sampling agree to roundoff at `h=0.06`
  (`3.59e-15` RMS on the audit grid) and to about `1.0e-09` RMS at `h=0.05`.
- Nextgen self-refinement is much smaller than the FEM/FV discrepancy:
  nextgen symmetric `160/480` vs `192/576` has Omega RMS `0.0649`
  (`3.4%` relative to the `192/576` RMS) on the audit grid.
- FEM vs nextgen symmetric remains large:
  `h=0.06` direct FEM vs nextgen `192/576` has Omega RMS `0.546`
  (`28.6%` relative to the nextgen RMS) and Linf `14.68`.
  `h=0.05` direct FEM vs nextgen `192/576` has Omega RMS `0.569`
  (`29.9%` relative to the nextgen RMS) and Linf `15.73`.
- Small rigid rotation does not explain the difference: the best sweep angle
  was `0` degrees for both full and active masks. A scalar least-squares
  rescaling improves RMS only mildly.
- The close-up plots show that FEM is producing sharper/higher core peaks than
  the current nextgen `fv_upwind` state; the difference is not only a
  visualization artifact.

Assessment:
- The user's concern is valid. The symmetric nextgen/FEM discrepancy is real,
  not an artifact of boundary convention, psi gauge, plotting interpolation,
  or a small rotational misalignment.
- The most likely cause is a discretization mismatch: current nextgen
  `fv_upwind` solves its own upwind-stabilized discrete equation accurately,
  but for this large-amplitude symmetric profile it appears too diffusive
  around the sharp cores compared with the baseline/FEM Galerkin equation.
- Therefore the current FEM cross-check should be interpreted as qualitative
  evidence that FEM also has separated symmetric and broken roots, not as a
  high-accuracy quantitative match between nextgen and FEM branches.
- Before making a paper-level continuum nonuniqueness claim, M3 should either
  produce a baseline-closer nextgen discretization for `{c_3=140}` or add
  a direct FEM refinement/branch study showing the nonunique branches in FEM
  independently of the upwind FV profiles.

Artifacts:
- Script:
  `experiments/run_m3_c3_140_symmetric_discrepancy_audit.py`.
- Summaries:
  `experiments/summaries/m3_c3_140_symmetric_discrepancy_audit_2026-04-30.json`,
  `experiments/summaries/m3_c3_140_symmetric_discrepancy_audit_h0p05_0p05_2026-04-30.json`.
- Plots:
  `experiments/raw/nextgen/m3_c3_140_symmetric_discrepancy_audit/plots/symmetric_direct_fem_vs_nextgen.png`,
  `experiments/raw/nextgen/m3_c3_140_symmetric_discrepancy_audit/plots/symmetric_top_core_closeup.png`,
  `experiments/raw/nextgen/m3_c3_140_symmetric_discrepancy_audit/plots/h0p05_0p05/`.
- Registry rows appended for the direct `h=0.06` and `h=0.05` discrepancy
  audits.

## 2026-04-30 — M3 c3=140 symmetric FV/FEM cross-initialization test

Scope stayed in M3; no M4 operator assembly or eigensolver work was started.

Motivation:
- A possible interpretation of the symmetric nextgen/FEM discrepancy is
  nonuniqueness even inside the threefold-symmetric class. A direct numerical
  test is to interpolate each solver's symmetric solution onto the other
  solver's grid and ask whether it is already a good native discrete solution,
  or whether Newton moves it to a different root.

Experiment added:
- Added `experiments/run_m3_c3_140_symmetric_cross_initialization.py`.
- The script performs both directions:
  1. FV `192/576` symmetric state seeds the FEM `h=0.06` or `h=0.05`
     line-search Newton solve.
  2. The converged FEM symmetric state is interpolated onto the FV `192/576`
     polar cell centers, projected through the FV Poisson solve, and used as
     the initial state for the nextgen reduced FV Newton solve.
- It reports native residuals before and after Newton and distances on the
  common `193x193` audit grid.

Validation commands:
- `PYTHONDONTWRITEBYTECODE=1 python -m py_compile experiments/run_m3_c3_140_symmetric_cross_initialization.py`
  passed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_nextgen_profile.py -q`
  passed: `13` tests.
- `PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n femx_spc python experiments/run_m3_c3_140_symmetric_cross_initialization.py --h-min 0.06 --h-max 0.06`
  completed.
- `PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n femx_spc python experiments/run_m3_c3_140_symmetric_cross_initialization.py --h-min 0.05 --h-max 0.05`
  completed.
- JSON validation passed for both summaries. The usual non-fatal OpenMPI TCP
  bind warning appeared during FEM runs.

Results at FEM `h=0.06`:
- FV-to-FEM: the FV seed had FEM Newton residual norm `4.94e-01`; FEM Newton
  converged in `7` iterations to residual norm `7.32e-15`. The final FEM
  profile stayed far from the FV reference: Omega RMS `0.546`, Linf `14.68`
  on the audit grid.
- FEM-to-FV: the FEM state interpolated to the FV grid and Poisson-projected
  had FV relative residual `1.77` and Omega residual norm `46.84`; FV Newton
  converged to relative residual `2.95e-12`.
- The FV final from the FEM seed returned to the original FV reference:
  Omega RMS `2.00e-10`, Linf `1.02e-08`; psi RMS `8.40e-13`.
- Interpolating FEM through the FV grid itself was not the limiting error:
  FEM-to-FV-grid representation vs direct FEM had Omega RMS `0.0551`
  on the same audit grid, much smaller than the FEM-vs-FV discrepancy.

Results at FEM `h=0.05`:
- FV-to-FEM: the FV seed had FEM Newton residual norm `4.45e-01`; FEM Newton
  converged in `7` iterations to residual norm `2.34e-14`. The final FEM
  profile was still far from FV: Omega RMS `0.569`, Linf `15.73`.
- FEM-to-FV: the FEM state interpolated to the FV grid and Poisson-projected
  had FV relative residual `1.34` and Omega residual norm `35.65`; FV Newton
  converged to relative residual `2.16e-12`.
- The FV final again returned to the original FV reference: Omega RMS
  `2.00e-10`, Linf `1.02e-08`; psi RMS `8.38e-13`.

Assessment:
- This cross-initialization test does not support the hypothesis that the
  current FV and FEM symmetric profiles are two well-resolved symmetric
  continuum branches. Each interpolated profile has a large native residual
  on the other solver's grid, and Newton moves it back to the receiving
  solver's own symmetric root.
- The result also strengthens the previous discretization-mismatch diagnosis:
  current `fv_upwind` gives an accurate root of the upwind FV system, FEM
  gives an accurate root of the Galerkin baseline system, but these two
  discrete systems are not yet approximating the same symmetric profile at
  the tested resolutions/discretizations.
- This does not prove symmetric-class uniqueness for the PDE. It only says
  that the present FV/FEM discrepancy is not positive evidence for
  same-symmetry nonuniqueness.

Artifacts:
- Script:
  `experiments/run_m3_c3_140_symmetric_cross_initialization.py`.
- Summaries:
  `experiments/summaries/m3_c3_140_symmetric_cross_initialization_h0p06_0p06_2026-04-30.json`,
  `experiments/summaries/m3_c3_140_symmetric_cross_initialization_h0p05_0p05_2026-04-30.json`.
- Plots:
  `experiments/raw/nextgen/m3_c3_140_symmetric_cross_initialization/plots/h0p06_0p06/symmetric_cross_initialization.png`,
  `experiments/raw/nextgen/m3_c3_140_symmetric_cross_initialization/plots/h0p05_0p05/symmetric_cross_initialization.png`.
- Registry rows appended for both cross-initialization runs.

## 2026-04-30 — M3 c3=140 paper-level validation protocol

Scope stayed in M3; no M4 operator assembly or eigensolver work was started.
No numerical experiments were run in this step.

Documentation added:
- Added `docs/m3_c3_140_validation_protocol.md`.
- The protocol separates three claims that must not be conflated:
  1. Newton solved a discrete nonlinear system.
  2. A mesh sequence approximates a continuum profile.
  3. Two distinct continuum profiles exist for the same boundary data.
- It defines five validation gates:
  discrete solve quality, single-discretization refinement,
  cross-discretization consistency, nonuniqueness evidence, and branch
  discovery/continuation.

Main protocol decisions:
- Small native residual is necessary but not sufficient; it certifies only
  the chosen discrete equations.
- The current `{c_3=140}` nextgen `fv_upwind` branches pass discrete-solver
  quality checks, but the symmetric FV/FEM cross-initialization failure blocks
  quantitative continuum validation.
- Paper-level evidence should require refinement error estimates that are
  much smaller than branch separation, preferably with branch separation at
  least `5x` to `10x` larger than estimated discretization error.
- Future evidence bundles should include native residuals, boundary and
  compatibility defects, refinement tables, cross-initialization tables,
  branch separation, symmetry metrics, plots, saved states, and exact
  commands/tolerances.

Assessment:
- The protocol makes the current M3 blocker explicit: the `{c_3=140}`
  nonuniqueness evidence is strong for the upwind FV discrete system and
  qualitatively supported by FEM, but not yet a paper-level continuum claim.
- The next M3 work should target a baseline-closer nextgen discretization or
  a FEM-only branch/refinement study, then run the protocol gates before any
  M4 profile consumption.
