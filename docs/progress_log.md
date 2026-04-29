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
