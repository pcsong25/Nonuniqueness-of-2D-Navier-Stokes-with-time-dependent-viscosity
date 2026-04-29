# Implementation plan

## Goal

Build a new sparse-matrix finite-difference codebase that reproduces the verified baseline
workflow for the self-similar 2D Euler time-dependent-viscosity problem, while improving
efficiency and keeping validation explicit.

## Success criteria

The new implementation should ultimately provide:
- vortex/profile solution,
- first k eigenvalues,
- first k eigenfunctions,
- error / residual / diagnostic measures,
- preliminary debug visualizations,
- meaningful baseline-vs-nextgen validation and timing comparisons.

## Milestones

### M0. Baseline intake and repo-local fixes
Status: completed

Objectives:
- identify the active baseline workflow from `src/baseline/driver.py`,
- determine which baseline files are actually used,
- make minimal repo-local fixes so baseline code is runnable,
- summarize the mathematical/computational workflow.

Deliverables:
- updated `docs/progress_log.md`
- baseline dependency/workflow summary

### M1. Nextgen numerical foundations
Status: completed

Objectives:
- create `src/nextgen/` structure,
- implement grid/boundary/operator/diagnostic foundations,
- validate basic conventions and low-level building blocks.

Deliverables:
- foundation modules
- low-level tests

### M2. Poisson solver
Status: completed

Objectives:
- implement sparse Neumann Poisson solve,
- pin gauge consistently,
- validate boundary conventions and solver residuals.

Deliverables:
- `src/nextgen/poisson.py`
- tests for Neumann conventions / manufactured checks / gauge handling

### M3. Profile solver
Status: implemented / moderate fine-grid acceptance passed / high-amplitude mixed robustness open

Objectives:
- implement the nonlinear self-similar profile solve on top of nextgen foundations,
- preserve baseline mathematical conventions,
- include robust Newton-type solve strategy and diagnostics,
- compare against baseline on reduced cases.

Required validation:
- unit/integration tests where appropriate,
- at least one simplified or manufactured check if feasible,
- at least one baseline comparison on a reduced case,
- runtime and discrepancy summary.

Deliverables:
- `src/nextgen/profile.py`
- `src/nextgen/polar_profile.py`
- `tests/test_nextgen_profile.py`
- `experiments/run_m3_profile_comparison.py`
- `experiments/run_m3_c3_profile_visual_check.py`
- `experiments/run_m3_fine_upwind_acceptance_suite.py`
- `experiments/run_m3_high_amplitude_stress_suite.py`
- logged reduced comparison results in `experiments/registry.csv` and
  `experiments/summaries/`

Validation note:
- The square-grid solver is retained as a diagnostic implementation but is not
  valid for c3 disk/whole-space validation because it preserves square-domain
  symmetry instead of threefold symmetry.
- A general full-disk polar upwind Newton solver is now the M3 validation path.
  It does not customize the mesh to c3 or any other n-fold symmetry. On the
  active `c_3=60`, `c_4=0` case, the latest common sampled strong residual is
  `0.589` for nextgen versus `1.118` for baseline. Runtime is slower than the
  baseline (`49.409 s` vs `10.050 s`), and Omega Linf discrepancies near the
  central vortex arms remain.
- A full-disk reduced validation suite also passed for `c_4=5`, mixed
  `c_2/c_5` complex data, and reduced mixed `c_3/c_4` data.
- A full-disk refinement study has now been run for the active
  `c_3=60`, `c_4=0` case and a mixed non-symmetric reduced
  `c_2=3+i`, `c_5=-2+0.5i` case. The mixed reduced case shows monotone
  improvement in common strong residual and probe RMS errors as
  `nr, ntheta` are refined from `30/90` to `36/108`.
- The active full-amplitude c3 case now has a viable runtime/accuracy point:
  optimized nextgen `nr=54`, `ntheta=162` completed in `17.585 s` with common
  sampled strong residual `0.650`; the finer baseline `h_min=0.08`,
  `h_max=0.12` completed in `48.056 s` with common sampled strong residual
  `0.637`. The previous `h_min=0.12`, `h_max=0.18` baseline was slightly
  slower than nextgen and less accurate under the common strong residual.
- Robustness is improved but not complete on coarser active grids:
  adaptive continuation moved `nr=48`, `ntheta=144` from the old sigma `0.7`
  stall to final accepted sigma `0.98125`, but it still did not reach sigma
  `1` and was too slow (`227.059 s`) to use as the production path.
- A general boundary visual suite was run for non-c3 inputs on medium and fine
  full-disk polar grids. Completed cases include `c_4=20`, phase-shifted
  `c_2=15+8i`, reduced mixed `c_1/c_4/c_7`, and reduced mixed `c_2/c_5/c_8`;
  each has baseline comparisons and plots under
  `experiments/raw/nextgen/m3_general_boundary_visual_suite/`.
- Coarser grids are not a reliable general-boundary production path:
  moderate mixed cases `{c_1=4-2i,c_4=8,c_7=-3+2i}` and
  `{c_2=6+3i,c_5=-5+i,c_8=2-1.5i}` stalled in nextgen continuation on
  `54/162`, even though the baseline profile workflow completes. A tighter
  continuation stress probe for the latter still stalled near sigma `0.658`.
- A focused robustness pass added secant prediction, affine coefficient homotopy,
  pinned-gauge Poisson diagnostics, LSMR least-squares fallback, and selectable
  centered/upwind/blended advection. These changes showed that the target
  `{c_2=6+3i,c_5=-5+i,c_8=2-1.5i}` case can be forced to sigma `1` with
  centered or lightly blended advection on `54/162`, but those fields exhibit
  radial ringing and very poor common sampled strong residuals. Upwind remains
  the smoother/more credible accuracy path, but still stalls near sigma `0.658`
  even with `min_sigma_step=1e-5`.
- A finer upwind polar grid resolves the target mixed case without centered
  ringing. On `nr=72`, `ntheta=216`, `{c_2=6+3i,c_5=-5+i,c_8=2-1.5i}` reaches
  sigma `1` in `13.227 s`, with native relative residual `1.25e-10`, common
  sampled strong residual `0.421`, Omega probe RMS `0.1065`, and offset-removed
  psi RMS `0.0101` versus the baseline. The baseline `h_min=0.12`,
  `h_max=0.18` run took `14.964 s` with common sampled strong residual `0.964`.
- A fine-grid upwind acceptance suite passed on `nr=72`, `ntheta=216` for:
  active `c_3=60,c_4=0`, the target mixed
  `{c_2=6+3i,c_5=-5+i,c_8=2-1.5i}`, and a `1.5x` larger mixed case. The
  nextgen native relative residuals were `2.18e-11`, `1.25e-10`, and
  `5.06e-10`. Common sampled strong residuals were `0.544`, `0.421`, and
  `0.462`, versus baseline `0.537`, `0.964`, and `1.104`. Runtimes were
  `33.274 s`, `13.491 s`, and `16.855 s`, versus baseline `53.414 s`,
  `15.032 s`, and `15.756 s`.
- M3 profile generation can be used for M4 input only within this tested
  full-disk upwind polar envelope. Very large arbitrary mixed Fourier data
  (`60`- to `100+`-scale combinations) still need a dedicated stress/refinement
  study before claiming general high-amplitude robustness.
- A high-amplitude mixed stress run with
  `{c_2=60,c_5=-5+i,c_8=2-1.5i}` shows that the current full-disk upwind
  Newton path is not yet robust enough for the larger boundary values needed
  before M4/eigenvalue searches. Nextgen `72/216` reached only sigma `0.45`
  after `248.238 s` and failed at attempted sigma `0.4515625`; `96/288` was
  skipped as not operationally feasible after the coarser failure. The fine
  baseline `h_min=0.08,h_max=0.12` completed sigma `1` in `105.068 s` with
  common sampled strong residual `0.651`. The coarser baseline
  `h_min=0.12,h_max=0.18` converged to a bad branch with common residual
  `5.94e3`, so baseline refinement is essential at this amplitude.
- A focused pseudo-transient fallback was added to the polar Newton solver:
  if the exact Newton step fails the residual line search, the solver tries
  shifted Omega-block solves `(J + beta M_Omega) delta = -F`. On the same
  high-amplitude stress case this did not improve the accepted sigma: nextgen
  `72/216` again stopped at sigma `0.45`, with the same partial-state common
  residual `0.468`, and runtime increased to `472.999 s`. The high-amplitude
  M3 blocker therefore remains a discretization/continuation issue, not just
  an undamped Newton-step issue.
- A conservative polar finite-volume transport option `fv_upwind` was added
  for the Omega advection term. It discretizes `v . grad(Omega)` as
  `div(v Omega) - Omega div(v)` with monotone face fluxes. On the
  high-amplitude stress case it is a real improvement: `30/90` reached sigma
  `1` in `68.953 s` with common sampled residual `0.738` versus fine baseline
  `0.651`, but refinement `36/108` still stalled at sigma `0.933203125` after
  `354.832 s`; therefore the requested `72/216` fine-grid run remains
  operationally blocked and M3 is still open for high-amplitude production.

### M4. Linearized operator assembly
Status: pending

Objectives:
- assemble the linearized operator around the computed profile,
- document sign conventions and normalization choices carefully,
- validate consistency against baseline logic.

Required validation:
- operator consistency checks,
- reduced-case comparisons,
- documented assumptions.

### M5. Eigensolver
Status: pending

Objectives:
- compute the first k eigenvalues/eigenfunctions,
- choose sparse eigensolver strategy appropriate for lowest modes,
- include normalization and residual diagnostics.

Required validation:
- eigenpair residual checks,
- stability under discretization/domain refinement,
- baseline comparison on shared cases.

### M6. Benchmarking and output refinement
Status: pending

Objectives:
- benchmark nextgen vs baseline,
- improve debug plots and numerical summaries,
- document where nextgen wins and where it does not.

Required validation:
- timing tables,
- accuracy comparisons,
- parameter-sensitivity summaries.

## Operating rules

- One milestone at a time.
- Commit after each accepted milestone.
- Do not rely on chat history as the only project memory.
- Use `docs/progress_log.md` for narrative status and `experiments/registry.csv` for run-by-run records.

## Current recommended next milestone

Continue M3 validation/fix work: fix robust nonlinear continuation for
moderate-amplitude mixed Fourier boundary data before M4.
Do not implement the linearized operator or eigensolver yet.
