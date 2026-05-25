# FEM Manuscript Experiments Plan

Created: 2026-05-22

## Purpose

This branch uses the verified baseline FEM workflow as the main numerical
engine for manuscript-quality experiments. The previous `src/nextgen/`
finite-difference / finite-volume work remains documented and available for
reference, but it is not the starting point for the current manuscript data
pipeline.

The near-term manuscript experiments are:

- Branch A: continue the boundary amplitude in `s cos(3 theta)`, compute the
  first 10 left-edge eigenvalues, and plot `Re(lambda_j)` versus `s`.
- Branch B: at large amplitudes suggested by Branch A, compare symmetric and
  symmetry-breaking FEM profiles with the same boundary data.

## Baseline FEM Reuse Map

The active baseline workflow is `src/baseline/driver.py`. The reusable entry
points are:

- `config.SimulationParams`: physical parameters, Newton controls, sigma
  continuation grid, and mesh sizes.
- `mesh.build_disk`: gmsh disk mesh on `B_Rdom`.
- `spaces.create_spaces`: scalar CG1 spaces for Omega and psi.
- `boundary_data_fourier.FourierBC`: Fourier far-field data for Omega and the
  derived Neumann data for psi.
- `poisson_neumann.PoissonNeumann`: solves `Delta psi = Omega` with the
  inhomogeneous Neumann data and one pinned gauge dof.
- `newton_block.NewtonAssembler`: assembles the coupled manifold Newton block.
- `continuation.ContinuationSolver.run`: zero-start sigma continuation for the
  profile.
- `newton_direct.DirectNewtonSolver.run_from`: fixed-sigma Newton from an
  externally supplied profile seed; useful for Branch B perturbation starts.
- `residuals_quick.residuals_quick`: raw weak residual diagnostics.
- `eigs_linearized_new.ManifoldEigenSolver.solve`: reduced manifold
  shift-invert eigensolver; use `k=10` for Branch A.
- `viz.py` and `eigs_linearized_new.save_modes_plots_real`: debug plots. New
  manuscript plots should be written under `experiments/fem_manuscript/`.

## Boundary Parameterization

The manuscript boundary amplitude is defined analytically as

```text
barOmega(theta) = s cos(3 theta).
```

The baseline `FourierBC` mirrors positive modes: if `coeffs={3: c}`, then the
stored full series contains both `c_3=c` and `c_-3=conj(c)`. Therefore an exact
analytic amplitude `s cos(3 theta)` must be passed to the baseline as:

```python
coeffs = {3: 0.5 * s + 0.0j}
```

The physical boundary condition imposed by the baseline continuation is then:

```text
Omega|boundary = sigma * R^(-1/mu) * s cos(3 theta).
```

For the manuscript branch, use `sigma=1` for a target amplitude `s`; reserve
the internal `sigma_grid` only for nonlinear continuation from zero to that
target problem.

## Directory Structure

```text
experiments/fem_manuscript/
  common.py
  c3_spectrum_continuation/
    run_spectrum_continuation.py
    data/
    checkpoints/
    logs/
    plots/
  symmetry_breaking_profiles/
    run_profile_comparison.py
    data/
    checkpoints/
    logs/
    plots/
```

Recommended persisted artifacts:

- CSV tables for scalar results and diagnostics.
- JSON manifests for run configuration and status.
- XDMF/HDF5 or compressed sampled-field checkpoints for profiles and selected
  eigenvectors.
- Plain log files for failures and partial runs.
- Debug plots first; final manuscript plots only after validation gates pass.

## Branch A Staged Plan

1. Smoke test at coarse mesh, for example `h_max >= 0.18`, with one or two
   small amplitudes and `k=2` or `k=3`. The goal is to verify script wiring,
   output paths, CSV rows, profile residual capture, and eigenvalue capture.
2. Preliminary exploratory continuation with coarse-to-medium meshes, for
   example `h_max` near `0.12` or `0.10`, `k=10`, and a sparse set of `s`
   values. Record success/failure and solver wall time for every point.
3. Implement adaptive amplitude sampling. Refine where eigenvalue real parts
   cross zero, where adjacent samples show sharp spectral movement, and near
   the first point where `min_j Re(lambda_j) < -0.1`.
4. Production runs use `h_max = 0.06` or finer. Preferred mesh checks compare
   `h_max = 0.08`, `0.06`, and a finer mesh if feasible.
5. Validate before final plotting: profile weak residuals, Poisson consistency,
   eigenpair residuals, solver tolerances, mesh sensitivity, and continuity of
   eigenvalue branches.
6. Produce the final academic plot only after the validated table is complete:
   `Re(lambda_j)` versus `s` for `j=1,...,10`, with crossing/instability
   regions clearly legible.

Branch A stop criterion:

```text
stop amplitude continuation once at least one computed eigenvalue has
real part smaller than -0.1.
```

## Branch B Staged Plan

1. Use amplitudes selected from Branch A, especially where the spectrum suggests
   instability or sharp changes.
2. Compute the symmetric profile for the exact same boundary data
   `s cos(3 theta)`.
3. Construct interior-only perturbations of the initial guess or continuation
   path. Do not alter boundary dofs or Fourier data.
4. Run fixed-sigma or continuation-based FEM Newton from perturbed starts.
5. Validate both profiles against the same boundary-value problem:
   boundary dof error, weak residuals, Poisson/compatibility diagnostics, and
   mesh checks.
6. Quantify symmetry breaking using non-threefold component fractions and
   separation norms between symmetric and perturbed profiles, including
   RMS/L2/Linf-type metrics on Omega and offset-adjusted psi.
7. Generate heat maps and streamlines only after the numerical evidence is
   stable under mesh refinement.

Do not claim bifurcation or nonuniqueness from a single run. A manuscript claim
requires same-boundary residual-clean profiles, visible separation, symmetry
diagnostics, and mesh robustness.

## Risks And Ambiguities

- Fourier amplitude convention is easy to misread. This plan uses `c_3=s/2`
  for exact `s cos(3 theta)`, even though historical code labels such as
  `{c3=140}` used the baseline coefficient dictionary directly.
- The current `eigs_linearized_new` solver sorts returned modes by real part
  after a shift-invert solve. Branch tracking across `s` still needs explicit
  matching or careful plotting conventions.
- Eigenpair residual computation is not yet exposed as a CSV diagnostic and
  must be added before production.
- FEniCS function checkpoint/restart needs a robust format. XDMF/HDF5 is
  natural for full fields; compressed sampled fields are useful for plots and
  cross-run diagnostics.
- The baseline continuation has no built-in checkpointing, so manuscript
  scripts must checkpoint after each successful amplitude/mesh point.
- The unstructured FEM mesh can introduce small apparent symmetry defects;
  symmetry diagnostics should compare against mesh/self-reference and use
  separation metrics, not raw visual symmetry alone.

## Immediate Next Milestone

The first Branch A runner was implemented and run on 2026-05-22. A very coarse
`h_max=0.45` smoke test confirmed the plumbing but gave qualitatively wrong
spectral data. Per user direction, Branch A should now use the tested FEM mesh
scale `h_min=h_max=0.06` directly and tune only the `s` step sizes.

The first `h=0.06`, `s=10`, `k=10` point completed with weak residuals
`||R_Omega||_2=1.175e-05`, `||R_Psi||_2=4.781e-14`, and
`min Re(lambda)=0.5813`.

The first accepted coarse ladder found `min Re(lambda)=3.44e-02` at `s=120`
and `min Re(lambda)=-3.33e-01` at `s=160`, so the Branch A stop criterion was
already met at `s=160`. The active `s=240` eigensolve was stopped after this
was detected; `s=280` was not launched.

The first refinement pass in `[120,140]` completed at `s=125,130,135,140`
with `h=0.06`, `k=10`, `sigma_steps=21`, and `max_newton=48`. It bracketed the
first zero crossing in `s in [120,125]` and the `min Re(lambda)<-0.1` stop
threshold in `s in [130,135]`. A combined first plot for `s in [0,160]`
actually uses available samples `s=10,40,80,120,125,130,135,140,160` and is
stored under
`experiments/fem_manuscript/c3_spectrum_continuation/plots/branchA_h006_s10_160/`.

Next milestone: refine the zero crossing in `[120,125]` at `h=0.06`, `k=10`,
using the existing profile/weak residual norms and eigensolver tolerance.
Eigenpair residual diagnostics remain deferred for now. After the zero is
localized, refine `[130,135]` only if the manuscript needs a precise
`-0.1` threshold location.
