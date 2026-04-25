# Progress log

## Current objective

Build and validate a new sparse-matrix finite-difference implementation for the self-similar
2D Euler time-dependent-viscosity eigenvalue workflow, while preserving the logic and outputs
of the verified baseline implementation.

## Current status

Initial intake completed on 2026-04-25.
The active baseline workflow has been traced, and repo-local output path issues in
`src/baseline/driver.py` have been fixed without changing the numerical method.

The baseline could not be live-imported or run in the current system Python because `gmsh`
is not installed there. This is an external environment issue, not a repo-local code issue.
Syntax compilation of the active baseline files passed with Python cache redirected to `/tmp`.

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
   active input, only mode `m=3` is nonzero with coefficient `70`, mirrored to
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
