# M3 c3=140 Nonuniqueness Validation Protocol

Status: active M3 protocol.

Scope: boundary data `{c_3=140}` with all other Fourier modes zero.

This protocol defines the validation gates needed before claiming
paper-level numerical evidence for continuum nonuniqueness. It separates
three different claims:

1. A Newton solver has accurately solved one discrete nonlinear system.
2. A sequence of discrete solutions is converging to a continuum PDE profile.
3. Two distinct continuum profiles exist for the same boundary data.

Small native residuals certify only the first claim. The second and third
claims require refinement, branch separation, and cross-discretization tests.

## Target Continuum Claim

For the same boundary data `{c_3=140}`, there exist at least two profiles
`(Omega, psi)` satisfying

```text
v . grad(Omega) - Omega - eps Lap(Omega) = 0,
Lap(psi) = Omega,
v = grad_perp(psi) - mu x,
```

with the same far-field `Omega` Dirichlet data and `psi` Neumann data. One
profile should preserve threefold symmetry, while another should have a
nonzero non-threefold component.

The current evidence does not yet establish this continuum claim.

## Gate 1: Discrete Solve Quality

Purpose: certify that each reported profile is a reliable root of its own
discrete equations.

Required for every branch, every mesh, and every discretization:

- Native nonlinear residual is small in that method's own norm.
  - FV: weighted algebraic residual
    `sqrt(||R_Omega||_w^2 + ||R_poisson||_w^2) / max(||Omega||_w, ||Lap psi||_w)`.
  - FEM: assembled Galerkin weak residual norms for the Omega and psi
    equations.
- Final Newton update is small.
- Solver tolerance is stricter than the observed mesh-refinement error.
- Poisson residual is small.
  - FV: `||Lap_h psi - Omega||_w`, plus gauge/compatibility diagnostic.
  - FEM: weak Poisson residual, plus compatibility diagnostic.
- Boundary defects are small.
  - Strongly imposed `Omega` Dirichlet data should be at roundoff or assembly
    tolerance.
  - `psi` Neumann data are natural in FEM; report compatibility and weak
    residual rather than a pointwise recovered flux error.
- Compatibility defect is small:
  `int_D Omega dx - int_boundary d_n psi ds`.
- Results are reproducible from saved state and script inputs.

Suggested acceptance:

- FV relative residual: at or below the chosen Newton tolerance, and at least
  two orders of magnitude below estimated discretization error.
- FEM weak residuals: stable under Newton tolerance tightening and small
  relative to the assembled residual at the initial seed.
- Boundary/compatibility defects: at least one to two orders of magnitude
  smaller than the branch-separation observables.

Current status:

- Pass for nextgen `fv_upwind` as a discrete solver.
- Pass for FEM line-search Newton as a discrete solver.
- This gate alone is not enough for continuum validation.

## Gate 2: Single-Discretization Refinement

Purpose: show that each branch is converging under mesh refinement within one
method.

Required branch-by-branch:

- Compute the same branch on a mesh sequence.
  - Nextgen candidate sequence: `160/480`, `192/576`, then finer.
  - FEM candidate sequence: `h=0.08`, `h=0.06`, `h=0.05`, then finer if
    feasible.
- Compare refined solutions on a common physical sampling grid.
- Remove the mean/gauge from `psi` before comparing `psi`.
- Track both global and localized metrics:
  - `Omega` RMS and Linf differences.
  - `psi` RMS and Linf differences after mean removal.
  - peak amplitudes and peak locations of the core structures.
  - integral quantities such as `||Omega||`, enstrophy-like norms, and
    compatibility defects.
  - symmetry metrics.
- Estimate a discretization error from consecutive refinements.

Suggested acceptance:

- Consecutive refinement differences decrease.
- Branch observables converge to stable values.
- The estimated branch error is much smaller than the branch separation.
  A practical target is branch separation at least `5x` to `10x` larger than
  the estimated refinement error.
- Symmetric branch has non-threefold fraction tending to zero.
- Broken branch has non-threefold fraction bounded away from zero.

Current status:

- Nextgen `fv_upwind` branch refinement supports two distinct discrete
  branches, but the baseline/FEM comparison shows the `fv_upwind` continuum
  accuracy is not yet established for `{c_3=140}`.
- FEM refinement has been sampled at `h=0.06` and `h=0.05`, but a complete
  FEM branch-refinement sequence and independent branch discovery remain open.

## Gate 3: Cross-Discretization Consistency

Purpose: determine whether two methods approximate the same continuum branch.

Required for each branch:

1. FV-to-FEM seed test:
   - Interpolate the FV solution into the FEM space.
   - Measure the FEM native residual before Newton.
   - Run FEM Newton.
   - Measure the correction from the FV seed to the FEM root.
2. FEM-to-FV seed test:
   - Interpolate the FEM solution onto the FV grid.
   - Project `psi` through the FV Poisson solve when appropriate.
   - Measure the FV native residual before Newton.
   - Run FV Newton.
   - Measure the correction from the FEM seed to the FV root.
3. Compare final profiles on a common grid.
4. Repeat under refinement.

Suggested acceptance for same-branch agreement:

- Cross-seeded native residual is comparable to the receiving method's
  truncation/refinement error.
- Newton correction from cross seed is small compared with branch separation.
- Cross-method difference decreases under refinement.
- Cross-method branch labels agree: symmetric maps to symmetric, broken maps
  to broken.

Current status:

- The symmetric branch fails this gate for current `fv_upwind` versus FEM.
- At FEM `h=0.06`, the FV seed has FEM Newton residual norm `4.94e-01`; the
  FEM-to-FV projected seed has FV relative residual `1.77`; FV Newton returns
  to the FV root with Omega RMS about `2e-10`.
- At FEM `h=0.05`, the FV seed has FEM Newton residual norm `4.45e-01`; the
  FEM-to-FV projected seed has FV relative residual `1.34`; FV Newton again
  returns to the FV root with Omega RMS about `2e-10`.
- Therefore current FV/FEM symmetric mismatch should be treated as
  discretization mismatch, not as evidence for symmetric-class
  nonuniqueness.

## Gate 4: Nonuniqueness Evidence

Purpose: certify that two branches are not numerical artifacts.

Required:

- Same boundary data verified analytically and numerically.
- Two distinct roots on the same discretization, each satisfying Gate 1.
- Branch separation remains positive under Gate 2 refinement.
- Estimated discretization error is much smaller than branch separation.
- Broken branch non-threefold metric remains bounded away from zero under
  refinement.
- Symmetric branch non-threefold metric tends to zero.
- Basin or continuation evidence shows organized branch structure, not a
  single isolated Newton endpoint.
- Cross-discretization evidence from Gate 3 supports the same branch
  structure.

Suggested acceptance:

- Branch separation stable under refinement and at least `5x` to `10x`
  larger than estimated branch error.
- Broken branch non-threefold fraction remains clearly separated from zero.
- Cross-method validation either matches the branches quantitatively or an
  independent method, such as FEM-only continuation, confirms the same
  nonunique structure.

Current status:

- Strong evidence for nonuniqueness of the nextgen `fv_upwind` discrete
  system.
- Qualitative FEM evidence for separated symmetric and broken roots.
- Not yet paper-level continuum evidence because quantitative FV/FEM
  agreement fails for the symmetric branch.

## Gate 5: Branch Discovery and Continuation

Purpose: show that branch structure is robust and not dependent on one
hand-picked seed.

Required:

- Continue symmetric and broken branches in amplitude.
- Locate or bracket the symmetry-breaking/fold region sufficiently to show
  organized branch behavior.
- Map Newton basins near the branch split.
- Prefer pseudo-arclength or fold-aware continuation near turning points.
- For FEM, attempt branch discovery that does not rely exclusively on
  nextgen branch seeds.

Suggested acceptance:

- Broken branch can be followed across a nontrivial amplitude interval.
- Basin boundary is reproducible under mesh refinement.
- FEM or a baseline-closer nextgen method can discover or continue the broken
  branch independently.

Current status:

- Nextgen `fv_upwind` continuation gives a coherent branch picture.
- The apparent bifurcation/merge window moved below `sigma=0.650` at
  `192/576`.
- FEM branch discovery independent of nextgen seeds remains open.

## Required Evidence Bundle

Every candidate paper-level run should produce:

- Native residual table.
- Boundary and compatibility table.
- Refinement table by branch.
- Cross-initialization table.
- Branch separation table.
- Symmetry metric table.
- Common-grid plots for:
  - each branch,
  - branch differences,
  - cross-method differences,
  - close-ups of core regions.
- Saved states or fields sufficient to reproduce the comparisons.
- Exact commands, mesh parameters, tolerances, and runtime.

The existing residual-audit script is a useful starting point, but future
runs must include the cross-initialization and refinement gates before being
used as continuum evidence.

## Current M3 Action Items

Before making a continuum nonuniqueness claim for `{c_3=140}`:

1. Build or use a baseline-closer nextgen profile discretization, for example
   `fv_centered` or a low-upwind-blend continuation from the existing
   `fv_upwind` branches.
2. Re-run the residual audit on symmetric and broken branches.
3. Re-run the symmetric cross-initialization audit.
4. If symmetric cross-method agreement improves, extend cross-initialization
   to the broken branch.
5. Run a method-specific refinement study for both branches.
6. Attempt FEM branch discovery or FEM pseudo-arclength continuation that does
   not rely exclusively on nextgen branch seeds.

Do not start M4 or the eigensolver while this validation gap is unresolved.
