# Decision log

This file records important project decisions so that work can be resumed from the repository
without relying on chat history alone.

## How to use this file

For every major milestone or meaningful design change, record:
- the decision,
- why it was made,
- what alternatives were considered,
- what evidence supported the choice,
- what consequences or follow-ups it creates.

Keep entries short but specific.

---

## Initial project decisions

### 2026-04-26 — Repository-centered workflow
Decision:
Use repository files (`AGENTS.md`, `docs/progress_log.md`, `docs/implementation_plan.md`,
`docs/decision_log.md`, `experiments/registry.csv`) as the primary long-term project memory.

Why:
Chat sessions may become long, interrupted, or split across threads. Important project state
should remain recoverable from the working tree.

Alternatives considered:
- Rely mainly on one long Codex conversation
- Keep only minimal documentation

Consequences:
All major milestones should update these files before completion.

### 2026-04-26 — Milestone-based implementation
Decision:
Implement the new code in narrow milestones:
foundations -> Poisson solver -> profile solver -> linearized operator assembly ->
eigensolver -> benchmarking/output refinement.

Why:
This reduces risk, keeps diffs reviewable, and makes validation easier.

Alternatives considered:
- Large end-to-end rewrite
- Simultaneous implementation of profile and eigen solver

Consequences:
Each milestone should have explicit validation before moving forward.

### 2026-04-26 — New implementation location
Decision:
Place all genuinely new implementation work in `src/nextgen/`.

Why:
This keeps the baseline workflow available for comparison and debugging.

Alternatives considered:
- Rewrite directly in `src/baseline/`
- Interleave baseline and nextgen code

Consequences:
Baseline may receive minimal compatibility/path fixes, but new architecture belongs in `src/nextgen/`.

### 2026-04-26 — Domain flexibility
Decision:
Allow a square computational domain in the nextgen implementation if it improves simplicity
or efficiency, while preserving the same underlying mathematics and target outputs.

Why:
The new sparse finite-difference code may be more natural on a square grid than on a disk.

Alternatives considered:
- Force the nextgen code to remain on a disk domain

Open questions:
- How boundary data and far-field conventions should be represented most cleanly on the square
- How to benchmark square-domain outputs against the baseline disk-domain implementation

### 2026-04-26 — M3 profile residual and Jacobian form
Decision:
Implement the nextgen profile solver as a monolithic sparse finite-difference Newton solve on
`(Omega_interior, psi_full)`.

Why:
This preserves the baseline manifold-Newton structure while reusing the M1/M2 square-grid
operators and gauged Neumann Poisson matrix directly. The Omega equation is assembled in
strong form as `v . grad(Omega) - Omega - eps * Delta(Omega)`, with boundary Omega values
eliminated into the interior derivative stencils.

Alternatives considered:
- Re-solve Poisson as a nested operation inside each residual evaluation
- Implement only Picard iteration before Newton
- Start the eigen/linearized operator work in the same milestone

Evidence:
Jacobian finite-difference tests pass on a small grid, and the reduced `c_3=6`, `c_4=5`
profile run converged with final nextgen relative residual `1.505e-13`.

Consequences:
The M4 linearized-operator assembly should reuse the same discrete derivative, Poisson,
and sign conventions instead of re-deriving separate stencils.

### 2026-04-26 — Historical active baseline coefficient correction
Decision:
At this point in the M3 work, treat `src/baseline/driver.py` as using both
`c_3=60+0j` and `c_4=50+0j`, with `h_min=0.05` and `h_max=0.10`.

Why:
Direct inspection of the authoritative driver showed the earlier m4-only notes are stale.

Alternatives considered:
- Continue using the historical m4-only baseline reference as the current default
- Revert the baseline driver to match the old documentation

Evidence:
The active driver coefficient dictionary includes modes `3: 60.0+0j` and `4: 50.0+0j`.

Consequences:
Nextgen defaults were updated to the current active shape. Existing m4-only run records remain
historical validation data but should not be described as the current driver state.
This entry was superseded later on 2026-04-26 when the active driver was corrected to
`c_3=60+0j`, `c_4=0+0j`.

### 2026-04-26 — M3 reduced square-vs-disk validation
Decision:
Validate M3 with a reduced profile-only comparison using the active boundary shape scaled by
`0.1`, nextgen on a square `n=65` grid, and baseline on a coarse disk mesh.

Why:
This gives a feasible same-shape workflow comparison without entering eigensolver scope.

Alternatives considered:
- Run the full active-amplitude baseline case
- Compare only residuals without a baseline run
- Delay baseline comparison until after eigensolver work

Evidence:
The profile-only comparison completed in `experiments/run_m3_profile_comparison.py`.
Nextgen reached relative residual `1.505e-13`; the coarse baseline profile run completed with
raw weak residuals `||R_Omega||_2=1.166` and `||R_Psi||_2=4.510e-13`.

Consequences:
The measured probe discrepancies are preliminary because square-vs-disk domain shape,
coarse baseline mesh, and first-order square Neumann closure are all active error sources.

### 2026-04-26 — C3-only profile validation blocks M4
Decision:
Do not move to M4 until the c3-only profile mismatch is understood.

Why:
The user corrected the active baseline input to `c_3=60+0j`, `c_4=0+0j` and requested
visual/residual validation before moving forward. The nextgen solver reached a tiny algebraic
residual on its square-grid equations, but the vorticity heat map and streamfunction contours
do not qualitatively match the baseline disk solve at comparable practical resolution.

Alternatives considered:
- Treat low nextgen residual alone as sufficient and start linearized-operator assembly
- Attribute all discrepancy to square-vs-disk geometry without further checks

Evidence:
For `experiments/run_m3_c3_profile_visual_check.py`, nextgen `n=129` reached relative residual
`7.944e-14` in `43.373 s`; the baseline disk profile solve completed in `10.000 s` with raw
weak residuals `||R_Omega||_2=1.541e-01` and `||R_Psi||_2=3.551e-12`. Probe discrepancies on
radius `<=8` were large: Omega RMS `8.017e-01`, psi RMS after gauge-offset removal `1.436`.
On common sampled strong residuals over `[-7,7]^2`, nextgen relative residual was `1.638e+01`
and baseline relative residual was `1.118e+00`. The nextgen Omega threefold relative RMS defect
was `1.416`.

Consequences:
M3 is implemented but not validated as a baseline replacement. The next task should investigate
profile residual/sign conventions, square boundary treatment, and disk-vs-square effects before
any M4 work.

### 2026-04-26 — Square domain is not acceptable for c3 profile validation
Decision:
The nextgen profile validation for c3 data should move away from the full square boundary
truncation before starting M4.

Why:
The c3 whole-space problem should have threefold symmetry. A square computational domain is
not invariant under 120-degree rotation, so the finite square-domain boundary value problem
is not the same symmetry class as the disk baseline problem. Applying radially scaled far-field
vorticity data on the square boundary fixed an inconsistency with the streamfunction Neumann
scaling, but it did not restore c3 symmetry.

Alternatives considered:
- Keep the square domain and rely on larger `Rdom`
- Use low native algebraic residual as the validation criterion
- Proceed to M4 while documenting the mismatch

Evidence:
Small nextgen `n=65` symmetry probes showed `c_3` Omega threefold relative RMS defect `1.417`.
The same solver/grid with `c_4` data had fourfold relative RMS defect `2.05e-15`, matching the
square grid symmetry. This isolates the issue to the square-domain symmetry class rather than
the Newton residual alone.

Consequences:
The next implementation step should be a disk/rotationally compatible profile discretization
or a circular truncation strategy for the FD solver. M4 and eigensolver work remain blocked.

### 2026-04-26 — Diagnostic polar c3-sector experiment
Decision:
Add a circular finite-difference profile solver in `src/nextgen/polar_profile.py`. The initial
diagnostic experiment used a `2*pi/3` periodic polar sector for the active `c_3=60`, `c_4=0`
case.

Why:
The square truncation cannot preserve threefold symmetry. A full polar disk with arbitrary
gauge-row replacement can also introduce avoidable asymmetry, while a c3 periodic sector
enforces the physical threefold class directly. The centered polar advection stencil converged
to a bounded radial profile with poor common strong residual, so the Omega advection term now
uses first-order upwind derivatives while retaining the same strong-form PDE and centered
diffusion.

Alternatives considered:
- Keep the square solver and document the c3 mismatch
- Use centered polar advection because it is closer to the original algebraic square M3 form
- Move directly to M4 with the square profile

Evidence:
The latest c3 profile-only comparison produced nextgen native relative residual `2.245e-14`,
common sampled strong residual `0.589`, and threefold Omega/psi relative RMS defects
`4.842e-15` / `1.419e-15`. The corresponding baseline run had common sampled strong residual
`1.118`; the square nextgen run had common sampled strong residual `16.38` and Omega symmetry
defect `1.416`. Probe discrepancies also improved to Omega RMS `0.302` and offset-removed psi
RMS `0.0381`.

Consequences:
This sector experiment is superseded by the 2026-04-27 full-disk decision below. It remains
useful debugging evidence but is not the production validation path because the solver must work
for arbitrary boundary data without a mesh customized to one symmetry.

### 2026-04-27 — Use a general full-disk polar FD profile solver for M3 validation
Decision:
Use the full polar disk (`symmetry=1`) as the M3 disk finite-difference profile validation path.
Keep n-fold symmetry only as a diagnostic when the boundary data happens to have such symmetry.

Why:
The project goal is a profile solver for arbitrary Fourier boundary data, not a solver tied to
one mode family. The mesh and gauge treatment therefore must not encode c3 symmetry. The polar
Poisson block now uses a bordered mean-gauge system rather than replacing one angular row, and
the validation scripts use full-disk grids.

Alternatives considered:
- Keep the c3-sector mesh for the active test because it is cheaper
- Keep the square-grid profile solver and document its c3 failure
- Use only symmetry-preserving cases as validation

Evidence:
The active full-amplitude `c_3=60`, `c_4=0` full-disk run completed with native relative
residual `2.282e-14`, common sampled strong residual `0.589`, Omega probe RMS `0.302`, and
offset-removed psi probe RMS `0.0381`. Runtime was `49.409 s` versus baseline `10.050 s`.
A reduced full-disk validation suite also completed for `c_4=5`, mixed complex `c_2/c_5` data,
and reduced mixed `c_3/c_4` data. In those cases, nextgen common sampled strong residuals were
`0.705`, `0.691`, and `0.740`, respectively, versus baseline sampled residuals `2.938`,
`2.093`, and `3.629` on the selected coarse grids.

Consequences:
The full-disk polar solver is the current M3 validation target. The square solver remains useful
for algebraic diagnostics, and the c3-sector experiment remains historical debugging evidence,
but neither should be treated as the production profile-validation path.

### 2026-04-27 — Full-disk polar refinement study keeps M3 open
Decision:
Keep M3 open after the first full-disk refinement study; do not start M4 yet.

Why:
The refinement study supports the full-disk polar formulation for general boundary data, but it
also shows that active-amplitude c3 continuation has a resolution/robustness threshold and that
the current nextgen implementation is slower than the baseline profile workflow.

Alternatives considered:
- Treat the lower common sampled strong residual on the completed nextgen grids as enough to
  declare M3 complete.
- Start linearized-operator assembly while leaving profile robustness as a later cleanup.

Evidence:
For the mixed non-symmetric reduced case `c_2=3+i`, `c_5=-2+0.5i`, nextgen grids
`30/90`, `32/96`, and `36/108` all completed and improved monotonically in common sampled
strong residual (`0.759`, `0.691`, `0.654`) and Omega probe RMS (`0.262`, `0.199`, `0.142`).
For active `c_3=60`, `c_4=0`, nextgen `48/144` stalled at sigma `0.7`, while `54/162`
completed with common sampled strong residual `0.650` versus the fine baseline value `0.835`.
The coarse active-c3 baseline mesh was unstable under the common strong residual metric, so fine
baseline meshes are required for meaningful comparison.

Consequences:
The common sampled strong residual remains the primary cross-code residual metric, supplemented
by probe RMS/Linf errors and runtime. Next M3 work should target Newton continuation robustness,
linear-solve performance, and perhaps better preconditioning/reuse before moving to M4.

### 2026-04-27 — Optimize polar upwind assembly before M4
Decision:
Precompute the polar upwind derivative sparsity patterns and use row-scaled sparse matrices
during residual/Jacobian assembly. Cache the bordered Poisson LU factor and use explicit SuperLU
factor objects for Newton linear solves.

Why:
Profiling by wall-clock experiments showed that direct Newton linear solves were not the only
cost. Rebuilding radial/angular upwind matrices with Python loops inside every residual and
Jacobian evaluation dominated runtime on the active full-disk grid.

Evidence:
On active `c_3=60`, `c_4=0`, nextgen `nr=54`, `ntheta=162` retained native relative residual
`2.317e-14` and common sampled strong residual `0.650`, while runtime dropped to `17.585 s`.
The finer baseline mesh `h_min=0.08`, `h_max=0.12` took `48.056 s` and reached common sampled
strong residual `0.637`; the baseline `0.12/0.18` mesh took `18.945 s` with common sampled
strong residual `0.835`.

Consequences:
M3 now has a credible profile-solver runtime/accuracy point for the active c3 case. The main
remaining robustness issue is the coarser `48/144` active grid: adaptive continuation reached
accepted sigma `0.98125` but not sigma `1`, and took `227.059 s`.

### 2026-04-28 — Do not accept M3 as general-purpose yet
Decision:
Keep M3 open. Do not start M4 until mixed-mode nonlinear continuation is more robust, or until
the accepted input envelope is explicitly narrowed.

Why:
The project goal for M3 is an efficient profile solver for arbitrary Fourier boundary values on
fine grids. The optimized full-disk polar solver is strong on active c3, single non-c3 modes,
and reduced mixed cases, but it still fails on moderate-amplitude mixed boundary data that the
baseline profile workflow can solve.

Evidence:
The general-boundary visual suite completed for `c_4=20`, phase-shifted `c_2=15+8i`, reduced
`c_1/c_4/c_7`, and reduced `c_2/c_5/c_8`, with plots and baseline comparisons recorded in
`experiments/summaries/m3_general_boundary_visual_suite_2026-04-28.json`. However, the moderate
mixed cases `{c_1=4-2i,c_4=8,c_7=-3+2i}` and `{c_2=6+3i,c_5=-5+i,c_8=2-1.5i}` stalled before
sigma `1`. A tighter continuation probe for the latter still stalled near sigma `0.658` after
`247.168 s`.

Consequences:
M3 remains the active milestone. The next work should target a more robust nonlinear solve path
for mixed modes, such as pseudo-transient/Newton continuation, better globalization, or a nested
Picard/Newton strategy, while keeping M4/eigensolver work out of scope.

### 2026-04-28 — Centered/blended continuation is diagnostic, not accepted accuracy
Decision:
Keep the new centered and blended advection paths as diagnostics/continuation experiments, but do
not treat them as proof that M3 is complete for general mixed Fourier data.

Why:
For the target moderate mixed case `{c_2=6+3i,c_5=-5+i,c_8=2-1.5i}`, pure centered advection and
a `1%` upwind blend both reached sigma `1` on `nr=54`, `ntheta=162`. However, the generated
Omega heat maps show radial ringing and the common sampled strong residual remains much worse
than the baseline. The upwind residual is visually smoother and better aligned with earlier
validation, but its scalar continuation path has a real fine-grid barrier near sigma `0.658`.

Evidence:
Upwind with `min_sigma_step=1e-5` stalled at accepted sigma `0.657702` after `125.996 s`.
Centered reached sigma `1` in `11.298 s` with native relative residual `4.76e-14`, but produced
ringing. A `1%` blend reached sigma `1` in `8.988 s` with native relative residual `3.14e-14`
and physical Poisson residual `3.91e-7`, but still had ringing and common sampled strong residual
about `20.36` on the target mixed case. The baseline profile solve for the same case completed
in about `15.0 s` with common sampled strong residual about `0.964`.

Consequences:
The next M3 step should focus on a discretization/globalization strategy that preserves the
smooth upwind-like profile quality while crossing the mixed-mode continuation barrier. Do not
move to M4 until this is resolved or the accepted boundary-data envelope is explicitly narrowed.
