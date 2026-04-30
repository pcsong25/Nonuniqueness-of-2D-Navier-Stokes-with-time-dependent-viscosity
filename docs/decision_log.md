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

### 2026-04-28 — Resolve target mixed case by refining the upwind polar grid
Decision:
Keep upwind full-disk polar FD as the M3 production profile path for mixed Fourier data, and use
finer grids for moderate/large mixed cases rather than switching to centered, blended, square, or
smooth-stencil variants.

Why:
The target mixed case failure at `54/162` was a resolution/continuation threshold. The same
upwind discretization reaches sigma `1` on `72/216` and produces smooth baseline-like plots.
Centered/blended and square FD alternatives converge algebraically but produce visible ringing
and poor common sampled strong residuals.

Evidence:
For `{c_2=6+3i,c_5=-5+i,c_8=2-1.5i}`, polar upwind `nr=72`, `ntheta=216` completed in
`13.227 s`, native relative residual `1.25e-10`, common sampled strong residual `0.421`, Omega
probe RMS `0.1065`, and offset-removed psi probe RMS `0.0101`. The baseline `h_min=0.12`,
`h_max=0.18` completed in `14.964 s` with common sampled strong residual `0.964`. Polar blended
`54/162` and square FD `n=97` had common sampled strong residuals `20.36` and `8.03`,
respectively, and their plots were visibly oscillatory.

Consequences:
Native algebraic residuals are not sufficient for accepting profile solves; visual checks and the
common sampled strong residual remain mandatory. The next M3 step should run a broader fine-grid
acceptance suite before M4, but the target mixed blocker no longer requires changing numerical
frameworks.

### 2026-04-28 — Fine-grid upwind acceptance suite passes tested M3 cases
Decision:
Accept the full-disk polar upwind solver on `nr=72`, `ntheta=216` as the current M3 profile
generation path for the tested active c3 and moderate mixed boundary data. Keep the square,
centered, blended, and smooth-upwind variants diagnostic only.

Why:
The broader acceptance suite crossed sigma `1` for the active `c_3=60,c_4=0` case, the target
mixed case `{c_2=6+3i,c_5=-5+i,c_8=2-1.5i}`, and a larger `1.5x` version of that mixed case.
The plots are smooth and baseline-like, and the common sampled strong residuals are comparable to
or better than the baseline on the same domain.

Evidence:
- `c_3=60,c_4=0`: nextgen `33.274 s`, native relative residual `2.18e-11`, common sampled
  strong residual `0.544`; baseline `53.414 s`, common sampled strong residual `0.537`.
- Target mixed `c_2/c_5/c_8`: nextgen `13.491 s`, native relative residual `1.25e-10`, common
  sampled strong residual `0.421`; baseline `15.032 s`, common sampled strong residual `0.964`.
- Larger `1.5x` mixed `c_2/c_5/c_8`: nextgen `16.855 s`, native relative residual `5.06e-10`,
  common sampled strong residual `0.462`; baseline `15.756 s`, common sampled strong residual
  `1.104`.

Consequences:
This clears the current M3 acceptance gate for the tested fine-grid cases, but it does not prove
robustness for arbitrary `60`- to `100+`-amplitude mixed Fourier boundary data. Before claiming a
general high-amplitude profile generator, run a separate stress/refinement suite with larger
mixed coefficients and, where runtime permits, one finer polar grid or domain-sensitivity check.

### 2026-04-28 — High-amplitude mixed stress keeps M3 open
Decision:
Do not move to M4 yet if the target is robust profile generation for large mixed Fourier data.
The current full-disk polar upwind Newton path is accepted for the moderate cases above, but not
for `60`-scale mixed coefficients.

Why:
For `{c_2=60,c_5=-5+i,c_8=2-1.5i}`, nextgen `nr=72`, `ntheta=216` did not reach sigma `1`.
It accepted sigma `0.45` with native relative residual `7.59e-09`, then failed the line search at
attempted sigma `0.4515625` after `248.238 s`. The requested `96/288` refinement was skipped
after that coarser failure because the direct sparse Newton path was not operationally feasible.

Evidence:
- Nextgen partial accepted state at sigma `0.45`: accepted-state sampled strong residual `0.468`,
  runtime `248.238 s`, `436` Newton iterations, physical Poisson residual `2.78e-02`.
- Fine baseline `h_min=0.08`, `h_max=0.12`: completed sigma `1` in `105.068 s`, common sampled
  strong residual `0.651`, and smooth Omega/psi plots.
- Coarse baseline `h_min=0.12`, `h_max=0.18`: reported completion but produced a bad branch with
  common sampled strong residual `5.94e3` and visibly noisy Omega, so coarse baseline comparisons
  are unreliable at this amplitude.

Consequences:
The M3 solver needs one more numerical-method pass before it can support the high-amplitude
profiles required for eigenvalue searches. Likely directions are a continuation method that can
cross the high-amplitude branch point more cheaply, a pseudo-transient or defect-correction
globalization, or a finite-volume/monotone discretization that preserves the smooth branch without
requiring many direct Newton factorizations.

### 2026-04-28 — Pseudo-transient Newton fallback is not enough for high amplitude
Decision:
Keep the pseudo-transient shifted-Newton fallback in `src/nextgen/polar_profile.py` as a
diagnostic/globalization option, but do not mark M3 high-amplitude mixed profile generation as
solved.

Why:
The fallback directly targets the observed line-search barrier by trying shifted Omega-block
systems `(J + beta M_Omega) delta = -F` after an exact Newton direction fails. This made the
failure mode less abrupt but did not move the accepted high-amplitude continuation threshold.

Evidence:
For `{c_2=60,c_5=-5+i,c_8=2-1.5i}` on `nr=72`, `ntheta=216`, pseudo-transient polar upwind
again stopped at sigma `0.45` and failed at attempted sigma `0.4515625`. Runtime increased from
`248.238 s` to `472.999 s`, Newton iteration records increased from `436` to `647`, and the
accepted partial-state common sampled strong residual remained `0.468`. The fine baseline
`h_min=0.08`, `h_max=0.12` still completes sigma `1` in about `106.815 s` with common sampled
strong residual `0.651`.

Consequences:
The next useful M3 move should not be another scalar Newton damping variant. It should change the
discrete nonlinear path, for example a conservative finite-volume/monotone transport form,
pseudo-arclength continuation, or a baseline-like Galerkin/SUPG finite-difference analogue with
explicit rejection criteria for the low-amplitude centered branch.

### 2026-04-28 — Conservative FV transport improves but does not close M3
Decision:
Add `fv_upwind` as an M3 polar transport option, but keep high-amplitude mixed profile generation
open until it reaches the refined/fine grids needed for production.

Why:
The conservative finite-volume form changes the nonlinear discrete path more substantially than
Newton damping. It evaluates the same continuous advection term as
`div(v Omega) - Omega div(v)` and uses monotone radial/angular face fluxes.

Evidence:
For `{c_2=60,c_5=-5+i,c_8=2-1.5i}`, `fv_upwind` on `nr=30`, `ntheta=90` reached sigma `1` in
`68.953 s`, with native relative residual `9.47e-09`, common sampled strong residual `0.738`,
Omega probe RMS `0.602`, and offset-removed psi probe RMS `0.0397` versus the fine baseline.
The fine baseline `h_min=0.08`, `h_max=0.12` completed in `108.811 s` with common sampled strong
residual `0.651`. The refinement `nr=36`, `ntheta=108` improved far beyond the old upwind sigma
`0.45` barrier but still stalled at sigma `0.933203125` after `354.832 s`; its accepted partial
state common sampled strong residual was `0.668`.

Consequences:
FV upwind is the best high-amplitude direction so far and should be the next M3 target, but it is
not yet a fine-grid production solver. The immediate blocker is refinement robustness/runtime,
possibly caused by using only an approximate psi-coupling Jacobian for the conservative fluxes.

### 2026-04-29 — Corrected FV psi Jacobian closes M3 profile acceptance
Decision:
Mark M3 profile generation complete for the documented acceptance envelope and move the next
milestone recommendation to M4 operator assembly.

Why:
The high-amplitude mixed blocker was caused by the `fv_upwind` Newton Jacobian using the old
centered strong-form omega-psi derivative while the residual used conservative active-upwind
finite-volume face fluxes. Differentiating the active FV face fluxes gives Newton the correct
psi-coupling block and removes the sigma `0.933` refinement stall.

Alternatives considered:
- Add pseudo-arclength continuation before fixing the mismatched Jacobian.
- Accept the direct `72/216` nextgen run using only the recorded baseline residual.
- Keep M3 open for arbitrary `100+`-scale Fourier stress cases.

Evidence:
The `fv_upwind` finite-difference Jacobian check improved from about `4.48e-02` omega-block
relative error to about `1.18e-08`. The full high-amplitude stress suite for
`{c_2=60,c_5=-5+i,c_8=2-1.5i}` completed in the `femx_spc` environment:
nextgen `30/90`, `36/108`, and `72/216` all reached sigma `1`; the `72/216` run had native
relative residual `8.43e-10`, common sampled strong residual `0.482`, Omega probe RMS `0.559`,
and offset-removed psi probe RMS `0.0202` versus the live fine baseline. The fine baseline
`h_min=0.08,h_max=0.12` completed in `107.966 s` with common sampled strong residual `0.651`.

Consequences:
M4 should build the linearized operator around the accepted full-disk polar profile path and
explicitly reuse/document the same `fv_upwind` conservative transport conventions where relevant.
The eigensolver remains M5 and should not start until M4 operator consistency checks pass.

### 2026-04-29 — c4/c7 high-amplitude spot check does not expand M3 envelope
Decision:
Do not expand the M3 validated profile envelope to arbitrary `100`-scale Fourier boundary data
based on the `{c_4=100,c_7=40}` spot check.

Why:
The corrected `fv_upwind` Newton path converges this case to sigma `1`, but the comparison with
the refined baseline is not close enough to treat it as production-grade M3 accuracy. The direct
sparse solve cost also prevented completing larger nextgen probes in this session.

Evidence:
Nextgen `64/256` and `72/288` completed with native relative residuals `1.785e-10` and
`8.221e-09`, and runtimes `226.561 s` and `251.747 s`. The refined baseline
`h_min=0.05,h_max=0.08` completed in `503.579 s` with common sampled strong residual `0.456`.
Nextgen `72/288` had common sampled strong residual `0.881`, Omega probe RMS `1.273`, and
offset-removed psi probe RMS `0.141` versus that fine baseline. By comparison, the baseline
mesh-refinement difference was Omega RMS `0.218` and offset-removed psi RMS `0.0175`. Nextgen
probes at `80/320` and `96/384` were stopped after extended runtime without completion output.

Consequences:
M4 may proceed only if it consumes profiles inside the already accepted M3 envelope. If
`{c_4=100,c_7=40}` or similar arbitrary `100`-scale data is required for M4, keep work in M3 and
target either finer-grid performance or a baseline-closer discretization before operator assembly.

### 2026-04-29 — Reduced Poisson Newton improves cost but does not close large-boundary M3
Decision:
Keep the reduced Poisson Newton path as an M3 performance option, but do not declare general
large-boundary profile generation solved.

Why:
Eliminating the linear Poisson constraint reduces the Newton linear system and improves the
validated c4/c7 high-branch `72/288` runtime. However, the current Schur-complement GMRES
preconditioner is not strong enough for corrected high-branch `96/384`, and the field
discrepancy versus the refined baseline is unchanged at `72/288`.

Evidence:
The corrected reduced `72/288` c4/c7 run reached sigma `1` with native relative residual
`8.22e-09`, common sampled strong residual `0.881`, Omega L2 `17.754`, and runtime `201.333 s`
with `83.082 s` in reduced linear solves. A corrected monolithic check on the same branch and
grid reached the same field with runtime `395.715 s` and `279.212 s` in linear solves. Corrected
reduced `96/384` high-branch probes, including a looser GMRES tolerance `1e-6`, were stopped
after extended runtime without completion. The previous low-branch reduced `96/384` and
`128/512` results were invalidated by a face-orientation bug introduced during the FV
centered/blended refactor and are not acceptance evidence.

Consequences:
M3 should stay open if the next work requires arbitrary large boundary data such as
`{c_4=100,c_7=40}`. The next M3 target should be a stronger reduced-system preconditioner
(for example a better Schur approximation or multilevel/ILU strategy) or a baseline-closer
large-boundary discretization that still follows the high-amplitude branch.

### 2026-04-29 — Nested reduced refinement is the operational fine-grid path for c4/c7
Decision:
Use nested high-branch refinement as the current operational path for large c4/c7 nextgen
profile solves, but keep M3 open for accuracy against the baseline.

Why:
Starting reduced Newton from zero on corrected high-branch `96/384` is not practical with the
current Schur preconditioner. However, solving the high branch at `72/288`, prolonging that
state, and applying reduced Poisson Newton at sigma `1` makes `96/384` and `128/512`
operational. This addresses the immediate fine-grid reachability problem without changing the
accepted FV residual.

Evidence:
For `{c_4=100,c_7=40}`, final validation against a fresh `h_min=0.05,h_max=0.08` baseline
completed: `72/288` seed runtime `285.952 s`, nested reduced `96/384` runtime `32.771 s`, and
nested reduced `128/512` runtime `92.078 s`. All reached sigma `1`. The `128/512` common sampled
strong residual improved to `0.616` compared with `0.881` at `72/288`, while the fine baseline
was `0.456`. Probe errors remained large: `128/512` Omega probe RMS was `1.612` and
offset-removed psi probe RMS was `0.291` versus the fine baseline. Seeded `fv_blended` corrections
at `96/384` with blends `0.5`, `0.25`, and `0.10` had worse common residuals than upwind.

Consequences:
The next M3 target should focus on the remaining branch/discretization mismatch rather than only
making finer grids run. M4 should not consume arbitrary large-boundary profiles until this
accuracy gap is understood or a documented acceptance criterion is chosen.

### 2026-04-29 — c4/c7 mismatch is branch selection, not boundary/Poisson convention
Decision:
Keep M3 open for arbitrary large-boundary profiles and target a baseline-free branch-selection
strategy before allowing `{c_4=100,c_7=40}`-scale profiles into M4.

Why:
The structure diagnostic rules out the leading convention hypotheses. The analytic nextgen and
baseline Fourier boundary formulas agree to roundoff, the baseline boundary dofs match the same
Omega data, and the nextgen Poisson solve produces the baseline streamfunction accurately when fed
the fine-baseline Omega. However, nextgen `128/512` has at least two FV profile branches: the
nested zero-continuation branch and a baseline-close branch reached when reduced Newton is seeded
from the fine baseline.

Evidence:
`experiments/run_m3_c4_c7_structure_diagnostics.py` completed the nested `72/288 -> 96/384 ->
128/512` path, a fresh `h_min=0.05,h_max=0.08` baseline, and a baseline-seeded `128/512` reduced
FV correction. Boundary convention errors were roundoff-size: Omega Dirichlet max `2.22e-16`, psi
Neumann max `5.55e-17`, and baseline boundary dof max `5.55e-16`. Nextgen Poisson from
fine-baseline Omega gave offset-removed psi RMS `2.21e-03`. The nested `128/512` branch had common
sampled strong residual `0.616`, Omega RMS `1.476`, offset-removed psi RMS `0.289`, Omega integral
defect `2.477`, and Poisson multiplier `387.6`. The baseline-seeded `128/512` branch converged in
`56.291 s` with common sampled strong residual `0.433`, Omega RMS `0.651`, offset-removed psi RMS
`0.022`, Omega integral defect `-0.154`, and Poisson multiplier `-24.0`. The radial error plots
show the nested-branch discrepancy is concentrated in the inner core, not at the outer boundary.

Consequences:
The next M3 work should not spend effort on Fourier sign/scale changes or a Poisson gauge rewrite.
It should implement a way to reach the baseline-close branch without using the baseline as an
oracle, such as anchored or pseudo-arclength continuation, an explicit circulation/compatibility
constraint, or a coarse Galerkin/less-dissipative predictor followed by reduced FV correction.
M4 remains pending for arbitrary large-boundary profiles.

### 2026-04-29 — c3=140 is the cleaner M3 nonuniqueness target
Decision:
Treat `{c_3=140}` as the primary near-term M3 nonuniqueness/symmetry-breaking investigation.
Do not reinterpret the broken-symmetry roots as solver failures unless refinement or independent
discretizations invalidate them.

Why:
The c4/c7 case showed path-dependent branches, but one branch had a larger Poisson compatibility
defect. The c3=140 experiment is cleaner: the boundary data has exact threefold symmetry, the
zero-continuation solution keeps that symmetry, and several deliberately symmetry-breaking
initial guesses converge to distinct non-threefold roots with the same boundary data and
roundoff-level Poisson compatibility.

Evidence:
`experiments/run_m3_c3_140_symmetry_probe.py` constructed the symmetric branch at `72/216`,
`96/288`, and `128/384`, then ran fixed-sigma reduced Newton from five non-threefold
perturbations. The `128/384` symmetric branch had native residual `2.16e-13`, common sampled
strong residual `0.598`, Omega non-threefold fraction `5.35e-14`, and Poisson compatibility
mismatch about `6e-14`. A small m1 perturbation and an m4 perturbation returned to the symmetric
branch. Larger m1, m2, and mixed non-threefold perturbations converged to symmetry-breaking
candidates with Omega non-threefold fractions about `0.168`, Omega RMS distance from the
symmetric branch about `0.293`, local Omega Linf separation about `8.3`, and common sampled
strong residuals `0.586` to `0.602`. Their Omega integral/Neumann-flux mismatches were around
`1e-14` to `1e-12`, with physical Poisson residual norms around `5e-12`. A quick orbit check
showed the m1 and mixed perturbations reach the same broken branch to numerical precision, while
the m2 candidate is not identical under the tested C3 rotations/reflections.

Consequences:
The next M3 work should verify persistence, not remove the effect. Refine the symmetric and
broken branches to at least `160/480`, map the Newton basin boundary by interpolating between
branches, and trace both branches in amplitude. If the broken branch survives refinement and
independent discretization/FEM checks, it becomes strong numerical evidence for nonuniqueness of
profiles with symmetric boundary data.

### 2026-04-29 — c3=140 survives 160/480 as strong discrete nonuniqueness evidence
Decision:
Treat the `{c_3=140}` symmetric and symmetry-breaking nextgen FV roots as genuine discrete branch
candidates. Keep M3 focused on continuum/refinement and independent-discretization validation
before making a final continuum nonuniqueness claim.

Why:
The `160/480` refinement keeps both the symmetric branch and the m1 broken branch at native
residuals near roundoff and Poisson compatibility near roundoff, while the broken branch retains
order-one non-threefold structure. A fixed-sigma basin sweep and a downward amplitude trace show
organized branch behavior rather than an isolated Newton artifact.

Evidence:
`experiments/run_m3_c3_140_nonuniqueness_validation.py` refined the saved `128/384` states to
`160/480`. The symmetric branch converged in `22.798 s` with native residual `3.46e-13`, common
sampled residual `0.602`, Omega non-threefold fraction `6.88e-14`, and physical Poisson residual
`8.39e-12`. The m1 broken branch converged in `47.573 s` with native residual `3.55e-13`, common
sampled residual `0.588`, Omega non-threefold fraction `0.252`, and physical Poisson residual
`8.40e-12`. The m2 broken-like branch also retained Omega non-threefold fraction `0.253` with
native residual `3.74e-09`. Basin interpolation from symmetric to m1 broken converged to the
symmetric branch for `t <= 0.4` and the broken branch for `t >= 0.5`. The m1 broken amplitude
trace stayed non-threefold through sigma `0.70` and collapsed to an essentially symmetric state
between sigma `0.70` and `0.65`.

FEM caveat:
The `h_min=0.08,h_max=0.12` baseline/FEM checks did not provide independent validation. The
zero-continuation baseline run took `1317.256 s` but had common sampled residual `6.27e4`; the
nextgen-seeded FEM fixed-sigma run was nonconverged with last relative update `1.033` and common
sampled residual `8.38e3`. These are failed stress checks, not counterevidence against the FV
branches.

Consequences:
The next M3 step should push the branches to `192/576` or finer, sharpen the apparent bifurcation
window near sigma `0.65` to `0.70`, and improve the FEM/baseline continuation so it converges to
a credible solution for the same boundary data. M4 and the eigensolver remain blocked for this
nonuniqueness investigation.
