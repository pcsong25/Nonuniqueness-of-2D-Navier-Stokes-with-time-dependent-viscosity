# AGENTS.md

## Project goal

Develop a new numerical codebase for computing the lowest eigenvalues and eigenfunctions
of the linearized operator around the forward self-similar profile of the 2D Euler equation with
time-dependent viscosity.

The new code should:
1. use a direct sparse-matrix-based finite-difference method,
2. follow the same mathematics and core logic as the existing verified code and project notes,
3. produce the same main outputs as the baseline workflow,
4. be tuned to outperform the previous implementation in efficiency whenever possible,
5. remain easy to validate and debug.

A square computational domain is acceptable if it is numerically and implementation-wise
more convenient than a disk.

## Authoritative sources

When reconstructing the intended mathematics or workflow, use the following priority:

1. `src/baseline/driver.py`
2. the actually imported/called files used by that workflow
3. the project notes in `docs/` (especially the LaTeX source if present)
4. the PDF only when the LaTeX is ambiguous or incomplete

Ignore commented-out code in `src/baseline/driver.py` unless explicitly told otherwise.

## Important repository rules

- `src/baseline/` is a copied baseline, not a sacred archive.
- You may modify files in `src/baseline/` when necessary to make the old workflow runnable
  inside this repository, especially for path fixes and repo-local compatibility.
- Do not do conceptual refactors in `src/baseline/` unless explicitly necessary.
- All genuinely new implementation work should go under `src/nextgen/`.
- Keep the new code modular and easy to benchmark against the baseline.

## Required first step

Before implementing the new solver, do an intake pass.

That intake must:
1. inspect `src/baseline/driver.py`,
2. determine which baseline files are actually used by the main workflow,
3. identify missing path fixes or repo-compatibility issues,
4. summarize the current mathematical and computational workflow,
5. propose a clean architecture for `src/nextgen/`,
6. propose a validation and benchmarking plan.

Do not jump directly into a large rewrite without this intake.

## Baseline and new-code expectations

The new code should ultimately accept the same class of boundary/input data as the baseline
workflow and should output, at minimum:

- the vortex/profile solution,
- the first `k` eigenfunctions (defaulted to 6),
- the corresponding eigenvalues,
- meaningful error/diagnostic measures,
- preliminary visualizations for debugging.

The debug visualizations do not need to be publication-quality yet.

## Validation requirements

Any numerical claim should be supported by at least one of the following:

- comparison with the baseline code on the same input,
- convergence/refinement studies,
- truncation/domain-size sensitivity checks,
- consistency checks of residuals and reconstructed equations,
- agreement between independently implemented components when possible.

Residuals alone are not sufficient validation.

When comparing new vs baseline code, keep track of:
- runtime,
- memory use when practical,
- discretization parameters,
- domain/truncation parameters,
- solver tolerances,
- numerical outputs and discrepancies.

## Performance objective

“Outperform” should be interpreted as improving one or both of:
- runtime for comparable accuracy,
- accuracy for comparable runtime.

When there is a tradeoff, document it clearly rather than hiding it.

## Documentation and experiment logging

Maintain these files as the project evolves:

- `docs/progress_log.md`
- `experiments/registry.csv`

For every meaningful experiment, add a row to `experiments/registry.csv`.

For every major code or validation milestone, update `docs/progress_log.md` with:
- what changed,
- why it changed,
- what was tested,
- what remains uncertain,
- recommended next steps.

## Done criteria for a major task

A major implementation task is not done until:
1. the code runs,
2. the relevant checks/tests have been executed,
3. the outputs are summarized,
4. the experiment log is updated,
5. the progress log is updated.

## Communication style

Be concise, explicit, and technically rigorous.
Separate facts, hypotheses, and open questions.
When something is ambiguous, state the ambiguity and propose a concrete resolution.

## Milestone discipline

Work in narrow milestones. Do not expand scope prematurely.

For this project, the intended implementation order is:
1. foundations (grid, operators, boundary handling, diagnostics),
2. Poisson solver,
3. profile solver,
4. linearized operator assembly,
5. eigensolver,
6. benchmarking and visualization refinement.

Unless explicitly instructed otherwise, only work on one milestone at a time.

## Session and context discipline

Do not rely on prior chat history as the only source of project state.

At the start of any substantial task:
1. read `AGENTS.md`,
2. read `docs/progress_log.md`,
3. read `docs/implementation_plan.md`,
4. inspect recent relevant files in `src/baseline/`, `src/nextgen/`, `tests/`, and `experiments/` as needed.

When a session may have been interrupted or context may be incomplete, first reconstruct state from the repository before making further changes.

## Progress bookkeeping

After every substantial milestone or validation step:
- update `docs/progress_log.md`,
- update `docs/implementation_plan.md`,
- append meaningful runs to `experiments/registry.csv`.

## Response footer requirement

At the end of every substantial response, provide:
1. a brief milestone summary,
2. the main unresolved risks or uncertainties,
3. the exact recommended next prompt.

The recommended next prompt must be narrowly scoped and should not jump ahead by more than one milestone.

## Prompt economy

Prefer short prompts that reference repo files and existing documentation rather than repeating long project background in chat.

When the task depends on project conventions or earlier work, rely on:
- `AGENTS.md`
- `docs/progress_log.md`
- `docs/implementation_plan.md`
- `experiments/registry.csv`

Keep conversational restatements brief unless the user explicitly asks for a longer explanation.

## Decision tracking

Maintain `docs/decision_log.md`.

Whenever a major design or workflow decision is made, update that file with:
- the decision,
- the reason,
- relevant evidence,
- and any follow-up consequences.

For major milestones, the repository should be understandable from:
- `AGENTS.md`
- `docs/progress_log.md`
- `docs/implementation_plan.md`
- `docs/decision_log.md`
- `experiments/registry.csv`

Do not rely on chat history as the only source of project state.