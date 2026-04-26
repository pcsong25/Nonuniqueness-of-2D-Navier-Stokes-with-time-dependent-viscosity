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