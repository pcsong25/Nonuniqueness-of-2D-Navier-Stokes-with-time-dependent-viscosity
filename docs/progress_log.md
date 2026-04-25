# Progress log

## Current objective

Build and validate a new sparse-matrix finite-difference implementation for the self-similar
2D Euler time-dependent-viscosity eigenvalue workflow, while preserving the logic and outputs
of the verified baseline implementation.

## Current status

Initial repository imported.
Need intake of baseline workflow and project notes.

## Open questions

- Exact performance target for “outperform”
- Default value(s) of `k`
- Allowed extra dependencies
- Boundary-data representation on square domain
- Baseline authoritative run command

## Latest completed step

- Repository created
- Baseline code and project documentation imported

## Next requested step

- Inspect `src/baseline/driver.py`
- Determine actually used files
- Fix repo-local path issues if needed
- Summarize old workflow
- Propose `src/nextgen/` architecture
- Propose validation/benchmark plan