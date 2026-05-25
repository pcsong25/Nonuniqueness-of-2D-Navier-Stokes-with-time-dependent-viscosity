# FEM Manuscript Experiments

This area is for manuscript-oriented experiments driven by the baseline FEM
code in `src/baseline/`.

The current scaffolds are dry-run only:

```bash
python experiments/fem_manuscript/c3_spectrum_continuation/run_spectrum_continuation.py
python experiments/fem_manuscript/symmetry_breaking_profiles/run_profile_comparison.py
```

They write JSON manifests under their `logs/` directories and do not launch
FEM solves unless a future implementation wires `--execute`.
