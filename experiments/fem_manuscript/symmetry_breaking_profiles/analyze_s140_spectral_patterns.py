#!/usr/bin/env python3
"""Summarize spectral patterns for s=140 symmetric and broken profiles."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import defaultdict
import sys


BRANCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BRANCH_DIR.parents[2]
MANUSCRIPT_ROOT = BRANCH_DIR.parents[0]
sys.path.insert(0, str(MANUSCRIPT_ROOT))

from common import append_registry_row, today  # noqa: E402
SYM_METADATA = (
    BRANCH_DIR.parents[0]
    / "c3_spectrum_continuation/checkpoints/branchA_h006_refine_s125_140_k10/s140_h0p06_0p06_metadata.json"
)
SPECTRA_CSV = BRANCH_DIR / "data/branchB_s140_broken_profile_spectra_eigenvalues.csv"
MEMBER_CSV = BRANCH_DIR / "data/branchB_s140_c3_orbit_classes_with_spectra_class_members_spectra.csv"
OUT_MATCH = BRANCH_DIR / "data/branchB_s140_spectral_matches_to_symmetric.csv"
OUT_SUMMARY = BRANCH_DIR / "data/branchB_s140_spectral_pattern_summary.csv"


REPRESENTATIVES = [
    ("symmetric", "symmetric", None),
    ("class_1_eigenmode1", "eigenmode1", "broken_eigenmode_mode1_eta0p5_signp1"),
    ("class_2_eigenmode3_minus", "eigenmode3_minus", "broken_eigenmode_mode3_eta0p5_signm1"),
    ("class_3_eigenmode3_plus", "eigenmode3_plus", "broken_eigenmode_mode3_eta0p5_signp1"),
    ("class_4_m2", "analytic_m2", "analytic_m2_phase0_eta0p25"),
    ("class_5_m1", "artificial_m1", "broken_artificial_m1_phase0_eta0p25"),
]


def load_symmetric() -> list[complex]:
    data = json.loads(SYM_METADATA.read_text())
    return [complex(float(ev["real"]), float(ev["imag"])) for ev in data["eigs"]["eigenvalues"][:10]]


def load_spectra() -> dict[str, list[complex]]:
    by: dict[str, list[tuple[int, complex]]] = defaultdict(list)
    with SPECTRA_CSV.open() as f:
        for row in csv.DictReader(f):
            if row.get("status") != "completed":
                continue
            by[row["profile_id"]].append(
                (int(row["eigen_index"]), complex(float(row["real_lambda"]), float(row["imag_lambda"])))
            )
    return {key: [z for _, z in sorted(vals)] for key, vals in by.items()}


def fmt(z: complex) -> str:
    return f"{z.real:.12g}{z.imag:+.12g}i"


def main() -> int:
    sym = load_symmetric()
    spectra = load_spectra()
    summary_rows = []
    match_rows = []

    all_spectra = {"symmetric": sym}
    for key, _, spectrum_id in REPRESENTATIVES:
        if spectrum_id is not None:
            all_spectra[key] = spectra[spectrum_id]

    for key, label, spectrum_id in REPRESENTATIVES:
        vals = all_spectra[key]
        summary_rows.append(
            {
                "representative_key": key,
                "profile_label": label,
                "spectrum_id": spectrum_id or "symmetric_metadata",
                "min_real_lambda": min(z.real for z in vals),
                "num_negative_real_first10": sum(1 for z in vals if z.real < 0.0),
                "num_real_first10": sum(1 for z in vals if abs(z.imag) < 1e-9),
                "num_complex_first10": sum(1 for z in vals if abs(z.imag) >= 1e-9),
                "first_three_eigenvalues": "; ".join(fmt(z) for z in vals[:3]),
                "full_first10": "; ".join(fmt(z) for z in vals),
            }
        )

        if key == "symmetric":
            continue
        for j, z in enumerate(vals, start=1):
            dist, sym_idx, sym_z = min((abs(z - w), i, w) for i, w in enumerate(sym, start=1))
            match_rows.append(
                {
                    "representative_key": key,
                    "profile_label": label,
                    "eigen_index": j,
                    "real_lambda": z.real,
                    "imag_lambda": z.imag,
                    "nearest_symmetric_index": sym_idx,
                    "nearest_symmetric_real": sym_z.real,
                    "nearest_symmetric_imag": sym_z.imag,
                    "complex_plane_distance": dist,
                }
            )

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with OUT_SUMMARY.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with OUT_MATCH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(match_rows[0]))
        writer.writeheader()
        writer.writerows(match_rows)
    append_registry_row(
        {
            "date": today(),
            "tag": "fem-branchB-s140-spectral-pattern-analysis",
            "codebase": "baseline",
            "script": str(Path(__file__).relative_to(REPO_ROOT)),
            "input_case": "s_cos3theta_s140",
            "domain": "disk_Rdom10",
            "grid_or_mesh": "h=0.06/0.06",
            "solver": "postprocess_spectral_pattern_matching",
            "parameters": "representatives=symmetric,C1,C2,C3,C4,C5; nearest_complex_match_to_symmetric",
            "wall_time_sec": "n/a",
            "error_metrics": f"summary={OUT_SUMMARY.relative_to(REPO_ROOT)}",
            "eigenvalues_summary": f"matches={OUT_MATCH.relative_to(REPO_ROOT)}",
            "status": "completed",
            "notes": "Compared first ten eigenvalues of s=140 symmetric and C3 orbit-class representative broken profiles.",
        }
    )

    print(f"[done] {OUT_SUMMARY}")
    print(f"[done] {OUT_MATCH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
