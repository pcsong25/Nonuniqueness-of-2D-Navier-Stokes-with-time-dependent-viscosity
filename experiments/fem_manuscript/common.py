from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
BASELINE_DIR = SRC / "baseline"
MANUSCRIPT_ROOT = REPO_ROOT / "experiments" / "fem_manuscript"

REGISTRY = REPO_ROOT / "experiments" / "registry.csv"
REGISTRY_COLUMNS = [
    "date",
    "tag",
    "codebase",
    "script",
    "input_case",
    "domain",
    "grid_or_mesh",
    "solver",
    "parameters",
    "wall_time_sec",
    "error_metrics",
    "eigenvalues_summary",
    "status",
    "notes",
]


def today() -> str:
    return date.today().isoformat()


def add_baseline_to_path() -> None:
    baseline = str(BASELINE_DIR)
    if baseline not in sys.path:
        sys.path.insert(0, baseline)


def coeffs_for_exact_c3_cos(amplitude_s: float) -> dict[int, complex]:
    """Return baseline FourierBC coeffs for barOmega = s cos(3 theta)."""
    return {3: 0.5 * float(amplitude_s) + 0.0j}


def coeffs_for_exact_cos_mode(amplitude_s: float, mode: int) -> dict[int, complex]:
    """Return baseline FourierBC coeffs for barOmega = s cos(mode theta)."""
    return {int(mode): 0.5 * float(amplitude_s) + 0.0j}


def boundary_description(amplitude_s: float, mode: int = 3) -> dict[str, Any]:
    return {
        "analytic": f"barOmega(theta) = s cos({int(mode)} theta)",
        "s": float(amplitude_s),
        "baseline_coeffs": {str(int(mode)): 0.5 * float(amplitude_s)},
        "note": f"FourierBC mirrors positive modes, so c{int(mode)}=s/2 gives exact s cos({int(mode)} theta).",
    }


def ensure_branch_tree(branch_dir: Path) -> None:
    for name in ("data", "checkpoints", "logs", "plots"):
        (branch_dir / name).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def append_registry_row(row: dict[str, Any]) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    exists = REGISTRY.exists()
    with REGISTRY.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "n/a") for key in REGISTRY_COLUMNS})


def mesh_label(h_min: float, h_max: float) -> str:
    return f"h{float(h_min):g}_{float(h_max):g}".replace(".", "p")


def amplitude_label(amplitude_s: float) -> str:
    return f"s{float(amplitude_s):g}".replace(".", "p")
