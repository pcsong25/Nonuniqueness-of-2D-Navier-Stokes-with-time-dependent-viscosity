from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


MANUSCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MANUSCRIPT_ROOT))

from common import (  # noqa: E402
    amplitude_label,
    boundary_description,
    ensure_branch_tree,
    mesh_label,
    today,
    write_json,
)


BRANCH_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Branch B scaffold: FEM symmetric vs symmetry-breaking profiles."
    )
    parser.add_argument("--s", type=float, default=140.0)
    parser.add_argument("--mesh", nargs=2, type=float, metavar=("H_MIN", "H_MAX"), default=[0.12, 0.18])
    parser.add_argument("--perturbation-family", default="m1_core")
    parser.add_argument("--perturbation-amplitude", type=float, default=0.2)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Reserved for the next milestone; without it this writes only a dry-run manifest.",
    )
    return parser.parse_args()


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    h_min, h_max = (float(args.mesh[0]), float(args.mesh[1]))
    amplitude_s = float(args.s)
    run_tag = args.run_tag or (
        f"branchB_{today()}_{amplitude_label(amplitude_s)}_{mesh_label(h_min, h_max)}"
    )
    return {
        "status": "scaffold_dry_run",
        "branch": "B_symmetry_breaking_profiles",
        "run_tag": run_tag,
        "mesh": {"h_min": h_min, "h_max": h_max},
        "boundary": boundary_description(amplitude_s),
        "profile_targets": ["symmetric", "interior_perturbed"],
        "perturbation": {
            "family": str(args.perturbation_family),
            "amplitude": float(args.perturbation_amplitude),
            "boundary_policy": "interior perturbation only; boundary data remain exact s cos(3 theta)",
        },
        "planned_outputs": {
            "profile_table": str((BRANCH_DIR / "data" / f"{run_tag}_profiles.csv").relative_to(MANUSCRIPT_ROOT.parents[1])),
            "checkpoint_dir": str((BRANCH_DIR / "checkpoints" / run_tag).relative_to(MANUSCRIPT_ROOT.parents[1])),
            "log_file": str((BRANCH_DIR / "logs" / f"{run_tag}.log").relative_to(MANUSCRIPT_ROOT.parents[1])),
            "plot_dir": str((BRANCH_DIR / "plots" / run_tag).relative_to(MANUSCRIPT_ROOT.parents[1])),
        },
        "required_diagnostics": [
            "same boundary data check",
            "weak residuals for both profiles",
            "Poisson/compatibility diagnostics",
            "non-threefold component fraction",
            "Omega and psi separation norms",
            "mesh refinement comparison",
        ],
        "notes": [
            "No FEM solve is launched by the scaffold mode.",
            "A single perturbed run is not enough to claim bifurcation or nonuniqueness.",
        ],
    }


def main() -> int:
    args = parse_args()
    ensure_branch_tree(BRANCH_DIR)
    manifest = build_manifest(args)
    manifest_path = BRANCH_DIR / "logs" / f"{manifest['run_tag']}_manifest.json"
    write_json(manifest_path, manifest)
    print(f"[manifest] wrote {manifest_path}")

    if args.execute:
        raise SystemExit(
            "Execution is intentionally not implemented in this scaffold. "
            "Next milestone: wire baseline FEM symmetric and perturbed profile smoke tests."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
