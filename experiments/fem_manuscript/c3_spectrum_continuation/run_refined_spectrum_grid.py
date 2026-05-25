#!/usr/bin/env python3
"""Refined Branch A spectrum grid runner and postprocessor.

Target experiment:
- compute exact boundary data barOmega(theta) = s cos(3 theta), encoded in
  the baseline runner as c3 = s/2;
- use h_min = h_max = 0.06, k = 10, sigma_steps = 21, max_newton = 48;
- target amplitudes for new computation are s = 0:5:120 plus s = 150;
- skip amplitudes that already have completed h0.06/k10 eigenvalue rows;
- after computation, combine all available completed h0.06/k10 data in
  0 <= s <= 160 and regenerate the adopted complex-plane tracked plot.

This wrapper is intentionally dry-run by default. Run with --execute inside the
FEM environment to launch the missing expensive jobs.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANUSCRIPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MANUSCRIPT_ROOT.parents[1]
BRANCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BRANCH_DIR / "data"
LOG_DIR = BRANCH_DIR / "logs"
PLOT_DIR = BRANCH_DIR / "plots/branchA_h006_s0_160_refined_grid"
RUNNER = BRANCH_DIR / "run_spectrum_continuation.py"
TRACKED_PLOTTER = BRANCH_DIR / "plot_tracked_spectrum.py"

sys.path.insert(0, str(MANUSCRIPT_ROOT))
from common import append_registry_row, today, write_json  # noqa: E402

BASE_COLUMNS = [
    "s",
    "mesh_id",
    "h_min",
    "h_max",
    "mesh_vertices",
    "mesh_cells",
    "eigen_index",
    "real_lambda",
    "imag_lambda",
    "eig_residual",
    "profile_residual_omega_l2",
    "profile_residual_psi_l2",
    "profile_wall_time",
    "eigs_wall_time",
    "wall_time",
    "status",
    "profile_checkpoint",
    "metadata_checkpoint",
    "message",
    "source_run_tag",
    "source_csv",
]


@dataclass(frozen=True)
class PointRows:
    s: float
    source_csv: Path
    rows: tuple[dict[str, str], ...]


def target_amplitudes() -> list[float]:
    values = [float(s) for s in range(0, 121, 5)]
    values.append(150.0)
    return values


def amplitude_key(value: float) -> int:
    return int(round(float(value) * 1_000_000_000))


def same_float(a: str | float, b: float, tol: float = 1e-12) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def source_priority(path: Path, run_tag: str) -> tuple[int, float, str]:
    name = path.name
    if name == f"{run_tag}_eigenvalues.csv":
        rank = 0
    elif "combined" in name:
        rank = 2
    else:
        rank = 1
    # Prefer newer raw files within a rank, but keep deterministic fallback.
    try:
        mtime = -path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (rank, mtime, name)


def read_candidate_csv(path: Path, h_min: float, h_max: float) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            required = {"s", "h_min", "h_max", "eigen_index", "real_lambda", "imag_lambda", "status"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                return grouped
            for row in reader:
                if row.get("status") != "completed":
                    continue
                if not same_float(row.get("h_min", "nan"), h_min):
                    continue
                if not same_float(row.get("h_max", "nan"), h_max):
                    continue
                s = float(row["s"])
                copied = dict(row)
                copied.setdefault("source_run_tag", "")
                copied["source_csv"] = str(path.relative_to(REPO_ROOT))
                grouped.setdefault(amplitude_key(s), []).append(copied)
    except Exception as exc:
        print(f"[warn] could not read {path}: {exc}")
    return grouped


def discover_completed_points(
    *,
    h_min: float,
    h_max: float,
    n_eigs: int,
    run_tag: str,
    s_min: float | None = None,
    s_max: float | None = None,
) -> dict[int, PointRows]:
    candidates: dict[int, list[PointRows]] = {}
    for path in sorted(DATA_DIR.glob("*_eigenvalues.csv")):
        grouped = read_candidate_csv(path, h_min, h_max)
        for key, rows in grouped.items():
            s = float(rows[0]["s"])
            if s_min is not None and s < s_min - 1e-12:
                continue
            if s_max is not None and s > s_max + 1e-12:
                continue
            rows_by_idx: dict[int, dict[str, str]] = {}
            for row in rows:
                try:
                    idx = int(row["eigen_index"])
                except Exception:
                    continue
                if 1 <= idx <= n_eigs:
                    rows_by_idx.setdefault(idx, row)
            if len(rows_by_idx) < n_eigs:
                continue
            selected = tuple(rows_by_idx[idx] for idx in range(1, n_eigs + 1))
            candidates.setdefault(key, []).append(PointRows(s=s, source_csv=path, rows=selected))

    chosen: dict[int, PointRows] = {}
    for key, point_options in candidates.items():
        chosen[key] = sorted(point_options, key=lambda p: source_priority(p.source_csv, run_tag))[0]
    return chosen


def format_s_values(values: list[float]) -> list[str]:
    return [f"{value:g}" for value in values]


def build_runner_command(args: argparse.Namespace, missing: list[float]) -> list[str]:
    return [
        sys.executable,
        str(RUNNER),
        "--execute",
        "--run-tag",
        args.run_tag,
        "--s-values",
        *format_s_values(missing),
        "--mesh",
        f"{args.h_min:g}",
        f"{args.h_max:g}",
        "--k",
        str(args.k),
        "--eig-shift",
        f"{args.eig_shift:g}",
        "--eig-tol",
        f"{args.eig_tol:g}",
        "--eig-max-it",
        str(args.eig_max_it),
        "--max-newton",
        str(args.max_newton),
        "--sigma-steps",
        str(args.sigma_steps),
    ]


def write_combined_outputs(args: argparse.Namespace) -> dict[str, Any]:
    points = discover_completed_points(
        h_min=args.h_min,
        h_max=args.h_max,
        n_eigs=args.k,
        run_tag=args.run_tag,
        s_min=args.plot_s_min,
        s_max=args.plot_s_max,
    )
    sorted_points = [points[key] for key in sorted(points, key=lambda k: points[k].s)]
    if not sorted_points:
        raise RuntimeError("no completed points available for postprocessing")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined_csv = DATA_DIR / f"{args.output_tag}_eigenvalues.csv"
    summary_csv = DATA_DIR / f"{args.output_tag}_min_re_summary.csv"
    tracked_csv = DATA_DIR / f"{args.output_tag}_tracked_eigenvalues.csv"

    all_fields: list[str] = []
    for field in BASE_COLUMNS:
        if field not in all_fields:
            all_fields.append(field)
    for point in sorted_points:
        for row in point.rows:
            for field in row:
                if field not in all_fields:
                    all_fields.append(field)

    with combined_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        for point in sorted_points:
            for row in point.rows:
                writer.writerow({field: row.get(field, "") for field in all_fields})

    with summary_csv.open("w", newline="") as f:
        fieldnames = [
            "s",
            "min_real_lambda",
            "min_real_eigen_index",
            "n_eigenvalues",
            "profile_residual_omega_l2",
            "profile_residual_psi_l2",
            "source_csv",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point in sorted_points:
            min_row = min(point.rows, key=lambda row: float(row["real_lambda"]))
            writer.writerow(
                {
                    "s": f"{point.s:.12g}",
                    "min_real_lambda": f"{float(min_row['real_lambda']):.16g}",
                    "min_real_eigen_index": min_row["eigen_index"],
                    "n_eigenvalues": str(len(point.rows)),
                    "profile_residual_omega_l2": min_row.get("profile_residual_omega_l2", ""),
                    "profile_residual_psi_l2": min_row.get("profile_residual_psi_l2", ""),
                    "source_csv": str(point.source_csv.relative_to(REPO_ROOT)),
                }
            )

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_stem = f"{args.output_tag}_tracked_Re_lambda_first{args.k}"
    plot_cmd = [
        sys.executable,
        str(TRACKED_PLOTTER),
        "--input",
        str(combined_csv),
        "--output-dir",
        str(PLOT_DIR),
        "--tracked-csv",
        str(tracked_csv),
        "--n-eigs",
        str(args.k),
        "--stem",
        plot_stem,
    ]
    subprocess.run(plot_cmd, check=True, cwd=REPO_ROOT)

    png = PLOT_DIR / f"{plot_stem}.png"
    pdf = PLOT_DIR / f"{plot_stem}.pdf"
    return {
        "completed_plot_s_values": [point.s for point in sorted_points],
        "combined_csv": str(combined_csv.relative_to(REPO_ROOT)),
        "summary_csv": str(summary_csv.relative_to(REPO_ROOT)),
        "tracked_csv": str(tracked_csv.relative_to(REPO_ROOT)),
        "plot_png": str(png.relative_to(REPO_ROOT)),
        "plot_pdf": str(pdf.relative_to(REPO_ROOT)),
        "plot_command": plot_cmd,
    }


def append_postprocess_registry(args: argparse.Namespace, outputs: dict[str, Any], wall_time: float) -> None:
    append_registry_row(
        {
            "date": today(),
            "tag": f"fem-branchA-h006-{args.output_tag}",
            "codebase": "baseline",
            "script": str(Path(__file__).relative_to(REPO_ROOT)),
            "input_case": "s_cos3theta",
            "domain": "disk_Rdom10",
            "grid_or_mesh": f"h={args.h_min:g}/{args.h_max:g}",
            "solver": "baseline_fem_continuation_plus_shift_invert_eigs_and_tracked_plot",
            "parameters": (
                f"target_compute_s=0:5:120 plus 150; plot_range=[{args.plot_s_min:g},{args.plot_s_max:g}]; "
                f"k={args.k}; sigma_steps={args.sigma_steps}; max_newton={args.max_newton}; "
                f"run_tag={args.run_tag}"
            ),
            "wall_time_sec": f"{wall_time:.6f}",
            "error_metrics": "existing weak/profile residual columns preserved in combined CSV",
            "eigenvalues_summary": (
                f"combined_s_values={','.join(format_s_values(outputs['completed_plot_s_values']))}; "
                f"plot={outputs['plot_png']}"
            ),
            "status": "completed",
            "notes": "Skipped already completed amplitudes and regenerated adopted complex-plane tracked Re(lambda) plot.",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Launch missing FEM/eigenvalue jobs.")
    parser.add_argument("--postprocess-only", action="store_true", help="Only rebuild combined CSVs and plots from existing data.")
    parser.add_argument("--no-postprocess", action="store_true", help="Skip combined CSV and plot generation after --execute.")
    parser.add_argument("--run-tag", default="branchA_h006_refined_s0_120_step5_plus150_k10")
    parser.add_argument("--output-tag", default="branchA_h006_refined_s0_160_step5_plus150_k10")
    parser.add_argument("--h-min", type=float, default=0.06)
    parser.add_argument("--h-max", type=float, default=0.06)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--eig-shift", type=float, default=-1.0)
    parser.add_argument("--eig-tol", type=float, default=1e-8)
    parser.add_argument("--eig-max-it", type=int, default=300)
    parser.add_argument("--max-newton", type=int, default=48)
    parser.add_argument("--sigma-steps", type=int, default=21)
    parser.add_argument("--plot-s-min", type=float, default=0.0)
    parser.add_argument("--plot-s-max", type=float, default=160.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    target = target_amplitudes()
    completed_before = discover_completed_points(
        h_min=args.h_min,
        h_max=args.h_max,
        n_eigs=args.k,
        run_tag=args.run_tag,
    )
    existing_target = [s for s in target if amplitude_key(s) in completed_before]
    missing = [s for s in target if amplitude_key(s) not in completed_before]
    runner_cmd = build_runner_command(args, missing) if missing else []

    manifest: dict[str, Any] = {
        "status": "planned",
        "run_tag": args.run_tag,
        "output_tag": args.output_tag,
        "target_compute_s_values": target,
        "already_completed_target_s_values": existing_target,
        "missing_target_s_values": missing,
        "plot_range": [args.plot_s_min, args.plot_s_max],
        "mesh": {"h_min": args.h_min, "h_max": args.h_max},
        "k": args.k,
        "sigma_steps": args.sigma_steps,
        "max_newton": args.max_newton,
        "runner_command": runner_cmd,
        "notes": [
            "Run this wrapper with --execute inside the FEM conda environment to launch missing jobs.",
            "The wrapper scans existing CSVs and skips amplitudes with completed h0.06/k10 rows.",
            "Postprocessing uses the adopted complex-plane tracked plotting method.",
        ],
    }

    print("Target compute s-values:", format_s_values(target))
    print("Already completed target s-values:", format_s_values(existing_target))
    print("Missing target s-values:", format_s_values(missing))

    if args.postprocess_only and args.execute:
        raise SystemExit("Use either --execute or --postprocess-only, not both.")

    if args.execute:
        if missing:
            print("Launching missing amplitudes with command:")
            print(" ".join(runner_cmd))
            subprocess.run(runner_cmd, check=True, cwd=REPO_ROOT)
            manifest["runner_status"] = "completed"
        else:
            print("No missing target amplitudes; skipping FEM/eigensolve run.")
            manifest["runner_status"] = "skipped_no_missing_values"
        if not args.no_postprocess:
            outputs = write_combined_outputs(args)
            manifest["postprocess_outputs"] = outputs
            append_postprocess_registry(args, outputs, time.perf_counter() - start)
        manifest["status"] = "completed"
    elif args.postprocess_only:
        outputs = write_combined_outputs(args)
        manifest["postprocess_outputs"] = outputs
        append_postprocess_registry(args, outputs, time.perf_counter() - start)
        manifest["status"] = "postprocess_completed"
    else:
        if runner_cmd:
            print("Dry run only. Command to execute missing amplitudes:")
            print(" ".join(runner_cmd))
        else:
            print("Dry run only. No missing target amplitudes were found.")
        print("Add --execute to run missing amplitudes, or --postprocess-only to rebuild plots from existing data.")
        manifest["status"] = "dry_run"

    manifest_path = LOG_DIR / f"{args.output_tag}_plan_manifest.json"
    write_json(manifest_path, manifest)
    print(f"manifest={manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
