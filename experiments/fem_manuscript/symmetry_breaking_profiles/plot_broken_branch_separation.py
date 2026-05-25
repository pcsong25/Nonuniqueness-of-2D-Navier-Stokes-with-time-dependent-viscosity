#!/usr/bin/env python3
"""Plot relative separation of the tracked broken branch.

The continuation runs are intentionally split into resumable chunks.  This
script merges those result CSVs, removes duplicate sigma rows by keeping the
largest measured separation, and writes a single summary table and plot.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

BRANCH_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS = [
    "branchB_broken_branch_sigma_track_s150_110",
    "branchB_broken_branch_sigma_track_extra_135_145",
    "branchB_broken_branch_sigma_track_extra_105_100",
]


def read_rows(run_tags: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for tag in run_tags:
        path = BRANCH_DIR / "data" / f"{tag}_results.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in {"", "n/a", "nan"}:
        return float("nan")
    return float(value)


def select_by_sigma(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected: dict[float, dict[str, str]] = {}
    for row in rows:
        if row.get("status") not in {"completed", "seed_loaded"}:
            continue
        sigma = as_float(row, "sigma")
        sep = as_float(row, "omega_relative_rms_separation")
        if not (sep > 0.0):
            continue
        old = selected.get(sigma)
        if old is None or sep > as_float(old, "omega_relative_rms_separation"):
            selected[sigma] = row
    return [selected[sigma] for sigma in sorted(selected)]


def write_summary(rows: list[dict[str, str]], path: Path) -> None:
    columns = [
        "sigma",
        "status",
        "direction",
        "source_sigma",
        "omega_relative_rms_separation",
        "omega_rms_separation",
        "omega_linf_separation",
        "psi_relative_rms_separation",
        "psi_rms_separation",
        "psi_linf_separation",
        "profile_residual_omega_l2",
        "profile_residual_psi_l2",
        "boundary_max_abs_diff",
        "omega_c3_fraction_symmetric",
        "omega_c3_fraction_broken",
        "omega_c3_fraction_excess",
        "omega_difference_non3_fraction",
        "broken_profile_checkpoint",
        "symmetric_profile_h5",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def plot(rows: list[dict[str, str]], stem: Path) -> None:
    sigma = [as_float(row, "sigma") for row in rows]
    sep = [as_float(row, "omega_relative_rms_separation") for row in rows]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "mathtext.fontset": "cm",
        }
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.1), constrained_layout=True)
    ax.plot(
        sigma,
        sep,
        color="#1f4e79",
        linewidth=1.8,
        marker="o",
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label=r"tracked branch",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"relative RMS separation in $\Omega$")
    ax.grid(True, which="major", color="0.86", linewidth=0.8)
    ax.grid(True, which="minor", color="0.92", linewidth=0.5)
    ax.legend(frameon=False, loc="best")
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_dual_legacy_style(rows: list[dict[str, str]], stem: Path) -> None:
    sigma = [as_float(row, "sigma") for row in rows]
    omega_sep = [as_float(row, "omega_relative_rms_separation") for row in rows]
    psi_sep = [as_float(row, "psi_relative_rms_separation") for row in rows]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titlesize": 22,
            "axes.labelsize": 17,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "mathtext.fontset": "dejavusans",
        }
    )
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(
        sigma,
        omega_sep,
        color="#1f77b4",
        linewidth=2.8,
        marker="o",
        markersize=7.5,
        markerfacecolor="white",
        markeredgewidth=1.8,
        label=r"$d_\Omega$",
    )
    ax.plot(
        sigma,
        psi_sep,
        color="#d62728",
        linewidth=2.4,
        marker="s",
        markersize=6.8,
        markerfacecolor="white",
        markeredgewidth=1.6,
        label=r"$d_\psi$",
    )
    ax.axvspan(120, 125, color="0.75", alpha=0.4, label="spectral zero bracket")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel("relative RMS separation")
    ax.set_title("Symmetric vs symmetry-breaking profile separation")
    ax.grid(True, which="major", color="0.85", linewidth=1.0)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(min(sigma) - 2, max(sigma) + 2)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tags", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument(
        "--summary",
        type=Path,
        default=BRANCH_DIR / "data" / "branchB_broken_branch_separation_summary_refined.csv",
    )
    parser.add_argument(
        "--plot-stem",
        type=Path,
        default=BRANCH_DIR
        / "plots"
        / "branchB_broken_branch_sigma_track_s150_110"
        / "branchB_broken_branch_relative_separation_vs_sigma_refined_log",
    )
    parser.add_argument(
        "--legacy-style-plot-stem",
        type=Path,
        default=BRANCH_DIR
        / "plots"
        / "branchB_broken_branch_sigma_track_s150_110"
        / "branchB_broken_branch_relative_separation_vs_s_log_refined",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = select_by_sigma(read_rows(args.run_tags))
    write_summary(rows, args.summary)
    plot(rows, args.plot_stem)
    plot_dual_legacy_style(rows, args.legacy_style_plot_stem)
    print(f"[done] summary={args.summary}")
    print(f"[done] plot_png={args.plot_stem.with_suffix('.png')}")
    print(f"[done] plot_pdf={args.plot_stem.with_suffix('.pdf')}")
    print(f"[done] legacy_style_plot_png={args.legacy_style_plot_stem.with_suffix('.png')}")
    print(f"[done] legacy_style_plot_pdf={args.legacy_style_plot_stem.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
