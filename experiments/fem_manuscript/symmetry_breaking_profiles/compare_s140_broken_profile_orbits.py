#!/usr/bin/env python3
"""Compare s=140 broken profiles up to rotations and reflections.

The comparison is performed on a common polar sample grid.  For each pair, the
second profile is rotated and optionally reflected to minimize the relative
RMS distance to the first profile.  We report both the full vorticity distance
and the distance between symmetry-breaking defects Omega - Omega_sym.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.interpolate import LinearNDInterpolator

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig-euler")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

BRANCH_DIR = Path(__file__).resolve().parent
MANUSCRIPT_ROOT = BRANCH_DIR.parents[0]
REPO_ROOT = MANUSCRIPT_ROOT.parents[1]
C3_BRANCH_DIR = MANUSCRIPT_ROOT / "c3_spectrum_continuation"
sys.path.insert(0, str(MANUSCRIPT_ROOT))

from common import append_registry_row, today, write_json  # noqa: E402


PROFILE_SPECS = [
    (
        "saved_artificial_m1",
        BRANCH_DIR / "checkpoints/branchB_s140_artificial_localized_probe/m1_phase0_eta0p25_profile.h5",
    ),
    (
        "saved_eigenmode_mode1",
        BRANCH_DIR / "checkpoints/branchB_s140_eigenmode_perturb_probe/mode1_eta0p5_signp1_profile.h5",
    ),
    (
        "analytic_m2",
        BRANCH_DIR / "checkpoints/branchB_s140_broken_profile_spectra/analytic_m2_phase0_eta0p25_profile.h5",
    ),
    (
        "eigenmode_mode2_signp1",
        BRANCH_DIR / "checkpoints/branchB_s140_eigenmode23_perturb_probe/mode2_eta0p5_signp1_profile.h5",
    ),
    (
        "eigenmode_mode2_signm1",
        BRANCH_DIR / "checkpoints/branchB_s140_eigenmode23_perturb_probe/mode2_eta0p5_signm1_profile.h5",
    ),
    (
        "eigenmode_mode3_signp1",
        BRANCH_DIR / "checkpoints/branchB_s140_eigenmode23_perturb_probe/mode3_eta0p5_signp1_profile.h5",
    ),
    (
        "eigenmode_mode3_signm1",
        BRANCH_DIR / "checkpoints/branchB_s140_eigenmode23_perturb_probe/mode3_eta0p5_signm1_profile.h5",
    ),
]


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_profile(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        xy = np.asarray(h5["/Mesh/mesh/geometry"][:], dtype=float)
        cells = np.asarray(h5["/Mesh/mesh/topology"][:], dtype=np.int64)
        omega = np.asarray(h5["/Function/Omega/0"][:], dtype=float).reshape(-1)
        psi = np.asarray(h5["/Function/psi/0"][:], dtype=float).reshape(-1)
    return {"path": path, "xy": xy, "cells": cells, "omega": omega, "psi": psi}


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def sample_field(profile: dict[str, Any], radii: np.ndarray, theta: np.ndarray, field: str = "omega") -> np.ndarray:
    interp = LinearNDInterpolator(profile["xy"], profile[field], fill_value=np.nan)
    rr, tt = np.meshgrid(radii, theta, indexing="ij")
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)
    vals = interp(x.reshape(-1), y.reshape(-1)).reshape(rr.shape)
    if np.any(~np.isfinite(vals)):
        raise RuntimeError(f"non-finite polar samples for {profile['path']}")
    return vals


def reflected(values: np.ndarray) -> np.ndarray:
    out = values[:, ::-1]
    return np.roll(out, 1, axis=1)


def best_shift_distance(a: np.ndarray, b: np.ndarray, theta: np.ndarray) -> dict[str, Any]:
    denom = max(rms(a), 1e-30)
    best = {"relative_rms": float("inf"), "shift_index": 0, "angle_rad": 0.0, "angle_deg": 0.0}
    ntheta = len(theta)
    for k in range(ntheta):
        diff = a - np.roll(b, shift=k, axis=1)
        dist = rms(diff) / denom
        if dist < best["relative_rms"]:
            angle = 2.0 * np.pi * k / ntheta
            best = {
                "relative_rms": dist,
                "shift_index": int(k),
                "angle_rad": float(angle),
                "angle_deg": float(180.0 * angle / np.pi),
            }
    return best


def best_orbit_distance(a: np.ndarray, b: np.ndarray, theta: np.ndarray) -> dict[str, Any]:
    rot = best_shift_distance(a, b, theta)
    refl = best_shift_distance(a, reflected(b), theta)
    if refl["relative_rms"] < rot["relative_rms"]:
        refl["transform"] = "reflection_plus_rotation"
        return refl
    rot["transform"] = "rotation"
    return rot


def shifted(values: np.ndarray, best: dict[str, Any]) -> np.ndarray:
    base = reflected(values) if best["transform"] == "reflection_plus_rotation" else values
    return np.roll(base, shift=int(best["shift_index"]), axis=1)


def plot_three_profiles(profiles: dict[str, dict[str, Any]], labels: list[str], out_base: Path) -> None:
    tri = mtri.Triangulation(
        profiles[labels[0]]["xy"][:, 0],
        profiles[labels[0]]["xy"][:, 1],
        triangles=profiles[labels[0]]["cells"],
    )
    vmax = max(float(np.max(np.abs(profiles[label]["omega"]))) for label in labels)
    pretty = {
        "saved_artificial_m1": "artificial m=1",
        "saved_eigenmode_mode1": "eigenmode 1",
        "analytic_m2": "analytic m=2",
    }
    fig, axes = plt.subplots(1, len(labels), figsize=(9.6, 3.1), dpi=240, constrained_layout=True)
    for ax, label in zip(axes, labels):
        cf = ax.tricontourf(tri, profiles[label]["omega"], levels=72, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(pretty.get(label, label.replace("_", " ")), fontsize=10)
        ax.set_aspect("equal", "box")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(cf, ax=axes, fraction=0.035, pad=0.02)
    fig.suptitle(r"$\sigma = 140$: broken vorticity profiles", fontsize=11)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_all_profiles(profiles: dict[str, dict[str, Any]], labels: list[str], out_base: Path) -> None:
    tri = mtri.Triangulation(
        profiles[labels[0]]["xy"][:, 0],
        profiles[labels[0]]["xy"][:, 1],
        triangles=profiles[labels[0]]["cells"],
    )
    vmax = max(float(np.max(np.abs(profiles[label]["omega"]))) for label in labels)
    ncols = 4
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 3.0 * nrows), dpi=240, constrained_layout=True)
    axes_flat = np.ravel(axes)
    for ax, label in zip(axes_flat, labels):
        cf = ax.tricontourf(tri, profiles[label]["omega"], levels=72, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(label.replace("_", " "), fontsize=8)
        ax.set_aspect("equal", "box")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_flat[len(labels) :]:
        ax.axis("off")
    fig.colorbar(cf, ax=axes_flat[: len(labels)], fraction=0.035, pad=0.02)
    fig.suptitle(r"$\sigma = 140$: broken-profile candidates", fontsize=11)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_aligned_defects(
    sampled: dict[str, np.ndarray],
    sym_sample: np.ndarray,
    labels: list[str],
    theta: np.ndarray,
    radii: np.ndarray,
    pair_table: list[dict[str, Any]],
    out_base: Path,
) -> None:
    ref = labels[0]
    ref_defect = sampled[ref] - sym_sample
    aligned = {ref: ref_defect}
    for label in labels[1:]:
        row = next(r for r in pair_table if r["profile_a"] == ref and r["profile_b"] == label)
        best = {
            "transform": row["defect_best_transform"],
            "shift_index": int(row["defect_best_shift_index"]),
        }
        aligned[label] = shifted(sampled[label] - sym_sample, best)
    vmax = max(float(np.max(np.abs(aligned[label]))) for label in labels)
    rr, tt = np.meshgrid(radii, theta, indexing="ij")
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)
    pretty = {
        "saved_artificial_m1": "reference: artificial m=1",
        "saved_eigenmode_mode1": "eigenmode 1 aligned",
        "analytic_m2": "analytic m=2 aligned",
    }
    fig, axes = plt.subplots(2, len(labels), figsize=(9.8, 5.8), dpi=240, constrained_layout=True)
    for j, label in enumerate(labels):
        cf = axes[0, j].pcolormesh(x, y, aligned[label], shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[0, j].set_title(pretty.get(label, label.replace("_", " ")), fontsize=9)
        diff = aligned[label] - ref_defect
        local = max(float(np.max(np.abs(diff))), 1e-14)
        cf2 = axes[1, j].pcolormesh(x, y, diff, shading="auto", cmap="RdBu_r", vmin=-local, vmax=local)
        axes[1, j].set_title("difference from reference", fontsize=9)
        for ax in (axes[0, j], axes[1, j]):
            ax.set_aspect("equal", "box")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(cf, ax=axes[0, j], fraction=0.046, pad=0.02)
        fig.colorbar(cf2, ax=axes[1, j], fraction=0.046, pad=0.02)
    fig.suptitle(r"$\sigma = 140$: aligned symmetry-breaking defects", fontsize=11)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-count", type=int, default=72)
    parser.add_argument("--theta-count", type=int, default=720)
    parser.add_argument("--r-max", type=float, default=9.5)
    parser.add_argument("--run-tag", default="branchB_s140_orbit_comparison")
    parser.add_argument(
        "--symmetric-h5",
        type=Path,
        default=C3_BRANCH_DIR / "checkpoints/branchA_h006_refine_s125_140_k10/s140_h0p06_0p06_profile.h5",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_data = BRANCH_DIR / "data"
    out_plots = BRANCH_DIR / "plots" / args.run_tag
    out_logs = BRANCH_DIR / "logs"
    for d in (out_data, out_plots, out_logs):
        d.mkdir(parents=True, exist_ok=True)

    profiles = {name: load_profile(path) for name, path in PROFILE_SPECS if path.exists()}
    profiles["symmetric"] = load_profile(args.symmetric_h5)
    requested_three = ["saved_artificial_m1", "saved_eigenmode_mode1", "analytic_m2"]
    missing = [label for label in requested_three if label not in profiles]
    if missing:
        raise RuntimeError(f"missing requested profiles: {missing}")

    radii = np.linspace(0.05, float(args.r_max), int(args.r_count))
    theta = np.linspace(0.0, 2.0 * np.pi, int(args.theta_count), endpoint=False)
    sampled = {name: sample_field(profile, radii, theta, "omega") for name, profile in profiles.items()}
    sym_sample = sampled["symmetric"]

    labels = [name for name in profiles if name != "symmetric"]
    rows = []
    for i, a_name in enumerate(labels):
        for b_name in labels[i + 1 :]:
            a = sampled[a_name]
            b = sampled[b_name]
            a_defect = a - sym_sample
            b_defect = b - sym_sample
            raw = rms(a - b) / max(rms(a), 1e-30)
            full_best = best_orbit_distance(a, b, theta)
            defect_best = best_orbit_distance(a_defect, b_defect, theta)
            rows.append(
                {
                    "profile_a": a_name,
                    "profile_b": b_name,
                    "raw_relative_rms_full_omega": raw,
                    "full_best_relative_rms": full_best["relative_rms"],
                    "full_best_transform": full_best["transform"],
                    "full_best_angle_deg": full_best["angle_deg"],
                    "full_best_shift_index": full_best["shift_index"],
                    "defect_best_relative_rms": defect_best["relative_rms"],
                    "defect_best_transform": defect_best["transform"],
                    "defect_best_angle_deg": defect_best["angle_deg"],
                    "defect_best_shift_index": defect_best["shift_index"],
                    "classification_hint": (
                        "same_orbit_candidate"
                        if defect_best["relative_rms"] < 5e-2
                        else "distinct_orbit_candidate"
                    ),
                }
            )

    csv_path = out_data / f"{args.run_tag}_pairwise_distances.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plot_three_profiles(profiles, requested_three, out_plots / "s140_three_broken_profiles")
    plot_all_profiles(profiles, labels, out_plots / "s140_all_broken_profile_candidates")
    plot_aligned_defects(
        sampled,
        sym_sample,
        requested_three,
        theta,
        radii,
        rows,
        out_plots / "s140_three_broken_profiles_aligned_defects",
    )

    manifest = {
        "status": "completed",
        "run_tag": args.run_tag,
        "profiles": {name: relpath(profile["path"]) for name, profile in profiles.items()},
        "polar_grid": {"r_count": int(args.r_count), "theta_count": int(args.theta_count), "r_max": float(args.r_max)},
        "outputs": {
            "pairwise_csv": relpath(csv_path),
            "three_profile_plot_png": relpath((out_plots / "s140_three_broken_profiles").with_suffix(".png")),
            "all_profile_plot_png": relpath((out_plots / "s140_all_broken_profile_candidates").with_suffix(".png")),
            "aligned_defects_plot_png": relpath((out_plots / "s140_three_broken_profiles_aligned_defects").with_suffix(".png")),
        },
    }
    write_json(out_logs / f"{args.run_tag}_manifest.json", manifest)
    append_registry_row(
        {
            "date": today(),
            "tag": f"fem-branchB-{args.run_tag}",
            "codebase": "baseline",
            "script": relpath(Path(__file__)),
            "input_case": "s_cos3theta_s140",
            "domain": "disk_Rdom10",
            "grid_or_mesh": "h=0.06/0.06",
            "solver": "postprocess_orbit_alignment_distance",
            "parameters": f"r_count={args.r_count}; theta_count={args.theta_count}; r_max={args.r_max}",
            "wall_time_sec": "n/a",
            "error_metrics": f"pairwise_csv={relpath(csv_path)}",
            "eigenvalues_summary": "n/a",
            "status": "completed",
            "notes": "Pairwise broken-profile distances minimized over rotations and reflections.",
        }
    )
    print(f"[done] pairwise distances: {csv_path}")
    print(f"[done] plots: {out_plots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
