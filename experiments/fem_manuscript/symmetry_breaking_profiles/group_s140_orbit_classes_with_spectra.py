#!/usr/bin/env python3
"""Group s=140 broken profiles into C3-rotation orbit classes with spectra."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
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
    ("artificial_m1", "artificial localized m=1", BRANCH_DIR / "checkpoints/branchB_s140_artificial_localized_probe/m1_phase0_eta0p25_profile.h5"),
    ("eigenmode1", "eigenmode 1 perturbation", BRANCH_DIR / "checkpoints/branchB_s140_eigenmode_perturb_probe/mode1_eta0p5_signp1_profile.h5"),
    ("analytic_m2", "analytic localized m=2", BRANCH_DIR / "checkpoints/branchB_s140_broken_profile_spectra/analytic_m2_phase0_eta0p25_profile.h5"),
    ("eigenmode2_plus", "eigenmode 2 positive", BRANCH_DIR / "checkpoints/branchB_s140_eigenmode23_perturb_probe/mode2_eta0p5_signp1_profile.h5"),
    ("eigenmode2_minus", "eigenmode 2 negative", BRANCH_DIR / "checkpoints/branchB_s140_eigenmode23_perturb_probe/mode2_eta0p5_signm1_profile.h5"),
    ("eigenmode3_plus", "eigenmode 3 positive", BRANCH_DIR / "checkpoints/branchB_s140_eigenmode23_perturb_probe/mode3_eta0p5_signp1_profile.h5"),
    ("eigenmode3_minus", "eigenmode 3 negative", BRANCH_DIR / "checkpoints/branchB_s140_eigenmode23_perturb_probe/mode3_eta0p5_signm1_profile.h5"),
]

SPECTRUM_ID_MAP = {
    "artificial_m1": "broken_artificial_m1_phase0_eta0p25",
    "eigenmode1": "broken_eigenmode_mode1_eta0p5_signp1",
    "analytic_m2": "analytic_m2_phase0_eta0p25",
    "eigenmode2_plus": "broken_eigenmode_mode2_eta0p5_signp1",
    "eigenmode2_minus": "broken_eigenmode_mode2_eta0p5_signm1",
    "eigenmode3_plus": "broken_eigenmode_mode3_eta0p5_signp1",
    "eigenmode3_minus": "broken_eigenmode_mode3_eta0p5_signm1",
}


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
    return {"path": path, "xy": xy, "cells": cells, "omega": omega}


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def sample_omega(profile: dict[str, Any], radii: np.ndarray, theta: np.ndarray) -> np.ndarray:
    interp = LinearNDInterpolator(profile["xy"], profile["omega"], fill_value=np.nan)
    rr, tt = np.meshgrid(radii, theta, indexing="ij")
    vals = interp((rr * np.cos(tt)).reshape(-1), (rr * np.sin(tt)).reshape(-1)).reshape(rr.shape)
    if np.any(~np.isfinite(vals)):
        raise RuntimeError(f"non-finite polar samples for {profile['path']}")
    return vals


def c3_best_distance(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    ntheta = a.shape[1]
    shifts = [0, ntheta // 3, (2 * ntheta) // 3]
    denom = max(rms(a), 1e-30)
    best = {"relative_rms": float("inf"), "angle_deg": 0.0, "shift_index": 0}
    for shift, angle in zip(shifts, (0.0, 120.0, 240.0)):
        dist = rms(a - np.roll(b, shift=shift, axis=1)) / denom
        if dist < best["relative_rms"]:
            best = {"relative_rms": dist, "angle_deg": angle, "shift_index": int(shift)}
    return best


def load_spectra(path: Path) -> dict[str, list[complex]]:
    out: dict[str, list[tuple[int, complex]]] = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("status") != "completed":
                continue
            out[row["profile_id"]].append(
                (int(row["eigen_index"]), complex(float(row["real_lambda"]), float(row["imag_lambda"])))
            )
    return {key: [z for _, z in sorted(vals)] for key, vals in out.items()}


def connected_components(labels: list[str], pair_rows: list[dict[str, Any]], threshold: float) -> list[list[str]]:
    parent = {label: label for label in labels}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for row in pair_rows:
        if float(row["c3_defect_relative_rms"]) <= threshold:
            union(row["profile_a"], row["profile_b"])
    comps: dict[str, list[str]] = defaultdict(list)
    for label in labels:
        comps[find(label)].append(label)
    return sorted(comps.values(), key=lambda c: (len(c), c[0]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_class_representatives(profiles: dict[str, dict[str, Any]], class_rows: list[dict[str, Any]], out_base: Path) -> None:
    reps = [row["representative"] for row in class_rows]
    tri = mtri.Triangulation(profiles[reps[0]]["xy"][:, 0], profiles[reps[0]]["xy"][:, 1], triangles=profiles[reps[0]]["cells"])
    vmax = max(float(np.max(np.abs(profiles[rep]["omega"]))) for rep in reps)
    fig, axes = plt.subplots(1, len(reps), figsize=(3.2 * len(reps), 3.2), dpi=240, constrained_layout=True)
    if len(reps) == 1:
        axes = [axes]
    for ax, rep, row in zip(axes, reps, class_rows):
        cf = ax.tricontourf(tri, profiles[rep]["omega"], levels=72, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"class {row['class_id']}: {rep.replace('_', ' ')}", fontsize=8)
        ax.set_aspect("equal", "box")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(cf, ax=axes, fraction=0.035, pad=0.02)
    fig.suptitle(r"$\sigma = 140$: C3 orbit-class representatives", fontsize=11)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-count", type=int, default=72)
    parser.add_argument("--theta-count", type=int, default=720)
    parser.add_argument("--r-max", type=float, default=9.5)
    parser.add_argument("--class-threshold", type=float, default=0.08)
    parser.add_argument("--run-tag", default="branchB_s140_c3_orbit_classes_with_spectra")
    parser.add_argument(
        "--symmetric-h5",
        type=Path,
        default=C3_BRANCH_DIR / "checkpoints/branchA_h006_refine_s125_140_k10/s140_h0p06_0p06_profile.h5",
    )
    parser.add_argument(
        "--spectra-csv",
        type=Path,
        default=BRANCH_DIR / "data/branchB_s140_broken_profile_spectra_eigenvalues.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_data = BRANCH_DIR / "data"
    out_plots = BRANCH_DIR / "plots" / args.run_tag
    out_logs = BRANCH_DIR / "logs"
    for d in (out_data, out_plots, out_logs):
        d.mkdir(parents=True, exist_ok=True)

    profiles = {name: load_profile(path) for name, _, path in PROFILE_SPECS if path.exists()}
    sym = load_profile(args.symmetric_h5)
    spectra = load_spectra(args.spectra_csv)
    labels = [name for name, _, _ in PROFILE_SPECS if name in profiles]
    label_to_desc = {name: desc for name, desc, _ in PROFILE_SPECS}

    radii = np.linspace(0.05, float(args.r_max), int(args.r_count))
    theta = np.linspace(0.0, 2.0 * np.pi, int(args.theta_count), endpoint=False)
    sampled = {name: sample_omega(profile, radii, theta) for name, profile in profiles.items()}
    sym_sample = sample_omega(sym, radii, theta)
    defects = {name: sampled[name] - sym_sample for name in labels}

    pair_rows = []
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            full = c3_best_distance(sampled[a], sampled[b])
            defect = c3_best_distance(defects[a], defects[b])
            pair_rows.append(
                {
                    "profile_a": a,
                    "profile_b": b,
                    "c3_full_relative_rms": full["relative_rms"],
                    "c3_full_best_angle_deg": full["angle_deg"],
                    "c3_defect_relative_rms": defect["relative_rms"],
                    "c3_defect_best_angle_deg": defect["angle_deg"],
                    "same_class_at_threshold": defect["relative_rms"] <= float(args.class_threshold),
                }
            )

    classes = connected_components(labels, pair_rows, float(args.class_threshold))
    class_rows = []
    member_rows = []
    for cid, members in enumerate(classes, start=1):
        representative = members[0]
        member_min_re = []
        member_unstable = []
        member_first3 = []
        for member in members:
            eigs = spectra.get(SPECTRUM_ID_MAP[member], [])
            min_re = min((z.real for z in eigs), default=float("nan"))
            unstable = sum(1 for z in eigs if z.real < 0.0)
            member_min_re.append(min_re)
            member_unstable.append(unstable)
            member_first3.append(
                member
                + ": "
                + "; ".join(f"{z.real:.6e}{z.imag:+.6e}i" for z in eigs[:3])
            )
            member_rows.append(
                {
                    "class_id": cid,
                    "profile_id": member,
                    "description": label_to_desc[member],
                    "spectrum_id": SPECTRUM_ID_MAP[member],
                    "min_real_lambda": min_re,
                    "num_negative_real_first10": unstable,
                    "first_three_eigenvalues": "; ".join(f"{z.real:.12g}{z.imag:+.12g}i" for z in eigs[:3]),
                }
            )
        class_rows.append(
            {
                "class_id": cid,
                "representative": representative,
                "members": "; ".join(members),
                "num_members": len(members),
                "min_real_lambda_range": f"{min(member_min_re):.12g} to {max(member_min_re):.12g}",
                "negative_count_range_first10": f"{min(member_unstable)} to {max(member_unstable)}",
                "first_three_eigenvalues_by_member": " | ".join(member_first3),
            }
        )

    pair_csv = out_data / f"{args.run_tag}_pairwise_c3_distances.csv"
    class_csv = out_data / f"{args.run_tag}_class_summary.csv"
    member_csv = out_data / f"{args.run_tag}_class_members_spectra.csv"
    write_csv(pair_csv, pair_rows)
    write_csv(class_csv, class_rows)
    write_csv(member_csv, member_rows)
    plot_class_representatives(profiles, class_rows, out_plots / "s140_c3_orbit_class_representatives")

    manifest = {
        "status": "completed",
        "run_tag": args.run_tag,
        "class_threshold": float(args.class_threshold),
        "rotation_group": "C3 rotations only: 0, 120, 240 degrees",
        "outputs": {
            "pairwise_csv": relpath(pair_csv),
            "class_csv": relpath(class_csv),
            "member_csv": relpath(member_csv),
            "class_representative_plot": relpath((out_plots / "s140_c3_orbit_class_representatives").with_suffix(".png")),
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
            "solver": "postprocess_c3_orbit_grouping_with_spectra",
            "parameters": f"class_threshold={args.class_threshold}; rotations=0,120,240; theta_count={args.theta_count}",
            "wall_time_sec": "n/a",
            "error_metrics": f"class_csv={relpath(class_csv)}",
            "eigenvalues_summary": f"member_csv={relpath(member_csv)}",
            "status": "completed",
            "notes": "Grouped s=140 broken candidates by C3-rotation defect distance and attached first-10 spectra.",
        }
    )
    print(f"[done] class summary: {class_csv}")
    print(f"[done] member spectra: {member_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
