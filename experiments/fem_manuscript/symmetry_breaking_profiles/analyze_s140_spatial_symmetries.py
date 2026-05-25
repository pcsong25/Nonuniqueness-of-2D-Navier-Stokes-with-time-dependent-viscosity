#!/usr/bin/env python3
"""Spatial symmetry diagnostics for s=140 profile classes."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
from typing import Any

import h5py
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
    ("symmetric", C3_BRANCH_DIR / "checkpoints/branchA_h006_refine_s125_140_k10/s140_h0p06_0p06_profile.h5"),
    ("class1_eigenmode1", BRANCH_DIR / "checkpoints/branchB_s140_eigenmode_perturb_probe/mode1_eta0p5_signp1_profile.h5"),
    ("class2_eigenmode3_minus", BRANCH_DIR / "checkpoints/branchB_s140_eigenmode23_perturb_probe/mode3_eta0p5_signm1_profile.h5"),
    ("class3_eigenmode3_plus", BRANCH_DIR / "checkpoints/branchB_s140_eigenmode23_perturb_probe/mode3_eta0p5_signp1_profile.h5"),
    ("class4_analytic_m2", BRANCH_DIR / "checkpoints/branchB_s140_broken_profile_spectra/analytic_m2_phase0_eta0p25_profile.h5"),
    ("class5_artificial_m1", BRANCH_DIR / "checkpoints/branchB_s140_artificial_localized_probe/m1_phase0_eta0p25_profile.h5"),
]


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def load_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        xy = np.asarray(h5["/Mesh/mesh/geometry"][:], dtype=float)
        omega = np.asarray(h5["/Function/Omega/0"][:], dtype=float).reshape(-1)
    return xy, omega


def sample_profile(path: Path, radii: np.ndarray, theta: np.ndarray) -> np.ndarray:
    xy, omega = load_profile(path)
    interp = LinearNDInterpolator(xy, omega, fill_value=np.nan)
    rr, tt = np.meshgrid(radii, theta, indexing="ij")
    vals = interp((rr * np.cos(tt)).reshape(-1), (rr * np.sin(tt)).reshape(-1)).reshape(rr.shape)
    if np.any(~np.isfinite(vals)):
        raise RuntimeError(f"non-finite samples for {path}")
    return vals


def c3_defect(values: np.ndarray) -> float:
    ntheta = values.shape[1]
    stack = np.stack([np.roll(values, shift=s, axis=1) for s in (0, ntheta // 3, 2 * ntheta // 3)])
    avg = stack.mean(axis=0)
    return rms(stack[0] - avg) / max(rms(stack[0]), 1e-30)


def reflect(values: np.ndarray, theta: np.ndarray, phi: float) -> np.ndarray:
    rr = values.shape[0]
    theta2 = (2.0 * phi - theta) % (2.0 * np.pi)
    extx = np.r_[theta, 2.0 * np.pi]
    out = np.empty_like(values)
    for i in range(rr):
        out[i] = np.interp(theta2, extx, np.r_[values[i], values[i, 0]])
    return out


def axis_residuals(values: np.ndarray, theta: np.ndarray, phi: float) -> dict[str, float]:
    mirrored = reflect(values, theta, phi)
    denom = max(rms(values), 1e-30)
    return {
        "even_residual": rms(mirrored - values) / denom,
        "odd_residual": rms(mirrored + values) / denom,
    }


def best_axis(values: np.ndarray, theta: np.ndarray, kind: str) -> tuple[float, float]:
    best_res = float("inf")
    best_phi = 0.0
    for phi in np.linspace(0.0, np.pi, 721, endpoint=False):
        res = axis_residuals(values, theta, float(phi))[f"{kind}_residual"]
        if res < best_res:
            best_res = res
            best_phi = float(phi)
    return best_res, best_phi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-count", type=int, default=96)
    parser.add_argument("--theta-count", type=int, default=720)
    parser.add_argument("--r-max", type=float, default=9.5)
    parser.add_argument("--run-tag", default="branchB_s140_spatial_symmetry_diagnostics")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_data = BRANCH_DIR / "data"
    out_logs = BRANCH_DIR / "logs"
    out_data.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    radii = np.linspace(0.05, float(args.r_max), int(args.r_count))
    theta = np.linspace(0.0, 2.0 * np.pi, int(args.theta_count), endpoint=False)
    samples = {name: sample_profile(path, radii, theta) for name, path in PROFILE_SPECS}
    sym = samples["symmetric"]

    summary_rows: list[dict[str, Any]] = []
    axis_rows: list[dict[str, Any]] = []
    named_axes = [
        ("x_axis_even_boundary", 0.0),
        ("60deg_even_boundary", np.pi / 3.0),
        ("120deg_even_boundary", 2.0 * np.pi / 3.0),
        ("30deg_odd_boundary", np.pi / 6.0),
        ("y_axis_odd_boundary", np.pi / 2.0),
        ("150deg_odd_boundary", 5.0 * np.pi / 6.0),
    ]

    for name, values in samples.items():
        defect = values - sym
        best_even_full = best_axis(values, theta, "even")
        best_odd_full = best_axis(values, theta, "odd")
        best_even_defect = best_axis(defect, theta, "even") if name != "symmetric" else (0.0, 0.0)
        best_odd_defect = best_axis(defect, theta, "odd") if name != "symmetric" else (0.0, 0.0)
        summary_rows.append(
            {
                "profile_id": name,
                "full_c3_defect": c3_defect(values),
                "defect_c3_defect": c3_defect(defect) if name != "symmetric" else 0.0,
                "relative_omega_separation_from_symmetric": rms(defect) / max(rms(sym), 1e-30),
                "best_even_full_residual": best_even_full[0],
                "best_even_full_axis_deg": 180.0 * best_even_full[1] / np.pi,
                "best_odd_full_residual": best_odd_full[0],
                "best_odd_full_axis_deg": 180.0 * best_odd_full[1] / np.pi,
                "best_even_defect_residual": best_even_defect[0],
                "best_even_defect_axis_deg": 180.0 * best_even_defect[1] / np.pi,
                "best_odd_defect_residual": best_odd_defect[0],
                "best_odd_defect_axis_deg": 180.0 * best_odd_defect[1] / np.pi,
            }
        )
        for axis_name, phi in named_axes:
            res = axis_residuals(values, theta, phi)
            dres = axis_residuals(defect, theta, phi) if name != "symmetric" else {"even_residual": 0.0, "odd_residual": 0.0}
            axis_rows.append(
                {
                    "profile_id": name,
                    "axis_name": axis_name,
                    "axis_deg": 180.0 * phi / np.pi,
                    "full_even_residual": res["even_residual"],
                    "full_odd_residual": res["odd_residual"],
                    "defect_even_residual": dres["even_residual"],
                    "defect_odd_residual": dres["odd_residual"],
                }
            )

    summary_csv = out_data / f"{args.run_tag}_summary.csv"
    axis_csv = out_data / f"{args.run_tag}_axis_residuals.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with axis_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(axis_rows[0]))
        writer.writeheader()
        writer.writerows(axis_rows)

    write_json(
        out_logs / f"{args.run_tag}_manifest.json",
        {
            "status": "completed",
            "run_tag": args.run_tag,
            "outputs": {"summary_csv": relpath(summary_csv), "axis_csv": relpath(axis_csv)},
            "notes": "C3 residuals and reflection even/odd residuals for s=140 profile classes.",
        },
    )
    append_registry_row(
        {
            "date": today(),
            "tag": f"fem-branchB-{args.run_tag}",
            "codebase": "baseline",
            "script": relpath(Path(__file__)),
            "input_case": "s_cos3theta_s140",
            "domain": "disk_Rdom10",
            "grid_or_mesh": "h=0.06/0.06",
            "solver": "postprocess_spatial_symmetry_diagnostics",
            "parameters": f"r_count={args.r_count}; theta_count={args.theta_count}; r_max={args.r_max}",
            "wall_time_sec": "n/a",
            "error_metrics": f"summary={relpath(summary_csv)}",
            "eigenvalues_summary": "n/a",
            "status": "completed",
            "notes": "Checked C3 symmetry and reflection/anti-reflection residuals for s=140 profile classes.",
        }
    )
    print(f"[done] {summary_csv}")
    print(f"[done] {axis_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
