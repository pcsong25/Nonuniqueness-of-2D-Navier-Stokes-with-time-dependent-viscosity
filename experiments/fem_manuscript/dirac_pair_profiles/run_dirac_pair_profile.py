#!/usr/bin/env python3
"""Profile solve for a regularized two-Dirac boundary datum in theta.

Boundary family:
    barOmega(theta) = sigma * (1 + 2 * sum_{m=1}^M exp(-4 eps^2 m^2) cos(2m theta)).

The baseline FourierBC mirrors positive coefficients, so the coefficient for
mode 2m is sigma * exp(-4 eps^2 m^2), and the constant coefficient is sigma.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig-euler")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

MANUSCRIPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MANUSCRIPT_ROOT.parents[1]
C3_BRANCH_DIR = MANUSCRIPT_ROOT / "c3_spectrum_continuation"
sys.path.insert(0, str(MANUSCRIPT_ROOT))
sys.path.insert(0, str(C3_BRANCH_DIR))

from common import add_baseline_to_path, append_registry_row, ensure_branch_tree, mesh_label, today, write_json  # noqa: E402
from run_spectrum_continuation import build_baseline_context, jsonable, mesh_stats  # noqa: E402


BRANCH_DIR = Path(__file__).resolve().parent
CSV_COLUMNS = [
    "run_tag",
    "sigma",
    "epsilon",
    "truncation_M",
    "fourier_cutoff",
    "h_min",
    "h_max",
    "mesh_vertices",
    "mesh_cells",
    "profile_residual_omega_l2",
    "profile_residual_psi_l2",
    "profile_wall_time",
    "wall_time",
    "status",
    "profile_checkpoint",
    "metadata_checkpoint",
    "plot_path",
    "message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--truncation-M", type=int, default=20)
    parser.add_argument("--mesh", nargs=2, type=float, metavar=("H_MIN", "H_MAX"), default=[0.06, 0.06])
    parser.add_argument("--mu", type=float, default=0.8)
    parser.add_argument("--R", type=float, default=150.0)
    parser.add_argument("--Rdom", type=float, default=10.0)
    parser.add_argument("--tol-newton", type=float, default=1e-8)
    parser.add_argument("--max-newton", type=int, default=48)
    parser.add_argument("--sigma-steps", type=int, default=21)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-registry", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually run the FEM solve; otherwise write a dry-run manifest.")
    return parser.parse_args()


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def safe_float_label(value: float, prefix: str = "") -> str:
    return (prefix + f"{float(value):g}").replace(".", "p").replace("-", "m")


def coeffs_for_regularized_dirac_pair(sigma: float, truncation_m: int, epsilon: float) -> dict[int, complex]:
    coeffs: dict[int, complex] = {0: complex(float(sigma), 0.0)}
    for m in range(1, int(truncation_m) + 1):
        coeffs[2 * m] = complex(float(sigma) * np.exp(-4.0 * float(epsilon) ** 2 * m * m), 0.0)
    return coeffs


def boundary_values(theta: np.ndarray, sigma: float, truncation_m: int, epsilon: float) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    out = np.full_like(theta, float(sigma), dtype=float)
    for m in range(1, int(truncation_m) + 1):
        out += 2.0 * float(sigma) * np.exp(-4.0 * float(epsilon) ** 2 * m * m) * np.cos(2 * m * theta)
    return out


def completed_metadata(path: Path, args: argparse.Namespace) -> bool:
    if not path.exists():
        return False
    try:
        import json

        with path.open() as f:
            data = json.load(f)
        params = data.get("parameters", {})
        mesh = data.get("mesh", {})
        return (
            data.get("status") == "completed"
            and abs(float(params.get("sigma", float("nan"))) - float(args.sigma)) < 1e-12
            and abs(float(params.get("epsilon", float("nan"))) - float(args.epsilon)) < 1e-12
            and int(params.get("truncation_M", -1)) == int(args.truncation_M)
            and abs(float(mesh.get("h_min", float("nan"))) - float(args.mesh[0])) < 1e-12
            and abs(float(mesh.get("h_max", float("nan"))) - float(args.mesh[1])) < 1e-12
        )
    except Exception:
        return False


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "n/a") for key in CSV_COLUMNS})


def write_failure_log(path: Path, exc: BaseException) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(f"\n[{datetime.now().isoformat()}] failed\n")
        f.write(repr(exc))
        f.write("\n")
        f.write(traceback.format_exc())
        f.write("\n")


def vec_array(fun: Any) -> np.ndarray:
    try:
        return fun.vector.array
    except AttributeError:
        return fun.x.array


def triangulation(mesh: Any) -> mtri.Triangulation:
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    cells = mesh.topology.connectivity(tdim, 0).array.reshape(-1, 3)
    xy = mesh.geometry.x[:, :2]
    return mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=cells)


def plot_vorticity_heatmap(mesh: Any, omega: Any, out_base: Path, args: argparse.Namespace) -> None:
    tri = triangulation(mesh)
    values = np.asarray(vec_array(omega), dtype=float).reshape(-1)
    vmax = max(float(np.max(np.abs(values))), 1e-14)
    levels = np.linspace(-vmax, vmax, 512)
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "figure.dpi": 300,
            "savefig.dpi": 600,
        }
    )
    fig, ax = plt.subplots(figsize=(5.6, 5.15), constrained_layout=True)
    cf = ax.tricontourf(tri, values, levels=levels, cmap="seismic")
    ax.add_patch(plt.Circle((0.0, 0.0), float(args.Rdom), fill=False, lw=0.75, color="0.2", alpha=0.55))
    ax.set_aspect("equal", "box")
    ax.set_xlim(-float(args.Rdom) * 1.02, float(args.Rdom) * 1.02)
    ax.set_ylim(-float(args.Rdom) * 1.02, float(args.Rdom) * 1.02)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("0.25")
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.025)
    cb.set_label(r"$\Omega$", fontsize=13)
    cb.ax.tick_params(labelsize=11, width=0.6, length=2.5)
    cb.outline.set_linewidth(0.6)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def build_manifest(args: argparse.Namespace, run_tag: str) -> dict[str, Any]:
    h_min, h_max = float(args.mesh[0]), float(args.mesh[1])
    fourier_cutoff = 2 * int(args.truncation_M)
    coeffs = coeffs_for_regularized_dirac_pair(args.sigma, args.truncation_M, args.epsilon)
    theta = np.linspace(0.0, 2.0 * np.pi, 4097)
    vals = boundary_values(theta, args.sigma, args.truncation_M, args.epsilon)
    return {
        "status": "initialized" if args.execute else "dry_run",
        "run_tag": run_tag,
        "boundary_formula": "sigma * (1 + 2 * sum_{m=1}^M exp(-4 epsilon^2 m^2) cos(2m theta))",
        "parameters": {
            "sigma": float(args.sigma),
            "epsilon": float(args.epsilon),
            "truncation_M": int(args.truncation_M),
            "fourier_cutoff": fourier_cutoff,
        },
        "baseline": {
            "mu": float(args.mu),
            "R": float(args.R),
            "Rdom": float(args.Rdom),
            "tol_newton": float(args.tol_newton),
            "max_newton": int(args.max_newton),
            "sigma_steps": int(args.sigma_steps),
        },
        "mesh": {"h_min": h_min, "h_max": h_max},
        "boundary_diagnostics": {
            "min_barOmega": float(np.min(vals)),
            "max_barOmega": float(np.max(vals)),
            "mean_sampled_barOmega": float(np.mean(vals)),
            "nonzero_positive_coefficients": {str(k): float(v.real) for k, v in coeffs.items() if abs(v) > 0.0},
        },
        "planned_outputs": {
            "data_csv": relpath(BRANCH_DIR / "data" / f"{run_tag}_profile_summary.csv"),
            "checkpoint_dir": relpath(BRANCH_DIR / "checkpoints" / run_tag),
            "plot_dir": relpath(BRANCH_DIR / "plots" / run_tag),
            "log_file": relpath(BRANCH_DIR / "logs" / f"{run_tag}.log"),
        },
    }


def run_profile(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    add_baseline_to_path()
    from io_utils import write_xdmf
    from residuals_quick import residuals_quick

    run_tag = manifest["run_tag"]
    h_min, h_max = float(args.mesh[0]), float(args.mesh[1])
    mesh_id = mesh_label(h_min, h_max)
    param_id = (
        f"sigma{safe_float_label(args.sigma)}_eps{safe_float_label(args.epsilon)}_"
        f"M{int(args.truncation_M)}_{mesh_id}"
    )
    checkpoint_dir = BRANCH_DIR / "checkpoints" / run_tag
    plot_dir = BRANCH_DIR / "plots" / run_tag
    data_csv = BRANCH_DIR / "data" / f"{run_tag}_profile_summary.csv"
    metadata_path = checkpoint_dir / f"{param_id}_metadata.json"
    profile_xdmf = checkpoint_dir / f"{param_id}_profile.xdmf"
    plot_base = plot_dir / f"{param_id}_omega_heatmap"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if completed_metadata(metadata_path, args) and not args.force:
        print(f"[resume] using completed profile metadata: {metadata_path}")
        return {
            "status": "skipped_completed",
            "metadata_checkpoint": metadata_path,
            "profile_checkpoint": profile_xdmf,
            "plot_path": plot_base.with_suffix(".pdf"),
        }

    metadata = dict(manifest)
    metadata.update(
        {
            "status": "started",
            "started_at": datetime.now().isoformat(),
            "profile_checkpoint": relpath(profile_xdmf),
            "plot_path": relpath(plot_base.with_suffix(".pdf")),
        }
    )
    write_json(metadata_path, jsonable(metadata))

    t0 = time.perf_counter()
    coeffs = coeffs_for_regularized_dirac_pair(args.sigma, args.truncation_M, args.epsilon)
    context_args = argparse.Namespace(
        mesh=[h_min, h_max],
        mu=float(args.mu),
        R=float(args.R),
        Rdom=float(args.Rdom),
        M=2 * int(args.truncation_M),
        tol_newton=float(args.tol_newton),
        max_newton=int(args.max_newton),
        sigma_steps=int(args.sigma_steps),
    )
    context = build_baseline_context(context_args, coeffs)
    stats = mesh_stats(context["mesh"], context["spaces"])
    metadata["mesh_stats"] = stats
    write_json(metadata_path, jsonable(metadata))

    profile_t0 = time.perf_counter()
    omega, psi = context["continuation"].run(context["mesh"], context["spaces"])
    profile_wall = time.perf_counter() - profile_t0
    write_xdmf(str(profile_xdmf), {"Omega": omega, "psi": psi}, context["mesh"])

    pinned = getattr(context["poisson"], "gauge_dof", None)
    residuals = residuals_quick(
        context["mesh"],
        context["spaces"],
        omega,
        psi,
        mu=context["params"].mu,
        R=context["params"].R,
        Rdom=context["params"].Rdom,
        facet_tags=context["facet_tags"],
        outer_tag=1,
        fourier_bc=context["fourier_bc"],
        sigma=1.0,
        pinned_psi_dof=pinned,
    )
    plot_vorticity_heatmap(context["mesh"], omega, plot_base, args)
    wall = time.perf_counter() - t0

    metadata.update(
        {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "wall_time_sec": wall,
            "profile": {
                "wall_time_sec": profile_wall,
                "weak_residuals": {
                    "Omega_l2": float(residuals["norms"]["Omega_l2"]),
                    "Psi_l2": float(residuals["norms"]["Psi_l2"]),
                },
            },
            "outputs": {
                "profile_xdmf": relpath(profile_xdmf),
                "profile_h5": relpath(profile_xdmf.with_suffix(".h5")),
                "heatmap_pdf": relpath(plot_base.with_suffix(".pdf")),
                "heatmap_png": relpath(plot_base.with_suffix(".png")),
                "summary_csv": relpath(data_csv),
            },
        }
    )
    write_json(metadata_path, jsonable(metadata))

    row = {
        "run_tag": run_tag,
        "sigma": float(args.sigma),
        "epsilon": float(args.epsilon),
        "truncation_M": int(args.truncation_M),
        "fourier_cutoff": 2 * int(args.truncation_M),
        "h_min": h_min,
        "h_max": h_max,
        "mesh_vertices": stats["vertices"],
        "mesh_cells": stats["cells"],
        "profile_residual_omega_l2": float(residuals["norms"]["Omega_l2"]),
        "profile_residual_psi_l2": float(residuals["norms"]["Psi_l2"]),
        "profile_wall_time": profile_wall,
        "wall_time": wall,
        "status": "completed",
        "profile_checkpoint": relpath(profile_xdmf),
        "metadata_checkpoint": relpath(metadata_path),
        "plot_path": relpath(plot_base.with_suffix(".pdf")),
        "message": "regularized_dirac_pair_profile_completed",
    }
    append_csv_row(data_csv, row)
    return {
        "status": "completed",
        "metadata_checkpoint": metadata_path,
        "profile_checkpoint": profile_xdmf,
        "plot_path": plot_base.with_suffix(".pdf"),
        "row": row,
    }


def main() -> int:
    ensure_branch_tree(BRANCH_DIR)
    args = parse_args()
    run_tag = args.run_tag
    if run_tag is None:
        run_tag = (
            f"dirac_pair_sigma{safe_float_label(args.sigma)}_eps{safe_float_label(args.epsilon)}_"
            f"M{int(args.truncation_M)}_{mesh_label(float(args.mesh[0]), float(args.mesh[1]))}"
        )
    args.run_tag = run_tag
    manifest = build_manifest(args, run_tag)
    manifest_path = BRANCH_DIR / "logs" / f"{run_tag}_manifest.json"
    write_json(manifest_path, jsonable(manifest))

    if not args.execute:
        print(f"[dry-run] wrote {manifest_path}")
        return 0

    t0 = time.perf_counter()
    try:
        result = run_profile(args, manifest)
        status = result["status"]
    except Exception as exc:
        write_failure_log(BRANCH_DIR / "logs" / f"{run_tag}.log", exc)
        raise
    wall = time.perf_counter() - t0
    if not args.no_registry:
        row = result.get("row", {})
        append_registry_row(
            {
                "date": today(),
                "tag": f"fem-dirac-pair-profile-{run_tag}",
                "codebase": "baseline",
                "script": relpath(Path(__file__)),
                "input_case": "regularized_two_dirac_pair_even_modes",
                "domain": f"disk_Rdom{float(args.Rdom):g}",
                "grid_or_mesh": f"h={float(args.mesh[0]):g}/{float(args.mesh[1]):g}",
                "solver": "baseline_fem_continuation_newton_profile_only",
                "parameters": (
                    f"sigma={float(args.sigma)}; epsilon={float(args.epsilon)}; "
                    f"truncation_M={int(args.truncation_M)}; fourier_cutoff={2 * int(args.truncation_M)}; "
                    f"mu={float(args.mu)}; R={float(args.R)}; sigma_steps={int(args.sigma_steps)}"
                ),
                "wall_time_sec": f"{wall:.6f}",
                "error_metrics": (
                    f"profile_weak_R_Omega={row.get('profile_residual_omega_l2', 'n/a')}; "
                    f"profile_weak_R_Psi={row.get('profile_residual_psi_l2', 'n/a')}"
                ),
                "eigenvalues_summary": "n/a",
                "status": status,
                "notes": f"Profile-only first run for regularized two-Dirac boundary; heatmap={relpath(result['plot_path'])}",
            }
        )
    print(f"[done] status={status}")
    print(f"[done] metadata={result['metadata_checkpoint']}")
    print(f"[done] profile={result['profile_checkpoint']}")
    print(f"[done] heatmap={result['plot_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
