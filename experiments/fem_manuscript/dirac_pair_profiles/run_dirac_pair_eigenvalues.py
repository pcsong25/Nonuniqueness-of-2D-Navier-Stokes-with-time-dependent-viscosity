#!/usr/bin/env python3
"""Compute linearized eigenvalues around a saved regularized two-Dirac FEM profile."""

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

import h5py
import numpy as np

MANUSCRIPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MANUSCRIPT_ROOT.parents[1]
C3_BRANCH_DIR = MANUSCRIPT_ROOT / "c3_spectrum_continuation"
sys.path.insert(0, str(MANUSCRIPT_ROOT))
sys.path.insert(0, str(C3_BRANCH_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import add_baseline_to_path, append_registry_row, ensure_branch_tree, mesh_label, today, write_json  # noqa: E402
from run_dirac_pair_profile import coeffs_for_regularized_dirac_pair, safe_float_label  # noqa: E402
from run_spectrum_continuation import build_baseline_context, jsonable, mesh_stats  # noqa: E402


BRANCH_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_TAG = "dirac_pair_sigma1_eps0p1_M20_h0p06_0p06"
DEFAULT_PROFILE_H5 = (
    BRANCH_DIR
    / "checkpoints"
    / DEFAULT_RUN_TAG
    / "sigma1_eps0p1_M20_h0p06_0p06_profile.h5"
)
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
    "eigen_index",
    "real_lambda",
    "imag_lambda",
    "eig_residual",
    "profile_residual_omega_l2",
    "profile_residual_psi_l2",
    "eigs_wall_time",
    "wall_time",
    "status",
    "profile_checkpoint",
    "metadata_checkpoint",
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
    parser.add_argument("--profile-h5", type=Path, default=DEFAULT_PROFILE_H5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--eig-shift", type=float, default=-1.0)
    parser.add_argument("--eig-tol", type=float, default=1e-8)
    parser.add_argument("--eig-max-it", type=int, default=300)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-registry", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually run the eigensolve; otherwise write a dry-run manifest.")
    return parser.parse_args()


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def vec_array(fun: Any) -> np.ndarray:
    try:
        return fun.vector.array
    except AttributeError:
        return fun.x.array


def ghost_forward(fun: Any) -> None:
    from petsc4py import PETSc

    try:
        fun.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    except AttributeError:
        if hasattr(fun.x, "scatter_forward"):
            fun.x.scatter_forward()


def copy_function(space: Any, values: np.ndarray):
    from dolfinx import fem

    out = fem.Function(space)
    vec_array(out)[:] = np.asarray(values).reshape(-1)
    ghost_forward(out)
    return out


def load_h5_profile(path: Path, mesh: Any, spaces: dict[str, Any]):
    with h5py.File(path, "r") as h5:
        saved_geom = np.asarray(h5["/Mesh/mesh/geometry"][:], dtype=float)
        current_geom = np.asarray(mesh.geometry.x[:, : saved_geom.shape[1]], dtype=float)
        geom_diff = float(np.max(np.abs(current_geom - saved_geom)))
        if geom_diff > 1e-12:
            raise RuntimeError(f"profile checkpoint mesh mismatch: max geometry diff={geom_diff:.3e}")
        omega = copy_function(spaces["Vomega"], np.asarray(h5["/Function/Omega/0"][:], dtype=float))
        psi = copy_function(spaces["Vpsi"], np.asarray(h5["/Function/psi/0"][:], dtype=float))
    return omega, psi, geom_diff


def completed_metadata(path: Path, args: argparse.Namespace) -> bool:
    if not path.exists():
        return False
    try:
        import json

        with path.open() as f:
            data = json.load(f)
        params = data.get("parameters", {})
        eigs = data.get("eigs", {})
        return (
            data.get("status") == "completed"
            and abs(float(params.get("sigma", float("nan"))) - float(args.sigma)) < 1e-12
            and abs(float(params.get("epsilon", float("nan"))) - float(args.epsilon)) < 1e-12
            and int(params.get("truncation_M", -1)) == int(args.truncation_M)
            and int(eigs.get("k_requested", -1)) == int(args.k)
            and abs(float(eigs.get("shift", float("nan"))) - float(args.eig_shift)) < 1e-12
            and abs(float(eigs.get("tol", float("nan"))) - float(args.eig_tol)) < 1e-20
        )
    except Exception:
        return False


def append_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "n/a") for key in CSV_COLUMNS})


def write_failure_log(path: Path, exc: BaseException) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(f"\n[{datetime.now().isoformat()}] failed\n")
        f.write(repr(exc))
        f.write("\n")
        f.write(traceback.format_exc())
        f.write("\n")


def make_run_tag(args: argparse.Namespace) -> str:
    if args.run_tag:
        return args.run_tag
    h_id = mesh_label(float(args.mesh[0]), float(args.mesh[1]))
    return (
        f"dirac_pair_sigma{safe_float_label(args.sigma)}_eps{safe_float_label(args.epsilon)}_"
        f"M{int(args.truncation_M)}_{h_id}_k{int(args.k)}_eigs"
    )


def build_manifest(args: argparse.Namespace, run_tag: str) -> dict[str, Any]:
    return {
        "status": "initialized" if args.execute else "dry_run",
        "run_tag": run_tag,
        "boundary_formula": "sigma * (1 + 2 * sum_{m=1}^M exp(-4 epsilon^2 m^2) cos(2m theta))",
        "parameters": {
            "sigma": float(args.sigma),
            "epsilon": float(args.epsilon),
            "truncation_M": int(args.truncation_M),
            "fourier_cutoff": 2 * int(args.truncation_M),
        },
        "mesh": {"h_min": float(args.mesh[0]), "h_max": float(args.mesh[1])},
        "baseline": {
            "mu": float(args.mu),
            "R": float(args.R),
            "Rdom": float(args.Rdom),
            "tol_newton": float(args.tol_newton),
            "max_newton": int(args.max_newton),
            "sigma_steps": int(args.sigma_steps),
        },
        "eigs": {
            "k_requested": int(args.k),
            "shift": float(args.eig_shift),
            "tol": float(args.eig_tol),
            "max_it": int(args.eig_max_it),
        },
        "profile_h5": relpath(args.profile_h5),
        "planned_outputs": {
            "data_csv": relpath(BRANCH_DIR / "data" / f"{run_tag}_eigenvalues.csv"),
            "metadata": relpath(BRANCH_DIR / "checkpoints" / run_tag / f"{run_tag}_metadata.json"),
            "log_file": relpath(BRANCH_DIR / "logs" / f"{run_tag}.log"),
        },
    }


def run_eigs(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    add_baseline_to_path()
    from eigs_linearized_new import ManifoldEigenSolver
    from residuals_quick import residuals_quick

    run_tag = manifest["run_tag"]
    checkpoint_dir = BRANCH_DIR / "checkpoints" / run_tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = checkpoint_dir / f"{run_tag}_metadata.json"
    data_csv = BRANCH_DIR / "data" / f"{run_tag}_eigenvalues.csv"

    if completed_metadata(metadata_path, args) and not args.force:
        print(f"[resume] using completed eigensolve metadata: {metadata_path}")
        return {"status": "skipped_completed", "metadata_checkpoint": metadata_path, "csv": data_csv}
    if args.force and data_csv.exists():
        data_csv.unlink()

    metadata = dict(manifest)
    metadata.update({"status": "started", "started_at": datetime.now().isoformat()})
    write_json(metadata_path, jsonable(metadata))

    t0 = time.perf_counter()
    coeffs = coeffs_for_regularized_dirac_pair(args.sigma, args.truncation_M, args.epsilon)
    context_args = argparse.Namespace(
        mesh=[float(args.mesh[0]), float(args.mesh[1])],
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
    omega, psi, geom_diff = load_h5_profile(args.profile_h5, context["mesh"], context["spaces"])

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
    metadata.update(
        {
            "mesh_stats": stats,
            "profile_checkpoint_geometry_diff": geom_diff,
            "profile": {
                "weak_residuals": {
                    "Omega_l2": float(residuals["norms"]["Omega_l2"]),
                    "Psi_l2": float(residuals["norms"]["Psi_l2"]),
                }
            },
        }
    )
    write_json(metadata_path, jsonable(metadata))

    eig_t0 = time.perf_counter()
    esolver = ManifoldEigenSolver(
        context["mesh"],
        context["spaces"],
        omega,
        psi,
        context["params"],
        context["facet_tags"],
        outer_tag=1,
    )
    modes = esolver.solve(
        k=int(args.k),
        sigma=float(args.eig_shift),
        tol=float(args.eig_tol),
        max_it=int(args.eig_max_it),
        verbose=True,
    )
    eig_wall = time.perf_counter() - eig_t0
    wall = time.perf_counter() - t0

    rows = []
    eig_values = []
    for idx, mode in enumerate(modes[: int(args.k)], start=1):
        lam = complex(mode.lam)
        eig_values.append({"index": idx, "real": float(lam.real), "imag": float(lam.imag)})
        rows.append(
            {
                "run_tag": run_tag,
                "sigma": float(args.sigma),
                "epsilon": float(args.epsilon),
                "truncation_M": int(args.truncation_M),
                "fourier_cutoff": 2 * int(args.truncation_M),
                "h_min": float(args.mesh[0]),
                "h_max": float(args.mesh[1]),
                "mesh_vertices": stats["vertices"],
                "mesh_cells": stats["cells"],
                "eigen_index": idx,
                "real_lambda": float(lam.real),
                "imag_lambda": float(lam.imag),
                "eig_residual": "n/a",
                "profile_residual_omega_l2": float(residuals["norms"]["Omega_l2"]),
                "profile_residual_psi_l2": float(residuals["norms"]["Psi_l2"]),
                "eigs_wall_time": eig_wall,
                "wall_time": wall,
                "status": "completed",
                "profile_checkpoint": relpath(args.profile_h5),
                "metadata_checkpoint": relpath(metadata_path),
                "message": "dirac_pair_linearized_eigenvalues_completed",
            }
        )
    append_csv_rows(data_csv, rows)

    metadata.update(
        {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "wall_time_sec": wall,
            "outputs": {"eigenvalues_csv": relpath(data_csv)},
            "eigs": {
                **metadata["eigs"],
                "wall_time_sec": eig_wall,
                "k_returned": len(eig_values),
                "eigenvalues": eig_values,
                "eig_residual_note": "not computed; baseline SLEPc tolerance was used",
            },
        }
    )
    write_json(metadata_path, jsonable(metadata))
    return {"status": "completed", "metadata_checkpoint": metadata_path, "csv": data_csv, "rows": rows, "wall_time_sec": wall}


def main() -> int:
    ensure_branch_tree(BRANCH_DIR)
    args = parse_args()
    run_tag = make_run_tag(args)
    args.run_tag = run_tag
    manifest = build_manifest(args, run_tag)
    manifest_path = BRANCH_DIR / "logs" / f"{run_tag}_manifest.json"
    write_json(manifest_path, jsonable(manifest))

    if not args.execute:
        print(f"[dry-run] wrote {manifest_path}")
        return 0

    try:
        result = run_eigs(args, manifest)
    except Exception as exc:
        write_failure_log(BRANCH_DIR / "logs" / f"{run_tag}.log", exc)
        raise

    rows = result.get("rows", [])
    if not args.no_registry:
        eig_summary = "; ".join(
            f"{row['real_lambda']:.6e}{row['imag_lambda']:+.6e}j" for row in rows
        )
        first = rows[0] if rows else {}
        append_registry_row(
            {
                "date": today(),
                "tag": f"fem-dirac-pair-eigs-{run_tag}",
                "codebase": "baseline",
                "script": relpath(Path(__file__)),
                "input_case": "regularized_two_dirac_pair_even_modes",
                "domain": f"disk_Rdom{float(args.Rdom):g}",
                "grid_or_mesh": f"h={float(args.mesh[0]):g}/{float(args.mesh[1]):g}",
                "solver": "baseline_linearized_shift_invert_eigs_from_saved_profile",
                "parameters": (
                    f"sigma={float(args.sigma)}; epsilon={float(args.epsilon)}; "
                    f"truncation_M={int(args.truncation_M)}; k={int(args.k)}; "
                    f"eig_shift={float(args.eig_shift)}; eig_tol={float(args.eig_tol)}"
                ),
                "wall_time_sec": f"{float(result.get('wall_time_sec', 0.0)):.6f}",
                "error_metrics": (
                    f"profile_weak_R_Omega={first.get('profile_residual_omega_l2', 'n/a')}; "
                    f"profile_weak_R_Psi={first.get('profile_residual_psi_l2', 'n/a')}"
                ),
                "eigenvalues_summary": eig_summary,
                "status": result["status"],
                "notes": f"Computed first {int(args.k)} linearized eigenvalues around saved regularized two-Dirac profile.",
            }
        )

    print(f"[done] status={result['status']}")
    print(f"[done] metadata={result['metadata_checkpoint']}")
    print(f"[done] eigenvalues_csv={result['csv']}")
    for row in rows:
        print(f"#{int(row['eigen_index']):2d}: lambda = {row['real_lambda']:.15e} {row['imag_lambda']:+.15e}j")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
