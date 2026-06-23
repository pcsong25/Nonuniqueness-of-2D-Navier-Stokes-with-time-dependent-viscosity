#!/usr/bin/env python3
"""Seeded sigma ladder for regularized two-Dirac FEM profiles and spectra."""

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

import numpy as np

MANUSCRIPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MANUSCRIPT_ROOT.parents[1]
C3_BRANCH_DIR = MANUSCRIPT_ROOT / "c3_spectrum_continuation"
SYMB_DIR = MANUSCRIPT_ROOT / "symmetry_breaking_profiles"
sys.path.insert(0, str(MANUSCRIPT_ROOT))
sys.path.insert(0, str(C3_BRANCH_DIR))
sys.path.insert(0, str(SYMB_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import add_baseline_to_path, append_registry_row, ensure_branch_tree, mesh_label, today, write_json  # noqa: E402
from run_dirac_pair_eigenvalues import copy_function, ghost_forward, load_h5_profile, vec_array  # noqa: E402
from run_dirac_pair_profile import coeffs_for_regularized_dirac_pair, plot_vorticity_heatmap, safe_float_label  # noqa: E402
from run_spectrum_continuation import build_baseline_context, jsonable, mesh_stats  # noqa: E402


BRANCH_DIR = Path(__file__).resolve().parent

CSV_COLUMNS = [
    "sigma",
    "epsilon",
    "truncation_M",
    "fourier_cutoff",
    "h_min",
    "h_max",
    "mesh_vertices",
    "mesh_cells",
    "source_sigma",
    "newton_iterations",
    "final_newton_residual",
    "final_relative_step",
    "profile_residual_omega_l2",
    "profile_residual_psi_l2",
    "profile_wall_time",
    "eigs_wall_time",
    "wall_time",
    "min_real_lambda",
    "unstable_count_re_lt_0",
    "status",
    "profile_checkpoint",
    "metadata_checkpoint",
    "heatmap_pdf",
    "message",
]

EIG_COLUMNS = [
    "sigma",
    "epsilon",
    "truncation_M",
    "fourier_cutoff",
    "h_min",
    "h_max",
    "eigen_index",
    "real_lambda",
    "imag_lambda",
    "profile_residual_omega_l2",
    "profile_residual_psi_l2",
    "profile_checkpoint",
    "metadata_checkpoint",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--sigma-values", nargs="+", type=float, default=None)
    parser.add_argument("--start-sigma", type=float, default=30.0)
    parser.add_argument("--max-sigma", type=float, default=150.0)
    parser.add_argument("--sigma-step", type=float, default=20.0)
    parser.add_argument("--seed-sigma", type=float, default=10.0)
    parser.add_argument(
        "--seed-profile-h5",
        type=Path,
        default=BRANCH_DIR
        / "checkpoints/dirac_pair_sigma10_eps0p01_M50_h0p06_0p06/sigma10_eps0p01_M50_h0p06_0p06_profile.h5",
    )
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--truncation-M", type=int, default=50)
    parser.add_argument("--mesh", nargs=2, type=float, metavar=("H_MIN", "H_MAX"), default=[0.06, 0.06])
    parser.add_argument("--mu", type=float, default=0.8)
    parser.add_argument("--R", type=float, default=150.0)
    parser.add_argument("--Rdom", type=float, default=10.0)
    parser.add_argument("--tol-newton", type=float, default=1e-8)
    parser.add_argument("--max-newton", type=int, default=48)
    parser.add_argument("--no-linesearch", action="store_true")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--eig-shift", type=float, default=-1.0)
    parser.add_argument("--eig-tol", type=float, default=1e-8)
    parser.add_argument("--eig-max-it", type=int, default=300)
    parser.add_argument("--stop-min-real-below", type=float, default=0.0)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-registry", action="store_true")
    return parser.parse_args()


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sigma_label(sigma: float) -> str:
    return f"sigma{safe_float_label(float(sigma))}"


def make_run_tag(args: argparse.Namespace) -> str:
    if args.run_tag:
        return args.run_tag
    return (
        f"dirac_pair_eps{safe_float_label(args.epsilon)}_M{int(args.truncation_M)}_"
        f"{mesh_label(float(args.mesh[0]), float(args.mesh[1]))}_sigma_ladder"
    )


def target_sigmas(args: argparse.Namespace) -> list[float]:
    if args.sigma_values:
        return [float(s) for s in args.sigma_values]
    vals = []
    s = float(args.start_sigma)
    while s <= float(args.max_sigma) + 1e-12:
        vals.append(float(s))
        s += float(args.sigma_step)
    return vals


def append_rows(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "n/a") for key in columns})


def write_failure_log(path: Path, exc: BaseException) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(f"\n[{datetime.now().isoformat()}] failed\n")
        f.write(repr(exc))
        f.write("\n")
        f.write(traceback.format_exc())
        f.write("\n")


def set_exact_boundary_values(context: dict[str, Any], omega: Any) -> None:
    from dolfinx import fem

    facets = context["facet_tags"].find(1)
    bdofs = fem.locate_dofs_topological(context["spaces"]["Vomega"], context["mesh"].topology.dim - 1, facets)
    raw = context["spaces"]["Vomega"].tabulate_dof_coordinates()
    if raw.ndim == 1:
        raw = raw.reshape(-1, context["mesh"].geometry.x.shape[1])
    xy = raw[bdofs.astype(np.int64), :2]
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    values = (context["params"].R ** (-1.0 / context["params"].mu)) * context["fourier_bc"].Omega_bar(theta)
    vec_array(omega)[bdofs.astype(np.int64)] = values
    ghost_forward(omega)


def build_context(args: argparse.Namespace, sigma: float) -> dict[str, Any]:
    ns = argparse.Namespace(
        mesh=[float(args.mesh[0]), float(args.mesh[1])],
        mu=float(args.mu),
        R=float(args.R),
        Rdom=float(args.Rdom),
        M=2 * int(args.truncation_M),
        tol_newton=float(args.tol_newton),
        max_newton=int(args.max_newton),
        sigma_steps=2,
    )
    coeffs = coeffs_for_regularized_dirac_pair(float(sigma), int(args.truncation_M), float(args.epsilon))
    return build_baseline_context(ns, coeffs)


def seeded_initial_guess(context: dict[str, Any], prev_omega: Any, source_sigma: float, target_sigma: float):
    ratio = float(target_sigma) / float(source_sigma)
    omega0 = copy_function(context["spaces"]["Vomega"], ratio * vec_array(prev_omega))
    set_exact_boundary_values(context, omega0)
    psi0 = context["poisson"].solve_state(omega0, sigma=1.0)
    return omega0, psi0


def completed_metadata(path: Path, args: argparse.Namespace, sigma: float) -> bool:
    if not path.exists():
        return False
    try:
        import json

        with path.open() as f:
            data = json.load(f)
        return (
            data.get("status") == "completed"
            and abs(float(data.get("sigma", float("nan"))) - float(sigma)) < 1e-12
            and int(data.get("eigs", {}).get("k_requested", -1)) == int(args.k)
        )
    except Exception:
        return False


def solve_point(
    args: argparse.Namespace,
    run_tag: str,
    sigma: float,
    source_sigma: float,
    source_profile_h5: Path,
    out_dirs: dict[str, Path],
) -> dict[str, Any]:
    add_baseline_to_path()
    from eigs_linearized_new import ManifoldEigenSolver
    from io_utils import write_xdmf
    from residuals_quick import residuals_quick
    from run_s140_bifurcation_probe import run_newton_from

    h_min, h_max = float(args.mesh[0]), float(args.mesh[1])
    point_id = f"{sigma_label(sigma)}_eps{safe_float_label(args.epsilon)}_M{int(args.truncation_M)}_{mesh_label(h_min, h_max)}"
    profile_xdmf = out_dirs["checkpoints"] / f"{point_id}_profile.xdmf"
    metadata_path = out_dirs["checkpoints"] / f"{point_id}_metadata.json"
    heatmap_base = out_dirs["plots"] / f"{point_id}_omega_heatmap"

    if completed_metadata(metadata_path, args, sigma) and not args.force:
        context = build_context(args, sigma)
        omega, _psi, _ = load_h5_profile(profile_xdmf.with_suffix(".h5"), context["mesh"], context["spaces"])
        return {
            "status": "skipped_completed",
            "sigma": sigma,
            "profile_h5": profile_xdmf.with_suffix(".h5"),
            "omega": omega,
            "metadata_path": metadata_path,
            "min_real_lambda": None,
        }

    metadata: dict[str, Any] = {
        "status": "started",
        "run_tag": run_tag,
        "sigma": float(sigma),
        "source_sigma": float(source_sigma),
        "source_profile_h5": relpath(source_profile_h5),
        "epsilon": float(args.epsilon),
        "truncation_M": int(args.truncation_M),
        "fourier_cutoff": 2 * int(args.truncation_M),
        "mesh": {"h_min": h_min, "h_max": h_max},
        "started_at": datetime.now().isoformat(),
    }
    write_json(metadata_path, jsonable(metadata))

    t0 = time.perf_counter()
    context = build_context(args, sigma)
    stats = mesh_stats(context["mesh"], context["spaces"])
    prev_omega, _prev_psi, geom_diff = load_h5_profile(source_profile_h5, context["mesh"], context["spaces"])
    omega0, psi0 = seeded_initial_guess(context, prev_omega, source_sigma, sigma)
    profile_t0 = time.perf_counter()
    omega, psi, history, converged = run_newton_from(
        context,
        omega0,
        psi0,
        max_newton=int(args.max_newton),
        tol=float(args.tol_newton),
        linesearch=not args.no_linesearch,
    )
    profile_wall = time.perf_counter() - profile_t0
    if not converged:
        metadata["status"] = "failed_newton_convergence"
        metadata["newton_history"] = history
        write_json(metadata_path, jsonable(metadata))
        raise RuntimeError(f"seeded Newton did not converge for sigma={sigma:g}")

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
    plot_vorticity_heatmap(context["mesh"], omega, heatmap_base, args)

    final = history[-1]
    profile_metadata = {
        "wall_time_sec": profile_wall,
        "newton_iterations": int(final["iteration"]),
        "final_newton_residual": float(final["residual_norm"]),
        "final_relative_step": float(final["relative_step"]),
        "weak_residuals": {
            "Omega_l2": float(residuals["norms"]["Omega_l2"]),
            "Psi_l2": float(residuals["norms"]["Psi_l2"]),
        },
    }
    metadata.update(
        {
            "status": "profile_completed_eigs_started",
            "source_profile_geometry_diff": geom_diff,
            "mesh_stats": stats,
            "profile": profile_metadata,
            "outputs": {
                "profile_xdmf": relpath(profile_xdmf),
                "profile_h5": relpath(profile_xdmf.with_suffix(".h5")),
                "heatmap_pdf": relpath(heatmap_base.with_suffix(".pdf")),
                "heatmap_png": relpath(heatmap_base.with_suffix(".png")),
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
    modes = esolver.solve(k=int(args.k), sigma=float(args.eig_shift), tol=float(args.eig_tol), max_it=int(args.eig_max_it), verbose=True)
    eig_wall = time.perf_counter() - eig_t0
    eigs = [complex(mode.lam) for mode in modes[: int(args.k)]]
    min_real = float(min(lam.real for lam in eigs))
    unstable_count = int(sum(1 for lam in eigs if lam.real < 0.0))
    wall = time.perf_counter() - t0

    metadata.update(
        {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "wall_time_sec": wall,
            "source_profile_geometry_diff": geom_diff,
            "mesh_stats": stats,
            "profile": profile_metadata,
            "eigs": {
                "wall_time_sec": eig_wall,
                "k_requested": int(args.k),
                "shift": float(args.eig_shift),
                "tol": float(args.eig_tol),
                "min_real_lambda": min_real,
                "unstable_count_re_lt_0": unstable_count,
                "eigenvalues": [
                    {"index": idx, "real": float(lam.real), "imag": float(lam.imag)}
                    for idx, lam in enumerate(eigs, start=1)
                ],
            },
            "outputs": {
                "profile_xdmf": relpath(profile_xdmf),
                "profile_h5": relpath(profile_xdmf.with_suffix(".h5")),
                "heatmap_pdf": relpath(heatmap_base.with_suffix(".pdf")),
                "heatmap_png": relpath(heatmap_base.with_suffix(".png")),
            },
        }
    )
    write_json(metadata_path, jsonable(metadata))

    summary_row = {
        "sigma": float(sigma),
        "epsilon": float(args.epsilon),
        "truncation_M": int(args.truncation_M),
        "fourier_cutoff": 2 * int(args.truncation_M),
        "h_min": h_min,
        "h_max": h_max,
        "mesh_vertices": stats["vertices"],
        "mesh_cells": stats["cells"],
        "source_sigma": float(source_sigma),
        "newton_iterations": int(final["iteration"]),
        "final_newton_residual": float(final["residual_norm"]),
        "final_relative_step": float(final["relative_step"]),
        "profile_residual_omega_l2": float(residuals["norms"]["Omega_l2"]),
        "profile_residual_psi_l2": float(residuals["norms"]["Psi_l2"]),
        "profile_wall_time": profile_wall,
        "eigs_wall_time": eig_wall,
        "wall_time": wall,
        "min_real_lambda": min_real,
        "unstable_count_re_lt_0": unstable_count,
        "status": "completed",
        "profile_checkpoint": relpath(profile_xdmf),
        "metadata_checkpoint": relpath(metadata_path),
        "heatmap_pdf": relpath(heatmap_base.with_suffix(".pdf")),
        "message": "seeded_sigma_ladder_completed",
    }
    eig_rows = [
        {
            "sigma": float(sigma),
            "epsilon": float(args.epsilon),
            "truncation_M": int(args.truncation_M),
            "fourier_cutoff": 2 * int(args.truncation_M),
            "h_min": h_min,
            "h_max": h_max,
            "eigen_index": idx,
            "real_lambda": float(lam.real),
            "imag_lambda": float(lam.imag),
            "profile_residual_omega_l2": float(residuals["norms"]["Omega_l2"]),
            "profile_residual_psi_l2": float(residuals["norms"]["Psi_l2"]),
            "profile_checkpoint": relpath(profile_xdmf),
            "metadata_checkpoint": relpath(metadata_path),
            "status": "completed",
        }
        for idx, lam in enumerate(eigs, start=1)
    ]
    print(f"[ladder] sigma={sigma:g} minRe={min_real:.6e} unstable_count={unstable_count}")
    return {
        "summary_row": summary_row,
        "eig_rows": eig_rows,
        "metadata": metadata,
        "profile_h5": profile_xdmf.with_suffix(".h5"),
        "omega": omega,
        "min_real_lambda": min_real,
        "unstable_count": unstable_count,
    }


def main() -> int:
    ensure_branch_tree(BRANCH_DIR)
    args = parse_args()
    run_tag = make_run_tag(args)
    out_dirs = {
        "data": BRANCH_DIR / "data",
        "checkpoints": BRANCH_DIR / "checkpoints" / run_tag,
        "plots": BRANCH_DIR / "plots" / run_tag,
        "logs": BRANCH_DIR / "logs",
    }
    for path in out_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    sigmas = target_sigmas(args)
    manifest = {
        "status": "initialized" if args.execute else "dry_run",
        "run_tag": run_tag,
        "sigma_values": sigmas,
        "seed_sigma": float(args.seed_sigma),
        "seed_profile_h5": relpath(args.seed_profile_h5),
        "epsilon": float(args.epsilon),
        "truncation_M": int(args.truncation_M),
        "mesh": {"h_min": float(args.mesh[0]), "h_max": float(args.mesh[1])},
        "stop_min_real_below": float(args.stop_min_real_below),
    }
    manifest_path = out_dirs["logs"] / f"{run_tag}_manifest.json"
    write_json(manifest_path, jsonable(manifest))
    if not args.execute:
        print(f"[dry-run] wrote {manifest_path}")
        return 0

    summary_csv = out_dirs["data"] / f"{run_tag}_summary.csv"
    eig_csv = out_dirs["data"] / f"{run_tag}_eigenvalues.csv"
    if args.force:
        for path in (summary_csv, eig_csv):
            if path.exists():
                path.unlink()

    prev_sigma = float(args.seed_sigma)
    prev_h5 = Path(args.seed_profile_h5)
    results = []
    status = "completed"
    t0 = time.perf_counter()
    for sigma in sigmas:
        try:
            result = solve_point(args, run_tag, float(sigma), prev_sigma, prev_h5, out_dirs)
            append_rows(summary_csv, CSV_COLUMNS, [result["summary_row"]])
            append_rows(eig_csv, EIG_COLUMNS, result["eig_rows"])
            results.append(result)
            prev_sigma = float(sigma)
            prev_h5 = result["profile_h5"]
            if float(result["min_real_lambda"]) < float(args.stop_min_real_below):
                status = "stopped_after_crossing"
                print(f"[stop] minRe={result['min_real_lambda']:.6e} below {args.stop_min_real_below:g}")
                break
        except Exception as exc:
            status = "partial_failed"
            write_failure_log(out_dirs["logs"] / f"{run_tag}.log", exc)
            print(f"[failed] sigma={sigma:g}: {exc!r}")
            break

    if not args.no_registry:
        if results:
            last = results[-1]["summary_row"]
            eig_summary = (
                f"last_sigma={last['sigma']}; min_Re={last['min_real_lambda']}; "
                f"unstable_count={last['unstable_count_re_lt_0']}"
            )
            err = (
                f"last_profile_weak_R_Omega={last['profile_residual_omega_l2']}; "
                f"last_profile_weak_R_Psi={last['profile_residual_psi_l2']}"
            )
        else:
            eig_summary = "n/a"
            err = "n/a"
        append_registry_row(
            {
                "date": today(),
                "tag": f"fem-dirac-pair-sigma-ladder-{run_tag}",
                "codebase": "baseline",
                "script": relpath(Path(__file__)),
                "input_case": "regularized_two_dirac_pair_even_modes_sigma_ladder",
                "domain": f"disk_Rdom{float(args.Rdom):g}",
                "grid_or_mesh": f"h={float(args.mesh[0]):g}/{float(args.mesh[1]):g}",
                "solver": "seeded_baseline_newton_plus_shift_invert_eigs",
                "parameters": f"epsilon={args.epsilon}; M={args.truncation_M}; sigma_values={sigmas}; seed_sigma={args.seed_sigma}; k={args.k}",
                "wall_time_sec": f"{time.perf_counter() - t0:.6f}",
                "error_metrics": err,
                "eigenvalues_summary": eig_summary,
                "status": status,
                "notes": f"Summary={relpath(summary_csv)}; eigenvalues={relpath(eig_csv)}",
            }
        )

    print(f"[done] status={status}")
    print(f"[done] summary_csv={summary_csv}")
    print(f"[done] eigenvalues_csv={eig_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
