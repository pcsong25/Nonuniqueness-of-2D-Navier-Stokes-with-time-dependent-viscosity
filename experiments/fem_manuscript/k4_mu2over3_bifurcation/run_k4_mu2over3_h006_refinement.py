#!/usr/bin/env python3
"""Refine the mu=2/3, s cos(4 theta) bifurcation scout from h=0.08 to h=0.06.

This script transfers the accepted h=0.08 symmetric and symmetry-breaking
profiles at s=240 onto a new h=0.06 FEM mesh, resets the exact boundary values,
recomputes psi with the baseline Poisson solver, and then runs fixed-boundary
Newton.  It then computes the first 10 eigenvalues of the refined symmetric
profile and compares h=0.08 versus h=0.06 profile-separation diagnostics.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

BRANCH_DIR = Path(__file__).resolve().parent
MANUSCRIPT_ROOT = BRANCH_DIR.parents[0]
REPO_ROOT = MANUSCRIPT_ROOT.parents[1]
C3_BRANCH_DIR = MANUSCRIPT_ROOT / "c3_spectrum_continuation"
SYMB_DIR = MANUSCRIPT_ROOT / "symmetry_breaking_profiles"
sys.path.insert(0, str(MANUSCRIPT_ROOT))
sys.path.insert(0, str(C3_BRANCH_DIR))
sys.path.insert(0, str(SYMB_DIR))

from common import add_baseline_to_path, append_registry_row, coeffs_for_exact_cos_mode, mesh_label, today, write_json  # noqa: E402
from run_k4_mu2over3_bifurcation import (  # noqa: E402
    EIG_COLUMNS,
    PERT_COLUMNS,
    append_rows,
    jsonable,
    plot_profile_comparison,
    relpath,
    set_exact_boundary_values,
    symmetry_fraction_on_probe,
)
from run_s140_bifurcation_probe import boundary_max_abs_diff, copy_function, linf, rms, run_newton_from, vec_array, write_xdmf_profile  # noqa: E402
from run_s140_artificial_perturb_probe import diff_function  # noqa: E402
from run_spectrum_continuation import build_baseline_context, mesh_stats  # noqa: E402


COMPARE_COLUMNS = [
    "mesh_id",
    "s",
    "mode",
    "mu",
    "unstable_count",
    "min_real_lambda",
    "omega_relative_rms_separation",
    "omega_rms_separation",
    "omega_linf_separation",
    "psi_relative_rms_separation",
    "psi_rms_separation",
    "psi_linf_separation",
    "omega_c4_fraction_symmetric",
    "omega_c4_fraction_broken",
    "omega_c4_fraction_excess",
    "omega_difference_non4_fraction",
    "psi_c4_fraction_symmetric",
    "psi_c4_fraction_broken",
    "psi_c4_fraction_excess",
    "psi_difference_non4_fraction",
    "profile_residual_omega_l2_symmetric",
    "profile_residual_psi_l2_symmetric",
    "profile_residual_omega_l2_broken",
    "profile_residual_psi_l2_broken",
    "boundary_max_abs_diff",
    "symmetric_profile_checkpoint",
    "broken_profile_checkpoint",
    "plot_path",
    "status",
]


def h5_field(path: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        xy = np.asarray(h5["/Mesh/mesh/geometry"][:, :2], dtype=float)
        vals = np.asarray(h5[f"/Function/{name}/0"][:], dtype=float).reshape(-1)
    return xy, vals


def interpolate_h5_field_to_space(path: Path, name: str, V: Any) -> Any:
    xy_src, vals_src = h5_field(path, name)
    coords = V.tabulate_dof_coordinates()
    if coords.ndim == 1:
        coords = coords.reshape(-1, V.mesh.geometry.x.shape[1])
    xy_tgt = coords[:, :2]
    linear = LinearNDInterpolator(xy_src, vals_src, fill_value=np.nan)
    vals = np.asarray(linear(xy_tgt), dtype=float)
    missing = ~np.isfinite(vals)
    if np.any(missing):
        nearest = NearestNDInterpolator(xy_src, vals_src)
        vals[missing] = nearest(xy_tgt[missing])
    return copy_function(V, vals)


def transfer_omega_guess(context: dict[str, Any], source_h5: Path) -> Any:
    omega0 = interpolate_h5_field_to_space(source_h5, "Omega", context["spaces"]["Vomega"])
    set_exact_boundary_values(context, omega0)
    return omega0


def solve_from_transferred_omega(
    context: dict[str, Any],
    omega0: Any,
    *,
    max_newton: int,
    tol_newton: float,
    no_linesearch: bool,
    label: str,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    psi0 = context["poisson"].solve_state(omega0, sigma=1.0)
    omega, psi, history, converged = run_newton_from(
        context,
        omega0,
        psi0,
        max_newton=max_newton,
        tol=tol_newton,
        linesearch=not no_linesearch,
    )
    if not converged:
        final = history[-1]
        raise RuntimeError(
            f"{label} Newton did not converge: "
            f"it={final['iteration']}, rel={final['relative_step']}, residual={final['residual_norm']}"
        )
    return omega, psi, history


def residuals_for(context: dict[str, Any], omega: Any, psi: Any) -> dict[str, Any]:
    add_baseline_to_path()
    from residuals_quick import residuals_quick

    pinned = getattr(context["poisson"], "gauge_dof", None)
    return residuals_quick(
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


def solve_eigenvalues(args: argparse.Namespace, context: dict[str, Any], omega: Any, psi: Any) -> list[complex]:
    add_baseline_to_path()
    from eigs_linearized_new import ManifoldEigenSolver

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
        k=int(args.k_eigs),
        sigma=float(args.eig_shift),
        tol=float(args.eig_tol),
        max_it=int(args.eig_max_it),
        verbose=True,
    )
    return [complex(mode.lam) for mode in modes[: int(args.k_eigs)]]


def metrics_row(
    args: argparse.Namespace,
    mesh_id: str,
    omega_sym: Any,
    psi_sym: Any,
    omega_broken: Any,
    psi_broken: Any,
    sym_res: dict[str, Any],
    broken_res: dict[str, Any],
    eigs: list[complex] | None,
    sym_checkpoint: Path,
    broken_checkpoint: Path,
    plot_path: Path,
) -> dict[str, Any]:
    context = args._context
    omega_sep = vec_array(omega_broken) - vec_array(omega_sym)
    psi_sep = vec_array(psi_broken) - vec_array(psi_sym)
    omega_diff = diff_function(context["spaces"]["Vomega"], omega_broken, omega_sym)
    psi_diff = diff_function(context["spaces"]["Vpsi"], psi_broken, psi_sym)
    omega_c4_sym = symmetry_fraction_on_probe(context["spaces"]["Vomega"], omega_sym, symmetry_order=args.mode)
    omega_c4 = symmetry_fraction_on_probe(context["spaces"]["Vomega"], omega_broken, symmetry_order=args.mode)
    psi_c4_sym = symmetry_fraction_on_probe(context["spaces"]["Vpsi"], psi_sym, symmetry_order=args.mode)
    psi_c4 = symmetry_fraction_on_probe(context["spaces"]["Vpsi"], psi_broken, symmetry_order=args.mode)
    return {
        "mesh_id": mesh_id,
        "s": float(args.s),
        "mode": int(args.mode),
        "mu": float(args.mu),
        "unstable_count": "n/a" if eigs is None else int(sum(lam.real < 0.0 for lam in eigs)),
        "min_real_lambda": "n/a" if eigs is None else min(lam.real for lam in eigs),
        "omega_relative_rms_separation": rms(omega_sep) / max(rms(vec_array(omega_sym)), 1e-30),
        "omega_rms_separation": rms(omega_sep),
        "omega_linf_separation": linf(omega_sep),
        "psi_relative_rms_separation": rms(psi_sep) / max(rms(vec_array(psi_sym)), 1e-30),
        "psi_rms_separation": rms(psi_sep),
        "psi_linf_separation": linf(psi_sep),
        "omega_c4_fraction_symmetric": omega_c4_sym,
        "omega_c4_fraction_broken": omega_c4,
        "omega_c4_fraction_excess": omega_c4 - omega_c4_sym,
        "omega_difference_non4_fraction": symmetry_fraction_on_probe(context["spaces"]["Vomega"], omega_diff, symmetry_order=args.mode),
        "psi_c4_fraction_symmetric": psi_c4_sym,
        "psi_c4_fraction_broken": psi_c4,
        "psi_c4_fraction_excess": psi_c4 - psi_c4_sym,
        "psi_difference_non4_fraction": symmetry_fraction_on_probe(context["spaces"]["Vpsi"], psi_diff, symmetry_order=args.mode),
        "profile_residual_omega_l2_symmetric": sym_res["norms"]["Omega_l2"],
        "profile_residual_psi_l2_symmetric": sym_res["norms"]["Psi_l2"],
        "profile_residual_omega_l2_broken": broken_res["norms"]["Omega_l2"],
        "profile_residual_psi_l2_broken": broken_res["norms"]["Psi_l2"],
        "boundary_max_abs_diff": boundary_max_abs_diff(context["spaces"], context["facet_tags"], omega_broken, omega_sym),
        "symmetric_profile_checkpoint": relpath(sym_checkpoint),
        "broken_profile_checkpoint": relpath(broken_checkpoint),
        "plot_path": relpath(plot_path.with_suffix(".png")),
        "status": "completed",
    }


def read_h008_summary(args: argparse.Namespace, plot_path: Path) -> dict[str, Any]:
    eig_rows = []
    with args.h008_eigen_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if abs(float(row["s"]) - float(args.s)) < 1e-12:
                eig_rows.append(row)
    pert_row = None
    with args.h008_perturb_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["trial_id"] == args.h008_trial_id:
                pert_row = row
                break
    if pert_row is None:
        raise RuntimeError(f"missing h=0.08 perturbation row {args.h008_trial_id}")
    eigs = [complex(float(row["real_lambda"]), float(row["imag_lambda"])) for row in eig_rows]
    return {
        "mesh_id": "h0p08_0p08",
        "s": float(args.s),
        "mode": int(args.mode),
        "mu": float(args.mu),
        "unstable_count": int(sum(lam.real < 0.0 for lam in eigs)),
        "min_real_lambda": min(lam.real for lam in eigs),
        "omega_relative_rms_separation": float(pert_row["omega_relative_rms_separation"]),
        "omega_rms_separation": float(pert_row["omega_rms_separation"]),
        "omega_linf_separation": float(pert_row["omega_linf_separation"]),
        "psi_relative_rms_separation": float(pert_row["psi_relative_rms_separation"]),
        "psi_rms_separation": float(pert_row["psi_rms_separation"]),
        "psi_linf_separation": float(pert_row["psi_linf_separation"]),
        "omega_c4_fraction_symmetric": float(pert_row["omega_c4_fraction_symmetric"]),
        "omega_c4_fraction_broken": float(pert_row["omega_c4_fraction_candidate"]),
        "omega_c4_fraction_excess": float(pert_row["omega_c4_fraction_excess"]),
        "omega_difference_non4_fraction": float(pert_row["omega_difference_non4_fraction"]),
        "psi_c4_fraction_symmetric": float(pert_row["psi_c4_fraction_symmetric"]),
        "psi_c4_fraction_broken": float(pert_row["psi_c4_fraction_candidate"]),
        "psi_c4_fraction_excess": float(pert_row["psi_c4_fraction_excess"]),
        "psi_difference_non4_fraction": float(pert_row["psi_difference_non4_fraction"]),
        "profile_residual_omega_l2_symmetric": eig_rows[0]["profile_residual_omega_l2"],
        "profile_residual_psi_l2_symmetric": eig_rows[0]["profile_residual_psi_l2"],
        "profile_residual_omega_l2_broken": float(pert_row["profile_residual_omega_l2"]),
        "profile_residual_psi_l2_broken": float(pert_row["profile_residual_psi_l2"]),
        "boundary_max_abs_diff": float(pert_row["boundary_max_abs_diff"]),
        "symmetric_profile_checkpoint": relpath(args.h008_symmetric_h5.with_suffix(".xdmf")),
        "broken_profile_checkpoint": relpath(args.h008_broken_h5.with_suffix(".xdmf")),
        "plot_path": relpath(plot_path.with_suffix(".png")),
        "status": "completed_reference",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-tag", default="branchB_k4_mu2over3_h006_refine_from_h008_s240")
    parser.add_argument("--s", type=float, default=240.0)
    parser.add_argument("--mode", type=int, default=4)
    parser.add_argument("--mu", type=float, default=2.0 / 3.0)
    parser.add_argument("--mesh", nargs=2, type=float, default=[0.06, 0.06])
    parser.add_argument("--R", type=float, default=150.0)
    parser.add_argument("--Rdom", type=float, default=10.0)
    parser.add_argument("--M", type=int, default=48)
    parser.add_argument("--sigma-steps", type=int, default=21)
    parser.add_argument("--max-newton", type=int, default=64)
    parser.add_argument("--broken-max-newton", type=int, default=64)
    parser.add_argument("--tol-newton", type=float, default=1e-8)
    parser.add_argument("--no-linesearch", action="store_true")
    parser.add_argument("--k-eigs", type=int, default=10)
    parser.add_argument("--eig-shift", type=float, default=-1.0)
    parser.add_argument("--eig-tol", type=float, default=1e-8)
    parser.add_argument("--eig-max-it", type=int, default=300)
    parser.add_argument(
        "--h008-symmetric-h5",
        type=Path,
        default=BRANCH_DIR / "checkpoints/branchB_k4_mu2over3_h008_s200_320_scout/s240_h0p08_0p08_profile.h5",
    )
    parser.add_argument(
        "--h008-broken-h5",
        type=Path,
        default=BRANCH_DIR / "checkpoints/branchB_k4_mu2over3_h008_s200_320_scout/s240_m3_phase0_eta0p5_profile.h5",
    )
    parser.add_argument(
        "--h008-eigen-csv",
        type=Path,
        default=BRANCH_DIR / "data/branchB_k4_mu2over3_h008_s200_320_scout_eigenvalues.csv",
    )
    parser.add_argument(
        "--h008-perturb-csv",
        type=Path,
        default=BRANCH_DIR / "data/branchB_k4_mu2over3_h008_s200_320_scout_perturbations.csv",
    )
    parser.add_argument("--h008-trial-id", default="m3_phase0_eta0p5")
    parser.add_argument("--no-registry", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dirs = {
        "data": BRANCH_DIR / "data",
        "logs": BRANCH_DIR / "logs",
        "plots": BRANCH_DIR / "plots" / args.run_tag,
        "checkpoints": BRANCH_DIR / "checkpoints" / args.run_tag,
    }
    for directory in out_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dirs["logs"] / f"{args.run_tag}_manifest.json"
    manifest = {
        "status": "dry_run",
        "run_tag": args.run_tag,
        "s": args.s,
        "mode": args.mode,
        "mu": args.mu,
        "target_mesh": args.mesh,
        "h008_symmetric_h5": relpath(args.h008_symmetric_h5),
        "h008_broken_h5": relpath(args.h008_broken_h5),
    }
    write_json(manifest_path, manifest)
    if not args.execute:
        print(f"[dry-run] wrote {manifest_path}")
        return 0

    t0 = time.perf_counter()
    ns = argparse.Namespace(
        mesh=[float(args.mesh[0]), float(args.mesh[1])],
        sigma_steps=int(args.sigma_steps),
        mu=float(args.mu),
        R=float(args.R),
        Rdom=float(args.Rdom),
        M=int(args.M),
        tol_newton=float(args.tol_newton),
        max_newton=int(args.max_newton),
    )
    context = build_baseline_context(ns, coeffs_for_exact_cos_mode(args.s, args.mode))
    args._context = context
    stats = mesh_stats(context["mesh"], context["spaces"])
    mesh_id = mesh_label(args.mesh[0], args.mesh[1])

    sym_omega0 = transfer_omega_guess(context, args.h008_symmetric_h5)
    omega_sym, psi_sym, sym_hist = solve_from_transferred_omega(
        context,
        sym_omega0,
        max_newton=int(args.max_newton),
        tol_newton=float(args.tol_newton),
        no_linesearch=bool(args.no_linesearch),
        label="symmetric h=0.06 transfer",
    )
    sym_res = residuals_for(context, omega_sym, psi_sym)
    sym_path = out_dirs["checkpoints"] / f"s{float(args.s):g}_{mesh_id}_symmetric_profile.xdmf"
    write_xdmf_profile(sym_path, context["mesh"], omega_sym, psi_sym)

    eigs = solve_eigenvalues(args, context, omega_sym, psi_sym)
    eig_rows = [
        {
            "s": float(args.s),
            "mode": int(args.mode),
            "mu": float(args.mu),
            "mesh_id": mesh_id,
            "eigen_index": idx,
            "real_lambda": float(lam.real),
            "imag_lambda": float(lam.imag),
            "profile_residual_omega_l2": sym_res["norms"]["Omega_l2"],
            "profile_residual_psi_l2": sym_res["norms"]["Psi_l2"],
            "profile_checkpoint": relpath(sym_path),
            "metadata_checkpoint": relpath(manifest_path),
            "status": "completed",
        }
        for idx, lam in enumerate(eigs, start=1)
    ]
    eig_csv = out_dirs["data"] / f"{args.run_tag}_eigenvalues.csv"
    append_rows(eig_csv, EIG_COLUMNS, eig_rows)

    broken_omega0 = transfer_omega_guess(context, args.h008_broken_h5)
    omega_broken, psi_broken, broken_hist = solve_from_transferred_omega(
        context,
        broken_omega0,
        max_newton=int(args.broken_max_newton),
        tol_newton=float(args.tol_newton),
        no_linesearch=bool(args.no_linesearch),
        label="broken h=0.06 transfer",
    )
    broken_res = residuals_for(context, omega_broken, psi_broken)
    broken_path = out_dirs["checkpoints"] / f"s{float(args.s):g}_{mesh_id}_broken_profile.xdmf"
    write_xdmf_profile(broken_path, context["mesh"], omega_broken, psi_broken)
    plot_path = out_dirs["plots"] / f"s{float(args.s):g}_{mesh_id}_omega_comparison"
    plot_profile_comparison(
        plot_path,
        context["mesh"],
        omega_sym,
        omega_broken,
        r"$\alpha = 3/4$, boundary data = $240\cos(4\theta)$",
    )

    compare_csv = out_dirs["data"] / f"{args.run_tag}_mesh_refinement_comparison.csv"
    h008_plot_ref = BRANCH_DIR / "plots/branchB_k4_mu2over3_h008_s200_320_scout/s240_m3_phase0_eta0p5_omega_comparison.png"
    h008_row = read_h008_summary(args, h008_plot_ref)
    h006_row = metrics_row(
        args,
        mesh_id,
        omega_sym,
        psi_sym,
        omega_broken,
        psi_broken,
        sym_res,
        broken_res,
        eigs,
        sym_path,
        broken_path,
        plot_path,
    )
    append_rows(compare_csv, COMPARE_COLUMNS, [h008_row, h006_row])

    manifest.update(
        {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "wall_time_sec": time.perf_counter() - t0,
            "mesh_stats": stats,
            "symmetric_newton_history": jsonable(sym_hist),
            "broken_newton_history": jsonable(broken_hist),
            "h006_eigenvalues": [{"index": i, "real": float(lam.real), "imag": float(lam.imag)} for i, lam in enumerate(eigs, start=1)],
            "h006_metrics": jsonable(h006_row),
            "h008_reference_metrics": jsonable(h008_row),
            "outputs": {
                "eigenvalues_csv": relpath(eig_csv),
                "comparison_csv": relpath(compare_csv),
                "symmetric_profile": relpath(sym_path),
                "broken_profile": relpath(broken_path),
                "plot": relpath(plot_path.with_suffix(".png")),
            },
        }
    )
    write_json(manifest_path, jsonable(manifest))
    if not args.no_registry:
        append_registry_row(
            {
                "date": today(),
                "tag": f"fem-branchB-{args.run_tag}",
                "codebase": "baseline",
                "script": relpath(Path(__file__)),
                "input_case": "s_cos4theta_mu0.666667",
                "domain": f"disk_Rdom{float(args.Rdom):g}",
                "grid_or_mesh": f"h={float(args.mesh[0]):g}/{float(args.mesh[1]):g}",
                "solver": "h008_to_h006_profile_transfer_eigs_newton_mesh_refinement",
                "parameters": f"s={args.s}; k_eigs={args.k_eigs}; source_trial={args.h008_trial_id}",
                "wall_time_sec": f"{manifest['wall_time_sec']:.6f}",
                "error_metrics": (
                    f"h006_Rsym={sym_res['norms']['Omega_l2']}; "
                    f"h006_Rbroken={broken_res['norms']['Omega_l2']}; "
                    f"h006_relOmega={h006_row['omega_relative_rms_separation']}"
                ),
                "eigenvalues_summary": (
                    f"unstable_count={h006_row['unstable_count']}; "
                    + "; ".join(f"{lam.real:.6e}{lam.imag:+.6e}j" for lam in eigs)
                ),
                "status": "completed",
                "notes": "Transferred h=0.08 C4/mu=2/3 bifurcation candidate to h=0.06 and compared mesh metrics.",
            }
        )
    print(f"[done] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
