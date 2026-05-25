#!/usr/bin/env python3
"""Eigenvalue spectra around symmetric and symmetry-breaking s=140 profiles.

This exploration reuses the accepted symmetric Branch A profile for
140 cos(3 theta), the previously saved Branch B broken candidates, and optional
new analytic localized perturbations.  Each new perturbation preserves the
boundary data because the perturbing factor vanishes on r=Rdom.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np

BRANCH_DIR = Path(__file__).resolve().parent
MANUSCRIPT_ROOT = BRANCH_DIR.parents[0]
REPO_ROOT = MANUSCRIPT_ROOT.parents[1]
C3_BRANCH_DIR = MANUSCRIPT_ROOT / "c3_spectrum_continuation"
sys.path.insert(0, str(MANUSCRIPT_ROOT))
sys.path.insert(0, str(BRANCH_DIR))
sys.path.insert(0, str(C3_BRANCH_DIR))

from common import add_baseline_to_path, append_registry_row, coeffs_for_exact_c3_cos, today, write_json  # noqa: E402
from run_s140_artificial_perturb_probe import artificial_pattern, make_artificial_initial_guess  # noqa: E402
from run_s140_bifurcation_probe import (  # noqa: E402
    boundary_max_abs_diff,
    c3_fraction_on_probe,
    copy_function,
    linf,
    load_h5_profile,
    plot_comparison,
    rms,
    run_newton_from,
    vec_array,
    write_xdmf_profile,
)
from run_spectrum_continuation import build_baseline_context, mesh_stats  # noqa: E402


PROFILE_COLUMNS = [
    "profile_id",
    "source",
    "status",
    "m_mode",
    "phase",
    "eta",
    "scale",
    "newton_iterations",
    "final_newton_residual",
    "final_relative_step",
    "profile_residual_omega_l2",
    "profile_residual_psi_l2",
    "boundary_max_abs_diff",
    "omega_rms_separation",
    "omega_relative_rms_separation",
    "omega_linf_separation",
    "omega_difference_non3_fraction",
    "nearest_existing_profile",
    "nearest_existing_relative_omega_rms",
    "profile_checkpoint",
    "plot_path",
    "message",
]

EIG_COLUMNS = [
    "profile_id",
    "source",
    "s",
    "h_min",
    "h_max",
    "mesh_vertices",
    "mesh_cells",
    "eigen_index",
    "real_lambda",
    "imag_lambda",
    "symmetric_real_lambda_same_index",
    "symmetric_imag_lambda_same_index",
    "real_minus_symmetric_same_index",
    "imag_minus_symmetric_same_index",
    "profile_residual_omega_l2",
    "profile_residual_psi_l2",
    "omega_relative_rms_separation",
    "omega_difference_non3_fraction",
    "eigs_wall_time_sec",
    "status",
    "profile_checkpoint",
    "message",
]


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def append_rows(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "n/a") for key in columns})


def existing_profile_specs() -> list[dict[str, Any]]:
    return [
        {
            "profile_id": "broken_artificial_m1_phase0_eta0p25",
            "source": "saved_artificial_localized_probe",
            "h5": BRANCH_DIR
            / "checkpoints/branchB_s140_artificial_localized_probe/m1_phase0_eta0p25_profile.h5",
        },
        {
            "profile_id": "broken_eigenmode_mode1_eta0p5_signp1",
            "source": "saved_eigenmode_perturb_probe",
            "h5": BRANCH_DIR
            / "checkpoints/branchB_s140_eigenmode_perturb_probe/mode1_eta0p5_signp1_profile.h5",
        },
        {
            "profile_id": "broken_eigenmode_mode2_eta0p5_signp1",
            "source": "saved_eigenmode23_perturb_probe",
            "h5": BRANCH_DIR
            / "checkpoints/branchB_s140_eigenmode23_perturb_probe/mode2_eta0p5_signp1_profile.h5",
        },
        {
            "profile_id": "broken_eigenmode_mode2_eta0p5_signm1",
            "source": "saved_eigenmode23_perturb_probe",
            "h5": BRANCH_DIR
            / "checkpoints/branchB_s140_eigenmode23_perturb_probe/mode2_eta0p5_signm1_profile.h5",
        },
        {
            "profile_id": "broken_eigenmode_mode3_eta0p5_signp1",
            "source": "saved_eigenmode23_perturb_probe",
            "h5": BRANCH_DIR
            / "checkpoints/branchB_s140_eigenmode23_perturb_probe/mode3_eta0p5_signp1_profile.h5",
        },
        {
            "profile_id": "broken_eigenmode_mode3_eta0p5_signm1",
            "source": "saved_eigenmode23_perturb_probe",
            "h5": BRANCH_DIR
            / "checkpoints/branchB_s140_eigenmode23_perturb_probe/mode3_eta0p5_signm1_profile.h5",
        },
    ]


def load_symmetric_eigenvalues(path: Path, k: int) -> list[complex]:
    import json

    data = json.loads(path.read_text())
    vals = []
    for ev in data.get("eigs", {}).get("eigenvalues", [])[: int(k)]:
        vals.append(complex(float(ev["real"]), float(ev["imag"])))
    return vals


def profile_metrics(context: dict[str, Any], omega_sym: Any, psi_sym: Any, omega: Any, psi: Any) -> dict[str, float]:
    add_baseline_to_path()
    from residuals_quick import residuals_quick

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
    omega_sep = vec_array(omega) - vec_array(omega_sym)
    omega_diff_fun = copy_function(context["spaces"]["Vomega"], omega_sep)
    return {
        "profile_residual_omega_l2": float(residuals["norms"]["Omega_l2"]),
        "profile_residual_psi_l2": float(residuals["norms"]["Psi_l2"]),
        "boundary_max_abs_diff": boundary_max_abs_diff(context["spaces"], context["facet_tags"], omega, omega_sym),
        "omega_rms_separation": rms(omega_sep),
        "omega_relative_rms_separation": rms(omega_sep) / max(rms(vec_array(omega_sym)), 1e-30),
        "omega_linf_separation": linf(omega_sep),
        "omega_difference_non3_fraction": c3_fraction_on_probe(context["spaces"]["Vomega"], omega_diff_fun),
    }


def nearest_profile(
    omega: Any,
    omega_sym: Any,
    accepted: list[tuple[str, Any]],
) -> tuple[str, float]:
    denom = max(rms(vec_array(omega_sym)), 1e-30)
    best_id = "none"
    best_dist = float("inf")
    for profile_id, other in accepted:
        dist = rms(vec_array(omega) - vec_array(other)) / denom
        if dist < best_dist:
            best_id = profile_id
            best_dist = dist
    return best_id, best_dist


def solve_eigenvalues(args: argparse.Namespace, context: dict[str, Any], omega: Any, psi: Any) -> tuple[list[complex], float]:
    add_baseline_to_path()
    from eigs_linearized_new import ManifoldEigenSolver

    t0 = time.perf_counter()
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
        ncv=args.eig_ncv,
        verbose=True,
    )
    wall = time.perf_counter() - t0
    return [complex(m.lam) for m in modes[: int(args.k)]], wall


def completed_profiles(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {row["profile_id"] for row in csv.DictReader(f) if row.get("status", "").startswith("completed")}


def completed_eigs(path: Path, k: int) -> set[str]:
    if not path.exists():
        return set()
    counts: dict[str, int] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("status") == "completed":
                counts[row["profile_id"]] = counts.get(row["profile_id"], 0) + 1
    return {profile_id for profile_id, count in counts.items() if count >= int(k)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--s", type=float, default=140.0)
    parser.add_argument("--mesh", nargs=2, type=float, default=[0.06, 0.06])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--eig-shift", type=float, default=-1.0)
    parser.add_argument("--eig-tol", type=float, default=1e-8)
    parser.add_argument("--eig-max-it", type=int, default=200)
    parser.add_argument("--eig-ncv", type=int, default=None)
    parser.add_argument("--mu", type=float, default=0.8)
    parser.add_argument("--R", type=float, default=150.0)
    parser.add_argument("--Rdom", type=float, default=10.0)
    parser.add_argument("--M", type=int, default=48)
    parser.add_argument("--sigma-steps", type=int, default=21)
    parser.add_argument("--max-newton", type=int, default=24)
    parser.add_argument("--tol-newton", type=float, default=1e-8)
    parser.add_argument("--no-linesearch", action="store_true")
    parser.add_argument("--analytic-modes", nargs="+", type=int, default=[2, 4, 5])
    parser.add_argument("--phases", nargs="+", type=float, default=[0.0])
    parser.add_argument("--etas", nargs="+", type=float, default=[0.25])
    parser.add_argument("--r0", type=float, default=2.5)
    parser.add_argument("--duplicate-relative-tol", type=float, default=1e-3)
    parser.add_argument(
        "--only-profile-ids",
        nargs="+",
        default=None,
        help="If provided, compute eigenvalues only for these profile ids after profile loading/probing.",
    )
    parser.add_argument(
        "--skip-symmetric-duplicates",
        action="store_true",
        help="Do not compute eigenvalues for analytic trials classified as duplicates of an existing profile.",
    )
    parser.add_argument("--run-tag", default="branchB_s140_broken_profile_spectra")
    parser.add_argument(
        "--symmetric-profile-h5",
        type=Path,
        default=C3_BRANCH_DIR / "checkpoints/branchA_h006_refine_s125_140_k10/s140_h0p06_0p06_profile.h5",
    )
    parser.add_argument(
        "--symmetric-metadata",
        type=Path,
        default=C3_BRANCH_DIR / "checkpoints/branchA_h006_refine_s125_140_k10/s140_h0p06_0p06_metadata.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_data = BRANCH_DIR / "data"
    out_logs = BRANCH_DIR / "logs"
    out_plots = BRANCH_DIR / "plots" / args.run_tag
    out_checkpoints = BRANCH_DIR / "checkpoints" / args.run_tag
    for d in (out_data, out_logs, out_plots, out_checkpoints):
        d.mkdir(parents=True, exist_ok=True)

    profile_csv = out_data / f"{args.run_tag}_profile_summary.csv"
    eig_csv = out_data / f"{args.run_tag}_eigenvalues.csv"
    manifest_path = out_logs / f"{args.run_tag}_manifest.json"
    manifest: dict[str, Any] = {
        "status": "dry_run",
        "run_tag": args.run_tag,
        "s": float(args.s),
        "boundary_data": "140 cos(3 theta), encoded as c3=70",
        "mesh": {"h_min": float(args.mesh[0]), "h_max": float(args.mesh[1])},
        "eigs": {"k": int(args.k), "shift": float(args.eig_shift), "tol": float(args.eig_tol)},
        "analytic_perturbations": {
            "formula": "(r/Rdom)^m exp(-(r/r0)^2) (1-(r/Rdom)^2)^2 cos(m theta + phase)",
            "modes": [int(m) for m in args.analytic_modes],
            "phases": [float(p) for p in args.phases],
            "etas": [float(e) for e in args.etas],
            "r0": float(args.r0),
        },
        "outputs": {
            "profile_csv": relpath(profile_csv),
            "eigenvalue_csv": relpath(eig_csv),
            "checkpoints": relpath(out_checkpoints),
            "plots": relpath(out_plots),
        },
    }
    write_json(manifest_path, manifest)
    if not args.execute:
        print(f"[dry-run] wrote {manifest_path}")
        return 0

    add_baseline_to_path()
    t_all = time.perf_counter()
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
    context = build_baseline_context(ns, coeffs_for_exact_c3_cos(args.s))
    stats = mesh_stats(context["mesh"], context["spaces"])
    omega_sym, psi_sym, geom_diff = load_h5_profile(args.symmetric_profile_h5, context["mesh"], context["spaces"])
    symmetric_eigs = load_symmetric_eigenvalues(args.symmetric_metadata, args.k)
    sym_metrics = profile_metrics(context, omega_sym, psi_sym, omega_sym, psi_sym)

    profiles_done = completed_profiles(profile_csv)
    eigs_done = completed_eigs(eig_csv, args.k)
    accepted_profiles: list[tuple[str, Any]] = [("symmetric_reference", omega_sym)]
    profile_records: dict[str, dict[str, Any]] = {}
    profile_fields: dict[str, tuple[Any, Any, str, str]] = {}
    duplicate_profile_ids: set[str] = set()

    for spec in existing_profile_specs():
        if not spec["h5"].exists():
            print(f"[missing] saved profile not found: {spec['h5']}")
            continue
        omega, psi, _ = load_h5_profile(spec["h5"], context["mesh"], context["spaces"])
        metrics = profile_metrics(context, omega_sym, psi_sym, omega, psi)
        nearest_id, nearest_dist = nearest_profile(omega, omega_sym, accepted_profiles)
        profile_id = spec["profile_id"]
        row = {
            "profile_id": profile_id,
            "source": spec["source"],
            "status": "completed_loaded",
            "nearest_existing_profile": nearest_id,
            "nearest_existing_relative_omega_rms": nearest_dist,
            "profile_checkpoint": relpath(spec["h5"]),
            "message": "loaded previously saved symmetry-breaking candidate",
            **metrics,
        }
        if profile_id not in profiles_done:
            append_rows(profile_csv, PROFILE_COLUMNS, [row])
            profiles_done.add(profile_id)
        accepted_profiles.append((profile_id, omega))
        profile_records[profile_id] = row
        profile_fields[profile_id] = (omega, psi, spec["source"], relpath(spec["h5"]))

    for m_mode in args.analytic_modes:
        if int(m_mode) % 3 == 0:
            print(f"[skip] analytic m={m_mode} is C3-compatible")
            continue
        for phase in args.phases:
            for eta in args.etas:
                profile_id = f"analytic_m{m_mode}_phase{phase:g}_eta{eta:g}".replace(".", "p").replace("-", "m")
                profile_xdmf = out_checkpoints / f"{profile_id}_profile.xdmf"
                profile_h5 = profile_xdmf.with_suffix(".h5")
                if profile_h5.exists():
                    omega, psi, _ = load_h5_profile(profile_h5, context["mesh"], context["spaces"])
                    metrics = profile_metrics(context, omega_sym, psi_sym, omega, psi)
                    nearest_id, nearest_dist = nearest_profile(omega, omega_sym, accepted_profiles)
                    if nearest_dist < float(args.duplicate_relative_tol):
                        duplicate_profile_ids.add(profile_id)
                    row = {
                        "profile_id": profile_id,
                        "source": "analytic_localized_perturbation",
                        "status": "completed_loaded_checkpoint",
                        "m_mode": int(m_mode),
                        "phase": float(phase),
                        "eta": float(eta),
                        "nearest_existing_profile": nearest_id,
                        "nearest_existing_relative_omega_rms": nearest_dist,
                        "profile_checkpoint": relpath(profile_h5),
                        "message": "loaded checkpoint from previous run",
                        **metrics,
                    }
                else:
                    row = {
                        "profile_id": profile_id,
                        "source": "analytic_localized_perturbation",
                        "status": "started",
                        "m_mode": int(m_mode),
                        "phase": float(phase),
                        "eta": float(eta),
                    }
                    try:
                        t0 = time.perf_counter()
                        omega0, psi0, scale = make_artificial_initial_guess(
                            context,
                            omega_sym,
                            eta=float(eta),
                            m_mode=int(m_mode),
                            phase=float(phase),
                            r0=float(args.r0),
                        )
                        _ = artificial_pattern(
                            context["spaces"]["Vomega"],
                            m_mode=int(m_mode),
                            phase=float(phase),
                            r0=float(args.r0),
                            Rdom=context["params"].Rdom,
                        )
                        omega, psi, history, converged = run_newton_from(
                            context,
                            omega0,
                            psi0,
                            max_newton=int(args.max_newton),
                            tol=float(args.tol_newton),
                            linesearch=not args.no_linesearch,
                        )
                        write_xdmf_profile(profile_xdmf, context["mesh"], omega, psi)
                        metrics = profile_metrics(context, omega_sym, psi_sym, omega, psi)
                        nearest_id, nearest_dist = nearest_profile(omega, omega_sym, accepted_profiles)
                        final = history[-1]
                        duplicate = nearest_dist < float(args.duplicate_relative_tol)
                        if duplicate:
                            duplicate_profile_ids.add(profile_id)
                        row.update(
                            {
                                "status": "completed_duplicate" if duplicate else "completed_distinct_candidate",
                                "scale": scale,
                                "newton_iterations": int(final["iteration"]),
                                "final_newton_residual": float(final["residual_norm"]),
                                "final_relative_step": float(final["relative_step"]),
                                "nearest_existing_profile": nearest_id,
                                "nearest_existing_relative_omega_rms": nearest_dist,
                                "profile_checkpoint": relpath(profile_h5),
                                "plot_path": relpath((out_plots / profile_id).with_suffix(".png")),
                                "message": f"converged={converged}; wall_time={time.perf_counter() - t0:.3f}",
                                **metrics,
                            }
                        )
                        plot_comparison(
                            out_plots / profile_id,
                            context["mesh"],
                            omega_sym,
                            omega,
                            r"$\sigma = 140$: bifurcation of vorticity profiles",
                        )
                    except Exception as exc:
                        row.update({"status": "failed", "message": repr(exc)})
                        append_rows(profile_csv, PROFILE_COLUMNS, [row])
                        print(f"[profile failed] {profile_id}: {exc!r}")
                        continue

                if profile_id not in profiles_done:
                    append_rows(profile_csv, PROFILE_COLUMNS, [row])
                    profiles_done.add(profile_id)
                accepted_profiles.append((profile_id, omega))
                profile_records[profile_id] = row
                profile_fields[profile_id] = (omega, psi, "analytic_localized_perturbation", relpath(profile_h5))

    selected_ids = set(args.only_profile_ids) if args.only_profile_ids else None
    for profile_id, (omega, psi, source, checkpoint) in profile_fields.items():
        if selected_ids is not None and profile_id not in selected_ids:
            print(f"[skip] {profile_id}: not selected")
            continue
        if args.skip_symmetric_duplicates and profile_id in duplicate_profile_ids:
            print(f"[skip] {profile_id}: duplicate of existing profile")
            continue
        if profile_id in eigs_done:
            print(f"[resume] eigs already completed for {profile_id}")
            continue
        record = profile_records[profile_id]
        try:
            lambdas, eig_wall = solve_eigenvalues(args, context, omega, psi)
            rows = []
            for idx, lam in enumerate(lambdas, start=1):
                sym_lam = symmetric_eigs[idx - 1] if idx <= len(symmetric_eigs) else complex(float("nan"), float("nan"))
                rows.append(
                    {
                        "profile_id": profile_id,
                        "source": source,
                        "s": float(args.s),
                        "h_min": float(args.mesh[0]),
                        "h_max": float(args.mesh[1]),
                        "mesh_vertices": stats["vertices"],
                        "mesh_cells": stats["cells"],
                        "eigen_index": idx,
                        "real_lambda": float(lam.real),
                        "imag_lambda": float(lam.imag),
                        "symmetric_real_lambda_same_index": float(sym_lam.real),
                        "symmetric_imag_lambda_same_index": float(sym_lam.imag),
                        "real_minus_symmetric_same_index": float(lam.real - sym_lam.real),
                        "imag_minus_symmetric_same_index": float(lam.imag - sym_lam.imag),
                        "profile_residual_omega_l2": record.get("profile_residual_omega_l2", "n/a"),
                        "profile_residual_psi_l2": record.get("profile_residual_psi_l2", "n/a"),
                        "omega_relative_rms_separation": record.get("omega_relative_rms_separation", "n/a"),
                        "omega_difference_non3_fraction": record.get("omega_difference_non3_fraction", "n/a"),
                        "eigs_wall_time_sec": eig_wall,
                        "status": "completed",
                        "profile_checkpoint": checkpoint,
                        "message": "linearized spectrum around symmetry-breaking profile",
                    }
                )
            append_rows(eig_csv, EIG_COLUMNS, rows)
            eigs_done.add(profile_id)
            print(f"[eigs] {profile_id}: min Re(lambda)={min(z.real for z in lambdas):.12g}")
        except Exception as exc:
            append_rows(
                eig_csv,
                EIG_COLUMNS,
                [
                    {
                        "profile_id": profile_id,
                        "source": source,
                        "s": float(args.s),
                        "h_min": float(args.mesh[0]),
                        "h_max": float(args.mesh[1]),
                        "status": "failed",
                        "profile_checkpoint": checkpoint,
                        "message": repr(exc),
                    }
                ],
            )
            print(f"[eigs failed] {profile_id}: {exc!r}")
            (out_logs / f"{args.run_tag}_failures.log").open("a").write(
                f"\n[{datetime.now().isoformat()}] {profile_id}\n{traceback.format_exc()}\n"
            )

    manifest.update(
        {
            "status": "completed",
            "mesh_stats": stats,
            "profile_checkpoint_geometry_diff": geom_diff,
            "symmetric_profile_metrics": sym_metrics,
            "wall_time_sec": time.perf_counter() - t_all,
        }
    )
    write_json(manifest_path, jsonable(manifest))
    append_registry_row(
        {
            "date": today(),
            "tag": f"fem-branchB-{args.run_tag}",
            "codebase": "baseline",
            "script": relpath(Path(__file__)),
            "input_case": "s_cos3theta_s140",
            "domain": f"disk_Rdom{float(args.Rdom):g}",
            "grid_or_mesh": f"h={float(args.mesh[0]):g}/{float(args.mesh[1]):g}",
            "solver": "fixed_boundary_newton_analytic_perturbations_plus_shift_invert_eigs",
            "parameters": (
                f"k={int(args.k)}; eig_shift={float(args.eig_shift)}; "
                f"analytic_modes={args.analytic_modes}; etas={args.etas}; phases={args.phases}"
            ),
            "wall_time_sec": f"{manifest['wall_time_sec']:.6f}",
            "error_metrics": f"symmetric_R_Omega={sym_metrics['profile_residual_omega_l2']:.6e}",
            "eigenvalues_summary": f"profile_spectra_csv={relpath(eig_csv)}",
            "status": "completed",
            "notes": "Spectra around saved and newly probed s=140 symmetry-breaking FEM profiles.",
        }
    )
    print(f"[done] profiles={profile_csv}")
    print(f"[done] eigenvalues={eig_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
