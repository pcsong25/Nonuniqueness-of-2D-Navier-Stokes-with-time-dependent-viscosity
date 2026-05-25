#!/usr/bin/env python3
"""h=0.06 seeded continuation ladder for the mu=2/3, C4 bifurcation case."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

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
from run_s140_bifurcation_probe import boundary_max_abs_diff, copy_function, linf, load_h5_profile, rms, run_newton_from, vec_array, write_xdmf_profile  # noqa: E402
from run_s140_artificial_perturb_probe import artificial_pattern, diff_function  # noqa: E402
from run_spectrum_continuation import build_baseline_context, mesh_stats  # noqa: E402


COMPARE_COLUMNS = [
    "mesh_id",
    "s",
    "source_s",
    "unstable_count",
    "min_real_lambda",
    "profile_residual_omega_l2",
    "profile_residual_psi_l2",
    "profile_checkpoint",
    "metadata_checkpoint",
    "status",
]


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

    esolver = ManifoldEigenSolver(context["mesh"], context["spaces"], omega, psi, context["params"], context["facet_tags"], outer_tag=1)
    modes = esolver.solve(k=args.k_eigs, sigma=args.eig_shift, tol=args.eig_tol, max_it=args.eig_max_it, verbose=True)
    return [complex(mode.lam) for mode in modes[: args.k_eigs]]


def build_context(args: argparse.Namespace, s_value: float) -> dict[str, Any]:
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
    return build_baseline_context(ns, coeffs_for_exact_cos_mode(s_value, args.mode))


def seeded_solve(args: argparse.Namespace, context: dict[str, Any], seed_h5: Path, seed_s: float, target_s: float):
    omega_seed, _psi_seed, _geom = load_h5_profile(seed_h5, context["mesh"], context["spaces"])
    ratio = float(target_s) / float(seed_s)
    omega0 = copy_function(context["spaces"]["Vomega"], ratio * vec_array(omega_seed))
    set_exact_boundary_values(context, omega0)
    psi0 = context["poisson"].solve_state(omega0, sigma=1.0)
    omega, psi, history, converged = run_newton_from(
        context,
        omega0,
        psi0,
        max_newton=args.max_newton,
        tol=args.tol_newton,
        linesearch=not args.no_linesearch,
    )
    if not converged:
        final = history[-1]
        raise RuntimeError(
            f"seeded solve failed target_s={target_s:g}: "
            f"it={final['iteration']} rel={final['relative_step']} residual={final['residual_norm']}"
        )
    return omega, psi, history


def artificial_probe(args: argparse.Namespace, selected: dict[str, Any], out_dirs: dict[str, Path]) -> dict[str, Any]:
    context = selected["context"]
    omega_sym = selected["omega"]
    psi_sym = selected["psi"]
    s_value = float(selected["s"])
    Vomega = context["spaces"]["Vomega"]
    pattern = artificial_pattern(Vomega, m_mode=args.perturb_mode, phase=args.phase, r0=args.r0, Rdom=context["params"].Rdom)
    scale = args.eta * rms(vec_array(omega_sym)) / max(rms(pattern), 1e-30)
    omega0 = copy_function(Vomega, vec_array(omega_sym) + scale * pattern)
    set_exact_boundary_values(context, omega0)
    psi0 = context["poisson"].solve_state(omega0, sigma=1.0)
    omega_broken, psi_broken, history, converged = run_newton_from(
        context,
        omega0,
        psi0,
        max_newton=args.perturb_max_newton,
        tol=args.tol_newton,
        linesearch=not args.no_linesearch,
    )
    sym_res = selected["residuals"]
    broken_res = residuals_for(context, omega_broken, psi_broken)
    omega_sep = vec_array(omega_broken) - vec_array(omega_sym)
    psi_sep = vec_array(psi_broken) - vec_array(psi_sym)
    omega_diff = diff_function(context["spaces"]["Vomega"], omega_broken, omega_sym)
    psi_diff = diff_function(context["spaces"]["Vpsi"], psi_broken, psi_sym)
    omega_c4_sym = symmetry_fraction_on_probe(context["spaces"]["Vomega"], omega_sym, symmetry_order=args.mode)
    omega_c4 = symmetry_fraction_on_probe(context["spaces"]["Vomega"], omega_broken, symmetry_order=args.mode)
    psi_c4_sym = symmetry_fraction_on_probe(context["spaces"]["Vpsi"], psi_sym, symmetry_order=args.mode)
    psi_c4 = symmetry_fraction_on_probe(context["spaces"]["Vpsi"], psi_broken, symmetry_order=args.mode)
    trial_id = f"m{args.perturb_mode}_phase{args.phase:g}_eta{args.eta:g}".replace(".", "p").replace("-", "m")
    profile_path = out_dirs["checkpoints"] / f"s{s_value:g}_{trial_id}_profile.xdmf"
    write_xdmf_profile(profile_path, context["mesh"], omega_broken, psi_broken)
    plot_path = out_dirs["plots"] / f"s{s_value:g}_{trial_id}_omega_comparison"
    plot_profile_comparison(
        plot_path,
        context["mesh"],
        omega_sym,
        omega_broken,
        r"$\alpha = 3/4$, boundary data = $240\cos(4\theta)$",
    )
    final = history[-1]
    rel_omega = rms(omega_sep) / max(rms(vec_array(omega_sym)), 1e-30)
    omega_diff_non4 = symmetry_fraction_on_probe(context["spaces"]["Vomega"], omega_diff, symmetry_order=args.mode)
    status = "completed_distinct_symmetry_breaking_candidate" if converged and rel_omega > args.distinct_relative_omega_rms and omega_diff_non4 > args.distinct_difference_non4 else ("completed" if converged else "nonconverged")
    return {
        "s": s_value,
        "mode": int(args.mode),
        "mu": float(args.mu),
        "trial_id": trial_id,
        "status": status,
        "m_mode": int(args.perturb_mode),
        "phase": float(args.phase),
        "eta": float(args.eta),
        "scale": scale,
        "newton_iterations": int(final["iteration"]),
        "final_newton_residual": float(final["residual_norm"]),
        "final_relative_step": float(final["relative_step"]),
        "profile_residual_omega_l2": broken_res["norms"]["Omega_l2"],
        "profile_residual_psi_l2": broken_res["norms"]["Psi_l2"],
        "boundary_max_abs_diff": boundary_max_abs_diff(context["spaces"], context["facet_tags"], omega_broken, omega_sym),
        "omega_rms_separation": rms(omega_sep),
        "omega_relative_rms_separation": rel_omega,
        "omega_linf_separation": linf(omega_sep),
        "psi_rms_separation": rms(psi_sep),
        "psi_relative_rms_separation": rms(psi_sep) / max(rms(vec_array(psi_sym)), 1e-30),
        "psi_linf_separation": linf(psi_sep),
        "omega_c4_fraction_symmetric": omega_c4_sym,
        "omega_c4_fraction_candidate": omega_c4,
        "omega_c4_fraction_excess": omega_c4 - omega_c4_sym,
        "omega_difference_non4_fraction": omega_diff_non4,
        "psi_c4_fraction_symmetric": psi_c4_sym,
        "psi_c4_fraction_candidate": psi_c4,
        "psi_c4_fraction_excess": psi_c4 - psi_c4_sym,
        "psi_difference_non4_fraction": symmetry_fraction_on_probe(context["spaces"]["Vpsi"], psi_diff, symmetry_order=args.mode),
        "wall_time_sec": "n/a",
        "profile_checkpoint": relpath(profile_path),
        "plot_path": relpath(plot_path.with_suffix(".png")),
        "message": f"symmetric_R_Omega={sym_res['norms']['Omega_l2']}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-tag", default="branchB_k4_mu2over3_h006_seeded_s225_240")
    parser.add_argument("--mode", type=int, default=4)
    parser.add_argument("--mu", type=float, default=2.0 / 3.0)
    parser.add_argument("--mesh", nargs=2, type=float, default=[0.06, 0.06])
    parser.add_argument("--s-values", nargs="+", type=float, default=[225.0, 230.0, 235.0, 240.0])
    parser.add_argument("--seed-s", type=float, default=220.0)
    parser.add_argument("--seed-profile-h5", type=Path, default=BRANCH_DIR / "checkpoints/branchB_k4_mu2over3_s220_bifurcation_probe/s220_h0p06_0p06_profile.h5")
    parser.add_argument("--required-unstable", type=int, default=4)
    parser.add_argument("--R", type=float, default=150.0)
    parser.add_argument("--Rdom", type=float, default=10.0)
    parser.add_argument("--M", type=int, default=48)
    parser.add_argument("--sigma-steps", type=int, default=21)
    parser.add_argument("--max-newton", type=int, default=96)
    parser.add_argument("--tol-newton", type=float, default=1e-8)
    parser.add_argument("--no-linesearch", action="store_true")
    parser.add_argument("--k-eigs", type=int, default=10)
    parser.add_argument("--eig-shift", type=float, default=-1.0)
    parser.add_argument("--eig-tol", type=float, default=1e-8)
    parser.add_argument("--eig-max-it", type=int, default=300)
    parser.add_argument("--perturb-mode", type=int, default=3)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--r0", type=float, default=2.5)
    parser.add_argument("--perturb-max-newton", type=int, default=64)
    parser.add_argument("--distinct-relative-omega-rms", type=float, default=1e-2)
    parser.add_argument("--distinct-difference-non4", type=float, default=0.1)
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
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dirs["logs"] / f"{args.run_tag}_manifest.json"
    manifest = {"status": "dry_run", "run_tag": args.run_tag, "s_values": args.s_values, "seed_s": args.seed_s, "seed_profile_h5": relpath(args.seed_profile_h5)}
    write_json(manifest_path, manifest)
    if not args.execute:
        print(f"[dry-run] wrote {manifest_path}")
        return 0

    t_all = time.perf_counter()
    mesh_id = mesh_label(args.mesh[0], args.mesh[1])
    seed_h5 = args.seed_profile_h5
    seed_s = float(args.seed_s)
    results: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    eig_csv = out_dirs["data"] / f"{args.run_tag}_eigenvalues.csv"
    summary_csv = out_dirs["data"] / f"{args.run_tag}_summary.csv"
    for target_s in args.s_values:
        context = build_context(args, target_s)
        omega, psi, history = seeded_solve(args, context, seed_h5, seed_s, target_s)
        residuals = residuals_for(context, omega, psi)
        profile_path = out_dirs["checkpoints"] / f"s{target_s:g}_{mesh_id}_profile.xdmf"
        write_xdmf_profile(profile_path, context["mesh"], omega, psi)
        eigs = solve_eigenvalues(args, context, omega, psi)
        unstable = int(sum(lam.real < 0.0 for lam in eigs))
        eig_rows = [
            {
                "s": float(target_s),
                "mode": int(args.mode),
                "mu": float(args.mu),
                "mesh_id": mesh_id,
                "eigen_index": idx,
                "real_lambda": float(lam.real),
                "imag_lambda": float(lam.imag),
                "profile_residual_omega_l2": residuals["norms"]["Omega_l2"],
                "profile_residual_psi_l2": residuals["norms"]["Psi_l2"],
                "profile_checkpoint": relpath(profile_path),
                "metadata_checkpoint": relpath(manifest_path),
                "status": "completed",
            }
            for idx, lam in enumerate(eigs, start=1)
        ]
        append_rows(eig_csv, EIG_COLUMNS, eig_rows)
        summary = {
            "mesh_id": mesh_id,
            "s": float(target_s),
            "source_s": seed_s,
            "unstable_count": unstable,
            "min_real_lambda": min(lam.real for lam in eigs),
            "profile_residual_omega_l2": residuals["norms"]["Omega_l2"],
            "profile_residual_psi_l2": residuals["norms"]["Psi_l2"],
            "profile_checkpoint": relpath(profile_path),
            "metadata_checkpoint": relpath(manifest_path),
            "status": "completed",
        }
        append_rows(summary_csv, COMPARE_COLUMNS, [summary])
        results.append({**summary, "eigenvalues": [{"index": i, "real": float(lam.real), "imag": float(lam.imag)} for i, lam in enumerate(eigs, start=1)]})
        print(f"[ladder] s={target_s:g} unstable={unstable} minRe={summary['min_real_lambda']:.6e}")
        seed_h5 = profile_path.with_suffix(".h5")
        seed_s = float(target_s)
        if unstable >= int(args.required_unstable):
            selected = {"s": target_s, "context": context, "omega": omega, "psi": psi, "residuals": residuals, "eigenvalues": eigs, "profile_path": profile_path}
            break

    perturb_row = None
    if selected is not None:
        perturb_row = artificial_probe(args, selected, out_dirs)
        append_rows(out_dirs["data"] / f"{args.run_tag}_perturbations.csv", PERT_COLUMNS, [perturb_row])

    manifest.update({"status": "completed", "completed_at": datetime.now().isoformat(), "wall_time_sec": time.perf_counter() - t_all, "results": jsonable(results), "selected_s": None if selected is None else selected["s"], "perturbation": jsonable(perturb_row)})
    write_json(manifest_path, jsonable(manifest))
    if not args.no_registry:
        append_registry_row({
            "date": today(),
            "tag": f"fem-branchB-{args.run_tag}",
            "codebase": "baseline",
            "script": relpath(Path(__file__)),
            "input_case": "s_cos4theta_mu0.666667",
            "domain": f"disk_Rdom{float(args.Rdom):g}",
            "grid_or_mesh": f"h={float(args.mesh[0]):g}/{float(args.mesh[1]):g}",
            "solver": "same_mesh_seeded_continuation_eigs_artificial_perturbation",
            "parameters": f"s_values={args.s_values}; seed={args.seed_s}; perturb=m{args.perturb_mode},eta={args.eta}",
            "wall_time_sec": f"{manifest['wall_time_sec']:.6f}",
            "error_metrics": f"selected_s={manifest['selected_s']}; perturb_status={None if perturb_row is None else perturb_row['status']}",
            "eigenvalues_summary": f"results={relpath(eig_csv)}",
            "status": "completed",
            "notes": "h=0.06 C4/mu=2/3 continuation refinement from accepted s=220 profile.",
        })
    print(f"[done] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
