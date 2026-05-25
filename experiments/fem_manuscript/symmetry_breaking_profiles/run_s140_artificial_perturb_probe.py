#!/usr/bin/env python3
"""Branch B: artificial localized perturbations of the s=140 profile.

The perturbations are interior-only, non-threefold angular modes:

    p_m,phase(r,theta) = (r/Rdom)^m exp(-(r/r0)^2)
                         (1 - (r/Rdom)^2)^2 cos(m theta + phase).

Here m is not divisible by 3, so the perturbation breaks C3 symmetry. The
boundary factor makes p=0 on r=Rdom, so the Omega boundary data remain exactly
the same. After perturbing Omega, psi is recomputed by the baseline Poisson
solver with the same boundary flux, keeping the initial guess on the
Delta psi = Omega manifold without needing eigenfunctions.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time
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


CSV_COLUMNS = [
    "trial_id",
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
    "psi_rms_separation",
    "psi_relative_rms_separation",
    "psi_linf_separation",
    "omega_c3_fraction_symmetric",
    "omega_c3_fraction_candidate",
    "omega_c3_fraction_excess",
    "omega_difference_non3_fraction",
    "psi_c3_fraction_symmetric",
    "psi_c3_fraction_candidate",
    "psi_c3_fraction_excess",
    "psi_difference_non3_fraction",
    "wall_time_sec",
    "profile_checkpoint",
    "plot_path",
    "message",
]


def artificial_pattern(V: Any, *, m_mode: int, phase: float, r0: float, Rdom: float) -> np.ndarray:
    raw = V.tabulate_dof_coordinates()
    if raw.ndim == 1:
        raw = raw.reshape(-1, V.mesh.geometry.x.shape[1])
    x = raw[:, 0]
    y = raw[:, 1]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    rho = np.clip(r / float(Rdom), 0.0, 1.0)
    envelope = (rho ** int(m_mode)) * np.exp(-((r / float(r0)) ** 2)) * ((1.0 - rho * rho) ** 2)
    values = envelope * np.cos(int(m_mode) * theta + float(phase))
    values[r >= float(Rdom) * (1.0 - 1e-12)] = 0.0
    return values


def make_artificial_initial_guess(context: dict[str, Any], omega_sym: Any, eta: float, m_mode: int, phase: float, r0: float):
    Vomega = context["spaces"]["Vomega"]
    pattern = artificial_pattern(Vomega, m_mode=m_mode, phase=phase, r0=r0, Rdom=context["params"].Rdom)
    scale = float(eta) * rms(vec_array(omega_sym)) / max(rms(pattern), 1e-30)
    omega0 = copy_function(Vomega, vec_array(omega_sym) + scale * pattern)
    psi0 = context["poisson"].solve_state(omega0, sigma=1.0)
    return omega0, psi0, scale


def diff_function(V: Any, a: Any, b: Any):
    return copy_function(V, vec_array(a) - vec_array(b))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--s", type=float, default=140.0)
    parser.add_argument("--m-modes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--phases", nargs="+", type=float, default=[0.0])
    parser.add_argument("--etas", nargs="+", type=float, default=[0.25, 0.5])
    parser.add_argument("--r0", type=float, default=2.5)
    parser.add_argument("--max-newton", type=int, default=24)
    parser.add_argument("--tol-newton", type=float, default=1e-8)
    parser.add_argument("--no-linesearch", action="store_true")
    parser.add_argument("--stop-on-distinct", action="store_true")
    parser.add_argument("--distinct-relative-omega-rms", type=float, default=1e-2)
    parser.add_argument("--distinct-difference-non3", type=float, default=0.1)
    parser.add_argument("--run-tag", default="branchB_s140_artificial_localized_probe")
    parser.add_argument(
        "--symmetric-profile-h5",
        type=Path,
        default=C3_BRANCH_DIR / "checkpoints/branchA_h006_refine_s125_140_k10/s140_h0p06_0p06_profile.h5",
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

    csv_path = out_data / f"{args.run_tag}_results.csv"
    manifest_path = out_logs / f"{args.run_tag}_manifest.json"
    manifest = {
        "status": "dry_run",
        "run_tag": args.run_tag,
        "s": args.s,
        "m_modes": args.m_modes,
        "phases": args.phases,
        "etas": args.etas,
        "r0": args.r0,
        "perturbation_formula": "(r/Rdom)^m exp(-(r/r0)^2) (1-(r/Rdom)^2)^2 cos(m theta + phase), scaled to eta*rms(Omega_sym)",
        "symmetry_metrics": [
            "raw C3 defect for symmetric and candidate on the same polar probe",
            "candidate minus symmetric C3-defect excess",
            "non-C3 fraction of the difference field candidate - symmetric, reducing mesh/interpolation bias",
        ],
        "outputs": {
            "csv": str(csv_path.relative_to(REPO_ROOT)),
            "plots": str(out_plots.relative_to(REPO_ROOT)),
            "checkpoints": str(out_checkpoints.relative_to(REPO_ROOT)),
        },
    }
    write_json(manifest_path, manifest)
    if not args.execute:
        print(f"[dry-run] wrote {manifest_path}")
        return 0

    add_baseline_to_path()
    from residuals_quick import residuals_quick

    t_all = time.perf_counter()
    ns = argparse.Namespace(
        mesh=[0.06, 0.06],
        sigma_steps=21,
        mu=0.8,
        R=150.0,
        Rdom=10.0,
        M=48,
        tol_newton=float(args.tol_newton),
        max_newton=int(args.max_newton),
    )
    context = build_baseline_context(ns, coeffs_for_exact_c3_cos(args.s))
    omega_sym, psi_sym, geom_diff = load_h5_profile(args.symmetric_profile_h5, context["mesh"], context["spaces"])
    pinned = getattr(context["poisson"], "gauge_dof", None)
    sym_res = residuals_quick(
        context["mesh"],
        context["spaces"],
        omega_sym,
        psi_sym,
        mu=context["params"].mu,
        R=context["params"].R,
        Rdom=context["params"].Rdom,
        facet_tags=context["facet_tags"],
        outer_tag=1,
        fourier_bc=context["fourier_bc"],
        sigma=1.0,
        pinned_psi_dof=pinned,
    )
    omega_c3_sym = c3_fraction_on_probe(context["spaces"]["Vomega"], omega_sym)
    psi_c3_sym = c3_fraction_on_probe(context["spaces"]["Vpsi"], psi_sym)

    rows: list[dict[str, Any]] = []
    stop_reason = None
    csv_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not csv_exists:
            writer.writeheader()

        for m_mode in args.m_modes:
            if int(m_mode) % 3 == 0:
                print(f"[skip] m={m_mode} is 3-fold compatible")
                continue
            for phase in args.phases:
                for eta in args.etas:
                    trial_id = f"m{m_mode}_phase{phase:g}_eta{eta:g}".replace(".", "p").replace("-", "m")
                    trial_t0 = time.perf_counter()
                    try:
                        omega0, psi0, scale = make_artificial_initial_guess(
                            context, omega_sym, eta=eta, m_mode=m_mode, phase=phase, r0=args.r0
                        )
                        init_bdiff = boundary_max_abs_diff(context["spaces"], context["facet_tags"], omega0, omega_sym)
                        omega_candidate, psi_candidate, history, converged = run_newton_from(
                            context,
                            omega0,
                            psi0,
                            max_newton=args.max_newton,
                            tol=args.tol_newton,
                            linesearch=not args.no_linesearch,
                        )
                        profile_path = out_checkpoints / f"{trial_id}_profile.xdmf"
                        write_xdmf_profile(profile_path, context["mesh"], omega_candidate, psi_candidate)
                        residuals = residuals_quick(
                            context["mesh"],
                            context["spaces"],
                            omega_candidate,
                            psi_candidate,
                            mu=context["params"].mu,
                            R=context["params"].R,
                            Rdom=context["params"].Rdom,
                            facet_tags=context["facet_tags"],
                            outer_tag=1,
                            fourier_bc=context["fourier_bc"],
                            sigma=1.0,
                            pinned_psi_dof=pinned,
                        )
                        omega_sep = vec_array(omega_candidate) - vec_array(omega_sym)
                        psi_sep = vec_array(psi_candidate) - vec_array(psi_sym)
                        omega_diff_fun = diff_function(context["spaces"]["Vomega"], omega_candidate, omega_sym)
                        psi_diff_fun = diff_function(context["spaces"]["Vpsi"], psi_candidate, psi_sym)
                        omega_c3 = c3_fraction_on_probe(context["spaces"]["Vomega"], omega_candidate)
                        psi_c3 = c3_fraction_on_probe(context["spaces"]["Vpsi"], psi_candidate)
                        omega_diff_non3 = c3_fraction_on_probe(context["spaces"]["Vomega"], omega_diff_fun)
                        psi_diff_non3 = c3_fraction_on_probe(context["spaces"]["Vpsi"], psi_diff_fun)
                        plot_path = out_plots / f"{trial_id}_omega_comparison"
                        plot_comparison(
                            plot_path,
                            context["mesh"],
                            omega_sym,
                            omega_candidate,
                            f"sigma=140 artificial m={m_mode}, eta={eta:g}, phase={phase:g}",
                        )
                        final = history[-1]
                        rel_omega = rms(omega_sep) / max(rms(vec_array(omega_sym)), 1e-30)
                        rel_psi = rms(psi_sep) / max(rms(vec_array(psi_sym)), 1e-30)
                        distinct = (
                            converged
                            and rel_omega > args.distinct_relative_omega_rms
                            and omega_diff_non3 > args.distinct_difference_non3
                        )
                        status = "completed_distinct_symmetry_breaking_candidate" if distinct else ("completed" if converged else "nonconverged")
                        row = {
                            "trial_id": trial_id,
                            "status": status,
                            "m_mode": m_mode,
                            "phase": phase,
                            "eta": eta,
                            "scale": scale,
                            "newton_iterations": int(final["iteration"]),
                            "final_newton_residual": float(final["residual_norm"]),
                            "final_relative_step": float(final["relative_step"]),
                            "profile_residual_omega_l2": residuals["norms"]["Omega_l2"],
                            "profile_residual_psi_l2": residuals["norms"]["Psi_l2"],
                            "boundary_max_abs_diff": boundary_max_abs_diff(context["spaces"], context["facet_tags"], omega_candidate, omega_sym),
                            "omega_rms_separation": rms(omega_sep),
                            "omega_relative_rms_separation": rel_omega,
                            "omega_linf_separation": linf(omega_sep),
                            "psi_rms_separation": rms(psi_sep),
                            "psi_relative_rms_separation": rel_psi,
                            "psi_linf_separation": linf(psi_sep),
                            "omega_c3_fraction_symmetric": omega_c3_sym,
                            "omega_c3_fraction_candidate": omega_c3,
                            "omega_c3_fraction_excess": omega_c3 - omega_c3_sym,
                            "omega_difference_non3_fraction": omega_diff_non3,
                            "psi_c3_fraction_symmetric": psi_c3_sym,
                            "psi_c3_fraction_candidate": psi_c3,
                            "psi_c3_fraction_excess": psi_c3 - psi_c3_sym,
                            "psi_difference_non3_fraction": psi_diff_non3,
                            "wall_time_sec": time.perf_counter() - trial_t0,
                            "profile_checkpoint": str(profile_path.relative_to(REPO_ROOT)),
                            "plot_path": str(plot_path.with_suffix(".png").relative_to(REPO_ROOT)),
                            "message": f"initial_boundary_diff={init_bdiff:.3e}",
                        }
                    except Exception as exc:
                        row = {
                            "trial_id": trial_id,
                            "status": "failed",
                            "m_mode": m_mode,
                            "phase": phase,
                            "eta": eta,
                            "wall_time_sec": time.perf_counter() - trial_t0,
                            "message": repr(exc),
                        }
                    writer.writerow({key: row.get(key, "n/a") for key in CSV_COLUMNS})
                    f.flush()
                    rows.append(row)
                    print(
                        f"[trial] {trial_id}: {row['status']} "
                        f"relOmega={row.get('omega_relative_rms_separation', 'n/a')} "
                        f"diffNon3={row.get('omega_difference_non3_fraction', 'n/a')}"
                    )
                    if args.stop_on_distinct and row["status"] == "completed_distinct_symmetry_breaking_candidate":
                        stop_reason = f"distinct candidate found: {trial_id}"
                        break
                if stop_reason:
                    break
            if stop_reason:
                break

    manifest.update(
        {
            "status": "completed",
            "mesh_stats": mesh_stats(context["mesh"], context["spaces"]),
            "profile_checkpoint_geometry_diff": geom_diff,
            "symmetric_profile_residuals": sym_res["norms"],
            "symmetric_c3": {"Omega": omega_c3_sym, "psi": psi_c3_sym},
            "num_trials": len(rows),
            "stop_reason": stop_reason,
            "wall_time_sec": time.perf_counter() - t_all,
        }
    )
    write_json(manifest_path, manifest)
    append_registry_row(
        {
            "date": today(),
            "tag": f"fem-branchB-{args.run_tag}",
            "codebase": "baseline",
            "script": str(Path(__file__).relative_to(REPO_ROOT)),
            "input_case": "s_cos3theta_s140",
            "domain": "disk_Rdom10",
            "grid_or_mesh": "h=0.06/0.06",
            "solver": "artificial_localized_perturbation_fixed_boundary_newton",
            "parameters": f"m_modes={args.m_modes}; etas={args.etas}; phases={args.phases}; r0={args.r0}; max_newton={args.max_newton}",
            "wall_time_sec": f"{manifest['wall_time_sec']:.6f}",
            "error_metrics": f"symmetric_R_Omega={sym_res['norms']['Omega_l2']}; symmetric_c3_Omega={omega_c3_sym}",
            "eigenvalues_summary": f"trials={len(rows)}; stop_reason={stop_reason}",
            "status": "completed",
            "notes": "Artificial localized non-C3 perturbation Branch B probe; no eigenfunction computation needed.",
        }
    )
    print(f"[done] results={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
