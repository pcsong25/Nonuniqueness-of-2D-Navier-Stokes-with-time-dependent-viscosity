from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-euler")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from nextgen.polar_profile import PolarProfileParams, PolarProfileResult, PolarProfileState
from run_m3_c3_140_residual_audit import (
    SAMPLE_X,
    SAMPLE_Y,
    load_nextgen_state,
    run_baseline_seeded_linesearch,
    sample_baseline,
    sanitize,
)
from run_m3_c3_140_symmetry_probe import common_residual, compatibility_metrics, symmetry_metrics
from run_m3_c3_140_symmetric_discrepancy_audit import (
    COEFFS,
    PHYSICAL,
    boundary_convention_check,
    diff_metrics,
    grid_points,
    mesh_label,
    nextgen_sample,
)


RUN_DATE = "2026-04-30"
RAW_ROOT = REPO_ROOT / "experiments" / "raw" / "nextgen" / "m3_c3_140_symmetric_cross_initialization"
SUMMARY_DIR = REPO_ROOT / "experiments" / "summaries"
PLOT_DIR = RAW_ROOT / "plots"
STATE_192 = REPO_ROOT / "experiments/raw/nextgen/m3_c3_140_refine192_fem_crosscheck/states/symmetric_192_576_state.npz"


def make_result(problem: Any, omega: np.ndarray, psi: np.ndarray, diagnostics: dict[str, Any], message: str) -> PolarProfileResult:
    return PolarProfileResult(
        omega=np.asarray(omega, dtype=float).reshape(problem.grid.shape),
        psi=np.asarray(psi, dtype=float).reshape(problem.grid.shape),
        grid=problem.grid,
        converged=True,
        iterations=(),
        runtime_sec=0.0,
        message=message,
        diagnostics=diagnostics,
    )


def sample_fem_on_audit_grid(fem_run: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    xx, yy, points = grid_points()
    omega = sample_baseline(fem_run["omega"], fem_run["spaces"]["Vomega"], points).reshape(xx.shape)
    psi = sample_baseline(fem_run["psi"], fem_run["spaces"]["Vpsi"], points).reshape(xx.shape)
    return omega, psi


def sample_fem_on_nextgen_grid(fem_run: dict[str, Any], problem: Any) -> tuple[np.ndarray, np.ndarray]:
    grid = problem.grid
    rr, tt = np.meshgrid(grid.r, grid.theta, indexing="ij")
    points = np.column_stack((rr.reshape(-1) * np.cos(tt.reshape(-1)), rr.reshape(-1) * np.sin(tt.reshape(-1))))
    omega = sample_baseline(fem_run["omega"], fem_run["spaces"]["Vomega"], points).reshape(grid.shape)
    psi = sample_baseline(fem_run["psi"], fem_run["spaces"]["Vpsi"], points).reshape(grid.shape)
    if np.any(~np.isfinite(omega)) or np.any(~np.isfinite(psi)):
        raise RuntimeError("FEM-to-FV interpolation produced non-finite values on polar cell centers.")
    return omega, psi


def solve_nextgen_from_state(problem: Any, state: PolarProfileState) -> dict[str, Any]:
    t0 = time.perf_counter()
    result = problem.solve(
        PolarProfileParams(
            sigma_grid=(1.0,),
            tol_step=1e-8,
            tol_residual=1e-8,
            tol_residual_only=1e-8,
            max_iter=140,
            max_backtracks=30,
            adaptive_continuation=True,
            min_sigma_step=5e-4,
            sigma_step_growth=1.2,
            sigma_step_shrink=0.5,
            least_squares_fallback=False,
            pseudo_transient_fallback=True,
            pseudo_transient_shifts=(1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
            reduced_poisson_newton=True,
            reduced_gmres_rtol=1e-8,
            reduced_gmres_maxiter=360,
            reduced_gmres_restart=180,
        ),
        initial_state=state,
    )
    run = {
        "label": "nextgen_from_fem_symmetric_seed_192_576",
        "status": "completed" if result.converged else "nonconverged",
        "runtime_sec": time.perf_counter() - t0,
        "problem": problem,
        "result": result,
        "diagnostics": result.diagnostics,
        "message": result.message,
    }
    if result.converged:
        run["common_residuals"] = common_residual(problem, result)
        run["compatibility"] = compatibility_metrics(problem, result)
        run["symmetry"] = symmetry_metrics(problem, result)
    return run


def branch_metrics_on_audit_grid(nextgen_ref: dict[str, Any], fem_run: dict[str, Any], fv_seed_result: PolarProfileResult, fv_final: dict[str, Any]) -> dict[str, Any]:
    xx, yy, _ = grid_points()
    radius = np.hypot(xx, yy)
    full_mask = radius <= 7.0
    ng_ref_o, ng_ref_p = nextgen_sample(nextgen_ref)
    fem_o, fem_p = sample_fem_on_audit_grid(fem_run)
    seed_o, seed_p = nextgen_ref["problem"].sample_cartesian(fv_seed_result, xx.reshape(-1), yy.reshape(-1))
    seed_o = seed_o.reshape(xx.shape)
    seed_p = seed_p.reshape(xx.shape)
    final_o, final_p = nextgen_ref["problem"].sample_cartesian(fv_final["result"], xx.reshape(-1), yy.reshape(-1))
    final_o = final_o.reshape(xx.shape)
    final_p = final_p.reshape(xx.shape)
    active_mask = full_mask & ((np.abs(ng_ref_o) > 0.5) | (np.abs(fem_o) > 0.5) | (np.abs(final_o) > 0.5))
    return {
        "omega_distances": {
            "fem_final_vs_nextgen_reference_full": diff_metrics(fem_o, ng_ref_o, SAMPLE_X, SAMPLE_Y, mask=full_mask),
            "fem_final_vs_nextgen_reference_active": diff_metrics(fem_o, ng_ref_o, SAMPLE_X, SAMPLE_Y, mask=active_mask),
            "fem_to_fv_seed_representation_vs_fem_full": diff_metrics(seed_o, fem_o, SAMPLE_X, SAMPLE_Y, mask=full_mask),
            "fem_to_fv_seed_representation_vs_fem_active": diff_metrics(seed_o, fem_o, SAMPLE_X, SAMPLE_Y, mask=active_mask),
            "fv_from_fem_final_vs_nextgen_reference_full": diff_metrics(final_o, ng_ref_o, SAMPLE_X, SAMPLE_Y, mask=full_mask),
            "fv_from_fem_final_vs_nextgen_reference_active": diff_metrics(final_o, ng_ref_o, SAMPLE_X, SAMPLE_Y, mask=active_mask),
            "fv_from_fem_final_vs_fem_seed_full": diff_metrics(final_o, fem_o, SAMPLE_X, SAMPLE_Y, mask=full_mask),
            "fv_from_fem_final_vs_fem_seed_active": diff_metrics(final_o, fem_o, SAMPLE_X, SAMPLE_Y, mask=active_mask),
        },
        "psi_distances": {
            "fem_final_vs_nextgen_reference_full": diff_metrics(fem_p - np.nanmean(fem_p), ng_ref_p - np.nanmean(ng_ref_p), SAMPLE_X, SAMPLE_Y, mask=full_mask),
            "fem_to_fv_seed_representation_vs_fem_full": diff_metrics(seed_p - np.nanmean(seed_p), fem_p - np.nanmean(fem_p), SAMPLE_X, SAMPLE_Y, mask=full_mask),
            "fv_from_fem_final_vs_nextgen_reference_full": diff_metrics(final_p - np.nanmean(final_p), ng_ref_p - np.nanmean(ng_ref_p), SAMPLE_X, SAMPLE_Y, mask=full_mask),
        },
    }


def write_plots(nextgen_ref: dict[str, Any], fem_run: dict[str, Any], fv_seed_result: PolarProfileResult, fv_final: dict[str, Any], label: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_dir = PLOT_DIR / label
    plot_dir.mkdir(parents=True, exist_ok=True)
    xx, yy, _ = grid_points()
    ng_o, _ = nextgen_sample(nextgen_ref)
    fem_o, _ = sample_fem_on_audit_grid(fem_run)
    seed_o, _ = nextgen_ref["problem"].sample_cartesian(fv_seed_result, xx.reshape(-1), yy.reshape(-1))
    seed_o = seed_o.reshape(xx.shape)
    final_o, _ = nextgen_ref["problem"].sample_cartesian(fv_final["result"], xx.reshape(-1), yy.reshape(-1))
    final_o = final_o.reshape(xx.shape)
    extent = (float(SAMPLE_X[0]), float(SAMPLE_X[-1]), float(SAMPLE_Y[0]), float(SAMPLE_Y[-1]))
    panels = [
        ("FV reference", ng_o),
        ("FEM final from FV seed", fem_o),
        ("FEM sampled to FV grid", seed_o),
        ("FV final from FEM seed", final_o),
        ("FEM final - FV reference", fem_o - ng_o),
        ("FV final - FEM seed", final_o - fem_o),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, (title, values) in zip(axes.ravel(), panels):
        image = ax.imshow(values.T, origin="lower", extent=extent, cmap="RdBu_r")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    path = plot_dir / "symmetric_cross_initialization.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [str(path.relative_to(REPO_ROOT))]


def append_registry(summary: dict[str, Any]) -> None:
    registry = REPO_ROOT / "experiments" / "registry.csv"
    with registry.open("a", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                RUN_DATE,
                "m3-c3-140-symmetric-cross-initialization",
                "nextgen_vs_baseline",
                "experiments/run_m3_c3_140_symmetric_cross_initialization.py",
                "{c3=140}",
                f"nextgen_192/576_and_baseline_fem_{summary['mesh_label']}",
                "FV_seed_to_FEM; FEM_seed_to_FV",
                "symmetric_branch_cross_initialization",
                "M3 c3=140 symmetric FV/FEM cross-initialization; no M4; no eigensolver",
                summary.get("runtime_sec", "n/a"),
                (
                    f"fem_seed_bnorm={summary['fv_to_fem']['initial_newton_bnorm']}; "
                    f"fv_seed_rel={summary['fem_to_fv']['initial_projected_relative_residual']}; "
                    f"fv_final_vs_ref={summary['distances']['omega_distances']['fv_from_fem_final_vs_nextgen_reference_full']['rms']}; "
                    f"fem_final_vs_ref={summary['distances']['omega_distances']['fem_final_vs_nextgen_reference_full']['rms']}"
                ),
                "n/a",
                "completed",
                "Symmetric branch cross-initialization completed.",
            ]
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-initialize c3=140 symmetric FV and FEM profiles.")
    parser.add_argument("--h-min", type=float, default=0.06)
    parser.add_argument("--h-max", type=float, default=0.06)
    args = parser.parse_args()

    t0 = time.perf_counter()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    label = mesh_label(args.h_min, args.h_max)

    nextgen_ref = load_nextgen_state(STATE_192, "symmetric_192_576", "symmetric", "192/576")
    problem = nextgen_ref["problem"]

    fem_run = run_baseline_seeded_linesearch(nextgen_ref, (float(args.h_min), float(args.h_max)))
    if fem_run.get("status") != "completed":
        raise RuntimeError(f"FEM solve from FV seed failed: {sanitize(fem_run)}")
    fem_history = fem_run.get("history", [])
    initial_bnorm = float(fem_history[0]["bnorm_before"]) if fem_history else float("nan")
    final_bnorm = float(fem_history[-1]["bnorm_after"]) if fem_history else float("nan")

    fem_omega_on_fv, fem_psi_on_fv = sample_fem_on_nextgen_grid(fem_run, problem)
    sampled_summary = problem.residual_summary(fem_omega_on_fv, fem_psi_on_fv, 1.0, 0.0)
    projected_psi, projected_lambda = problem.solve_poisson(fem_omega_on_fv, 1.0)
    projected_summary = problem.residual_summary(fem_omega_on_fv, projected_psi, 1.0, projected_lambda)
    seed_result = make_result(problem, fem_omega_on_fv, projected_psi.reshape(problem.grid.shape), projected_summary, "fem_sampled_to_fv_grid_poisson_projected")
    fv_from_fem = solve_nextgen_from_state(
        problem,
        PolarProfileState(
            omega=fem_omega_on_fv,
            psi=projected_psi.reshape(problem.grid.shape),
            poisson_lambda=projected_lambda,
            sigma=0.0,
        ),
    )
    if fv_from_fem.get("status") != "completed":
        raise RuntimeError(f"FV solve from FEM seed failed: {sanitize(fv_from_fem)}")

    distances = branch_metrics_on_audit_grid(nextgen_ref, fem_run, seed_result, fv_from_fem)
    plot_paths = write_plots(nextgen_ref, fem_run, seed_result, fv_from_fem, label)
    summary = {
        "status": "completed",
        "case": {"coeffs": COEFFS, "name": "c3_140_symmetric_cross_initialization"},
        "mesh": {"h_min": float(args.h_min), "h_max": float(args.h_max)},
        "mesh_label": label,
        "runtime_sec": time.perf_counter() - t0,
        "boundary_convention": boundary_convention_check(),
        "fv_reference": {
            "native_relative_residual": nextgen_ref["diagnostics"].get("relative_residual"),
            "native_residual_norm": nextgen_ref["diagnostics"].get("residual_norm"),
            "omega_non3": nextgen_ref["symmetry"].get("omega_non_threefold_fraction"),
        },
        "fv_to_fem": {
            "status": fem_run["status"],
            "runtime_sec": fem_run.get("runtime_sec"),
            "newton_iterations": fem_run.get("newton_iterations"),
            "initial_newton_bnorm": initial_bnorm,
            "final_newton_bnorm": final_bnorm,
            "initial_to_final_bnorm_ratio": initial_bnorm / max(final_bnorm, 1e-300),
            "final_weak_residuals": fem_run.get("residuals", {}).get("norms", {}),
            "omega_non3": fem_run.get("symmetry", {}).get("omega_non_threefold_fraction"),
        },
        "fem_to_fv": {
            "sampled_psi_relative_residual": sampled_summary.get("relative_residual"),
            "sampled_psi_residual_norm": sampled_summary.get("residual_norm"),
            "initial_projected_relative_residual": projected_summary.get("relative_residual"),
            "initial_projected_residual_norm": projected_summary.get("residual_norm"),
            "initial_projected_omega_residual_norm": projected_summary.get("omega_residual_norm"),
            "initial_projected_poisson_residual_norm": projected_summary.get("poisson_residual_norm"),
            "status": fv_from_fem.get("status"),
            "runtime_sec": fv_from_fem.get("runtime_sec"),
            "final_relative_residual": fv_from_fem.get("diagnostics", {}).get("relative_residual"),
            "final_residual_norm": fv_from_fem.get("diagnostics", {}).get("residual_norm"),
            "final_omega_non3": fv_from_fem.get("symmetry", {}).get("omega_non_threefold_fraction"),
            "message": fv_from_fem.get("message"),
        },
        "distances": distances,
        "plot_paths": plot_paths,
        "interpretation_notes": [
            "If FV and FEM were approximating the same symmetric continuum root, the FV seed should have a small FEM native residual and FEM-to-FV projected seed should have a small FV native residual.",
            "If they were two genuine symmetric continuum roots, each solver should ideally admit a nearby root after interpolation from the other discretization.",
            "Large native residuals at the interpolated seeds and Newton convergence back to the solver's own reference root point instead to discretization mismatch at the tested resolutions.",
        ],
    }
    path = SUMMARY_DIR / f"m3_c3_140_symmetric_cross_initialization_{label}_{RUN_DATE}.json"
    path.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    append_registry(summary)
    print(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
