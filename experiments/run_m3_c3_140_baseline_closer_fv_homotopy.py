from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
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

from nextgen.boundary import FourierBoundaryData
from nextgen.polar_profile import (
    PolarGrid,
    PolarProfileParams,
    PolarProfileProblem,
    PolarProfileResult,
    PolarProfileState,
)
from run_m3_c3_140_residual_audit import (
    SAMPLE_X,
    SAMPLE_Y,
    load_cached_fem,
    run_baseline_seeded_linesearch,
    sample_baseline,
    sanitize,
)
from run_m3_c3_140_symmetry_probe import (
    COEFFS,
    PHYSICAL,
    common_residual,
    compatibility_metrics,
    save_state,
    symmetry_metrics,
)
from run_m3_c3_140_symmetric_discrepancy_audit import (
    boundary_convention_check,
    diff_metrics,
    grid_points,
    mesh_label,
    nextgen_sample,
)


RUN_DATE = "2026-04-30"
RAW_ROOT = REPO_ROOT / "experiments" / "raw" / "nextgen" / "m3_c3_140_baseline_closer_fv_homotopy"
SUMMARY_DIR = REPO_ROOT / "experiments" / "summaries"
STATE_DIR = RAW_ROOT / "states"
TABLE_DIR = RAW_ROOT / "tables"
PLOT_DIR = RAW_ROOT / "plots"

SOURCE_STATES = {
    "160/480": {
        "symmetric": REPO_ROOT
        / "experiments/raw/nextgen/m3_c3_140_nonuniqueness_validation/states/symmetric_160_480_state.npz",
        "broken_m1": REPO_ROOT
        / "experiments/raw/nextgen/m3_c3_140_nonuniqueness_validation/states/broken_m1_160_480_state.npz",
    },
    "192/576": {
        "symmetric": REPO_ROOT
        / "experiments/raw/nextgen/m3_c3_140_refine192_fem_crosscheck/states/symmetric_192_576_state.npz",
        "broken_m1": REPO_ROOT
        / "experiments/raw/nextgen/m3_c3_140_refine192_fem_crosscheck/states/broken_m1_192_576_state.npz",
    },
}

DEFAULT_BLEND_PATH = (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
BASELINE_FAILURES = {
    "fem_vs_fv_omega_rms_h0p06": 0.546,
    "fem_vs_fv_omega_rms_h0p05": 0.569,
    "fv_to_fem_seed_residual_h0p06": 4.94e-1,
    "fv_to_fem_seed_residual_h0p05": 4.45e-1,
    "fem_to_fv_projected_residual_h0p06": 1.77,
    "fem_to_fv_projected_residual_h0p05": 1.34,
}


@dataclass(frozen=True)
class TransportSpec:
    label: str
    advection_scheme: str
    upwind_blend: float


def transport_specs(blends: tuple[float, ...], *, include_centered: bool) -> list[TransportSpec]:
    specs = [TransportSpec("fv_upwind_reference", "fv_upwind", 1.0)]
    specs.extend(TransportSpec(blend_label(b), "fv_blended", float(b)) for b in blends)
    if include_centered:
        specs.append(TransportSpec("fv_centered", "fv_centered", 0.0))
    return specs


def blend_label(blend: float) -> str:
    return f"fv_blend_{blend:g}".replace(".", "p")


def grid_label(grid: tuple[int, int]) -> str:
    return f"{int(grid[0])}/{int(grid[1])}"


def fem_short_label(mesh: tuple[float, float]) -> str:
    if abs(mesh[0] - mesh[1]) < 1e-14:
        return f"h{mesh[0]:g}".replace(".", "p")
    return mesh_label(mesh[0], mesh[1])


def make_problem(grid: tuple[int, int], spec: TransportSpec) -> PolarProfileProblem:
    return PolarProfileProblem(
        PolarGrid(radius=PHYSICAL.Rdom, nr=int(grid[0]), ntheta=int(grid[1])),
        PHYSICAL,
        FourierBoundaryData(M=64, coeffs=COEFFS),
        advection_scheme=spec.advection_scheme,
        upwind_blend=float(spec.upwind_blend),
    )


def result_from_arrays(
    problem: PolarProfileProblem,
    omega: np.ndarray,
    psi: np.ndarray,
    *,
    label: str,
    branch: str,
    transport: TransportSpec,
    sigma: float = 1.0,
    poisson_lambda: float = 0.0,
    message: str = "loaded_state",
) -> dict[str, Any]:
    omega_arr = np.asarray(omega, dtype=float).reshape(problem.grid.shape)
    psi_arr = np.asarray(psi, dtype=float).reshape(problem.grid.shape)
    diagnostics = problem.residual_summary(omega_arr, psi_arr, float(sigma), float(poisson_lambda))
    diagnostics.update(
        {
            "final_sigma": float(sigma),
            "grid_nr": problem.grid.nr,
            "grid_ntheta": problem.grid.ntheta,
            "grid_dr": problem.grid.dr,
            "grid_dtheta": problem.grid.dtheta,
            "advection_scheme": transport.advection_scheme,
            "upwind_blend": float(transport.upwind_blend),
            "poisson_lambda": float(poisson_lambda),
        }
    )
    result = PolarProfileResult(
        omega=omega_arr,
        psi=psi_arr,
        grid=problem.grid,
        converged=True,
        iterations=(),
        runtime_sec=0.0,
        message=message,
        diagnostics=diagnostics,
    )
    run = {
        "label": label,
        "branch": branch,
        "transport": {
            "label": transport.label,
            "advection_scheme": transport.advection_scheme,
            "upwind_blend": float(transport.upwind_blend),
        },
        "status": "completed",
        "grid": {
            "nr": problem.grid.nr,
            "ntheta": problem.grid.ntheta,
            "dr": problem.grid.dr,
            "dtheta": problem.grid.dtheta,
        },
        "grid_label": f"{problem.grid.nr}/{problem.grid.ntheta}",
        "problem": problem,
        "result": result,
        "runtime_sec": 0.0,
        "diagnostics": diagnostics,
        "message": message,
        "common_residuals": common_residual(problem, result),
        "symmetry": symmetry_metrics(problem, result),
        "compatibility": compatibility_metrics(problem, result),
        "boundary_defects": {
            "omega_dirichlet_max_abs": 0.0,
            "psi_neumann_max_abs": 0.0,
            "note": "polar FV boundary data are analytic RHS terms, not mutable state dofs",
        },
    }
    return run


def load_source_state(branch: str, grid: tuple[int, int], transport: TransportSpec) -> dict[str, Any]:
    label = grid_label(grid)
    path = SOURCE_STATES[label][branch]
    data = np.load(path)
    problem = make_problem(grid, transport)
    sigma = float(data["sigma"]) if "sigma" in data.files else 1.0
    return result_from_arrays(
        problem,
        data["omega"],
        data["psi"],
        label=f"{branch}_{transport.label}_{problem.grid.nr}_{problem.grid.ntheta}",
        branch=branch,
        transport=transport,
        sigma=sigma,
        poisson_lambda=0.0,
        message=f"loaded_source_state:{path.relative_to(REPO_ROOT)}",
    )


def reproject_state(problem: PolarProfileProblem, omega: np.ndarray, *, sigma: float = 1.0) -> PolarProfileState:
    psi, lam = problem.solve_poisson(np.asarray(omega, dtype=float).reshape(problem.grid.shape), sigma=float(sigma))
    return PolarProfileState(
        omega=np.asarray(omega, dtype=float).reshape(problem.grid.shape),
        psi=psi.reshape(problem.grid.shape),
        poisson_lambda=float(lam),
        sigma=0.0,
    )


def solve_stage(
    source: dict[str, Any],
    spec: TransportSpec,
    *,
    branch: str,
    grid: tuple[int, int],
    max_iter: int,
    gmres_maxiter: int,
    gmres_restart: int,
) -> dict[str, Any]:
    problem = make_problem(grid, spec)
    initial = reproject_state(problem, source["result"].omega, sigma=1.0)
    t0 = time.perf_counter()
    result = problem.solve(
        PolarProfileParams(
            sigma_grid=(1.0,),
            tol_step=1e-8,
            tol_residual=1e-8,
            tol_residual_only=1e-8,
            max_iter=int(max_iter),
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
            reduced_gmres_maxiter=int(gmres_maxiter),
            reduced_gmres_restart=int(gmres_restart),
        ),
        initial_state=initial,
    )
    run = {
        "label": f"{branch}_{spec.label}_{grid[0]}_{grid[1]}",
        "branch": branch,
        "transport": {
            "label": spec.label,
            "advection_scheme": spec.advection_scheme,
            "upwind_blend": float(spec.upwind_blend),
        },
        "status": "completed" if result.converged else "nonconverged",
        "grid": {"nr": grid[0], "ntheta": grid[1], "dr": problem.grid.dr, "dtheta": problem.grid.dtheta},
        "grid_label": grid_label(grid),
        "problem": problem,
        "result": result,
        "runtime_sec": time.perf_counter() - t0,
        "diagnostics": result.diagnostics,
        "message": result.message,
    }
    if result.converged:
        run["common_residuals"] = common_residual(problem, result)
        run["symmetry"] = symmetry_metrics(problem, result)
        run["compatibility"] = compatibility_metrics(problem, result)
        run["boundary_defects"] = {
            "omega_dirichlet_max_abs": 0.0,
            "psi_neumann_max_abs": 0.0,
            "note": "polar FV boundary data are analytic RHS terms, not mutable state dofs",
        }
    return run


def final_relative_update(run: dict[str, Any]) -> float:
    iterations = getattr(run.get("result"), "iterations", ())
    if not iterations:
        return float("nan")
    return float(iterations[-1].relative_omega_step)


def sample_run_on_audit_grid(run: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    xx, yy, _ = grid_points()
    omega, psi = run["problem"].sample_cartesian(run["result"], xx.reshape(-1), yy.reshape(-1))
    return omega.reshape(xx.shape), psi.reshape(xx.shape)


def fem_on_audit_grid(fem_run: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "sample_omega" in fem_run and "sample_psi" in fem_run:
        return np.asarray(fem_run["sample_omega"], dtype=float), np.asarray(fem_run["sample_psi"], dtype=float)
    _, _, points = grid_points()
    omega = sample_baseline(fem_run["omega"], fem_run["spaces"]["Vomega"], points)
    psi = sample_baseline(fem_run["psi"], fem_run["spaces"]["Vpsi"], points)
    xx, _, _ = grid_points()
    return omega.reshape(xx.shape), psi.reshape(xx.shape)


def sample_fem_on_nextgen_grid(fem_run: dict[str, Any], problem: PolarProfileProblem) -> tuple[np.ndarray, np.ndarray]:
    grid = problem.grid
    rr, tt = np.meshgrid(grid.r, grid.theta, indexing="ij")
    points = np.column_stack((rr.reshape(-1) * np.cos(tt.reshape(-1)), rr.reshape(-1) * np.sin(tt.reshape(-1))))
    omega = sample_baseline(fem_run["omega"], fem_run["spaces"]["Vomega"], points).reshape(grid.shape)
    psi = sample_baseline(fem_run["psi"], fem_run["spaces"]["Vpsi"], points).reshape(grid.shape)
    if np.any(~np.isfinite(omega)) or np.any(~np.isfinite(psi)):
        raise RuntimeError("FEM-to-FV interpolation produced non-finite values on polar cell centers.")
    return omega, psi


def make_loaded_result(problem: PolarProfileProblem, omega: np.ndarray, psi: np.ndarray, diagnostics: dict[str, Any], message: str) -> PolarProfileResult:
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


def baseline_fem_reference(reference: dict[str, Any], mesh: tuple[float, float]) -> dict[str, Any]:
    if abs(mesh[0] - 0.06) < 1e-14 and abs(mesh[1] - 0.06) < 1e-14:
        cached = load_cached_fem(str(reference.get("branch", "symmetric")))
        if cached is not None:
            cached["label"] = f"baseline_h0p06_{reference.get('branch', 'unknown')}_cached_reference"
            return cached
    run = run_baseline_seeded_linesearch(reference, mesh)
    if run.get("status") != "completed":
        raise RuntimeError(f"FEM reference solve failed for mesh={mesh}: {sanitize(run)}")
    return run


def distance_to_fem(run: dict[str, Any], fem_run: dict[str, Any]) -> dict[str, Any]:
    xx, yy, _ = grid_points()
    radius = np.hypot(xx, yy)
    full_mask = radius <= 7.0
    omega, psi = sample_run_on_audit_grid(run)
    fem_omega, fem_psi = fem_on_audit_grid(fem_run)
    active_mask = full_mask & ((np.abs(omega) > 0.5) | (np.abs(fem_omega) > 0.5))
    return {
        "omega_full": diff_metrics(omega, fem_omega, SAMPLE_X, SAMPLE_Y, mask=full_mask),
        "omega_active": diff_metrics(omega, fem_omega, SAMPLE_X, SAMPLE_Y, mask=active_mask),
        "psi_full_offset_removed": diff_metrics(
            psi - np.nanmean(psi),
            fem_psi - np.nanmean(fem_psi),
            SAMPLE_X,
            SAMPLE_Y,
            mask=full_mask,
        ),
    }


def distance_between_runs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    xx, yy, _ = grid_points()
    radius = np.hypot(xx, yy)
    full_mask = radius <= 7.0
    ao, ap = sample_run_on_audit_grid(a)
    bo, bp = sample_run_on_audit_grid(b)
    active_mask = full_mask & ((np.abs(ao) > 0.5) | (np.abs(bo) > 0.5))
    return {
        "omega_full": diff_metrics(ao, bo, SAMPLE_X, SAMPLE_Y, mask=full_mask),
        "omega_active": diff_metrics(ao, bo, SAMPLE_X, SAMPLE_Y, mask=active_mask),
        "psi_full_offset_removed": diff_metrics(
            ap - np.nanmean(ap),
            bp - np.nanmean(bp),
            SAMPLE_X,
            SAMPLE_Y,
            mask=full_mask,
        ),
    }


def residual_scan_rows(branch: str, grid: tuple[int, int], specs: list[TransportSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        loaded = load_source_state(branch, grid, spec)
        diag = loaded["diagnostics"]
        rows.append(
            {
                "branch": branch,
                "grid": grid_label(grid),
                "transport_label": spec.label,
                "advection_scheme": spec.advection_scheme,
                "upwind_blend": float(spec.upwind_blend),
                "relative_residual": diag.get("relative_residual"),
                "residual_norm": diag.get("residual_norm"),
                "omega_residual_norm": diag.get("omega_residual_norm"),
                "poisson_residual_norm": diag.get("poisson_residual_norm"),
                "physical_poisson_residual_norm": diag.get("physical_poisson_residual_norm"),
                "status": "evaluated_without_newton",
            }
        )
    return rows


def homotopy_table_row(run: dict[str, Any], reference: dict[str, Any], fem_refs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    diag = run.get("diagnostics", {})
    sym = run.get("symmetry", {})
    comp = run.get("compatibility", {})
    common = run.get("common_residuals", {})
    row: dict[str, Any] = {
        "label": run.get("label"),
        "branch": run.get("branch"),
        "grid": run.get("grid_label"),
        "transport_label": run.get("transport", {}).get("label"),
        "advection_scheme": run.get("transport", {}).get("advection_scheme"),
        "upwind_blend": run.get("transport", {}).get("upwind_blend"),
        "status": run.get("status"),
        "runtime_sec": run.get("runtime_sec"),
        "native_relative_residual": diag.get("relative_residual"),
        "native_residual_norm": diag.get("residual_norm"),
        "omega_residual_norm": diag.get("omega_residual_norm"),
        "poisson_residual_norm": diag.get("poisson_residual_norm"),
        "physical_poisson_residual_norm": diag.get("physical_poisson_residual_norm"),
        "final_relative_update": final_relative_update(run),
        "omega_non3": sym.get("omega_non_threefold_fraction"),
        "psi_non3": sym.get("psi_non_threefold_fraction"),
        "compatibility_omega_minus_flux": comp.get("omega_minus_flux"),
        "common_profile_strong_relative": common.get("profile_strong_relative"),
        "omega_rms_from_fv_upwind_reference": None,
        "omega_rms_to_fem_h0p06": None,
        "omega_rms_to_fem_h0p05": None,
    }
    if run.get("status") == "completed":
        row["omega_rms_from_fv_upwind_reference"] = distance_between_runs(run, reference)["omega_full"].get("rms")
        for label, fem_ref in fem_refs.items():
            key = f"omega_rms_to_fem_{label}"
            row[key] = distance_to_fem(run, fem_ref)["omega_full"].get("rms")
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path.relative_to(REPO_ROOT))


def select_best_blend(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("status") == "completed"
        and row.get("transport_label") != "fv_upwind_reference"
        and row.get("native_relative_residual") is not None
        and float(row["native_relative_residual"]) <= 1e-8
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            float(row.get("omega_rms_to_fem_h0p06") or float("inf")),
            float(row.get("omega_rms_to_fem_h0p05") or float("inf")),
        )
    )
    return candidates[0]


def passes_symmetric_gate(selected_row: dict[str, Any] | None) -> bool:
    if selected_row is None:
        return False
    rms_h06 = selected_row.get("omega_rms_to_fem_h0p06")
    if rms_h06 is None:
        return False
    return (
        float(selected_row.get("native_relative_residual") or float("inf")) <= 1e-8
        and float(selected_row.get("omega_non3") or float("inf")) <= 1e-6
        and float(rms_h06) <= 0.5 * BASELINE_FAILURES["fem_vs_fv_omega_rms_h0p06"]
    )


def run_fem_to_fv(problem: PolarProfileProblem, fem_run: dict[str, Any], *, label: str) -> dict[str, Any]:
    fem_omega_on_fv, fem_psi_on_fv = sample_fem_on_nextgen_grid(fem_run, problem)
    sampled_summary = problem.residual_summary(fem_omega_on_fv, fem_psi_on_fv, 1.0, 0.0)
    projected_psi, projected_lambda = problem.solve_poisson(fem_omega_on_fv, 1.0)
    projected_summary = problem.residual_summary(fem_omega_on_fv, projected_psi, 1.0, projected_lambda)
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
        initial_state=PolarProfileState(
            omega=fem_omega_on_fv,
            psi=projected_psi.reshape(problem.grid.shape),
            poisson_lambda=projected_lambda,
            sigma=0.0,
        ),
    )
    run = {
        "label": label,
        "status": "completed" if result.converged else "nonconverged",
        "result": result,
        "diagnostics": result.diagnostics,
        "message": result.message,
        "sampled_psi_summary": sampled_summary,
        "projected_summary": projected_summary,
    }
    return run


def cross_initialization(selected: dict[str, Any], meshes: tuple[tuple[float, float], ...]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for mesh in meshes:
        label = mesh_label(mesh[0], mesh[1])
        fem_run = run_baseline_seeded_linesearch(selected, mesh)
        if fem_run.get("status") != "completed":
            results[label] = {"status": "fem_failed", "fem_run": sanitize(fem_run)}
            continue
        fem_history = fem_run.get("history", [])
        initial_bnorm = float(fem_history[0]["bnorm_before"]) if fem_history else float("nan")
        final_bnorm = float(fem_history[-1]["bnorm_after"]) if fem_history else float("nan")
        fv_from_fem = run_fem_to_fv(selected["problem"], fem_run, label=f"fv_from_fem_{label}")

        fem_omega_on_fv, _ = sample_fem_on_nextgen_grid(fem_run, selected["problem"])
        projected_psi, _ = selected["problem"].solve_poisson(fem_omega_on_fv, 1.0)
        seed_projected = make_loaded_result(
            selected["problem"],
            fem_omega_on_fv,
            projected_psi,
            fv_from_fem["projected_summary"],
            "fem_sampled_to_fv_grid_poisson_projected",
        )
        distances = cross_initialization_distances(selected, fem_run, seed_projected, fv_from_fem)
        results[label] = {
            "status": "completed" if fv_from_fem.get("status") == "completed" else "fv_failed",
            "fv_to_fem": {
                "fem_status": fem_run.get("status"),
                "runtime_sec": fem_run.get("runtime_sec"),
                "newton_iterations": fem_run.get("newton_iterations"),
                "initial_newton_bnorm": initial_bnorm,
                "final_newton_bnorm": final_bnorm,
                "final_weak_residuals": fem_run.get("residuals", {}).get("norms", {}),
                "omega_non3": fem_run.get("symmetry", {}).get("omega_non_threefold_fraction"),
            },
            "fem_to_fv": {
                "sampled_psi_relative_residual": fv_from_fem["sampled_psi_summary"].get("relative_residual"),
                "sampled_psi_residual_norm": fv_from_fem["sampled_psi_summary"].get("residual_norm"),
                "initial_projected_relative_residual": fv_from_fem["projected_summary"].get("relative_residual"),
                "initial_projected_residual_norm": fv_from_fem["projected_summary"].get("residual_norm"),
                "initial_projected_omega_residual_norm": fv_from_fem["projected_summary"].get("omega_residual_norm"),
                "status": fv_from_fem.get("status"),
                "final_relative_residual": fv_from_fem.get("diagnostics", {}).get("relative_residual"),
                "message": fv_from_fem.get("message"),
            },
            "distances": distances,
        }
    return results


def cross_initialization_distances(
    selected: dict[str, Any],
    fem_run: dict[str, Any],
    seed_result: PolarProfileResult,
    fv_final: dict[str, Any],
) -> dict[str, Any]:
    xx, yy, _ = grid_points()
    radius = np.hypot(xx, yy)
    full_mask = radius <= 7.0
    sel_o, sel_p = nextgen_sample(selected)
    fem_o, fem_p = fem_on_audit_grid(fem_run)
    seed_o, seed_p = selected["problem"].sample_cartesian(seed_result, xx.reshape(-1), yy.reshape(-1))
    seed_o = seed_o.reshape(xx.shape)
    seed_p = seed_p.reshape(xx.shape)
    final_o, final_p = selected["problem"].sample_cartesian(fv_final["result"], xx.reshape(-1), yy.reshape(-1))
    final_o = final_o.reshape(xx.shape)
    final_p = final_p.reshape(xx.shape)
    active_mask = full_mask & ((np.abs(sel_o) > 0.5) | (np.abs(fem_o) > 0.5) | (np.abs(final_o) > 0.5))
    return {
        "omega_distances": {
            "fem_final_vs_selected_full": diff_metrics(fem_o, sel_o, SAMPLE_X, SAMPLE_Y, mask=full_mask),
            "fem_final_vs_selected_active": diff_metrics(fem_o, sel_o, SAMPLE_X, SAMPLE_Y, mask=active_mask),
            "fem_to_fv_seed_representation_vs_fem_full": diff_metrics(seed_o, fem_o, SAMPLE_X, SAMPLE_Y, mask=full_mask),
            "fv_from_fem_final_vs_selected_full": diff_metrics(final_o, sel_o, SAMPLE_X, SAMPLE_Y, mask=full_mask),
            "fv_from_fem_final_vs_fem_seed_full": diff_metrics(final_o, fem_o, SAMPLE_X, SAMPLE_Y, mask=full_mask),
        },
        "psi_distances": {
            "fem_final_vs_selected_full": diff_metrics(
                fem_p - np.nanmean(fem_p),
                sel_p - np.nanmean(sel_p),
                SAMPLE_X,
                SAMPLE_Y,
                mask=full_mask,
            ),
            "fv_from_fem_final_vs_selected_full": diff_metrics(
                final_p - np.nanmean(final_p),
                sel_p - np.nanmean(sel_p),
                SAMPLE_X,
                SAMPLE_Y,
                mask=full_mask,
            ),
        },
    }


def write_plots(runs: list[dict[str, Any]], selected: dict[str, Any] | None, fem_refs: dict[str, dict[str, Any]], rows: list[dict[str, Any]], *, branch: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    xx, yy, _ = grid_points()
    extent = (float(SAMPLE_X[0]), float(SAMPLE_X[-1]), float(SAMPLE_Y[0]), float(SAMPLE_Y[-1]))
    reference = runs[0]
    if selected is not None:
        ref_o, _ = sample_run_on_audit_grid(reference)
        sel_o, _ = sample_run_on_audit_grid(selected)
        panels = [
            ("fv_upwind reference", ref_o),
            ("selected blend", sel_o),
            ("selected - reference", sel_o - ref_o),
        ]
        for label, fem in fem_refs.items():
            fem_o, _ = fem_on_audit_grid(fem)
            panels.append((f"FEM {label}", fem_o))
            panels.append((f"selected - FEM {label}", sel_o - fem_o))
        ncols = min(5, len(panels))
        nrows = int(np.ceil(len(panels) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.5 * nrows), constrained_layout=True)
        axes_arr = np.asarray(axes).reshape(-1)
        for ax, (title, values) in zip(axes_arr, panels):
            image = ax.imshow(values.T, origin="lower", extent=extent, cmap="RdBu_r")
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=9)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        for ax in axes_arr[len(panels) :]:
            ax.axis("off")
        path = PLOT_DIR / f"{branch}_selected_full_domain.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        created.append(str(path.relative_to(REPO_ROOT)))

        x_window = (-1.0, 1.0)
        y_window = (3.3, 5.3)
        ix = (SAMPLE_X >= x_window[0]) & (SAMPLE_X <= x_window[1])
        iy = (SAMPLE_Y >= y_window[0]) & (SAMPLE_Y <= y_window[1])
        close_panels = [
            ("fv_upwind", ref_o[np.ix_(ix, iy)]),
            ("selected", sel_o[np.ix_(ix, iy)]),
            ("selected - upwind", (sel_o - ref_o)[np.ix_(ix, iy)]),
        ]
        for label, fem in fem_refs.items():
            fem_o, _ = fem_on_audit_grid(fem)
            close_panels.append((f"selected - FEM {label}", (sel_o - fem_o)[np.ix_(ix, iy)]))
        close_extent = (float(SAMPLE_X[ix][0]), float(SAMPLE_X[ix][-1]), float(SAMPLE_Y[iy][0]), float(SAMPLE_Y[iy][-1]))
        fig, axes = plt.subplots(1, len(close_panels), figsize=(3.3 * len(close_panels), 3.3), constrained_layout=True)
        for ax, (title, values) in zip(np.asarray(axes).reshape(-1), close_panels):
            image = ax.imshow(values.T, origin="lower", extent=close_extent, cmap="RdBu_r")
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=9)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        path = PLOT_DIR / f"{branch}_selected_top_core_closeup.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        created.append(str(path.relative_to(REPO_ROOT)))

    trend_rows = [row for row in rows if row.get("status") == "completed"]
    if trend_rows:
        xvals = [float(row["upwind_blend"]) for row in trend_rows]
        y06 = [row.get("omega_rms_to_fem_h0p06") for row in trend_rows]
        y05 = [row.get("omega_rms_to_fem_h0p05") for row in trend_rows]
        fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
        ax.plot(xvals, y06, marker="o", label="FEM h=0.06")
        if any(v is not None for v in y05):
            ax.plot(xvals, y05, marker="s", label="FEM h=0.05")
        ax.axhline(BASELINE_FAILURES["fem_vs_fv_omega_rms_h0p06"], color="0.4", linestyle="--", linewidth=1.0)
        ax.set_xscale("symlog", linthresh=0.02)
        ax.set_xlabel("upwind blend")
        ax.set_ylabel("Omega RMS to FEM")
        ax.legend()
        path = PLOT_DIR / f"{branch}_blend_vs_fem_rms.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        created.append(str(path.relative_to(REPO_ROOT)))
    return created


def append_registry(summary: dict[str, Any]) -> None:
    registry = REPO_ROOT / "experiments" / "registry.csv"
    selected = summary.get("selected_blend") or {}
    with registry.open("a", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                RUN_DATE,
                "m3-c3-140-baseline-closer-fv-homotopy",
                "nextgen_vs_baseline",
                "experiments/run_m3_c3_140_baseline_closer_fv_homotopy.py",
                "{c3=140}",
                "nextgen_polar_full_disk_Rdom10_and_baseline_fem_disk_Rdom10",
                f"grid={summary.get('grid_label')}; branch={summary.get('stage')}",
                "fv_blend_homotopy_residual_and_cross_initialization",
                "M3 c3=140 baseline-closer FV homotopy; no M4; no eigensolver",
                summary.get("runtime_sec", "n/a"),
                (
                    f"selected={selected.get('transport_label')}; "
                    f"selected_rel={selected.get('native_relative_residual')}; "
                    f"selected_rms_h06={selected.get('omega_rms_to_fem_h0p06')}; "
                    f"passes_gate={summary.get('passes_symmetric_gate')}"
                ),
                "n/a",
                summary.get("status", "completed"),
                "Baseline-closer FV homotopy generated diagnostics.",
            ]
        )


def summary_path(stage: str) -> Path:
    suffix = "" if stage == "symmetric" else f"_{stage}"
    return SUMMARY_DIR / f"m3_c3_140_baseline_closer_fv_homotopy{suffix}_{RUN_DATE}.json"


def run_branch_homotopy(
    *,
    branch: str,
    grid: tuple[int, int],
    blends: tuple[float, ...],
    include_centered: bool,
    fem_meshes: tuple[tuple[float, float], ...],
    max_iter: int,
    gmres_maxiter: int,
    gmres_restart: int,
    run_cross_init: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    upwind_spec = TransportSpec("fv_upwind_reference", "fv_upwind", 1.0)
    reference = load_source_state(branch, grid, upwind_spec)
    fem_refs = {fem_short_label(mesh): baseline_fem_reference(reference, mesh) for mesh in fem_meshes}
    specs = transport_specs(blends, include_centered=include_centered)
    scan_rows = residual_scan_rows(branch, grid, specs)

    runs = [reference]
    previous = reference
    for spec in specs[1:]:
        run = solve_stage(
            previous,
            spec,
            branch=branch,
            grid=grid,
            max_iter=max_iter,
            gmres_maxiter=gmres_maxiter,
            gmres_restart=gmres_restart,
        )
        runs.append(run)
        if run.get("status") != "completed":
            break
        save_state(run, STATE_DIR)
        previous = run

    rows = [homotopy_table_row(run, reference, fem_refs) for run in runs]
    selected_row = select_best_blend(rows)
    selected_run = None
    if selected_row is not None:
        selected_run = next((run for run in runs if run.get("label") == selected_row.get("label")), None)
        if selected_run is not None:
            save_state(selected_run, STATE_DIR)
    cross_init: dict[str, Any] = {}
    if run_cross_init and selected_run is not None:
        cross_init = cross_initialization(selected_run, fem_meshes)

    scan_path = write_csv(TABLE_DIR / f"{branch}_residual_scan_table.csv", scan_rows)
    table_path = write_csv(TABLE_DIR / f"{branch}_blend_homotopy_table.csv", rows)
    closeness_path = write_csv(TABLE_DIR / f"{branch}_baseline_closeness_table.csv", rows)
    plot_paths = write_plots(runs, selected_run, fem_refs, rows, branch=branch)
    summary = {
        "status": "completed",
        "stage": branch,
        "case": {"coeffs": COEFFS, "name": "c3_140_baseline_closer_fv_homotopy"},
        "grid": {"nr": int(grid[0]), "ntheta": int(grid[1])},
        "grid_label": grid_label(grid),
        "runtime_sec": time.perf_counter() - t0,
        "boundary_convention": boundary_convention_check(),
        "transport_path": [spec.__dict__ for spec in specs],
        "baseline_failures_to_beat": BASELINE_FAILURES,
        "residual_scan_rows": scan_rows,
        "homotopy_rows": rows,
        "selected_blend": selected_row,
        "passes_symmetric_gate": passes_symmetric_gate(selected_row) if branch == "symmetric" else None,
        "cross_initialization": cross_init,
        "table_paths": {
            "residual_scan": scan_path,
            "blend_homotopy": table_path,
            "baseline_closeness": closeness_path,
        },
        "plot_paths": plot_paths,
        "state_dir": str(STATE_DIR.relative_to(REPO_ROOT)),
        "notes": [
            "This script stays in M3 and does not assemble or run the eigensolver.",
            "The symmetric branch is the validation blocker; broken-branch evidence should not be used to excuse a failed symmetric cross-discretization test.",
        ],
    }
    return summary


def load_symmetric_summary() -> dict[str, Any]:
    path = summary_path("symmetric")
    if not path.exists():
        raise RuntimeError(f"Missing symmetric summary {path}; run --stage symmetric first.")
    return json.loads(path.read_text())


def parse_meshes(values: list[str]) -> tuple[tuple[float, float], ...]:
    meshes = []
    for value in values:
        if "/" in value:
            a, b = value.split("/", 1)
        elif "," in value:
            a, b = value.split(",", 1)
        else:
            a = b = value
        meshes.append((float(a), float(b)))
    return tuple(meshes)


def main() -> int:
    parser = argparse.ArgumentParser(description="M3 c3=140 baseline-closer conservative FV homotopy.")
    parser.add_argument("--stage", choices=("symmetric", "broken_m1"), default="symmetric")
    parser.add_argument("--grid", nargs=2, type=int, default=(192, 576), metavar=("NR", "NTHETA"))
    parser.add_argument("--blends", nargs="*", type=float, default=list(DEFAULT_BLEND_PATH))
    parser.add_argument("--include-centered", action="store_true", help="attempt fv_centered after all requested blend stages converge")
    parser.add_argument("--fem-mesh", action="append", default=None, help="FEM h_min/h_max mesh label, e.g. 0.06 or 0.05/0.05")
    parser.add_argument("--max-iter", type=int, default=140)
    parser.add_argument("--gmres-maxiter", type=int, default=360)
    parser.add_argument("--gmres-restart", type=int, default=180)
    parser.add_argument("--skip-cross-initialization", action="store_true")
    parser.add_argument("--allow-broken", action="store_true", help="allow broken_m1 stage even if the saved symmetric gate did not pass")
    parser.add_argument("--no-registry", action="store_true", help="do not append a registry row after a completed run")
    args = parser.parse_args()

    if args.stage == "broken_m1" and not args.allow_broken:
        symmetric = load_symmetric_summary()
        if not symmetric.get("passes_symmetric_gate"):
            raise RuntimeError(
                "Broken branch is gated until the symmetric blend materially improves Gate 3. "
                "Use --allow-broken only for an explicit diagnostic override."
            )

    summary = run_branch_homotopy(
        branch=args.stage,
        grid=(int(args.grid[0]), int(args.grid[1])),
        blends=tuple(float(v) for v in args.blends),
        include_centered=bool(args.include_centered),
        fem_meshes=parse_meshes(args.fem_mesh or ["0.06", "0.05"]),
        max_iter=int(args.max_iter),
        gmres_maxiter=int(args.gmres_maxiter),
        gmres_restart=int(args.gmres_restart),
        run_cross_init=not bool(args.skip_cross_initialization),
    )
    path = summary_path(args.stage)
    path.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    if not args.no_registry:
        append_registry(summary)
    print(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
