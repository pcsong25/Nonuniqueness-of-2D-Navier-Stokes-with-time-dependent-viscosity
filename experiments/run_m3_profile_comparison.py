from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
from scipy.interpolate import LinearNDInterpolator

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from nextgen.boundary import FourierBoundaryData
from nextgen.config import PhysicalParams
from nextgen.grid import SquareGrid
from nextgen.profile import ProfileNewtonParams, ProfileProblem, save_profile_debug_plots


SIGMA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def reduced_coefficients(scale: float) -> dict[int, complex]:
    return {
        0: 0.0 + 0.0j,
        1: 0.0 + 0.0j,
        2: 0.0 + 0.0j,
        3: 60.0 * scale + 0.0j,
        4: 50.0 * scale + 0.0j,
        5: 0.0 + 0.0j,
        6: 0.0 + 0.0j,
        10: 0.0 + 0.0j,
        12: 0.0 + 0.0j,
        24: 0.0 + 0.0j,
        36: 0.0 + 0.0j,
    }


def run_nextgen(scale: float) -> dict[str, Any]:
    grid = SquareGrid(half_width=10.0, n=65)
    physical = PhysicalParams(mu=0.8, R=150.0, Rdom=10.0)
    boundary = FourierBoundaryData(M=48, coeffs=reduced_coefficients(scale))
    problem = ProfileProblem(grid, physical, boundary)
    result = problem.solve(
        ProfileNewtonParams(
            sigma_grid=SIGMA_GRID,
            tol_step=1e-8,
            tol_residual=1e-8,
            max_newton=24,
        )
    )
    plot_paths: list[str] = []
    try:
        outdir = REPO_ROOT / "experiments" / "raw" / "nextgen" / "m3_profile_reduced"
        plot_paths = [str(path.relative_to(REPO_ROOT)) for path in save_profile_debug_plots(result, outdir)]
    except Exception as exc:  # pragma: no cover - plotting is a diagnostic convenience.
        result.diagnostics["plot_error"] = repr(exc)

    return {
        "status": "completed" if result.converged else "nonconverged",
        "grid": grid,
        "result": result,
        "runtime_sec": result.runtime_sec,
        "diagnostics": result.diagnostics,
        "plot_paths": plot_paths,
    }


def run_baseline_profile(scale: float) -> dict[str, Any]:
    baseline_dir = SRC / "baseline"
    sys.path.insert(0, str(baseline_dir))
    try:
        from config import SimulationParams, eps_from_R
        from mesh import build_disk
        from spaces import create_spaces
        from boundary_data_fourier import FourierBC
        from poisson_neumann import PoissonNeumann
        from newton_block import NewtonAssembler
        from continuation import ContinuationSolver
        from residuals_quick import residuals_quick
    except Exception as exc:
        return {
            "status": "blocked_import",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }

    try:
        params = SimulationParams(
            mu=0.8,
            R=150.0,
            Rdom=10.0,
            M=48,
            tol_newton=1e-8,
            max_newton=32,
            sigma_grid=SIGMA_GRID,
            h_min=0.30,
            h_max=0.45,
        )
        t0 = time.perf_counter()
        mesh, facet_tags = build_disk(params.Rdom, params.h_min, params.h_max)
        spaces = create_spaces(mesh)
        fourier_bc = FourierBC(params.M, reduced_coefficients(scale))
        poisson = PoissonNeumann(
            mesh,
            spaces["Vpsi"],
            facet_tags,
            outer_tag=1,
            Rdom=params.Rdom,
            fourier_bc=fourier_bc,
            mu=params.mu,
            R=params.R,
        )
        assembler = NewtonAssembler(
            spaces["Vomega"],
            spaces["Vpsi"],
            facet_tags,
            outer_tag=1,
            eps=eps_from_R(params.R, params.Rdom),
            use_supg=False,
            tau_scale=1.0,
        )
        solver = ContinuationSolver(
            assembler,
            poisson,
            params,
            facet_tags,
            outer_tag=1,
            fourier_bc=fourier_bc,
        )
        omega, psi = solver.run(mesh, spaces)
        runtime = time.perf_counter() - t0
        residuals = residuals_quick(
            mesh,
            spaces,
            omega,
            psi,
            mu=params.mu,
            R=params.R,
            Rdom=params.Rdom,
            facet_tags=facet_tags,
            outer_tag=1,
            fourier_bc=fourier_bc,
            sigma=1.0,
            pinned_psi_dof=getattr(poisson, "gauge_dof", None),
        )
        residual_summary = {
            "norms": dict(residuals["norms"]),
            "meta": dict(residuals["meta"]),
        }
        ncell = mesh.topology.index_map(mesh.topology.dim).size_global
        nvert = mesh.topology.index_map(0).size_global
        return {
            "status": "completed",
            "runtime_sec": runtime,
            "mesh_vertices": int(nvert),
            "mesh_cells": int(ncell),
            "params": params,
            "spaces": spaces,
            "omega": omega,
            "psi": psi,
            "residuals": residual_summary,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def _function_array(function: Any) -> np.ndarray:
    try:
        return np.asarray(function.x.array, dtype=float)
    except AttributeError:
        return np.asarray(function.vector.array, dtype=float)


def _sample_function(function: Any, space: Any, points: np.ndarray) -> np.ndarray:
    coords = np.asarray(space.tabulate_dof_coordinates(), dtype=float)[:, :2]
    values = _function_array(function)
    interpolator = LinearNDInterpolator(coords, values, fill_value=np.nan)
    return np.asarray(interpolator(points), dtype=float)


def compare_on_probes(nextgen: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float | int]:
    grid: SquareGrid = nextgen["grid"]
    result = nextgen["result"]
    mask = grid.interior_mask & (grid.X * grid.X + grid.Y * grid.Y <= 64.0)
    points = np.column_stack((grid.X[mask], grid.Y[mask]))
    omega_next = result.omega[mask]
    psi_next = result.psi[mask]

    omega_base = _sample_function(baseline["omega"], baseline["spaces"]["Vomega"], points)
    psi_base = _sample_function(baseline["psi"], baseline["spaces"]["Vpsi"], points)
    valid = np.isfinite(omega_base) & np.isfinite(psi_base)

    def metric(next_values: np.ndarray, base_values: np.ndarray, *, remove_mean: bool) -> tuple[float, float, float]:
        diff = np.asarray(next_values[valid] - base_values[valid], dtype=float)
        offset = float(np.mean(diff)) if diff.size else float("nan")
        if remove_mean and diff.size:
            diff = diff - offset
        rms = float(np.sqrt(np.mean(diff * diff))) if diff.size else float("nan")
        linf = float(np.max(np.abs(diff))) if diff.size else float("nan")
        return rms, linf, offset

    omega_rms, omega_linf, omega_offset = metric(omega_next, omega_base, remove_mean=False)
    psi_rms, psi_linf, psi_offset = metric(psi_next, psi_base, remove_mean=True)
    return {
        "probe_count": int(points.shape[0]),
        "valid_probe_count": int(np.count_nonzero(valid)),
        "omega_probe_rms": omega_rms,
        "omega_probe_linf": omega_linf,
        "omega_probe_mean_offset": omega_offset,
        "psi_probe_rms_offset_removed": psi_rms,
        "psi_probe_linf_offset_removed": psi_linf,
        "psi_probe_mean_offset": psi_offset,
    }


def sanitize(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items() if k not in {"grid", "result", "params", "spaces", "omega", "psi"}}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return value


def append_registry(scale: float, summary: dict[str, Any]) -> None:
    registry = REPO_ROOT / "experiments" / "registry.csv"
    next_diag = summary["nextgen"]["diagnostics"]
    baseline = summary.get("baseline", {})
    comparison = summary.get("comparison", {})
    status = summary["status"]
    error_metrics = (
        f"nextgen_rel_res={next_diag.get('relative_residual'):.3e}; "
        f"nextgen_compat={next_diag.get('poisson_compatibility'):.3e}; "
        f"baseline_raw_R_Omega={baseline.get('residuals', {}).get('norms', {}).get('Omega_l2', 'n/a')}; "
        f"baseline_raw_R_Psi={baseline.get('residuals', {}).get('norms', {}).get('Psi_l2', 'n/a')}; "
        f"omega_probe_rms={comparison.get('omega_probe_rms', 'n/a')}; "
        f"psi_probe_rms_offset_removed={comparison.get('psi_probe_rms_offset_removed', 'n/a')}"
    )
    with registry.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "2026-04-26",
                f"m3-profile-reduced-scale-{scale:g}",
                "nextgen_vs_baseline",
                "experiments/run_m3_profile_comparison.py",
                f"current_shape_m3_60_m4_50_scaled_{scale:g}",
                "nextgen_square_half10_vs_baseline_disk_Rdom10",
                "nextgen_n65; baseline_gmsh_h0.30_0.45",
                "profile_continuation_newton_only",
                "mu=0.8; R=150; Rdom=10; M=48; sigma_grid=0,0.25,0.5,0.75,1; no eigensolver",
                f"nextgen={summary['nextgen'].get('runtime_sec'):.3f}; baseline={baseline.get('runtime_sec', 'n/a')}",
                error_metrics,
                "n/a",
                status,
                summary.get("notes", ""),
            ]
        )


def main() -> int:
    summaries_dir = REPO_ROOT / "experiments" / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    final_summary: dict[str, Any] | None = None
    for scale in (0.1, 0.05):
        nextgen = run_nextgen(scale)
        baseline = run_baseline_profile(scale)
        comparison: dict[str, float | int] = {}
        notes = ""
        if nextgen["status"] == "completed" and baseline["status"] == "completed":
            comparison = compare_on_probes(nextgen, baseline)
            status = "completed"
            notes = "Profile-only reduced comparison completed; psi discrepancy removes mean gauge offset."
        elif baseline["status"] == "blocked_import":
            status = "blocked_baseline_dependency"
            notes = f"Baseline profile import blocked: {baseline.get('error')}"
        elif nextgen["status"] != "completed":
            status = "nextgen_nonconverged"
            notes = f"Nextgen status: {nextgen['status']}"
        else:
            status = "baseline_failed"
            notes = f"Baseline status: {baseline['status']}; {baseline.get('error', '')}"

        final_summary = {
            "status": status,
            "scale": scale,
            "nextgen": nextgen,
            "baseline": baseline,
            "comparison": comparison,
            "notes": notes,
        }
        append_registry(scale, final_summary)
        if status in {"completed", "blocked_baseline_dependency"}:
            break

    assert final_summary is not None
    summary_path = summaries_dir / "m3_profile_reduced_comparison_2026-04-26.json"
    summary_path.write_text(json.dumps(sanitize(final_summary), indent=2, sort_keys=True))
    print(json.dumps(sanitize(final_summary), indent=2, sort_keys=True))
    return 0 if final_summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
