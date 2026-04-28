from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
from scipy.interpolate import LinearNDInterpolator

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-euler")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from nextgen.boundary import FourierBoundaryData
from nextgen.config import PhysicalParams
from nextgen.diagnostics import sampled_strong_profile_residual
from nextgen.polar_profile import PolarGrid, PolarProfileParams, PolarProfileProblem


RUN_DATE = "2026-04-28"
PHYSICAL = PhysicalParams(mu=0.8, R=150.0, Rdom=10.0)
SIGMA_GRID = tuple([0.0] + [i / 10.0 for i in range(1, 11)])
MEDIUM_GRID = (40, 120)
FINE_GRID = (54, 162)
BASELINE_MESH = (0.12, 0.18)
SAMPLE_X = np.linspace(-7.0, 7.0, 129)
SAMPLE_Y = np.linspace(-7.0, 7.0, 129)
PROBE_RADII = np.linspace(0.5, 8.0, 58)
PROBE_THETA = np.linspace(-np.pi, np.pi, 174, endpoint=False)

CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "single_c4_20",
        "description": "single non-c3 fourfold mode, moderate amplitude",
        "coeffs": {4: 20.0 + 0.0j},
    },
    {
        "name": "phase_shifted_c2_20",
        "description": "single phase-shifted twofold mode, moderate amplitude",
        "coeffs": {2: 15.0 + 8.0j},
    },
    {
        "name": "mixed_1_4_7_small",
        "description": "small non-symmetric mixed low/high Fourier modes",
        "coeffs": {1: 0.48 - 0.24j, 4: 0.96 + 0.0j, 7: -0.36 + 0.24j},
    },
    {
        "name": "mixed_2_5_8_reduced",
        "description": "reduced non-symmetric mixed even/odd Fourier modes",
        "coeffs": {2: 2.4 + 1.2j, 5: -2.0 + 0.4j, 8: 0.8 - 0.6j},
    },
    {
        "name": "mixed_2_5_8_moderate",
        "description": "moderate mixed even/odd Fourier modes that failed under upwind continuation",
        "coeffs": {2: 6.0 + 3.0j, 5: -5.0 + 1.0j, 8: 2.0 - 1.5j},
    },
)


def run_nextgen(coeffs: dict[int, complex], grid_spec: tuple[int, int]) -> dict[str, Any]:
    nr, ntheta = grid_spec
    grid = PolarGrid(radius=PHYSICAL.Rdom, nr=nr, ntheta=ntheta)
    boundary = FourierBoundaryData(M=48, coeffs=coeffs)
    problem = PolarProfileProblem(grid, PHYSICAL, boundary, advection_scheme="blended", upwind_blend=0.01)
    result = problem.solve(
        PolarProfileParams(
            sigma_grid=SIGMA_GRID,
            tol_step=1e-8,
            tol_residual=1e-8,
            max_iter=44,
            max_backtracks=20,
            adaptive_continuation=True,
            min_sigma_step=0.00625,
            sigma_step_growth=1.25,
            sigma_step_shrink=0.5,
            picard_predictor_steps=0,
        )
    )
    return {
        "status": "completed" if result.converged else "nonconverged",
        "grid": {"nr": nr, "ntheta": ntheta, "dr": grid.dr, "dtheta": grid.dtheta},
        "problem": problem,
        "result": result,
        "runtime_sec": result.runtime_sec,
        "diagnostics": result.diagnostics,
    }


def run_baseline(coeffs: dict[int, complex]) -> dict[str, Any]:
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
        return {"status": "blocked_import", "error": repr(exc), "traceback": traceback.format_exc()}

    try:
        h_min, h_max = BASELINE_MESH
        params = SimulationParams(
            mu=PHYSICAL.mu,
            R=PHYSICAL.R,
            Rdom=PHYSICAL.Rdom,
            M=48,
            tol_newton=1e-9,
            max_newton=48,
            sigma_grid=SIGMA_GRID,
            h_min=h_min,
            h_max=h_max,
        )
        t0 = time.perf_counter()
        mesh, facet_tags = build_disk(params.Rdom, params.h_min, params.h_max)
        spaces = create_spaces(mesh)
        fourier_bc = FourierBC(params.M, coeffs)
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
        return {
            "status": "completed",
            "runtime_sec": runtime,
            "mesh": {"h_min": h_min, "h_max": h_max},
            "mesh_vertices": int(mesh.topology.index_map(0).size_global),
            "mesh_cells": int(mesh.topology.index_map(mesh.topology.dim).size_global),
            "spaces": spaces,
            "omega": omega,
            "psi": psi,
            "residuals": {"norms": dict(residuals["norms"]), "meta": dict(residuals["meta"])},
        }
    except Exception as exc:
        return {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()}


def _function_array(function: Any) -> np.ndarray:
    try:
        return np.asarray(function.x.array, dtype=float)
    except AttributeError:
        return np.asarray(function.vector.array, dtype=float)


def _function_xy(space: Any) -> np.ndarray:
    return np.asarray(space.tabulate_dof_coordinates(), dtype=float)[:, :2]


def _sample_function(function: Any, space: Any, points: np.ndarray) -> np.ndarray:
    return np.asarray(
        LinearNDInterpolator(_function_xy(space), _function_array(function), fill_value=np.nan)(points),
        dtype=float,
    )


def _sample_nextgen(run: dict[str, Any], points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return run["problem"].sample_cartesian(run["result"], points[:, 0], points[:, 1])


def _probe_points() -> np.ndarray:
    rr, tt = np.meshgrid(PROBE_RADII, PROBE_THETA, indexing="ij")
    return np.column_stack((rr.reshape(-1) * np.cos(tt.reshape(-1)), rr.reshape(-1) * np.sin(tt.reshape(-1))))


def _field_metrics(a: np.ndarray, b: np.ndarray, valid: np.ndarray, *, remove_mean: bool) -> tuple[float, float, float]:
    diff = np.asarray(a[valid] - b[valid], dtype=float)
    offset = float(np.mean(diff)) if diff.size else float("nan")
    if remove_mean and diff.size:
        diff = diff - offset
    return (
        float(np.sqrt(np.mean(diff * diff))) if diff.size else float("nan"),
        float(np.max(np.abs(diff))) if diff.size else float("nan"),
        offset,
    )


def compare_nextgen_to_baseline(nextgen: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float | int]:
    points = _probe_points()
    omega_next, psi_next = _sample_nextgen(nextgen, points)
    omega_base = _sample_function(baseline["omega"], baseline["spaces"]["Vomega"], points)
    psi_base = _sample_function(baseline["psi"], baseline["spaces"]["Vpsi"], points)
    valid = np.isfinite(omega_next) & np.isfinite(psi_next) & np.isfinite(omega_base) & np.isfinite(psi_base)
    omega_rms, omega_linf, omega_offset = _field_metrics(omega_next, omega_base, valid, remove_mean=False)
    psi_rms, psi_linf, psi_offset = _field_metrics(psi_next, psi_base, valid, remove_mean=True)
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


def compare_nextgen_grids(coarse: dict[str, Any], fine: dict[str, Any]) -> dict[str, float | int]:
    points = _probe_points()
    omega_coarse, psi_coarse = _sample_nextgen(coarse, points)
    omega_fine, psi_fine = _sample_nextgen(fine, points)
    valid = np.isfinite(omega_coarse) & np.isfinite(psi_coarse) & np.isfinite(omega_fine) & np.isfinite(psi_fine)
    omega_rms, omega_linf, omega_offset = _field_metrics(omega_fine, omega_coarse, valid, remove_mean=False)
    psi_rms, psi_linf, psi_offset = _field_metrics(psi_fine, psi_coarse, valid, remove_mean=True)
    return {
        "valid_probe_count": int(np.count_nonzero(valid)),
        "omega_rms_fine_minus_medium": omega_rms,
        "omega_linf_fine_minus_medium": omega_linf,
        "omega_mean_offset_fine_minus_medium": omega_offset,
        "psi_rms_offset_removed_fine_minus_medium": psi_rms,
        "psi_linf_offset_removed_fine_minus_medium": psi_linf,
        "psi_mean_offset_fine_minus_medium": psi_offset,
    }


def common_residual_for_sample(omega: np.ndarray, psi: np.ndarray) -> dict[str, float]:
    return sampled_strong_profile_residual(
        omega,
        psi,
        SAMPLE_X,
        SAMPLE_Y,
        mu=PHYSICAL.mu,
        eps=PHYSICAL.eps,
    )


def common_residual_nextgen(run: dict[str, Any]) -> dict[str, float]:
    X, Y = np.meshgrid(SAMPLE_X, SAMPLE_Y, indexing="ij")
    points = np.column_stack((X.reshape(-1), Y.reshape(-1)))
    omega, psi = _sample_nextgen(run, points)
    return common_residual_for_sample(omega.reshape(X.shape), psi.reshape(X.shape))


def common_residual_baseline(baseline: dict[str, Any]) -> dict[str, float]:
    X, Y = np.meshgrid(SAMPLE_X, SAMPLE_Y, indexing="ij")
    points = np.column_stack((X.reshape(-1), Y.reshape(-1)))
    omega = _sample_function(baseline["omega"], baseline["spaces"]["Vomega"], points).reshape(X.shape)
    psi = _sample_function(baseline["psi"], baseline["spaces"]["Vpsi"], points).reshape(X.shape)
    return common_residual_for_sample(omega, psi)


def make_case_plots(case_name: str, fine: dict[str, Any], baseline: dict[str, Any], outdir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-PHYSICAL.Rdom, PHYSICAL.Rdom, 241)
    y = np.linspace(-PHYSICAL.Rdom, PHYSICAL.Rdom, 241)
    X, Y = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack((X.reshape(-1), Y.reshape(-1)))
    omega_next, psi_next = fine["problem"].sample_cartesian(fine["result"], X, Y)
    created: list[str] = []
    extent = (-PHYSICAL.Rdom, PHYSICAL.Rdom, -PHYSICAL.Rdom, PHYSICAL.Rdom)

    def save_heatmap(values: np.ndarray, filename: str, title: str, cmap: str = "RdBu_r") -> None:
        path = outdir / filename
        fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
        image = ax.imshow(values.T, origin="lower", extent=extent, cmap=cmap)
        ax.set_aspect("equal")
        ax.set_title(title)
        fig.colorbar(image, ax=ax)
        fig.savefig(path, dpi=135)
        plt.close(fig)
        created.append(str(path.relative_to(REPO_ROOT)))

    def save_contours(values: np.ndarray, filename: str, title: str) -> None:
        path = outdir / filename
        fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
        ax.contour(X, Y, values, levels=42, colors="black", linewidths=0.55)
        ax.set_aspect("equal")
        ax.set_title(title)
        fig.savefig(path, dpi=135)
        plt.close(fig)
        created.append(str(path.relative_to(REPO_ROOT)))

    save_heatmap(omega_next, "nextgen_omega.png", f"{case_name} nextgen Omega")
    save_contours(psi_next, "nextgen_psi_contours.png", f"{case_name} nextgen psi")

    if baseline["status"] == "completed":
        omega_base = _sample_function(baseline["omega"], baseline["spaces"]["Vomega"], points).reshape(X.shape)
        psi_base = _sample_function(baseline["psi"], baseline["spaces"]["Vpsi"], points).reshape(X.shape)
        save_heatmap(omega_base, "baseline_omega.png", f"{case_name} baseline Omega")
        save_contours(psi_base, "baseline_psi_contours.png", f"{case_name} baseline psi")
        omega_diff = omega_next - omega_base
        psi_diff = psi_next - psi_base
        finite_psi = np.isfinite(psi_diff)
        if np.any(finite_psi):
            psi_diff = psi_diff - float(np.nanmean(psi_diff))
        save_heatmap(omega_diff, "omega_nextgen_minus_baseline.png", f"{case_name} Omega diff")
        save_heatmap(psi_diff, "psi_nextgen_minus_baseline_offset_removed.png", f"{case_name} psi diff")

    return created


def sanitize(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        blocked = {"problem", "result", "spaces", "omega", "psi"}
        return {str(k): sanitize(v) for k, v in value.items() if k not in blocked}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return value


def append_registry(summary: dict[str, Any]) -> None:
    registry = REPO_ROOT / "experiments" / "registry.csv"
    with registry.open("a", newline="") as f:
        writer = csv.writer(f)
        for case in summary["cases"]:
            fine = case["nextgen_fine"]
            medium = case["nextgen_medium"]
            baseline = case["baseline"]
            comparison = case.get("comparison_to_baseline", {})
            self_compare = case.get("nextgen_fine_vs_medium", {})
            common = case.get("common_sampled_residuals", {})
            writer.writerow(
                [
                    RUN_DATE,
                    f"m3-general-boundary-visual-{case['name']}",
                    "nextgen_vs_baseline",
                    "experiments/run_m3_general_boundary_visual_suite.py",
                    case["description"],
                    "nextgen_polar_full_disk_vs_baseline_disk_Rdom10",
                    "nextgen_medium_nr40_ntheta120; nextgen_fine_nr54_ntheta162; baseline_h0.12_0.18",
                    "profile_continuation_newton_only_polar_fd",
                    "four non-c3/general boundary cases; plots; no eigensolver",
                    f"nextgen_fine={fine.get('runtime_sec'):.3f}; nextgen_medium={medium.get('runtime_sec'):.3f}; baseline={baseline.get('runtime_sec', 'n/a')}",
                    (
                        f"status={case['status']}; "
                        f"fine_rel_res={fine.get('diagnostics', {}).get('relative_residual', float('nan')):.3e}; "
                        f"fine_common_strong_rel={common.get('nextgen_fine', {}).get('profile_strong_relative', 'n/a')}; "
                        f"medium_common_strong_rel={common.get('nextgen_medium', {}).get('profile_strong_relative', 'n/a')}; "
                        f"baseline_common_strong_rel={common.get('baseline', {}).get('profile_strong_relative', 'n/a')}; "
                        f"omega_probe_rms={comparison.get('omega_probe_rms', 'n/a')}; "
                        f"psi_probe_rms_offset_removed={comparison.get('psi_probe_rms_offset_removed', 'n/a')}; "
                        f"omega_self_rms={self_compare.get('omega_rms_fine_minus_medium', 'n/a')}; "
                        f"psi_self_rms={self_compare.get('psi_rms_offset_removed_fine_minus_medium', 'n/a')}"
                    ),
                    "n/a",
                    case["status"],
                    case.get("notes", ""),
                ]
            )


def main() -> int:
    raw_root = REPO_ROOT / "experiments" / "raw" / "nextgen" / "m3_general_boundary_visual_suite"
    summary_dir = REPO_ROOT / "experiments" / "summaries"
    raw_root.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    cases_out = []
    overall_status = "completed"
    for case in CASES:
        medium = run_nextgen(case["coeffs"], MEDIUM_GRID)
        fine = run_nextgen(case["coeffs"], FINE_GRID)
        baseline = run_baseline(case["coeffs"])
        status = "completed"
        notes = "General boundary visual/profile validation completed."
        comparison: dict[str, Any] = {}
        self_compare: dict[str, Any] = {}
        common: dict[str, Any] = {}
        plots: list[str] = []

        if medium["status"] != "completed":
            status = "nextgen_medium_nonconverged"
            notes = medium["status"]
            overall_status = "partial"
        elif fine["status"] != "completed":
            status = "nextgen_fine_nonconverged"
            notes = fine["status"]
            overall_status = "partial"
        else:
            self_compare = compare_nextgen_grids(medium, fine)
            common = {
                "nextgen_medium": common_residual_nextgen(medium),
                "nextgen_fine": common_residual_nextgen(fine),
            }
            plots = make_case_plots(case["name"], fine, baseline, raw_root / case["name"])
            if baseline["status"] != "completed":
                status = f"baseline_{baseline['status']}"
                notes = baseline.get("error", baseline["status"])
                overall_status = "partial"
            else:
                comparison = compare_nextgen_to_baseline(fine, baseline)
                common["baseline"] = common_residual_baseline(baseline)

        cases_out.append(
            {
                "name": case["name"],
                "description": case["description"],
                "coeffs": case["coeffs"],
                "status": status,
                "notes": notes,
                "nextgen_medium": medium,
                "nextgen_fine": fine,
                "baseline": baseline,
                "nextgen_fine_vs_medium": self_compare,
                "comparison_to_baseline": comparison,
                "common_sampled_residuals": common,
                "plot_paths": plots,
            }
        )

    summary = {
        "status": overall_status,
        "input": {
            "physical": {"mu": PHYSICAL.mu, "R": PHYSICAL.R, "Rdom": PHYSICAL.Rdom, "eps": PHYSICAL.eps},
            "sigma_grid": SIGMA_GRID,
            "nextgen_medium_grid": {"nr": MEDIUM_GRID[0], "ntheta": MEDIUM_GRID[1]},
            "nextgen_fine_grid": {"nr": FINE_GRID[0], "ntheta": FINE_GRID[1]},
            "baseline_mesh": {"h_min": BASELINE_MESH[0], "h_max": BASELINE_MESH[1]},
            "sample_grid": {"x_min": float(SAMPLE_X[0]), "x_max": float(SAMPLE_X[-1]), "n": int(SAMPLE_X.size)},
            "probe_grid": {"r_min": float(PROBE_RADII[0]), "r_max": float(PROBE_RADII[-1]), "nr": int(PROBE_RADII.size), "ntheta": int(PROBE_THETA.size)},
        },
        "norm_definitions": {
            "common_strong_relative": "both codes sampled to the same Cartesian grid; sqrt(||R_Omega||_L2^2+||R_Psi||_L2^2)/max(||Omega||_L2,||Delta psi||_L2)",
            "probe_rms": "sqrt(mean((nextgen-baseline)^2)) over common polar probes r in [0.5,8]",
            "probe_linf": "max(abs(nextgen-baseline)) over valid common probes",
            "psi_probe_errors": "streamfunction probe differences after subtracting mean difference to remove gauge offset",
            "nextgen_self_convergence": "fine nextgen minus medium nextgen on the same common probes, with psi mean offset removed",
        },
        "cases": cases_out,
    }
    append_registry(summary)
    path = summary_dir / f"m3_general_boundary_visual_suite_{RUN_DATE}.json"
    path.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    print(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    return 0 if overall_status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
