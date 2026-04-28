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
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-euler")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from nextgen.boundary import FourierBoundaryData
from nextgen.config import PhysicalParams
from nextgen.diagnostics import sampled_strong_profile_residual
from nextgen.grid import SquareGrid
from nextgen.polar_profile import PolarGrid, PolarProfileParams, PolarProfileProblem
from nextgen.profile import ProfileNewtonParams, ProfileProblem


RUN_DATE = "2026-04-28"
PHYSICAL = PhysicalParams(mu=0.8, R=150.0, Rdom=10.0)
COEFFS = {2: 6.0 + 3.0j, 5: -5.0 + 1.0j, 8: 2.0 - 1.5j}
SIGMA_GRID = tuple([0.0] + [i / 10.0 for i in range(1, 11)])
SAMPLE_X = np.linspace(-7.0, 7.0, 129)
SAMPLE_Y = np.linspace(-7.0, 7.0, 129)
PROBE_RADII = np.linspace(0.5, 8.0, 58)
PROBE_THETA = np.linspace(-np.pi, np.pi, 174, endpoint=False)


def run_square() -> dict[str, Any]:
    grid = SquareGrid(half_width=PHYSICAL.Rdom, n=97)
    problem = ProfileProblem(grid, PHYSICAL, FourierBoundaryData(M=48, coeffs=COEFFS))
    result = problem.solve(
        ProfileNewtonParams(
            sigma_grid=SIGMA_GRID,
            tol_step=1e-8,
            tol_residual=1e-8,
            max_newton=40,
            max_backtracks=20,
        )
    )
    return {
        "status": "completed" if result.converged else "nonconverged",
        "grid": {"n": grid.n, "h": grid.h, "half_width": grid.half_width},
        "problem": problem,
        "result": result,
        "runtime_sec": result.runtime_sec,
        "diagnostics": result.diagnostics,
    }


def run_polar_blended() -> dict[str, Any]:
    grid = PolarGrid(radius=PHYSICAL.Rdom, nr=54, ntheta=162)
    problem = PolarProfileProblem(
        grid,
        PHYSICAL,
        FourierBoundaryData(M=48, coeffs=COEFFS),
        advection_scheme="blended",
        upwind_blend=0.01,
    )
    result = problem.solve(
        PolarProfileParams(
            sigma_grid=SIGMA_GRID,
            tol_step=1e-8,
            tol_residual=1e-8,
            tol_residual_only=1e-8,
            max_iter=80,
            max_backtracks=28,
            adaptive_continuation=True,
            min_sigma_step=5e-4,
            sigma_step_growth=1.25,
            sigma_step_shrink=0.5,
        )
    )
    return {
        "status": "completed" if result.converged else "nonconverged",
        "grid": {"nr": grid.nr, "ntheta": grid.ntheta, "dr": grid.dr, "dtheta": grid.dtheta},
        "problem": problem,
        "result": result,
        "runtime_sec": result.runtime_sec,
        "diagnostics": result.diagnostics,
    }


def run_polar_upwind_fine() -> dict[str, Any]:
    grid = PolarGrid(radius=PHYSICAL.Rdom, nr=72, ntheta=216)
    problem = PolarProfileProblem(
        grid,
        PHYSICAL,
        FourierBoundaryData(M=48, coeffs=COEFFS),
        advection_scheme="upwind",
    )
    result = problem.solve(
        PolarProfileParams(
            sigma_grid=SIGMA_GRID,
            tol_step=1e-8,
            tol_residual=1e-8,
            tol_residual_only=1e-8,
            max_iter=80,
            max_backtracks=28,
            adaptive_continuation=True,
            min_sigma_step=1e-3,
            sigma_step_growth=1.2,
            sigma_step_shrink=0.5,
            least_squares_fallback=False,
        )
    )
    return {
        "status": "completed" if result.converged else "nonconverged",
        "grid": {"nr": grid.nr, "ntheta": grid.ntheta, "dr": grid.dr, "dtheta": grid.dtheta},
        "problem": problem,
        "result": result,
        "runtime_sec": result.runtime_sec,
        "diagnostics": result.diagnostics,
    }


def run_baseline() -> dict[str, Any]:
    baseline_dir = SRC / "baseline"
    sys.path.insert(0, str(baseline_dir))
    try:
        from config import SimulationParams
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
        params = SimulationParams(
            mu=PHYSICAL.mu,
            R=PHYSICAL.R,
            Rdom=PHYSICAL.Rdom,
            M=48,
            tol_newton=1e-9,
            max_newton=48,
            sigma_grid=SIGMA_GRID,
            h_min=0.12,
            h_max=0.18,
        )
        t0 = time.perf_counter()
        mesh, facet_tags = build_disk(params.Rdom, params.h_min, params.h_max)
        spaces = create_spaces(mesh)
        fourier_bc = FourierBC(params.M, COEFFS)
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
            eps=PHYSICAL.eps,
        )
        omega, psi = ContinuationSolver(
            assembler,
            poisson,
            params,
            facet_tags=facet_tags,
            outer_tag=1,
            fourier_bc=fourier_bc,
        ).run(mesh, spaces)
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
        )
        return {
            "status": "completed",
            "runtime_sec": runtime,
            "mesh": {"h_min": params.h_min, "h_max": params.h_max},
            "mesh_cells": int(mesh.topology.index_map(mesh.topology.dim).size_local),
            "mesh_vertices": int(mesh.topology.index_map(0).size_local),
            "spaces": spaces,
            "omega": omega,
            "psi": psi,
            "residuals": residuals,
        }
    except Exception as exc:
        return {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()}


def probe_points() -> np.ndarray:
    rr, tt = np.meshgrid(PROBE_RADII, PROBE_THETA, indexing="ij")
    return np.column_stack((rr.reshape(-1) * np.cos(tt.reshape(-1)), rr.reshape(-1) * np.sin(tt.reshape(-1))))


def sample_baseline(function: Any, space: Any, points: np.ndarray) -> np.ndarray:
    coords = np.asarray(space.tabulate_dof_coordinates())[:, :2]
    values = np.asarray(function.x.array, dtype=float)
    return np.asarray(LinearNDInterpolator(coords, values, fill_value=np.nan)(points), dtype=float)


def sample_square(run: dict[str, Any], points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid = run["problem"].grid
    omega_interp = RegularGridInterpolator((grid.x, grid.y), run["result"].omega, bounds_error=False, fill_value=np.nan)
    psi_interp = RegularGridInterpolator((grid.x, grid.y), run["result"].psi, bounds_error=False, fill_value=np.nan)
    return np.asarray(omega_interp(points), dtype=float), np.asarray(psi_interp(points), dtype=float)


def sample_polar(run: dict[str, Any], points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return run["problem"].sample_cartesian(run["result"], points[:, 0], points[:, 1])


def common_residual_from_sampler(sampler: Any) -> dict[str, float]:
    X, Y = np.meshgrid(SAMPLE_X, SAMPLE_Y, indexing="ij")
    points = np.column_stack((X.reshape(-1), Y.reshape(-1)))
    omega, psi = sampler(points)
    return sampled_strong_profile_residual(
        omega.reshape(X.shape),
        psi.reshape(X.shape),
        SAMPLE_X,
        SAMPLE_Y,
        mu=PHYSICAL.mu,
        eps=PHYSICAL.eps,
    )


def compare_to_baseline(run: dict[str, Any], sampler: Any, baseline: dict[str, Any]) -> dict[str, float | int]:
    points = probe_points()
    omega_next, psi_next = sampler(points)
    omega_base = sample_baseline(baseline["omega"], baseline["spaces"]["Vomega"], points)
    psi_base = sample_baseline(baseline["psi"], baseline["spaces"]["Vpsi"], points)
    valid = np.isfinite(omega_next) & np.isfinite(psi_next) & np.isfinite(omega_base) & np.isfinite(psi_base)
    omega_diff = omega_next[valid] - omega_base[valid]
    psi_diff = psi_next[valid] - psi_base[valid]
    psi_diff = psi_diff - float(np.mean(psi_diff))
    return {
        "valid_probe_count": int(np.count_nonzero(valid)),
        "omega_probe_rms": float(np.sqrt(np.mean(omega_diff**2))),
        "omega_probe_linf": float(np.max(np.abs(omega_diff))),
        "psi_probe_rms_offset_removed": float(np.sqrt(np.mean(psi_diff**2))),
        "psi_probe_linf_offset_removed": float(np.max(np.abs(psi_diff))),
    }


def make_plots(
    square: dict[str, Any],
    polar: dict[str, Any],
    polar_upwind: dict[str, Any],
    baseline: dict[str, Any],
    outdir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-PHYSICAL.Rdom, PHYSICAL.Rdom, 241)
    y = np.linspace(-PHYSICAL.Rdom, PHYSICAL.Rdom, 241)
    X, Y = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack((X.reshape(-1), Y.reshape(-1)))
    fields = {
        "square_fd": sample_square(square, points),
        "polar_blended": sample_polar(polar, points),
        "polar_upwind_fine": sample_polar(polar_upwind, points),
    }
    if baseline["status"] == "completed":
        fields["baseline"] = (
            sample_baseline(baseline["omega"], baseline["spaces"]["Vomega"], points),
            sample_baseline(baseline["psi"], baseline["spaces"]["Vpsi"], points),
        )
    created: list[str] = []
    extent = (-PHYSICAL.Rdom, PHYSICAL.Rdom, -PHYSICAL.Rdom, PHYSICAL.Rdom)
    for name, (omega, psi) in fields.items():
        omega_grid = omega.reshape(X.shape)
        psi_grid = psi.reshape(X.shape)
        for values, suffix, cmap in ((omega_grid, "omega", "RdBu_r"), (psi_grid, "psi", "viridis")):
            path = outdir / f"{name}_{suffix}.png"
            fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
            if suffix == "omega":
                image = ax.imshow(values.T, origin="lower", extent=extent, cmap=cmap)
                fig.colorbar(image, ax=ax)
            else:
                ax.contour(X, Y, values, levels=42, colors="black", linewidths=0.55)
            ax.set_aspect("equal")
            ax.set_title(f"{name} {suffix}")
            fig.savefig(path, dpi=135)
            plt.close(fig)
            created.append(str(path.relative_to(REPO_ROOT)))
    return created


def sanitize(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        blocked = {"problem", "result", "spaces", "omega", "psi", "rOmega", "rPsi"}
        return {str(k): sanitize(v) for k, v in value.items() if k not in blocked}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return value


def append_registry(summary: dict[str, Any]) -> None:
    registry = REPO_ROOT / "experiments" / "registry.csv"
    with registry.open("a", newline="") as f:
        writer = csv.writer(f)
        square = summary["square_fd"]
        polar = summary["polar_blended"]
        polar_upwind = summary["polar_upwind_fine"]
        baseline = summary["baseline"]
        writer.writerow(
            [
                RUN_DATE,
                "m3-target-mixed-method-probe",
                "nextgen_vs_baseline",
                "experiments/run_m3_target_mixed_method_probe.py",
                "{c2=6+3i,c5=-5+i,c8=2-1.5i}",
                "square_fd_and_polar_disk_vs_baseline_disk_Rdom10",
                "square_n97; polar_blended_nr54_ntheta162; polar_upwind_nr72_ntheta216; baseline_h0.12_0.18",
                "profile_continuation_newton_only_no_eigensolver",
                "square FD fallback probe; polar blended diagnostic; no M4",
                f"square={square.get('runtime_sec'):.3f}; polar_blended={polar.get('runtime_sec'):.3f}; polar_upwind={polar_upwind.get('runtime_sec'):.3f}; baseline={baseline.get('runtime_sec', 'n/a')}",
                (
                    f"square_status={square['status']}; polar_status={polar['status']}; baseline_status={baseline['status']}; "
                    f"polar_upwind_status={polar_upwind['status']}; "
                    f"square_rel={square.get('diagnostics', {}).get('relative_residual', float('nan')):.3e}; "
                    f"polar_rel={polar.get('diagnostics', {}).get('relative_residual', float('nan')):.3e}; "
                    f"polar_upwind_rel={polar_upwind.get('diagnostics', {}).get('relative_residual', float('nan')):.3e}; "
                    f"square_common={summary.get('common_residuals', {}).get('square_fd', {}).get('profile_strong_relative', 'n/a')}; "
                    f"polar_common={summary.get('common_residuals', {}).get('polar_blended', {}).get('profile_strong_relative', 'n/a')}; "
                    f"polar_upwind_common={summary.get('common_residuals', {}).get('polar_upwind_fine', {}).get('profile_strong_relative', 'n/a')}; "
                    f"baseline_common={summary.get('common_residuals', {}).get('baseline', {}).get('profile_strong_relative', 'n/a')}"
                ),
                "n/a",
                summary["status"],
                summary["notes"],
            ]
        )


def main() -> int:
    square = run_square()
    polar = run_polar_blended()
    polar_upwind = run_polar_upwind_fine()
    baseline = run_baseline()
    common: dict[str, Any] = {}
    probe: dict[str, Any] = {}
    plots: list[str] = []
    status = "completed"
    notes = "Target mixed method probe completed."

    if square["status"] == "completed":
        common["square_fd"] = common_residual_from_sampler(lambda points: sample_square(square, points))
    else:
        status = "partial"
        notes = "square_fd_nonconverged"
    if polar["status"] == "completed":
        common["polar_blended"] = common_residual_from_sampler(lambda points: sample_polar(polar, points))
    else:
        status = "partial"
        notes = "polar_blended_nonconverged"
    if polar_upwind["status"] == "completed":
        common["polar_upwind_fine"] = common_residual_from_sampler(lambda points: sample_polar(polar_upwind, points))
    else:
        status = "partial"
        notes = "polar_upwind_fine_nonconverged"
    if baseline["status"] == "completed":
        common["baseline"] = common_residual_from_sampler(
            lambda points: (
                sample_baseline(baseline["omega"], baseline["spaces"]["Vomega"], points),
                sample_baseline(baseline["psi"], baseline["spaces"]["Vpsi"], points),
            )
        )
        if square["status"] == "completed":
            probe["square_fd_vs_baseline"] = compare_to_baseline(square, lambda points: sample_square(square, points), baseline)
        if polar["status"] == "completed":
            probe["polar_blended_vs_baseline"] = compare_to_baseline(polar, lambda points: sample_polar(polar, points), baseline)
        if polar_upwind["status"] == "completed":
            probe["polar_upwind_fine_vs_baseline"] = compare_to_baseline(polar_upwind, lambda points: sample_polar(polar_upwind, points), baseline)
        plots = make_plots(
            square,
            polar,
            polar_upwind,
            baseline,
            REPO_ROOT / "experiments" / "raw" / "nextgen" / "m3_target_mixed_method_probe",
        )
    else:
        status = "partial"
        notes = baseline.get("error", baseline["status"])

    summary = {
        "status": status,
        "notes": notes,
        "input": {
            "physical": {"mu": PHYSICAL.mu, "R": PHYSICAL.R, "Rdom": PHYSICAL.Rdom, "eps": PHYSICAL.eps},
            "coeffs": COEFFS,
            "sigma_grid": SIGMA_GRID,
        },
        "square_fd": square,
        "polar_blended": polar,
        "polar_upwind_fine": polar_upwind,
        "baseline": baseline,
        "common_residuals": common,
        "probe_errors": probe,
        "plot_paths": plots,
    }
    append_registry(summary)
    out = REPO_ROOT / "experiments" / "summaries" / f"m3_target_mixed_method_probe_{RUN_DATE}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    print(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
