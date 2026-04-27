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
from scipy.interpolate import RegularGridInterpolator

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-euler")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from nextgen.boundary import FourierBoundaryData
from nextgen.config import PhysicalParams
from nextgen.diagnostics import sampled_strong_profile_residual
from nextgen.polar_profile import PolarGrid, PolarProfileParams, PolarProfileProblem


COEFFS = {
    0: 0.0 + 0.0j,
    1: 0.0 + 0.0j,
    2: 0.0 + 0.0j,
    3: 60.0 + 0.0j,
    4: 0.0 + 0.0j,
    5: 0.0 + 0.0j,
    6: 0.0 + 0.0j,
    10: 0.0 + 0.0j,
    12: 0.0 + 0.0j,
    24: 0.0 + 0.0j,
    36: 0.0 + 0.0j,
}
BASELINE_SIGMA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
NEXTGEN_SIGMA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def run_nextgen() -> dict[str, Any]:
    grid = PolarGrid(radius=10.0, nr=54, ntheta=54, symmetry=3)
    physical = PhysicalParams(mu=0.8, R=150.0, Rdom=10.0)
    boundary = FourierBoundaryData(M=48, coeffs=COEFFS)
    problem = PolarProfileProblem(grid, physical, boundary)
    result = problem.solve(
        PolarProfileParams(
            sigma_grid=NEXTGEN_SIGMA_GRID,
            tol_step=1e-9,
            tol_residual=1e-9,
            max_iter=32,
            max_backtracks=16,
            initial_damping=1.0,
        )
    )
    return {
        "status": "completed" if result.converged else "nonconverged",
        "solver": "polar_c3_sector_upwind_newton",
        "grid": grid,
        "problem": problem,
        "result": result,
        "runtime_sec": result.runtime_sec,
        "diagnostics": result.diagnostics,
    }


def run_baseline_profile() -> dict[str, Any]:
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
            tol_newton=1e-9,
            max_newton=48,
            sigma_grid=BASELINE_SIGMA_GRID,
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
            "residuals": {
                "norms": dict(residuals["norms"]),
                "meta": dict(residuals["meta"]),
            },
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


def _function_xy(space: Any) -> np.ndarray:
    return np.asarray(space.tabulate_dof_coordinates(), dtype=float)[:, :2]


def _sample_function(function: Any, space: Any, points: np.ndarray) -> np.ndarray:
    coords = _function_xy(space)
    values = _function_array(function)
    interpolator = LinearNDInterpolator(coords, values, fill_value=np.nan)
    return np.asarray(interpolator(points), dtype=float)


def _sample_nextgen_field(nextgen: dict[str, Any], field_name: str, points: np.ndarray) -> np.ndarray:
    grid = nextgen["grid"]
    result = nextgen["result"]
    if isinstance(grid, PolarGrid):
        omega, psi = nextgen["problem"].sample_cartesian(result, points[:, 0], points[:, 1])
        return np.asarray(omega if field_name == "omega" else psi, dtype=float)
    values = getattr(result, field_name)
    interpolator = RegularGridInterpolator((grid.x, grid.y), values, bounds_error=False, fill_value=np.nan)
    return np.asarray(interpolator(points), dtype=float)


def compare_on_probes(nextgen: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float | int]:
    grid = nextgen["grid"]
    result = nextgen["result"]
    if isinstance(grid, PolarGrid):
        radii = np.linspace(max(0.3, 2.0 * grid.dr), 8.0, 58)
        theta = np.linspace(-np.pi, np.pi, 174, endpoint=False)
        rr, tt = np.meshgrid(radii, theta, indexing="ij")
        points = np.column_stack((rr.reshape(-1) * np.cos(tt.reshape(-1)), rr.reshape(-1) * np.sin(tt.reshape(-1))))
        omega_next = _sample_nextgen_field(nextgen, "omega", points)
        psi_next = _sample_nextgen_field(nextgen, "psi", points)
    else:
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


def common_sampled_residuals(nextgen: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    sample_x = np.linspace(-7.0, 7.0, 129)
    sample_y = np.linspace(-7.0, 7.0, 129)
    X, Y = np.meshgrid(sample_x, sample_y, indexing="ij")
    points = np.column_stack((X.reshape(-1), Y.reshape(-1)))

    omega_next = _sample_nextgen_field(nextgen, "omega", points).reshape(X.shape)
    psi_next = _sample_nextgen_field(nextgen, "psi", points).reshape(X.shape)
    out = {
        "sample_domain": "square_half_width_7",
        "nextgen": sampled_strong_profile_residual(
            omega_next,
            psi_next,
            sample_x,
            sample_y,
            mu=0.8,
            eps=(10.0 * 10.0) / (150.0 * 150.0),
        ),
    }
    if baseline.get("status") == "completed":
        omega_base = _sample_function(baseline["omega"], baseline["spaces"]["Vomega"], points).reshape(X.shape)
        psi_base = _sample_function(baseline["psi"], baseline["spaces"]["Vpsi"], points).reshape(X.shape)
        out["baseline"] = sampled_strong_profile_residual(
            omega_base,
            psi_base,
            sample_x,
            sample_y,
            mu=0.8,
            eps=(10.0 * 10.0) / (150.0 * 150.0),
        )
    return out


def threefold_symmetry_metrics(nextgen: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    radii = np.linspace(0.25, 7.0, 40)
    theta = np.linspace(-np.pi, np.pi, 181, endpoint=False)
    rr, tt = np.meshgrid(radii, theta, indexing="ij")
    points = np.column_stack((rr.reshape(-1) * np.cos(tt.reshape(-1)), rr.reshape(-1) * np.sin(tt.reshape(-1))))
    angle = 2.0 * np.pi / 3.0
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = points @ rot.T

    def metrics(sample_a: np.ndarray, sample_b: np.ndarray) -> dict[str, float]:
        valid = np.isfinite(sample_a) & np.isfinite(sample_b)
        diff = sample_a[valid] - sample_b[valid]
        base = sample_a[valid]
        rms = float(np.sqrt(np.mean(diff * diff))) if diff.size else float("nan")
        denom = float(np.sqrt(np.mean(base * base))) if base.size else float("nan")
        return {
            "rms": rms,
            "relative_rms": rms / max(denom, 1e-30),
            "max_abs": float(np.max(np.abs(diff))) if diff.size else float("nan"),
            "sample_count": int(np.count_nonzero(valid)),
        }

    out: dict[str, Any] = {}
    for name in ("omega", "psi"):
        values = _sample_nextgen_field(nextgen, name, points)
        rotated_values = _sample_nextgen_field(nextgen, name, rotated)
        out[f"nextgen_{name}"] = metrics(values, rotated_values)
    if baseline.get("status") == "completed":
        for name, space_name in (("omega", "Vomega"), ("psi", "Vpsi")):
            values = _sample_function(baseline[name], baseline["spaces"][space_name], points)
            rotated_values = _sample_function(baseline[name], baseline["spaces"][space_name], rotated)
            out[f"baseline_{name}"] = metrics(values, rotated_values)
    return out


def make_plots(nextgen: dict[str, Any], baseline: dict[str, Any], outdir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    grid = nextgen["grid"]
    result = nextgen["result"]

    path = outdir / "nextgen_vorticity_heatmap.png"
    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    if isinstance(grid, PolarGrid):
        x = np.linspace(-grid.radius, grid.radius, 241)
        y = np.linspace(-grid.radius, grid.radius, 241)
        X, Y = np.meshgrid(x, y, indexing="ij")
        omega_plot, psi_plot = nextgen["problem"].sample_cartesian(result, X, Y)
        extent = (-grid.radius, grid.radius, -grid.radius, grid.radius)
        image = ax.imshow(omega_plot.T, origin="lower", extent=extent, cmap="RdBu_r")
    else:
        X, Y = grid.X, grid.Y
        psi_plot = result.psi
        extent = (-grid.half_width, grid.half_width, -grid.half_width, grid.half_width)
        image = ax.imshow(result.omega.T, origin="lower", extent=extent, cmap="RdBu_r")
    ax.set_aspect("equal")
    ax.set_title("nextgen Omega")
    fig.colorbar(image, ax=ax)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(str(path.relative_to(REPO_ROOT)))

    path = outdir / "nextgen_streamlines.png"
    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    levels = 42
    ax.contour(X, Y, psi_plot, levels=levels, colors="black", linewidths=0.55)
    ax.set_aspect("equal")
    ax.set_title("nextgen streamfunction contours")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(str(path.relative_to(REPO_ROOT)))

    if baseline.get("status") == "completed":
        coords = _function_xy(baseline["spaces"]["Vomega"])
        omega_vals = _function_array(baseline["omega"])
        triang = mtri.Triangulation(coords[:, 0], coords[:, 1])

        path = outdir / "baseline_vorticity_heatmap.png"
        fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
        image = ax.tricontourf(triang, omega_vals, levels=64, cmap="RdBu_r")
        ax.set_aspect("equal")
        ax.set_title("baseline Omega")
        fig.colorbar(image, ax=ax)
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path.relative_to(REPO_ROOT)))

        coords_psi = _function_xy(baseline["spaces"]["Vpsi"])
        psi_vals = _function_array(baseline["psi"])
        triang_psi = mtri.Triangulation(coords_psi[:, 0], coords_psi[:, 1])

        path = outdir / "baseline_streamlines.png"
        fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
        ax.tricontour(triang_psi, psi_vals, levels=42, colors="black", linewidths=0.55)
        ax.set_aspect("equal")
        ax.set_title("baseline streamfunction contours")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path.relative_to(REPO_ROOT)))

    return paths


def sanitize(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        blocked = {"grid", "problem", "result", "params", "spaces", "omega", "psi"}
        return {str(k): sanitize(v) for k, v in value.items() if k not in blocked}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return value


def append_registry(summary: dict[str, Any]) -> None:
    registry = REPO_ROOT / "experiments" / "registry.csv"
    next_diag = summary["nextgen"]["diagnostics"]
    baseline = summary.get("baseline", {})
    comparison = summary.get("comparison", {})
    common = summary.get("common_sampled_residuals", {})
    symmetry = summary.get("threefold_symmetry", {})
    next_common = common.get("nextgen", {})
    base_common = common.get("baseline", {})
    next_sym = symmetry.get("nextgen_omega", {})
    base_sym = symmetry.get("baseline_omega", {})
    with registry.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "2026-04-26",
                "m3-profile-c3-60-c4-0-polar-upwind-visual-check",
                "nextgen_vs_baseline",
                "experiments/run_m3_c3_profile_visual_check.py",
                "c3_60_c4_0_mu0.8_R150_Rdom10",
                "nextgen_polar_c3_sector_disk_Rdom10_vs_baseline_disk_Rdom10",
                "nextgen_nr54_ntheta54_sector3_dr0.185; baseline_gmsh_h0.12_0.18",
                "profile_continuation_newton_only_polar_fd",
                "nextgen_sigma_grid=0:0.1:1; baseline_sigma_grid=0,0.25,0.5,0.75,1; c3 sector; no eigensolver",
                f"nextgen={summary['nextgen'].get('runtime_sec'):.3f}; baseline={baseline.get('runtime_sec', 'n/a')}",
                (
                    f"nextgen_rel_res={next_diag.get('relative_residual'):.3e}; "
                    f"nextgen_abs_res={next_diag.get('residual_norm'):.3e}; "
                    f"baseline_raw_R_Omega={baseline.get('residuals', {}).get('norms', {}).get('Omega_l2', 'n/a')}; "
                    f"baseline_raw_R_Psi={baseline.get('residuals', {}).get('norms', {}).get('Psi_l2', 'n/a')}; "
                    f"nextgen_common_strong_rel={next_common.get('profile_strong_relative', 'n/a')}; "
                    f"baseline_common_strong_rel={base_common.get('profile_strong_relative', 'n/a')}; "
                    f"nextgen_omega_sym_rel={next_sym.get('relative_rms', 'n/a')}; "
                    f"baseline_omega_sym_rel={base_sym.get('relative_rms', 'n/a')}; "
                    f"omega_probe_rms={comparison.get('omega_probe_rms', 'n/a')}; "
                    f"psi_probe_rms_offset_removed={comparison.get('psi_probe_rms_offset_removed', 'n/a')}"
                ),
                "n/a",
                summary["status"],
                summary.get("notes", ""),
            ]
        )


def main() -> int:
    outdir = REPO_ROOT / "experiments" / "raw" / "nextgen" / "m3_c3_60_c4_0_polar_visual"
    summaries = REPO_ROOT / "experiments" / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)

    nextgen = run_nextgen()
    baseline = run_baseline_profile()
    comparison: dict[str, Any] = {}
    common_residuals: dict[str, Any] = {}
    symmetry: dict[str, Any] = {}
    if nextgen["status"] == "completed" and baseline["status"] == "completed":
        comparison = compare_on_probes(nextgen, baseline)
        common_residuals = common_sampled_residuals(nextgen, baseline)
        symmetry = threefold_symmetry_metrics(nextgen, baseline)
        status = "completed"
        notes = "Profile-only visual/residual comparison completed; psi comparison removes mean gauge offset."
    elif baseline["status"] == "blocked_import":
        status = "blocked_baseline_dependency"
        notes = f"Baseline profile import blocked: {baseline.get('error')}"
    elif nextgen["status"] != "completed":
        status = "nextgen_nonconverged"
        notes = f"Nextgen status: {nextgen['status']}"
    else:
        status = "baseline_failed"
        notes = f"Baseline status: {baseline['status']}; {baseline.get('error', '')}"

    plot_paths = make_plots(nextgen, baseline, outdir)
    summary = {
        "status": status,
        "notes": notes,
        "input": {
            "coefficients": {"3": 60.0, "4": 0.0},
            "mu": 0.8,
            "R": 150.0,
            "Rdom": 10.0,
            "nextgen_sigma_grid": NEXTGEN_SIGMA_GRID,
            "baseline_sigma_grid": BASELINE_SIGMA_GRID,
            "nextgen_grid": {"type": "polar_c3_sector", "radius": 10.0, "nr": 54, "ntheta": 54, "symmetry": 3},
        },
        "nextgen": nextgen,
        "baseline": baseline,
        "comparison": comparison,
        "common_sampled_residuals": common_residuals,
        "threefold_symmetry": symmetry,
        "plot_paths": plot_paths,
    }
    append_registry(summary)

    summary_path = summaries / "m3_c3_60_c4_0_polar_visual_check_2026-04-26.json"
    summary_path.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    print(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
