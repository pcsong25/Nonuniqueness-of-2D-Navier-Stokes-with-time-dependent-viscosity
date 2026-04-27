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


PHYSICAL = PhysicalParams(mu=0.8, R=150.0, Rdom=10.0)
SAMPLE_X = np.linspace(-7.0, 7.0, 97)
SAMPLE_Y = np.linspace(-7.0, 7.0, 97)
PROBE_RADII = np.linspace(0.5, 8.0, 48)
PROBE_THETA = np.linspace(-np.pi, np.pi, 144, endpoint=False)

CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "active_c3_60_c4_0",
        "description": "active full-amplitude c3-only profile",
        "coeffs": {3: 60.0 + 0.0j, 4: 0.0 + 0.0j},
        "nextgen_grids": ((48, 144), (54, 162)),
        "baseline_meshes": ((0.20, 0.30), (0.12, 0.18)),
        "sigma_grid": tuple([0.0] + [i / 10.0 for i in range(1, 11)]),
        "params": {"tol_step": 1e-8, "tol_residual": 1e-8, "max_iter": 40, "max_backtracks": 18},
    },
    {
        "name": "mixed_2_5_reduced",
        "description": "reduced non-symmetric mixed complex modes",
        "coeffs": {2: 3.0 + 1.0j, 5: -2.0 + 0.5j},
        "nextgen_grids": ((30, 90), (32, 96), (36, 108)),
        "baseline_meshes": ((0.30, 0.45), (0.20, 0.30)),
        "sigma_grid": tuple([0.0] + [i / 10.0 for i in range(1, 11)]),
        "params": {"tol_step": 1e-8, "tol_residual": 1e-8, "max_iter": 50, "max_backtracks": 18},
    },
)


def run_nextgen(case: dict[str, Any], nr: int, ntheta: int) -> dict[str, Any]:
    grid = PolarGrid(radius=PHYSICAL.Rdom, nr=nr, ntheta=ntheta)
    boundary = FourierBoundaryData(M=48, coeffs=case["coeffs"])
    problem = PolarProfileProblem(grid, PHYSICAL, boundary)
    result = problem.solve(
        PolarProfileParams(
            sigma_grid=tuple(case["sigma_grid"]),
            **case["params"],
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


def run_baseline(case: dict[str, Any], h_min: float, h_max: float) -> dict[str, Any]:
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
        params = SimulationParams(
            mu=PHYSICAL.mu,
            R=PHYSICAL.R,
            Rdom=PHYSICAL.Rdom,
            M=48,
            tol_newton=1e-9,
            max_newton=48,
            sigma_grid=tuple(case["sigma_grid"]),
            h_min=float(h_min),
            h_max=float(h_max),
        )
        t0 = time.perf_counter()
        mesh, facet_tags = build_disk(params.Rdom, params.h_min, params.h_max)
        spaces = create_spaces(mesh)
        fourier_bc = FourierBC(params.M, case["coeffs"])
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
            "mesh": {"h_min": h_min, "h_max": h_max},
            "runtime_sec": runtime,
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


def _sample_nextgen(nextgen: dict[str, Any], points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return nextgen["problem"].sample_cartesian(nextgen["result"], points[:, 0], points[:, 1])


def _probe_points() -> np.ndarray:
    rr, tt = np.meshgrid(PROBE_RADII, PROBE_THETA, indexing="ij")
    return np.column_stack((rr.reshape(-1) * np.cos(tt.reshape(-1)), rr.reshape(-1) * np.sin(tt.reshape(-1))))


def compare_fields(nextgen: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float | int]:
    points = _probe_points()
    omega_next, psi_next = _sample_nextgen(nextgen, points)
    omega_base = _sample_function(baseline["omega"], baseline["spaces"]["Vomega"], points)
    psi_base = _sample_function(baseline["psi"], baseline["spaces"]["Vpsi"], points)
    valid = np.isfinite(omega_next) & np.isfinite(psi_next) & np.isfinite(omega_base) & np.isfinite(psi_base)

    def metric(next_values: np.ndarray, base_values: np.ndarray, *, remove_mean: bool) -> tuple[float, float, float]:
        diff = np.asarray(next_values[valid] - base_values[valid], dtype=float)
        offset = float(np.mean(diff)) if diff.size else float("nan")
        if remove_mean and diff.size:
            diff = diff - offset
        return (
            float(np.sqrt(np.mean(diff * diff))) if diff.size else float("nan"),
            float(np.max(np.abs(diff))) if diff.size else float("nan"),
            offset,
        )

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


def common_residual_for_fields(omega: np.ndarray, psi: np.ndarray) -> dict[str, float]:
    return sampled_strong_profile_residual(
        omega,
        psi,
        SAMPLE_X,
        SAMPLE_Y,
        mu=PHYSICAL.mu,
        eps=PHYSICAL.eps,
    )


def nextgen_common_residual(nextgen: dict[str, Any]) -> dict[str, float]:
    X, Y = np.meshgrid(SAMPLE_X, SAMPLE_Y, indexing="ij")
    points = np.column_stack((X.reshape(-1), Y.reshape(-1)))
    omega, psi = _sample_nextgen(nextgen, points)
    return common_residual_for_fields(omega.reshape(X.shape), psi.reshape(X.shape))


def baseline_common_residual(baseline: dict[str, Any]) -> dict[str, float]:
    X, Y = np.meshgrid(SAMPLE_X, SAMPLE_Y, indexing="ij")
    points = np.column_stack((X.reshape(-1), Y.reshape(-1)))
    omega = _sample_function(baseline["omega"], baseline["spaces"]["Vomega"], points).reshape(X.shape)
    psi = _sample_function(baseline["psi"], baseline["spaces"]["Vpsi"], points).reshape(X.shape)
    return common_residual_for_fields(omega, psi)


def baseline_pairwise_difference(coarse: dict[str, Any], fine: dict[str, Any]) -> dict[str, float | int]:
    points = _probe_points()
    omega_coarse = _sample_function(coarse["omega"], coarse["spaces"]["Vomega"], points)
    psi_coarse = _sample_function(coarse["psi"], coarse["spaces"]["Vpsi"], points)
    omega_fine = _sample_function(fine["omega"], fine["spaces"]["Vomega"], points)
    psi_fine = _sample_function(fine["psi"], fine["spaces"]["Vpsi"], points)
    valid = np.isfinite(omega_coarse) & np.isfinite(psi_coarse) & np.isfinite(omega_fine) & np.isfinite(psi_fine)

    def metric(a: np.ndarray, b: np.ndarray, *, remove_mean: bool) -> tuple[float, float]:
        diff = np.asarray(a[valid] - b[valid], dtype=float)
        if remove_mean and diff.size:
            diff = diff - float(np.mean(diff))
        return (
            float(np.sqrt(np.mean(diff * diff))) if diff.size else float("nan"),
            float(np.max(np.abs(diff))) if diff.size else float("nan"),
        )

    omega_rms, omega_linf = metric(omega_coarse, omega_fine, remove_mean=False)
    psi_rms, psi_linf = metric(psi_coarse, psi_fine, remove_mean=True)
    return {
        "valid_probe_count": int(np.count_nonzero(valid)),
        "omega_rms": omega_rms,
        "omega_linf": omega_linf,
        "psi_rms_offset_removed": psi_rms,
        "psi_linf_offset_removed": psi_linf,
    }


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
            baseline_fine = case["baseline_runs"][-1]
            for run in case["nextgen_runs"]:
                diag = run["diagnostics"]
                comparison = run.get("comparison_to_finest_baseline", {})
                common = run.get("common_sampled_residuals", {})
                writer.writerow(
                    [
                        "2026-04-27",
                        f"m3-polar-refinement-{case['name']}-nr{run['grid']['nr']}-ntheta{run['grid']['ntheta']}",
                        "nextgen_vs_baseline",
                        "experiments/run_m3_polar_refinement_study.py",
                        case["description"],
                        "nextgen_polar_full_disk_vs_finest_baseline_disk",
                        f"nextgen_nr{run['grid']['nr']}_ntheta{run['grid']['ntheta']}; baseline_h{baseline_fine['mesh']['h_min']}_{baseline_fine['mesh']['h_max']}",
                        "profile_continuation_newton_only_polar_fd",
                        "full-disk polar refinement; no eigensolver",
                        f"nextgen={run.get('runtime_sec'):.3f}; baseline_fine={baseline_fine.get('runtime_sec', 'n/a')}",
                        (
                            f"status={run['status']}; nextgen_rel_res={diag.get('relative_residual'):.3e}; "
                            f"nextgen_common_strong_rel={common.get('profile_strong_relative', 'n/a')}; "
                            f"baseline_fine_common_strong_rel={baseline_fine.get('common_sampled_residuals', {}).get('profile_strong_relative', 'n/a')}; "
                            f"omega_probe_rms={comparison.get('omega_probe_rms', 'n/a')}; "
                            f"psi_probe_rms_offset_removed={comparison.get('psi_probe_rms_offset_removed', 'n/a')}"
                        ),
                        "n/a",
                        run["status"],
                        case.get("notes", ""),
                    ]
                )


def main() -> int:
    outdir = REPO_ROOT / "experiments" / "summaries"
    outdir.mkdir(parents=True, exist_ok=True)
    cases_out = []
    overall_status = "completed"

    for case in CASES:
        baseline_runs = []
        for h_min, h_max in case["baseline_meshes"]:
            baseline = run_baseline(case, h_min, h_max)
            if baseline["status"] == "completed":
                baseline["common_sampled_residuals"] = baseline_common_residual(baseline)
            else:
                overall_status = "partial"
            baseline_runs.append(baseline)

        if len(baseline_runs) >= 2 and all(run["status"] == "completed" for run in baseline_runs[-2:]):
            baseline_mesh_difference = baseline_pairwise_difference(baseline_runs[-2], baseline_runs[-1])
        else:
            baseline_mesh_difference = {}

        finest_baseline = baseline_runs[-1]
        nextgen_runs = []
        for nr, ntheta in case["nextgen_grids"]:
            nextgen = run_nextgen(case, nr, ntheta)
            if nextgen["status"] == "completed":
                nextgen["common_sampled_residuals"] = nextgen_common_residual(nextgen)
                if finest_baseline["status"] == "completed":
                    nextgen["comparison_to_finest_baseline"] = compare_fields(nextgen, finest_baseline)
            else:
                nextgen["common_sampled_residuals"] = {}
                nextgen["comparison_to_finest_baseline"] = {}
                overall_status = "partial"
            nextgen_runs.append(nextgen)

        cases_out.append(
            {
                "name": case["name"],
                "description": case["description"],
                "coeffs": case["coeffs"],
                "status": "completed" if all(run["status"] == "completed" for run in baseline_runs + nextgen_runs) else "partial",
                "notes": "Nonconverged nextgen grids are retained as refinement/stability data.",
                "baseline_runs": baseline_runs,
                "baseline_mesh_difference": baseline_mesh_difference,
                "nextgen_runs": nextgen_runs,
            }
        )

    summary = {
        "status": overall_status,
        "input": {
            "physical": {"mu": PHYSICAL.mu, "R": PHYSICAL.R, "Rdom": PHYSICAL.Rdom, "eps": PHYSICAL.eps},
            "sample_grid": {"x_min": -7.0, "x_max": 7.0, "n": int(SAMPLE_X.size)},
            "probe_grid": {"r_min": float(PROBE_RADII[0]), "r_max": float(PROBE_RADII[-1]), "nr": int(PROBE_RADII.size), "ntheta": int(PROBE_THETA.size)},
        },
        "norm_definitions": {
            "native_relative_residual": "nextgen assembled polar residual weighted by polar cell areas, divided by max(||Omega||_L2,||Delta psi||_L2)",
            "common_strong_relative": "both codes sampled to the same Cartesian grid; sqrt(||R_Omega||_L2^2+||R_Psi||_L2^2)/max(||Omega||_L2,||Delta psi||_L2)",
            "probe_rms": "sqrt(mean((nextgen-baseline)^2)) over common polar probes r in [0.5,8]",
            "probe_linf": "max(abs(nextgen-baseline)) over valid common probes",
            "psi_probe_errors": "streamfunction probe differences after subtracting the mean difference to remove gauge offset",
            "baseline_mesh_difference": "coarser baseline minus finer baseline on the same common probes, with psi mean offset removed",
        },
        "cases": cases_out,
    }
    append_registry(summary)
    path = outdir / "m3_polar_full_disk_refinement_study_2026-04-27.json"
    path.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    print(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    # A partial status is a valid refinement/stability outcome: nonconverged
    # grids are recorded and interpreted rather than discarded.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
