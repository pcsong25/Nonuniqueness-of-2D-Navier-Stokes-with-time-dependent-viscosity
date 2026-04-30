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
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from nextgen.boundary import FourierBoundaryData
from nextgen.config import PhysicalParams
from nextgen.diagnostics import sampled_strong_profile_residual
from nextgen.polar_profile import (
    PolarGrid,
    PolarProfileParams,
    PolarProfileProblem,
    PolarProfileResult,
    PolarProfileState,
)
from run_m3_c3_140_symmetry_probe import (
    COEFFS,
    PHYSICAL,
    RUN_DATE,
    common_residual,
    compatibility_metrics,
    compare_runs,
    prolong_state,
    save_state,
    symmetry_metrics,
)


TARGET_GRID = (160, 480)
SOURCE_GRID = (128, 384)
SOURCE_STATE_DIR = REPO_ROOT / "experiments" / "raw" / "nextgen" / "m3_c3_140_symmetry_probe" / "states"
SAMPLE_X = np.linspace(-7.0, 7.0, 193)
SAMPLE_Y = np.linspace(-7.0, 7.0, 193)
PROBE_RADII = np.linspace(0.5, 9.5, 91)
PROBE_THETA = np.linspace(-np.pi, np.pi, 720, endpoint=False)
BASIN_T_VALUES = tuple(float(v) for v in np.linspace(0.0, 1.0, 11))
TRACE_SIGMAS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)
BASELINE_MESH = (0.08, 0.12)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        blocked = {"problem", "result", "spaces", "omega", "psi", "rOmega", "rPsi"}
        return {str(k): sanitize(v) for k, v in value.items() if k not in blocked}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def make_problem(grid_spec: tuple[int, int]) -> PolarProfileProblem:
    nr, ntheta = grid_spec
    return PolarProfileProblem(
        PolarGrid(radius=PHYSICAL.Rdom, nr=nr, ntheta=ntheta),
        PHYSICAL,
        FourierBoundaryData(M=64, coeffs=COEFFS),
        advection_scheme="fv_upwind",
        upwind_blend=1.0,
    )


def result_from_state(problem: PolarProfileProblem, omega: np.ndarray, psi: np.ndarray, label: str) -> dict[str, Any]:
    omega_arr = np.asarray(omega, dtype=float).reshape(problem.grid.shape)
    psi_arr = np.asarray(psi, dtype=float).reshape(problem.grid.shape)
    diagnostics = problem.residual_summary(omega_arr, psi_arr, 1.0, 0.0)
    diagnostics.update(
        {
            "runtime_sec": 0.0,
            "newton_steps": 0,
            "final_sigma": 1.0,
            "attempted_sigma": 1.0,
            "grid_nr": problem.grid.nr,
            "grid_ntheta": problem.grid.ntheta,
            "grid_dr": problem.grid.dr,
            "grid_dtheta": problem.grid.dtheta,
            "poisson_lambda": 0.0,
            "reduced_poisson_newton": False,
        }
    )
    result = PolarProfileResult(
        omega=omega_arr,
        psi=psi_arr,
        grid=problem.grid,
        converged=True,
        iterations=(),
        runtime_sec=0.0,
        message="loaded_state",
        diagnostics=diagnostics,
    )
    run = {
        "label": label,
        "status": "completed",
        "grid": {
            "nr": problem.grid.nr,
            "ntheta": problem.grid.ntheta,
            "dr": problem.grid.dr,
            "dtheta": problem.grid.dtheta,
        },
        "problem": problem,
        "result": result,
        "runtime_sec": 0.0,
        "diagnostics": diagnostics,
        "message": "loaded_state",
        "common_residuals": common_residual(problem, result),
        "symmetry": symmetry_metrics(problem, result),
        "compatibility": compatibility_metrics(problem, result),
    }
    return run


def load_source_state(label: str) -> dict[str, Any]:
    path = SOURCE_STATE_DIR / f"{label}_state.npz"
    data = np.load(path)
    problem = make_problem((int(data["grid_nr"]), int(data["grid_ntheta"])))
    return result_from_state(problem, data["omega"], data["psi"], label)


def solve_grid(
    grid_spec: tuple[int, int],
    *,
    label: str,
    initial_state: PolarProfileState,
    sigma: float = 1.0,
    max_iter: int = 100,
) -> dict[str, Any]:
    problem = make_problem(grid_spec)
    result = problem.solve(
        PolarProfileParams(
            sigma_grid=(float(sigma),),
            tol_step=1e-8,
            tol_residual=1e-8,
            tol_residual_only=1e-8,
            max_iter=max_iter,
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
            reduced_gmres_maxiter=240,
            reduced_gmres_restart=120,
        ),
        initial_state=initial_state,
    )
    run = {
        "label": label,
        "status": "completed" if result.converged else "nonconverged",
        "grid": {
            "nr": problem.grid.nr,
            "ntheta": problem.grid.ntheta,
            "dr": problem.grid.dr,
            "dtheta": problem.grid.dtheta,
        },
        "problem": problem,
        "result": result,
        "runtime_sec": result.runtime_sec,
        "diagnostics": result.diagnostics,
        "message": result.message,
        "common_residuals": common_residual(problem, result) if result.converged else None,
    }
    if result.converged:
        run["symmetry"] = symmetry_metrics(problem, result)
        run["compatibility"] = compatibility_metrics(problem, result)
    return run


def same_grid_state(source: dict[str, Any], *, sigma: float = 0.0) -> PolarProfileState:
    return PolarProfileState(
        omega=np.asarray(source["result"].omega, dtype=float).copy(),
        psi=np.asarray(source["result"].psi, dtype=float).copy(),
        sigma=float(sigma),
        poisson_lambda=float(source["diagnostics"].get("poisson_lambda", 0.0)),
    )


def mixed_poisson_state(
    problem: PolarProfileProblem,
    a: dict[str, Any],
    b: dict[str, Any],
    t: float,
    *,
    sigma: float = 1.0,
) -> PolarProfileState:
    omega = (1.0 - float(t)) * np.asarray(a["result"].omega, dtype=float) + float(t) * np.asarray(
        b["result"].omega, dtype=float
    )
    psi, lam = problem.solve_poisson(omega, sigma=float(sigma))
    return PolarProfileState(
        omega=omega.reshape(problem.grid.shape),
        psi=psi.reshape(problem.grid.shape),
        poisson_lambda=lam,
        sigma=0.0,
    )


def scaled_state_for_sigma(source: dict[str, Any], target_sigma: float, source_sigma: float) -> PolarProfileState:
    problem = source["problem"]
    ratio = float(target_sigma) / max(float(source_sigma), 1e-30)
    omega = ratio * np.asarray(source["result"].omega, dtype=float)
    psi, lam = problem.solve_poisson(omega, sigma=float(target_sigma))
    return PolarProfileState(
        omega=omega.reshape(problem.grid.shape),
        psi=psi.reshape(problem.grid.shape),
        poisson_lambda=lam,
        sigma=0.0,
    )


def branch_distance_summary(run: dict[str, Any], symmetric: dict[str, Any], broken: dict[str, Any]) -> dict[str, Any]:
    if run["status"] != "completed":
        return {}
    return {
        "to_symmetric": compare_runs(run, symmetric),
        "to_broken": compare_runs(run, broken),
    }


def classify_by_distance(run: dict[str, Any], symmetric: dict[str, Any], broken: dict[str, Any]) -> str:
    if run["status"] != "completed":
        return "nonconverged"
    distances = branch_distance_summary(run, symmetric, broken)
    ds = distances["to_symmetric"]["omega_probe_rms"]
    db = distances["to_broken"]["omega_probe_rms"]
    non3 = run.get("symmetry", {}).get("omega_non_threefold_fraction", np.nan)
    if ds < 1e-8:
        return "symmetric"
    if db < 1e-8:
        return "broken_m1"
    if non3 > 1e-3 and db < ds:
        return "broken_like"
    if non3 < 1e-6 and ds < db:
        return "symmetric_like"
    return "other_root_candidate"


def transformed_arrays(values: np.ndarray) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for reflect in (False, True):
        arr = values[:, ::-1] if reflect else values
        for shift in (0, values.shape[1] // 3, 2 * values.shape[1] // 3):
            name = f"{'reflect_' if reflect else ''}shift_{shift}"
            out.append((name, np.roll(arr, shift, axis=1)))
    return out


def orbit_distance(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float | str]:
    grid = a["problem"].grid
    weights = grid.quadrature_weights().reshape(grid.shape)
    a_omega = np.asarray(a["result"].omega, dtype=float).reshape(grid.shape)
    a_psi = np.asarray(a["result"].psi, dtype=float).reshape(grid.shape)
    best: tuple[float, float, str] | None = None
    for name, b_omega in transformed_arrays(np.asarray(b["result"].omega, dtype=float).reshape(grid.shape)):
        omega_diff = a_omega - b_omega
        omega_rms = float(np.sqrt(np.sum(omega_diff * omega_diff * weights) / np.sum(weights)))
        b_psi_base = np.asarray(b["result"].psi, dtype=float).reshape(grid.shape)
        if name.startswith("reflect_"):
            b_psi_base = b_psi_base[:, ::-1]
        shift = int(name.rsplit("_", 1)[-1])
        b_psi = np.roll(b_psi_base, shift, axis=1)
        psi_diff = a_psi - b_psi
        psi_diff -= float(np.sum(psi_diff * weights) / np.sum(weights))
        psi_rms = float(np.sqrt(np.sum(psi_diff * psi_diff * weights) / np.sum(weights)))
        if best is None or omega_rms < best[0]:
            best = (omega_rms, psi_rms, name)
    assert best is not None
    return {"omega_weighted_rms": best[0], "psi_weighted_rms": best[1], "transform": best[2]}


def sample_baseline(function: Any, space: Any, points: np.ndarray) -> np.ndarray:
    coords = np.asarray(space.tabulate_dof_coordinates())[:, :2]
    values = np.asarray(function.x.array, dtype=float)
    return np.asarray(LinearNDInterpolator(coords, values, fill_value=np.nan)(points), dtype=float)


def common_residual_baseline(baseline: dict[str, Any]) -> dict[str, float]:
    x, y = np.meshgrid(SAMPLE_X, SAMPLE_Y, indexing="ij")
    points = np.column_stack((x.reshape(-1), y.reshape(-1)))
    omega = sample_baseline(baseline["omega"], baseline["spaces"]["Vomega"], points).reshape(x.shape)
    psi = sample_baseline(baseline["psi"], baseline["spaces"]["Vpsi"], points).reshape(x.shape)
    return sampled_strong_profile_residual(
        omega,
        psi,
        SAMPLE_X,
        SAMPLE_Y,
        mu=PHYSICAL.mu,
        eps=PHYSICAL.eps,
    )


def compare_nextgen_to_baseline(nextgen: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float | int]:
    rr, tt = np.meshgrid(PROBE_RADII, PROBE_THETA, indexing="ij")
    points = np.column_stack((rr.reshape(-1) * np.cos(tt.reshape(-1)), rr.reshape(-1) * np.sin(tt.reshape(-1))))
    omega_next, psi_next = nextgen["problem"].sample_cartesian(nextgen["result"], points[:, 0], points[:, 1])
    omega_base = sample_baseline(baseline["omega"], baseline["spaces"]["Vomega"], points)
    psi_base = sample_baseline(baseline["psi"], baseline["spaces"]["Vpsi"], points)
    valid = np.isfinite(omega_next) & np.isfinite(psi_next) & np.isfinite(omega_base) & np.isfinite(psi_base)
    omega_diff = omega_next[valid] - omega_base[valid]
    psi_diff = psi_next[valid] - psi_base[valid]
    psi_offset = float(np.mean(psi_diff)) if psi_diff.size else float("nan")
    if psi_diff.size:
        psi_diff -= psi_offset
    return {
        "valid_probe_count": int(np.count_nonzero(valid)),
        "omega_probe_rms": float(np.sqrt(np.mean(omega_diff**2))) if omega_diff.size else float("nan"),
        "omega_probe_linf": float(np.max(np.abs(omega_diff))) if omega_diff.size else float("nan"),
        "psi_probe_mean_offset": psi_offset,
        "psi_probe_rms_offset_removed": float(np.sqrt(np.mean(psi_diff**2))) if psi_diff.size else float("nan"),
        "psi_probe_linf_offset_removed": float(np.max(np.abs(psi_diff))) if psi_diff.size else float("nan"),
    }


def baseline_symmetry_metrics(baseline: dict[str, Any]) -> dict[str, float]:
    rr, tt = np.meshgrid(PROBE_RADII, PROBE_THETA, indexing="ij")
    points = np.column_stack((rr.reshape(-1) * np.cos(tt.reshape(-1)), rr.reshape(-1) * np.sin(tt.reshape(-1))))
    omega = sample_baseline(baseline["omega"], baseline["spaces"]["Vomega"], points).reshape(
        PROBE_RADII.size, PROBE_THETA.size
    )
    psi = sample_baseline(baseline["psi"], baseline["spaces"]["Vpsi"], points).reshape(
        PROBE_RADII.size, PROBE_THETA.size
    )
    weights = np.repeat(PROBE_RADII * (PROBE_RADII[1] - PROBE_RADII[0]) * (2.0 * np.pi / PROBE_THETA.size), PROBE_THETA.size).reshape(
        PROBE_RADII.size, PROBE_THETA.size
    )

    def metric(values: np.ndarray) -> tuple[float, float]:
        values = np.asarray(values, dtype=float)
        valid = np.isfinite(values)
        fill = np.where(valid, values, 0.0)
        shift = values.shape[1] // 3
        roll = np.roll(fill, shift, axis=1)
        proj = (fill + roll + np.roll(fill, 2 * shift, axis=1)) / 3.0
        norm = max(float(np.sqrt(np.sum(fill * fill * weights))), 1e-30)
        return (
            float(np.sqrt(np.sum((fill - roll) ** 2 * weights)) / norm),
            float(np.sqrt(np.sum((fill - proj) ** 2 * weights)) / norm),
        )

    omega_defect, omega_non3 = metric(omega)
    psi -= float(np.nanmean(psi))
    psi_defect, psi_non3 = metric(psi)
    return {
        "omega_threefold_defect": omega_defect,
        "omega_non_threefold_fraction": omega_non3,
        "psi_threefold_defect": psi_defect,
        "psi_non_threefold_fraction": psi_non3,
    }


def run_baseline_continuation(mesh_spec: tuple[float, float]) -> dict[str, Any]:
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
        h_min, h_max = mesh_spec
        params = SimulationParams(
            mu=PHYSICAL.mu,
            R=PHYSICAL.R,
            Rdom=PHYSICAL.Rdom,
            M=64,
            tol_newton=1e-9,
            max_newton=72,
            sigma_grid=tuple(float(s) for s in np.linspace(0.0, 1.0, 41)),
            h_min=float(h_min),
            h_max=float(h_max),
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
        run = {
            "label": f"baseline_continuation_h{h_min:g}_{h_max:g}",
            "status": "completed",
            "runtime_sec": runtime,
            "mesh": {"h_min": float(h_min), "h_max": float(h_max)},
            "mesh_cells": int(mesh.topology.index_map(mesh.topology.dim).size_local),
            "mesh_vertices": int(mesh.topology.index_map(0).size_local),
            "spaces": spaces,
            "omega": omega,
            "psi": psi,
            "residuals": residuals,
        }
        run["common_residuals"] = common_residual_baseline(run)
        run["symmetry"] = baseline_symmetry_metrics(run)
        return run
    except Exception as exc:
        return {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()}


def seed_baseline_function_from_nextgen(function: Any, space: Any, nextgen: dict[str, Any]) -> None:
    coords_raw = space.tabulate_dof_coordinates()
    coords = coords_raw[:, :2] if coords_raw.ndim > 1 else coords_raw.reshape(-1, 3)[:, :2]
    omega, _ = nextgen["problem"].sample_cartesian(nextgen["result"], coords[:, 0], coords[:, 1])
    arr = function.x.array
    arr[: omega.size] = np.asarray(omega, dtype=float)
    if hasattr(function.x, "scatter_forward"):
        function.x.scatter_forward()


def run_baseline_seeded_fixed(nextgen_seed: dict[str, Any], mesh_spec: tuple[float, float]) -> dict[str, Any]:
    baseline_dir = SRC / "baseline"
    sys.path.insert(0, str(baseline_dir))
    try:
        from petsc4py import PETSc
        from dolfinx import fem
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
        h_min, h_max = mesh_spec
        params = SimulationParams(
            mu=PHYSICAL.mu,
            R=PHYSICAL.R,
            Rdom=PHYSICAL.Rdom,
            M=64,
            tol_newton=1e-9,
            max_newton=40,
            sigma_grid=(1.0,),
            h_min=float(h_min),
            h_max=float(h_max),
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
        continuation = ContinuationSolver(
            assembler,
            poisson,
            params,
            facet_tags=facet_tags,
            outer_tag=1,
            fourier_bc=fourier_bc,
        )
        omega = fem.Function(spaces["Vomega"])
        seed_baseline_function_from_nextgen(omega, spaces["Vomega"], nextgen_seed)
        continuation._apply_omega_dirichlet_on_boundary(omega, spaces["Vomega"], 1.0)
        psi = poisson.solve_state(omega, sigma=1.0)

        converged = False
        last_rel = float("nan")
        newton_iterations = 0
        for it in range(params.max_newton):
            A, b = assembler.assemble(omega, psi, mu=params.mu)
            ksp = PETSc.KSP().create(mesh.comm)
            ksp.setOperators(A)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.setFromOptions()
            x = b.duplicate()
            x.set(0.0)
            ksp.solve(b, x)
            (x1, is1), (x2, is2) = continuation._split_block_vec(x, spaces["Vomega"], spaces["Vpsi"])

            try:
                omega.vector.axpy(1.0, x1)
                continuation._ghost_forward(omega)
            except AttributeError:
                d1 = continuation._vec_local_array(x1)
                arr = continuation._vec_data(omega)
                arr[: d1.size] += d1
                continuation._ghost_forward(omega)

            try:
                psi.vector.axpy(1.0, x2)
                continuation._ghost_forward(psi)
            except AttributeError:
                d2 = continuation._vec_local_array(x2)
                arrp = continuation._vec_data(psi)
                arrp[: d2.size] += d2
                continuation._ghost_forward(psi)

            d1norm = np.linalg.norm(continuation._vec_local_array(x1))
            onorm = np.linalg.norm(continuation._vec_data(omega)) + 1e-16
            last_rel = float(d1norm / onorm)
            newton_iterations = it + 1
            x.restoreSubVector(is1, x1)
            x.restoreSubVector(is2, x2)
            if last_rel < params.tol_newton:
                converged = True
                break

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
        run = {
            "label": f"baseline_seeded_from_{nextgen_seed['label']}_h{h_min:g}_{h_max:g}",
            "status": "completed" if converged else "nonconverged",
            "runtime_sec": runtime,
            "mesh": {"h_min": float(h_min), "h_max": float(h_max)},
            "mesh_cells": int(mesh.topology.index_map(mesh.topology.dim).size_local),
            "mesh_vertices": int(mesh.topology.index_map(0).size_local),
            "newton_iterations": newton_iterations,
            "last_relative_update": last_rel,
            "spaces": spaces,
            "omega": omega,
            "psi": psi,
            "residuals": residuals,
        }
        run["common_residuals"] = common_residual_baseline(run)
        run["symmetry"] = baseline_symmetry_metrics(run)
        return run
    except Exception as exc:
        return {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()}


def make_plots(
    refined: list[dict[str, Any]],
    basin: list[dict[str, Any]],
    traces: dict[str, list[dict[str, Any]]],
    outdir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    x = np.linspace(-PHYSICAL.Rdom, PHYSICAL.Rdom, 321)
    y = np.linspace(-PHYSICAL.Rdom, PHYSICAL.Rdom, 321)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    extent = (-PHYSICAL.Rdom, PHYSICAL.Rdom, -PHYSICAL.Rdom, PHYSICAL.Rdom)

    for run in refined:
        if run["status"] != "completed":
            continue
        omega, psi = run["problem"].sample_cartesian(run["result"], points[:, 0], points[:, 1])
        omega = omega.reshape(xx.shape)
        psi = psi.reshape(xx.shape)
        for field_name, values, cmap in (("omega", omega, "RdBu_r"),):
            path = outdir / f"{run['label']}_{field_name}.png"
            fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
            image = ax.imshow(values.T, origin="lower", extent=extent, cmap=cmap)
            ax.set_aspect("equal")
            ax.set_title(f"c3_140 {run['label']} {field_name}")
            fig.colorbar(image, ax=ax)
            fig.savefig(path, dpi=135)
            plt.close(fig)
            created.append(str(path.relative_to(REPO_ROOT)))

        path = outdir / f"{run['label']}_psi_contours.png"
        fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
        ax.contour(xx, yy, psi, levels=50, colors="black", linewidths=0.5)
        ax.set_aspect("equal")
        ax.set_title(f"c3_140 {run['label']} psi")
        fig.savefig(path, dpi=135)
        plt.close(fig)
        created.append(str(path.relative_to(REPO_ROOT)))

    if len(refined) >= 2 and refined[0]["status"] == "completed" and refined[1]["status"] == "completed":
        omega_a, _ = refined[0]["problem"].sample_cartesian(refined[0]["result"], points[:, 0], points[:, 1])
        omega_b, _ = refined[1]["problem"].sample_cartesian(refined[1]["result"], points[:, 0], points[:, 1])
        path = outdir / "broken_m1_minus_symmetric_omega.png"
        fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
        image = ax.imshow((omega_b - omega_a).reshape(xx.shape).T, origin="lower", extent=extent, cmap="RdBu_r")
        ax.set_aspect("equal")
        ax.set_title("c3_140 broken m1 minus symmetric")
        fig.colorbar(image, ax=ax)
        fig.savefig(path, dpi=135)
        plt.close(fig)
        created.append(str(path.relative_to(REPO_ROOT)))

    path = outdir / "basin_sweep.png"
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    t_values = [run["t"] for run in basin if run["status"] == "completed"]
    non3 = [run.get("symmetry", {}).get("omega_non_threefold_fraction", np.nan) for run in basin if run["status"] == "completed"]
    dist_sym = [
        run.get("branch_distances", {}).get("to_symmetric", {}).get("omega_probe_rms", np.nan)
        for run in basin
        if run["status"] == "completed"
    ]
    dist_broken = [
        run.get("branch_distances", {}).get("to_broken", {}).get("omega_probe_rms", np.nan)
        for run in basin
        if run["status"] == "completed"
    ]
    ax.plot(t_values, non3, "o-", label="Omega non-3 fraction")
    ax.plot(t_values, dist_sym, "s-", label="distance to symmetric")
    ax.plot(t_values, dist_broken, "^-", label="distance to broken")
    ax.set_xlabel("initial interpolation t")
    ax.set_yscale("log")
    ax.set_title("c3_140 Newton basin sweep")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    created.append(str(path.relative_to(REPO_ROOT)))

    path = outdir / "amplitude_trace_non3.png"
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    for branch, runs in traces.items():
        sigmas = [run.get("sigma") for run in runs if run["status"] == "completed"]
        branch_non3 = [
            run.get("symmetry", {}).get("omega_non_threefold_fraction", np.nan)
            for run in runs
            if run["status"] == "completed"
        ]
        ax.plot(sigmas, branch_non3, "o-", label=branch)
    ax.set_xlabel("sigma")
    ax.set_ylabel("Omega non-3 fraction")
    ax.set_yscale("log")
    ax.set_title("c3_140 seeded amplitude trace")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    created.append(str(path.relative_to(REPO_ROOT)))
    return created


def append_registry(summary: dict[str, Any]) -> None:
    registry = REPO_ROOT / "experiments" / "registry.csv"
    with registry.open("a", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        for group_name, runs in (
            ("refinement", summary["refined_runs"]),
            ("basin", summary["basin_runs"]),
            ("trace_symmetric", summary["trace_runs"]["symmetric"]),
            ("trace_broken_m1", summary["trace_runs"]["broken_m1"]),
        ):
            for run in runs:
                sym = run.get("symmetry", {})
                common = run.get("common_residuals", {}) or {}
                writer.writerow(
                    [
                        RUN_DATE,
                        f"m3-c3-140-nonuniqueness-{group_name}-{run['label']}",
                        "nextgen",
                        "experiments/run_m3_c3_140_nonuniqueness_validation.py",
                        "{c3=140}",
                        "nextgen_polar_full_disk_Rdom10",
                        f"grid={run.get('grid', {}).get('nr', 'n/a')}/{run.get('grid', {}).get('ntheta', 'n/a')}",
                        "profile_fv_upwind_branch_refinement_basin_amplitude_trace",
                        "M3 c3=140 nonuniqueness validation; no M4; no eigensolver",
                        run.get("runtime_sec", "n/a"),
                        (
                            f"status={run['status']}; class={run.get('branch_class', 'n/a')}; "
                            f"sigma={run.get('sigma', run.get('diagnostics', {}).get('final_sigma', 'n/a'))}; "
                            f"rel={run.get('diagnostics', {}).get('relative_residual', 'n/a')}; "
                            f"common={common.get('profile_strong_relative', 'n/a')}; "
                            f"omega_non3={sym.get('omega_non_threefold_fraction', 'n/a')}"
                        ),
                        "n/a",
                        run["status"],
                        run.get("message", ""),
                    ]
                )
        for run in summary.get("baseline_runs", []):
            sym = run.get("symmetry", {})
            common = run.get("common_residuals", {}) or {}
            writer.writerow(
                [
                    RUN_DATE,
                    f"m3-c3-140-nonuniqueness-baseline-{run['label']}",
                    "baseline",
                    "experiments/run_m3_c3_140_nonuniqueness_validation.py",
                    "{c3=140}",
                    "baseline_fem_disk_Rdom10",
                    f"h={run.get('mesh', {}).get('h_min', 'n/a')}/{run.get('mesh', {}).get('h_max', 'n/a')}",
                    "baseline_fem_continuation_or_seeded_fixed_newton",
                    "M3 c3=140 FEM cross-check; no M4; no eigensolver",
                    run.get("runtime_sec", "n/a"),
                    (
                        f"status={run['status']}; common={common.get('profile_strong_relative', 'n/a')}; "
                        f"omega_non3={sym.get('omega_non_threefold_fraction', 'n/a')}; "
                        f"last_rel={run.get('last_relative_update', 'n/a')}"
                    ),
                    "n/a",
                    run["status"],
                    run.get("error", run.get("label", "")),
                ]
            )


def main() -> int:
    raw_root = REPO_ROOT / "experiments" / "raw" / "nextgen" / "m3_c3_140_nonuniqueness_validation"
    summary_dir = REPO_ROOT / "experiments" / "summaries"
    state_dir = raw_root / "states"
    plot_dir = raw_root / "plots"
    raw_root.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    print("[load] c3=140 source states", flush=True)
    source_symmetric = load_source_state("nested_128_384")
    source_broken_m1 = load_source_state("perturbed_m1_core_amp1p5_128_384")
    source_broken_m2 = load_source_state("perturbed_m2_core_amp1p5_128_384")

    refined_runs: list[dict[str, Any]] = []
    for label, source in (
        ("symmetric_160_480", source_symmetric),
        ("broken_m1_160_480", source_broken_m1),
        ("broken_m2_160_480", source_broken_m2),
    ):
        target_grid = PolarGrid(radius=PHYSICAL.Rdom, nr=TARGET_GRID[0], ntheta=TARGET_GRID[1])
        initial = prolong_state(source, target_grid, sigma=0.0)
        print(f"[refine start] {label}", flush=True)
        run = solve_grid(TARGET_GRID, label=label, initial_state=initial, sigma=1.0, max_iter=120)
        if run["status"] == "completed":
            run["branch_distances"] = {}
            run["state_path"] = save_state(run, state_dir)
        refined_runs.append(run)
        print(
            f"[refine done] {label} status={run['status']} runtime={run.get('runtime_sec'):.3f} "
            f"rel={run.get('diagnostics', {}).get('relative_residual')} "
            f"omega_non3={run.get('symmetry', {}).get('omega_non_threefold_fraction')}",
            flush=True,
        )

    symmetric = refined_runs[0]
    broken_m1 = refined_runs[1]
    broken_m2 = refined_runs[2]
    if symmetric["status"] == "completed" and broken_m1["status"] == "completed":
        for run in refined_runs:
            if run["status"] == "completed":
                run["branch_distances"] = branch_distance_summary(run, symmetric, broken_m1)
                run["branch_class"] = classify_by_distance(run, symmetric, broken_m1)
        orbit = {
            "broken_m1_vs_broken_m2": orbit_distance(broken_m1, broken_m2)
            if broken_m2["status"] == "completed"
            else {},
            "symmetric_vs_broken_m1": orbit_distance(symmetric, broken_m1),
        }
    else:
        orbit = {}

    basin_runs: list[dict[str, Any]] = []
    if symmetric["status"] == "completed" and broken_m1["status"] == "completed":
        for t in BASIN_T_VALUES:
            initial = mixed_poisson_state(symmetric["problem"], symmetric, broken_m1, t, sigma=1.0)
            label = f"basin_t{str(round(t, 2)).replace('.', 'p')}_160_480"
            print(f"[basin start] t={t:.2f}", flush=True)
            run = solve_grid(TARGET_GRID, label=label, initial_state=initial, sigma=1.0, max_iter=120)
            run["t"] = float(t)
            if run["status"] == "completed":
                run["branch_distances"] = branch_distance_summary(run, symmetric, broken_m1)
                run["branch_class"] = classify_by_distance(run, symmetric, broken_m1)
            basin_runs.append(run)
            print(
                f"[basin done] t={t:.2f} status={run['status']} class={run.get('branch_class')} "
                f"runtime={run.get('runtime_sec'):.3f} "
                f"omega_non3={run.get('symmetry', {}).get('omega_non_threefold_fraction')}",
                flush=True,
            )

    trace_runs: dict[str, list[dict[str, Any]]] = {"symmetric": [], "broken_m1": []}
    for branch_name, start in (("symmetric", symmetric), ("broken_m1", broken_m1)):
        if start["status"] != "completed":
            continue
        previous = start
        previous_sigma = 1.0
        start["sigma"] = 1.0
        start["branch_class"] = branch_name
        trace_runs[branch_name].append(start)
        for sigma in TRACE_SIGMAS[1:]:
            initial = scaled_state_for_sigma(previous, sigma, previous_sigma)
            label = f"trace_{branch_name}_sigma{str(sigma).replace('.', 'p')}_160_480"
            print(f"[trace start] branch={branch_name} sigma={sigma:.2f}", flush=True)
            run = solve_grid(TARGET_GRID, label=label, initial_state=initial, sigma=sigma, max_iter=120)
            run["sigma"] = float(sigma)
            if run["status"] == "completed":
                run["branch_distances"] = branch_distance_summary(run, symmetric, broken_m1)
                run["branch_class"] = classify_by_distance(run, symmetric, broken_m1)
                previous = run
                previous_sigma = sigma
            trace_runs[branch_name].append(run)
            print(
                f"[trace done] branch={branch_name} sigma={sigma:.2f} status={run['status']} "
                f"class={run.get('branch_class')} runtime={run.get('runtime_sec'):.3f} "
                f"omega_non3={run.get('symmetry', {}).get('omega_non_threefold_fraction')}",
                flush=True,
            )
            if run["status"] != "completed":
                break

    baseline_runs: list[dict[str, Any]] = []
    print(f"[baseline continuation start] h={BASELINE_MESH[0]}/{BASELINE_MESH[1]}", flush=True)
    baseline_cont = run_baseline_continuation(BASELINE_MESH)
    baseline_runs.append(baseline_cont)
    print(
        f"[baseline continuation done] status={baseline_cont['status']} "
        f"runtime={baseline_cont.get('runtime_sec', 'n/a')}",
        flush=True,
    )
    if broken_m1["status"] == "completed":
        print("[baseline seeded broken start]", flush=True)
        baseline_seeded = run_baseline_seeded_fixed(broken_m1, BASELINE_MESH)
        baseline_runs.append(baseline_seeded)
        print(
            f"[baseline seeded broken done] status={baseline_seeded['status']} "
            f"runtime={baseline_seeded.get('runtime_sec', 'n/a')} "
            f"omega_non3={baseline_seeded.get('symmetry', {}).get('omega_non_threefold_fraction')}",
            flush=True,
        )

    baseline_comparisons: dict[str, Any] = {}
    completed_baselines = [run for run in baseline_runs if run["status"] == "completed"]
    for baseline in completed_baselines:
        for run in (symmetric, broken_m1, broken_m2):
            if run["status"] == "completed":
                baseline_comparisons[f"{run['label']}_vs_{baseline['label']}"] = compare_nextgen_to_baseline(
                    run,
                    baseline,
                )

    plot_paths = make_plots(refined_runs, basin_runs, trace_runs, plot_dir)
    summary = {
        "status": "completed"
        if symmetric["status"] == "completed" and broken_m1["status"] == "completed"
        else "partial",
        "case": {
            "name": "c3_140",
            "coeffs": COEFFS,
            "description": "refinement, basin, amplitude, and FEM cross-checks for c3=140 nonuniqueness",
        },
        "input": {
            "physical": {"mu": PHYSICAL.mu, "R": PHYSICAL.R, "Rdom": PHYSICAL.Rdom, "eps": PHYSICAL.eps},
            "source_grid": {"nr": SOURCE_GRID[0], "ntheta": SOURCE_GRID[1]},
            "target_grid": {"nr": TARGET_GRID[0], "ntheta": TARGET_GRID[1]},
            "basin_t_values": BASIN_T_VALUES,
            "trace_sigmas": TRACE_SIGMAS,
            "baseline_mesh": {"h_min": BASELINE_MESH[0], "h_max": BASELINE_MESH[1]},
        },
        "validation_methods": [
            "branch refinement by prolongation from saved 128/384 states to 160/480 followed by reduced FV Newton",
            "fixed-sigma Newton basin map from convex combinations of symmetric and broken branch Omega fields with Poisson-projected psi",
            "seeded amplitude trace on both branches by stepping sigma downward and re-solving the same FV equations",
            "C3/reflection orbit distance checks between candidate branches",
            "independent sampled strong residuals on a common Cartesian grid",
            "Poisson compatibility diagnostics: Omega integral, Neumann flux, Poisson multiplier, physical Poisson residual",
            "FEM baseline zero-continuation cross-check and seeded fixed-sigma FEM Newton attempt from the broken nextgen branch",
        ],
        "refined_runs": refined_runs,
        "orbit_checks": orbit,
        "basin_runs": basin_runs,
        "trace_runs": trace_runs,
        "baseline_runs": baseline_runs,
        "baseline_comparisons": baseline_comparisons,
        "plot_paths": plot_paths,
    }
    append_registry(sanitize(summary))
    path = summary_dir / f"m3_c3_140_nonuniqueness_validation_{RUN_DATE}.json"
    path.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    print(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
