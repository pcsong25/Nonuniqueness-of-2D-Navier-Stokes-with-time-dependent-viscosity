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


RUN_DATE = "2026-04-29"
PHYSICAL = PhysicalParams(mu=0.8, R=150.0, Rdom=10.0)
COEFFS = {4: 100.0 + 0.0j, 7: 40.0 + 0.0j}
SIGMA_GRID = tuple(float(s) for s in np.linspace(0.0, 1.0, 41))
NEXTGEN_GRIDS = ((64, 256), (72, 288))
BASELINE_MESHES = ((0.08, 0.12), (0.05, 0.08))
SAMPLE_X = np.linspace(-7.0, 7.0, 193)
SAMPLE_Y = np.linspace(-7.0, 7.0, 193)
PROBE_RADII = np.linspace(0.5, 8.0, 90)
PROBE_THETA = np.linspace(-np.pi, np.pi, 360, endpoint=False)


def run_nextgen(grid_spec: tuple[int, int]) -> dict[str, Any]:
    nr, ntheta = grid_spec
    grid = PolarGrid(radius=PHYSICAL.Rdom, nr=nr, ntheta=ntheta)
    problem = PolarProfileProblem(
        grid,
        PHYSICAL,
        FourierBoundaryData(M=64, coeffs=COEFFS),
        advection_scheme="fv_upwind",
    )
    result = problem.solve(
        PolarProfileParams(
            sigma_grid=SIGMA_GRID,
            tol_step=1e-8,
            tol_residual=1e-8,
            tol_residual_only=1e-8,
            max_iter=64,
            max_backtracks=28,
            adaptive_continuation=True,
            min_sigma_step=5e-4,
            sigma_step_growth=1.25,
            sigma_step_shrink=0.5,
            least_squares_fallback=False,
            pseudo_transient_fallback=True,
            pseudo_transient_shifts=(1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
        )
    )
    return {
        "status": "completed" if result.converged else "nonconverged",
        "grid": {"nr": nr, "ntheta": ntheta, "dr": grid.dr, "dtheta": grid.dtheta},
        "problem": problem,
        "result": result,
        "runtime_sec": result.runtime_sec,
        "diagnostics": result.diagnostics,
        "message": result.message,
    }


def run_baseline(mesh_spec: tuple[float, float]) -> dict[str, Any]:
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
            max_newton=80,
            sigma_grid=SIGMA_GRID,
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
        return {
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
    except Exception as exc:
        return {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()}


def probe_points() -> np.ndarray:
    rr, tt = np.meshgrid(PROBE_RADII, PROBE_THETA, indexing="ij")
    return np.column_stack((rr.reshape(-1) * np.cos(tt.reshape(-1)), rr.reshape(-1) * np.sin(tt.reshape(-1))))


def sample_baseline(function: Any, space: Any, points: np.ndarray) -> np.ndarray:
    coords = np.asarray(space.tabulate_dof_coordinates())[:, :2]
    values = np.asarray(function.x.array, dtype=float)
    return np.asarray(LinearNDInterpolator(coords, values, fill_value=np.nan)(points), dtype=float)


def sample_nextgen(run: dict[str, Any], points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return run["problem"].sample_cartesian(run["result"], points[:, 0], points[:, 1])


def common_residual_from_sampler(sampler: Any) -> dict[str, float]:
    x, y = np.meshgrid(SAMPLE_X, SAMPLE_Y, indexing="ij")
    points = np.column_stack((x.reshape(-1), y.reshape(-1)))
    omega, psi = sampler(points)
    return sampled_strong_profile_residual(
        omega.reshape(x.shape),
        psi.reshape(x.shape),
        SAMPLE_X,
        SAMPLE_Y,
        mu=PHYSICAL.mu,
        eps=PHYSICAL.eps,
    )


def compare_samples(
    a_omega: np.ndarray,
    a_psi: np.ndarray,
    b_omega: np.ndarray,
    b_psi: np.ndarray,
) -> dict[str, float | int]:
    valid = np.isfinite(a_omega) & np.isfinite(a_psi) & np.isfinite(b_omega) & np.isfinite(b_psi)
    omega_diff = np.asarray(a_omega[valid] - b_omega[valid], dtype=float)
    psi_diff = np.asarray(a_psi[valid] - b_psi[valid], dtype=float)
    psi_offset = float(np.mean(psi_diff)) if psi_diff.size else float("nan")
    if psi_diff.size:
        psi_diff = psi_diff - psi_offset
    return {
        "valid_probe_count": int(np.count_nonzero(valid)),
        "omega_probe_rms": float(np.sqrt(np.mean(omega_diff**2))) if omega_diff.size else float("nan"),
        "omega_probe_linf": float(np.max(np.abs(omega_diff))) if omega_diff.size else float("nan"),
        "psi_probe_mean_offset": psi_offset,
        "psi_probe_rms_offset_removed": float(np.sqrt(np.mean(psi_diff**2))) if psi_diff.size else float("nan"),
        "psi_probe_linf_offset_removed": float(np.max(np.abs(psi_diff))) if psi_diff.size else float("nan"),
    }


def compare_nextgen_to_baseline(nextgen: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float | int]:
    points = probe_points()
    omega_next, psi_next = sample_nextgen(nextgen, points)
    omega_base = sample_baseline(baseline["omega"], baseline["spaces"]["Vomega"], points)
    psi_base = sample_baseline(baseline["psi"], baseline["spaces"]["Vpsi"], points)
    return compare_samples(omega_next, psi_next, omega_base, psi_base)


def compare_nextgen_pair(coarse: dict[str, Any], fine: dict[str, Any]) -> dict[str, float | int]:
    points = probe_points()
    omega_coarse, psi_coarse = sample_nextgen(coarse, points)
    omega_fine, psi_fine = sample_nextgen(fine, points)
    return compare_samples(omega_coarse, psi_coarse, omega_fine, psi_fine)


def compare_baseline_pair(coarse: dict[str, Any], fine: dict[str, Any]) -> dict[str, float | int]:
    points = probe_points()
    omega_coarse = sample_baseline(coarse["omega"], coarse["spaces"]["Vomega"], points)
    psi_coarse = sample_baseline(coarse["psi"], coarse["spaces"]["Vpsi"], points)
    omega_fine = sample_baseline(fine["omega"], fine["spaces"]["Vomega"], points)
    psi_fine = sample_baseline(fine["psi"], fine["spaces"]["Vpsi"], points)
    return compare_samples(omega_coarse, psi_coarse, omega_fine, psi_fine)


def make_plots(
    nextgen_runs: list[dict[str, Any]],
    baseline_runs: list[dict[str, Any]],
    outdir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-PHYSICAL.Rdom, PHYSICAL.Rdom, 321)
    y = np.linspace(-PHYSICAL.Rdom, PHYSICAL.Rdom, 321)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))

    fields: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for run in nextgen_runs:
        if run["status"] != "completed":
            continue
        nr = run["grid"]["nr"]
        ntheta = run["grid"]["ntheta"]
        omega, psi = sample_nextgen(run, points)
        fields[f"nextgen_{nr}_{ntheta}"] = (omega.reshape(xx.shape), psi.reshape(xx.shape))
    for run in baseline_runs:
        if run["status"] != "completed":
            continue
        h_min = run["mesh"]["h_min"]
        h_max = run["mesh"]["h_max"]
        label = f"baseline_h{h_min:g}_{h_max:g}".replace(".", "p")
        fields[label] = (
            sample_baseline(run["omega"], run["spaces"]["Vomega"], points).reshape(xx.shape),
            sample_baseline(run["psi"], run["spaces"]["Vpsi"], points).reshape(xx.shape),
        )

    created: list[str] = []
    extent = (-PHYSICAL.Rdom, PHYSICAL.Rdom, -PHYSICAL.Rdom, PHYSICAL.Rdom)
    for source, (omega, psi) in fields.items():
        path = outdir / f"{source}_omega.png"
        fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
        image = ax.imshow(omega.T, origin="lower", extent=extent, cmap="RdBu_r")
        ax.set_aspect("equal")
        ax.set_title(f"c4_100_c7_40 {source} Omega")
        fig.colorbar(image, ax=ax)
        fig.savefig(path, dpi=135)
        plt.close(fig)
        created.append(str(path.relative_to(REPO_ROOT)))

        path = outdir / f"{source}_psi_contours.png"
        fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
        ax.contour(xx, yy, psi, levels=50, colors="black", linewidths=0.5)
        ax.set_aspect("equal")
        ax.set_title(f"c4_100_c7_40 {source} psi")
        fig.savefig(path, dpi=135)
        plt.close(fig)
        created.append(str(path.relative_to(REPO_ROOT)))

    return created


def sanitize(value: Any) -> Any:
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
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
    baseline_fine = summary["baseline_runs"][-1]
    baseline_common = baseline_fine.get("common_residuals", {}).get("profile_strong_relative", "n/a")
    baseline_runtime = baseline_fine.get("runtime_sec", "n/a")
    baseline_label = f"h{baseline_fine.get('mesh', {}).get('h_min', 'n/a')}_{baseline_fine.get('mesh', {}).get('h_max', 'n/a')}"
    with registry.open("a", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        for run in summary["nextgen_runs"]:
            grid = run["grid"]
            common = run.get("common_residuals", {})
            probe = run.get("probe_errors_vs_fine_baseline", {})
            writer.writerow(
                [
                    RUN_DATE,
                    f"m3-c4-c7-high-amplitude-nr{grid['nr']}_ntheta{grid['ntheta']}",
                    "nextgen_vs_baseline",
                    "experiments/run_m3_c4_c7_high_amplitude_validation.py",
                    "{c4=100,c7=40}",
                    "nextgen_polar_full_disk_vs_baseline_disk_Rdom10",
                    f"nextgen_nr{grid['nr']}_ntheta{grid['ntheta']}; baseline_ref={baseline_label}",
                    "profile_continuation_fv_upwind_exact_psi_jacobian_pseudo_transient",
                    "M3 high-amplitude c4/c7 validation; no M4; no eigensolver",
                    f"nextgen={run.get('runtime_sec'):.3f}; baseline_ref={baseline_runtime}",
                    (
                        f"status={run['status']}; "
                        f"final_sigma={run.get('diagnostics', {}).get('final_sigma', 'n/a')}; "
                        f"nextgen_rel={run.get('diagnostics', {}).get('relative_residual', float('nan')):.3e}; "
                        f"nextgen_common={common.get('profile_strong_relative', 'n/a')}; "
                        f"baseline_common={baseline_common}; "
                        f"omega_probe_rms={probe.get('omega_probe_rms', 'n/a')}; "
                        f"psi_probe_rms_offset_removed={probe.get('psi_probe_rms_offset_removed', 'n/a')}"
                    ),
                    "n/a",
                    run["status"],
                    "High-amplitude c4/c7 M3 validation completed." if run["status"] == "completed" else run.get("message", ""),
                ]
            )


def main() -> int:
    raw_root = REPO_ROOT / "experiments" / "raw" / "nextgen" / "m3_c4_c7_high_amplitude_validation"
    summary_dir = REPO_ROOT / "experiments" / "summaries"
    raw_root.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    print("[case] c4_100_c7_40", flush=True)
    nextgen_runs: list[dict[str, Any]] = []
    baseline_runs: list[dict[str, Any]] = []
    overall_status = "completed"

    for grid in NEXTGEN_GRIDS:
        print(f"[nextgen start] grid={grid[0]}/{grid[1]}", flush=True)
        run = run_nextgen(grid)
        print(
            "[nextgen done] "
            f"grid={grid[0]}/{grid[1]} status={run['status']} "
            f"runtime={run.get('runtime_sec'):.3f} "
            f"final_sigma={run.get('diagnostics', {}).get('final_sigma', 'n/a')}",
            flush=True,
        )
        if run["status"] == "completed":
            run["common_residuals"] = common_residual_from_sampler(
                lambda points, current=run: sample_nextgen(current, points)
            )
        else:
            overall_status = "partial"
        nextgen_runs.append(run)

    for mesh in BASELINE_MESHES:
        print(f"[baseline start] h={mesh[0]}/{mesh[1]}", flush=True)
        run = run_baseline(mesh)
        print(
            "[baseline done] "
            f"h={mesh[0]}/{mesh[1]} status={run['status']} "
            f"runtime={run.get('runtime_sec', 'n/a')}",
            flush=True,
        )
        if run["status"] == "completed":
            run["label"] = f"h{run['mesh']['h_min']:g}_{run['mesh']['h_max']:g}"
            run["common_residuals"] = common_residual_from_sampler(
                lambda points, current=run: (
                    sample_baseline(current["omega"], current["spaces"]["Vomega"], points),
                    sample_baseline(current["psi"], current["spaces"]["Vpsi"], points),
                )
            )
        else:
            overall_status = "partial"
        baseline_runs.append(run)

    completed_baselines = [run for run in baseline_runs if run["status"] == "completed"]
    fine_baseline = completed_baselines[-1] if completed_baselines else {}
    if fine_baseline:
        for run in nextgen_runs:
            if run["status"] == "completed":
                run["probe_errors_vs_fine_baseline"] = compare_nextgen_to_baseline(run, fine_baseline)

    nextgen_grid_difference = (
        compare_nextgen_pair(nextgen_runs[-2], nextgen_runs[-1])
        if len(nextgen_runs) >= 2 and all(run["status"] == "completed" for run in nextgen_runs[-2:])
        else {}
    )
    baseline_mesh_difference = (
        compare_baseline_pair(baseline_runs[-2], baseline_runs[-1])
        if len(baseline_runs) >= 2 and all(run["status"] == "completed" for run in baseline_runs[-2:])
        else {}
    )
    plots = make_plots(nextgen_runs, baseline_runs, raw_root / "c4_100_c7_40")

    summary = {
        "status": overall_status,
        "case": {
            "name": "c4_100_c7_40",
            "description": "large c4/c7 mixed high-amplitude profile validation",
            "coeffs": COEFFS,
        },
        "input": {
            "physical": {"mu": PHYSICAL.mu, "R": PHYSICAL.R, "Rdom": PHYSICAL.Rdom, "eps": PHYSICAL.eps},
            "nextgen_grids": [{"nr": nr, "ntheta": ntheta} for nr, ntheta in NEXTGEN_GRIDS],
            "baseline_meshes": [{"h_min": h_min, "h_max": h_max} for h_min, h_max in BASELINE_MESHES],
            "sigma_grid": SIGMA_GRID,
            "sample_grid": {"x_min": float(SAMPLE_X[0]), "x_max": float(SAMPLE_X[-1]), "n": int(SAMPLE_X.size)},
            "probe_grid": {
                "r_min": float(PROBE_RADII[0]),
                "r_max": float(PROBE_RADII[-1]),
                "nr": int(PROBE_RADII.size),
                "ntheta": int(PROBE_THETA.size),
            },
            "aborted_probe_grids": [
                {
                    "nr": 96,
                    "ntheta": 384,
                    "reason": "stopped after extended runtime without completion output",
                },
                {
                    "nr": 80,
                    "ntheta": 320,
                    "reason": "stopped after extended runtime without completion output",
                },
            ],
        },
        "norm_definitions": {
            "common_strong_relative": "fields sampled to the same Cartesian grid; sqrt(||R_Omega||_L2^2+||R_Psi||_L2^2)/max(||Omega||_L2,||Delta psi||_L2)",
            "probe_rms": "sqrt(mean((field_a-field_b)^2)) over common polar probes r in [0.5,8]",
            "psi_probe_errors": "streamfunction probe differences after subtracting mean difference to remove gauge offset",
        },
        "nextgen_runs": nextgen_runs,
        "baseline_runs": baseline_runs,
        "nextgen_grid_difference": nextgen_grid_difference,
        "baseline_mesh_difference": baseline_mesh_difference,
        "plot_paths": plots,
    }
    append_registry(summary)
    path = summary_dir / f"m3_c4_c7_high_amplitude_validation_{RUN_DATE}.json"
    path.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    print(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    return 0 if overall_status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
