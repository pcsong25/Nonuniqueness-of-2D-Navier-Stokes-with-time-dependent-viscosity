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
SIGMA_GRID = tuple(float(s) for s in np.linspace(0.0, 1.0, 21))
NEXTGEN_GRIDS = ((30, 90), (36, 108), (72, 216))
BASELINE_MESHES = ((0.08, 0.12),)
NEXTGEN_ADVECTION_SCHEME = "fv_upwind"
SAMPLE_X = np.linspace(-7.0, 7.0, 129)
SAMPLE_Y = np.linspace(-7.0, 7.0, 129)
PROBE_RADII = np.linspace(0.5, 8.0, 58)
PROBE_THETA = np.linspace(-np.pi, np.pi, 174, endpoint=False)

CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "large_leading_mixed_2_5_8",
        "description": "mixed case with a 60-scale leading coefficient and target-size higher modes",
        "coeffs": {2: 60.0 + 0.0j, 5: -5.0 + 1.0j, 8: 2.0 - 1.5j},
    },
)


def run_nextgen(coeffs: dict[int, complex], grid_spec: tuple[int, int]) -> dict[str, Any]:
    nr, ntheta = grid_spec
    grid = PolarGrid(radius=PHYSICAL.Rdom, nr=nr, ntheta=ntheta)
    problem = PolarProfileProblem(
        grid,
        PHYSICAL,
        FourierBoundaryData(M=64, coeffs=coeffs),
        advection_scheme=NEXTGEN_ADVECTION_SCHEME,
    )
    result = problem.solve(
        PolarProfileParams(
            sigma_grid=SIGMA_GRID,
            tol_step=1e-8,
            tol_residual=1e-8,
            tol_residual_only=1e-8,
            max_iter=48,
            max_backtracks=24,
            adaptive_continuation=True,
            min_sigma_step=1e-3,
            sigma_step_growth=1.25,
            sigma_step_shrink=0.5,
            least_squares_fallback=False,
            pseudo_transient_fallback=True,
            pseudo_transient_shifts=(1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
            diffusion_continuation=False,
            diffusion_scales=(4.0, 2.0, 1.0),
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


def run_baseline(coeffs: dict[int, complex], mesh_spec: tuple[float, float]) -> dict[str, Any]:
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
            max_newton=64,
            sigma_grid=SIGMA_GRID,
            h_min=float(h_min),
            h_max=float(h_max),
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


def compare_to_baseline(nextgen: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float | int]:
    points = probe_points()
    omega_next, psi_next = sample_nextgen(nextgen, points)
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


def make_case_plots(
    case_name: str,
    nextgen_runs: list[dict[str, Any]],
    baseline_runs: list[dict[str, Any]],
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

    fields: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for run in nextgen_runs:
        if run["status"] == "skipped_after_coarser_nonconvergence":
            continue
        nr = run["grid"]["nr"]
        ntheta = run["grid"]["ntheta"]
        final_sigma = run.get("diagnostics", {}).get("final_sigma", 1.0)
        suffix = "" if run["status"] == "completed" else f"_partial_sigma{float(final_sigma):.3f}".replace(".", "p")
        omega, psi = sample_nextgen(run, points)
        fields[f"nextgen_{nr}_{ntheta}{suffix}"] = (omega.reshape(X.shape), psi.reshape(X.shape))
    for run in baseline_runs:
        if run["status"] != "completed":
            continue
        h_min = run["mesh"]["h_min"]
        h_max = run["mesh"]["h_max"]
        label = f"baseline_h{h_min:g}_{h_max:g}".replace(".", "p")
        fields[label] = (
            sample_baseline(run["omega"], run["spaces"]["Vomega"], points).reshape(X.shape),
            sample_baseline(run["psi"], run["spaces"]["Vpsi"], points).reshape(X.shape),
        )

    created: list[str] = []
    extent = (-PHYSICAL.Rdom, PHYSICAL.Rdom, -PHYSICAL.Rdom, PHYSICAL.Rdom)
    for source, (omega, psi) in fields.items():
        path = outdir / f"{source}_omega.png"
        fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
        image = ax.imshow(omega.T, origin="lower", extent=extent, cmap="RdBu_r")
        ax.set_aspect("equal")
        ax.set_title(f"{case_name} {source} Omega")
        fig.colorbar(image, ax=ax)
        fig.savefig(path, dpi=135)
        plt.close(fig)
        created.append(str(path.relative_to(REPO_ROOT)))

        path = outdir / f"{source}_psi_contours.png"
        fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
        ax.contour(X, Y, psi, levels=42, colors="black", linewidths=0.55)
        ax.set_aspect("equal")
        ax.set_title(f"{case_name} {source} psi")
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
    with registry.open("a", newline="") as f:
        writer = csv.writer(f)
        for case in summary["cases"]:
            best_baseline = case.get("baseline_reference", {})
            baseline_label = best_baseline.get("label", "n/a")
            baseline_common = best_baseline.get("common_residuals", {}).get("profile_strong_relative", "n/a")
            baseline_runtime = best_baseline.get("runtime_sec", "n/a")
            for run in case["nextgen_runs"]:
                common = run.get("common_residuals", run.get("accepted_state_common_residuals", {}))
                probe = run.get("probe_errors_vs_reference_baseline", {})
                grid = run["grid"]
                writer.writerow(
                    [
                        RUN_DATE,
                        f"m3-high-amplitude-stress-{case['name']}-nr{grid['nr']}_ntheta{grid['ntheta']}",
                        "nextgen_vs_baseline",
                        "experiments/run_m3_high_amplitude_stress_suite.py",
                        case["description"],
                        "nextgen_polar_full_disk_vs_baseline_disk_Rdom10",
                        f"nextgen_nr{grid['nr']}_ntheta{grid['ntheta']}; baseline_ref={baseline_label}",
                        f"profile_continuation_{NEXTGEN_ADVECTION_SCHEME}_exact_psi_jacobian_pseudo_transient",
                        "high-amplitude M3 conservative-FV psi-Jacobian stress/refinement suite; no eigensolver",
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
                        run.get("message", ""),
                    ]
                )


def main() -> int:
    raw_root = REPO_ROOT / "experiments" / "raw" / "nextgen" / "m3_high_amplitude_stress_suite"
    summary_dir = REPO_ROOT / "experiments" / "summaries"
    raw_root.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    cases_out: list[dict[str, Any]] = []
    overall_status = "completed"
    for case in CASES:
        print(f"[case] {case['name']}", flush=True)
        nextgen_runs = []
        skip_finer = False
        for grid in NEXTGEN_GRIDS:
            if skip_finer:
                print(
                    f"[nextgen skip] grid={grid[0]}/{grid[1]} after coarser nonconvergence",
                    flush=True,
                )
                nextgen_runs.append(
                    {
                        "status": "skipped_after_coarser_nonconvergence",
                        "grid": {
                            "nr": grid[0],
                            "ntheta": grid[1],
                            "dr": PHYSICAL.Rdom / float(grid[0]),
                            "dtheta": 2.0 * np.pi / float(grid[1]),
                        },
                        "runtime_sec": 0.0,
                        "diagnostics": {"relative_residual": float("nan"), "final_sigma": float("nan")},
                        "message": "Skipped because the coarser high-amplitude nextgen solve did not reach sigma 1.",
                    }
                )
                continue
            print(f"[nextgen start] grid={grid[0]}/{grid[1]}", flush=True)
            run = run_nextgen(case["coeffs"], grid)
            print(
                "[nextgen done] "
                f"grid={grid[0]}/{grid[1]} status={run['status']} "
                f"runtime={run.get('runtime_sec'):.3f} "
                f"final_sigma={run.get('diagnostics', {}).get('final_sigma', 'n/a')}",
                flush=True,
            )
            nextgen_runs.append(run)
            if run["status"] != "completed":
                skip_finer = True
        baseline_runs = []
        for mesh in BASELINE_MESHES:
            print(f"[baseline start] h={mesh[0]}/{mesh[1]}", flush=True)
            run = run_baseline(case["coeffs"], mesh)
            print(
                "[baseline done] "
                f"h={mesh[0]}/{mesh[1]} status={run['status']} "
                f"runtime={run.get('runtime_sec', 'n/a')}",
                flush=True,
            )
            baseline_runs.append(run)

        for run in nextgen_runs:
            if run["status"] != "completed":
                overall_status = "partial"
            if run["status"] == "completed":
                run["common_residuals"] = common_residual_from_sampler(
                    lambda points, current=run: sample_nextgen(current, points)
                )
            elif np.isfinite(run.get("diagnostics", {}).get("final_sigma", np.nan)):
                run["accepted_state_common_residuals"] = common_residual_from_sampler(
                    lambda points, current=run: sample_nextgen(current, points)
                )

        for run in baseline_runs:
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

        completed_baselines = [run for run in baseline_runs if run["status"] == "completed"]
        baseline_reference = completed_baselines[-1] if completed_baselines else {}
        if baseline_reference:
            for run in nextgen_runs:
                if run["status"] == "completed":
                    run["probe_errors_vs_reference_baseline"] = compare_to_baseline(run, baseline_reference)

        plots = make_case_plots(case["name"], nextgen_runs, baseline_runs, raw_root / case["name"])
        cases_out.append(
            {
                "name": case["name"],
                "description": case["description"],
                "coeffs": case["coeffs"],
                "status": "completed"
                if all(run["status"] == "completed" for run in nextgen_runs + baseline_runs)
                else "partial",
                "nextgen_runs": nextgen_runs,
                "baseline_runs": baseline_runs,
                "baseline_reference": baseline_reference,
                "plot_paths": plots,
            }
        )

    summary = {
        "status": overall_status,
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
        },
        "norm_definitions": {
            "common_strong_relative": "both codes sampled to the same Cartesian grid; sqrt(||R_Omega||_L2^2+||R_Psi||_L2^2)/max(||Omega||_L2,||Delta psi||_L2)",
            "probe_rms": "sqrt(mean((nextgen-baseline)^2)) over common polar probes r in [0.5,8]",
            "psi_probe_errors": "streamfunction probe differences after subtracting mean difference to remove gauge offset",
        },
        "cases": cases_out,
    }
    append_registry(summary)
    path = summary_dir / f"m3_high_amplitude_stress_suite_{RUN_DATE}.json"
    path.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    print(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    return 0 if overall_status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
