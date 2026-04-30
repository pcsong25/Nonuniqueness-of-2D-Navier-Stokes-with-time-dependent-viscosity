from __future__ import annotations

import argparse
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
from nextgen.polar_profile import PolarGrid, PolarProfileProblem, PolarProfileResult, PolarProfileState
from run_m3_c3_140_symmetry_probe import (
    COEFFS,
    PHYSICAL,
    common_residual,
    compatibility_metrics,
    compare_runs,
    prolong_state,
    save_state,
    symmetry_metrics,
)
from run_m3_c3_140_nonuniqueness_validation import (
    branch_distance_summary,
    classify_by_distance,
    compare_nextgen_to_baseline,
    common_residual_baseline,
    baseline_symmetry_metrics,
)


RUN_DATE = "2026-04-30"
TARGET_GRID = (192, 576)
SOURCE_STATE_DIR = REPO_ROOT / "experiments" / "raw" / "nextgen" / "m3_c3_140_nonuniqueness_validation" / "states"
RAW_ROOT = REPO_ROOT / "experiments" / "raw" / "nextgen" / "m3_c3_140_refine192_fem_crosscheck"
SUMMARY_DIR = REPO_ROOT / "experiments" / "summaries"
BASELINE_MESH = (0.06, 0.06)
TRACE_SIGMAS = (
    1.0,
    0.9,
    0.8,
    0.75,
    0.72,
    0.70,
    0.695,
    0.690,
    0.685,
    0.680,
    0.675,
    0.670,
    0.665,
    0.660,
    0.655,
    0.650,
)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        blocked = {"problem", "result", "spaces", "omega", "psi", "rOmega", "rPsi", "facet_tags"}
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


def write_summary(summary: dict[str, Any]) -> Path:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARY_DIR / f"m3_c3_140_refine192_fem_crosscheck_{RUN_DATE}.json"
    path.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    return path


def make_problem(grid_spec: tuple[int, int]) -> PolarProfileProblem:
    nr, ntheta = grid_spec
    return PolarProfileProblem(
        PolarGrid(radius=PHYSICAL.Rdom, nr=nr, ntheta=ntheta),
        PHYSICAL,
        FourierBoundaryData(M=64, coeffs=COEFFS),
        advection_scheme="fv_upwind",
        upwind_blend=1.0,
    )


def result_from_arrays(
    problem: PolarProfileProblem,
    omega: np.ndarray,
    psi: np.ndarray,
    label: str,
    *,
    sigma: float = 1.0,
    message: str = "loaded_state",
) -> dict[str, Any]:
    omega_arr = np.asarray(omega, dtype=float).reshape(problem.grid.shape)
    psi_arr = np.asarray(psi, dtype=float).reshape(problem.grid.shape)
    diagnostics = problem.residual_summary(omega_arr, psi_arr, float(sigma), 0.0)
    diagnostics.update(
        {
            "runtime_sec": 0.0,
            "newton_steps": 0,
            "final_sigma": float(sigma),
            "attempted_sigma": float(sigma),
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
        message=message,
        diagnostics=diagnostics,
    )
    return {
        "label": label,
        "status": "completed",
        "grid": {"nr": problem.grid.nr, "ntheta": problem.grid.ntheta, "dr": problem.grid.dr, "dtheta": problem.grid.dtheta},
        "problem": problem,
        "result": result,
        "runtime_sec": 0.0,
        "diagnostics": diagnostics,
        "message": message,
        "common_residuals": common_residual(problem, result),
        "symmetry": symmetry_metrics(problem, result),
        "compatibility": compatibility_metrics(problem, result),
    }


def load_saved_state(label: str) -> dict[str, Any]:
    path = SOURCE_STATE_DIR / f"{label}_state.npz"
    data = np.load(path)
    problem = make_problem((int(data["grid_nr"]), int(data["grid_ntheta"])))
    sigma = float(data["sigma"]) if "sigma" in data.files else 1.0
    return result_from_arrays(problem, data["omega"], data["psi"], label, sigma=sigma)


def solve_grid(
    grid_spec: tuple[int, int],
    *,
    label: str,
    initial_state: PolarProfileState,
    sigma: float = 1.0,
    max_iter: int = 140,
) -> dict[str, Any]:
    from nextgen.polar_profile import PolarProfileParams

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
            reduced_gmres_maxiter=320,
            reduced_gmres_restart=160,
        ),
        initial_state=initial_state,
    )
    run = {
        "label": label,
        "status": "completed" if result.converged else "nonconverged",
        "grid": {"nr": problem.grid.nr, "ntheta": problem.grid.ntheta, "dr": problem.grid.dr, "dtheta": problem.grid.dtheta},
        "problem": problem,
        "result": result,
        "runtime_sec": result.runtime_sec,
        "diagnostics": result.diagnostics,
        "message": result.message,
        "sigma": float(sigma),
        "common_residuals": common_residual(problem, result) if result.converged else None,
    }
    if result.converged:
        run["symmetry"] = symmetry_metrics(problem, result)
        run["compatibility"] = compatibility_metrics(problem, result)
    return run


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


def sample_baseline(function: Any, space: Any, points: np.ndarray) -> np.ndarray:
    coords = np.asarray(space.tabulate_dof_coordinates())[:, :2]
    values = np.asarray(function.x.array, dtype=float)
    return np.asarray(LinearNDInterpolator(coords, values, fill_value=np.nan)(points), dtype=float)


def seed_baseline_function_from_nextgen(function: Any, space: Any, nextgen: dict[str, Any]) -> None:
    coords_raw = space.tabulate_dof_coordinates()
    coords = coords_raw[:, :2] if coords_raw.ndim > 1 else coords_raw.reshape(-1, 3)[:, :2]
    omega, _ = nextgen["problem"].sample_cartesian(nextgen["result"], coords[:, 0], coords[:, 1])
    arr = function.x.array
    arr[: omega.size] = np.asarray(omega, dtype=float)
    if hasattr(function.x, "scatter_forward"):
        function.x.scatter_forward()


def compare_baseline_pair(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float | int]:
    radii = np.linspace(0.5, 9.5, 91)
    theta = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    rr, tt = np.meshgrid(radii, theta, indexing="ij")
    points = np.column_stack((rr.reshape(-1) * np.cos(tt.reshape(-1)), rr.reshape(-1) * np.sin(tt.reshape(-1))))
    omega_a = sample_baseline(a["omega"], a["spaces"]["Vomega"], points)
    psi_a = sample_baseline(a["psi"], a["spaces"]["Vpsi"], points)
    omega_b = sample_baseline(b["omega"], b["spaces"]["Vomega"], points)
    psi_b = sample_baseline(b["psi"], b["spaces"]["Vpsi"], points)
    valid = np.isfinite(omega_a) & np.isfinite(psi_a) & np.isfinite(omega_b) & np.isfinite(psi_b)
    omega_diff = omega_a[valid] - omega_b[valid]
    psi_diff = psi_a[valid] - psi_b[valid]
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


def run_baseline_seeded_linesearch(
    nextgen_seed: dict[str, Any],
    mesh_spec: tuple[float, float],
    *,
    max_newton: int = 36,
    tol_newton: float = 1e-9,
    ls_max_steps: int = 10,
    ls_shrink: float = 0.5,
) -> dict[str, Any]:
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

    def vec_data(fn: Any) -> np.ndarray:
        try:
            return fn.vector.array
        except AttributeError:
            return fn.x.array

    def ghost(fn: Any) -> None:
        try:
            fn.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        except AttributeError:
            if hasattr(fn.x, "scatter_forward"):
                fn.x.scatter_forward()

    def split_block_vec(x: Any, Vomega: Any, Vpsi: Any) -> Any:
        n_o = ContinuationSolver._owned_size(Vomega)
        n_p = ContinuationSolver._owned_size(Vpsi)
        is_o = PETSc.IS().createStride(n_o, 0, 1)
        is_p = PETSc.IS().createStride(n_p, n_o, 1)
        return (x.getSubVector(is_o), is_o), (x.getSubVector(is_p), is_p)

    def subvec_array(v: Any) -> np.ndarray:
        try:
            return v.array.copy()
        except AttributeError:
            with v.localForm() as lf:
                return lf.array.copy()

    try:
        h_min, h_max = mesh_spec
        params = SimulationParams(
            mu=PHYSICAL.mu,
            R=PHYSICAL.R,
            Rdom=PHYSICAL.Rdom,
            M=64,
            tol_newton=float(tol_newton),
            max_newton=int(max_newton),
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

        history: list[dict[str, float | int]] = []
        converged = False
        last_rel = float("nan")
        last_bnorm = float("nan")
        accepted_alpha = float("nan")
        for it in range(1, int(max_newton) + 1):
            A, b = assembler.assemble(omega, psi, mu=params.mu)
            bnorm = float(b.norm(PETSc.NormType.NORM_2))
            last_bnorm = bnorm
            ksp = PETSc.KSP().create(mesh.comm)
            ksp.setOperators(A)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.setFromOptions()
            x = b.duplicate()
            x.set(0.0)
            ksp.solve(b, x)
            (x1, is1), (x2, is2) = split_block_vec(x, spaces["Vomega"], spaces["Vpsi"])
            d_omega = subvec_array(x1)
            d_psi = subvec_array(x2)
            x.restoreSubVector(is1, x1)
            x.restoreSubVector(is2, x2)
            x.destroy()

            omega0 = vec_data(omega).copy()
            psi0 = vec_data(psi).copy()
            alpha = 1.0
            accepted = False
            trial_bnorm = bnorm
            for _ in range(int(ls_max_steps)):
                vec_data(omega)[:] = omega0
                vec_data(psi)[:] = psi0
                vec_data(omega)[: d_omega.size] += alpha * d_omega
                vec_data(psi)[: d_psi.size] += alpha * d_psi
                ghost(omega)
                ghost(psi)
                A_trial, b_trial = assembler.assemble(omega, psi, mu=params.mu)
                trial_bnorm = float(b_trial.norm(PETSc.NormType.NORM_2))
                if trial_bnorm < bnorm or alpha <= 2.0 ** (1 - int(ls_max_steps)):
                    accepted = True
                    break
                alpha *= float(ls_shrink)
            if not accepted:
                vec_data(omega)[:] = omega0
                vec_data(psi)[:] = psi0
                ghost(omega)
                ghost(psi)
                alpha = 0.0
                trial_bnorm = bnorm

            accepted_alpha = float(alpha)
            last_rel = float(alpha * np.linalg.norm(d_omega) / (np.linalg.norm(vec_data(omega)) + 1e-16))
            history.append({"iteration": it, "bnorm_before": bnorm, "bnorm_after": trial_bnorm, "rel_omega_step": last_rel, "alpha": accepted_alpha})
            print(
                f"[baseline seeded] {nextgen_seed['label']} it={it} "
                f"bnorm={bnorm:.3e}->{trial_bnorm:.3e} rel={last_rel:.3e} alpha={accepted_alpha:.3g}",
                flush=True,
            )
            if last_rel < float(tol_newton):
                converged = True
                last_bnorm = trial_bnorm
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
            "label": f"baseline_seeded_linesearch_from_{nextgen_seed['label']}_h{h_min:g}_{h_max:g}",
            "status": "completed" if converged else "nonconverged",
            "runtime_sec": runtime,
            "mesh": {"h_min": float(h_min), "h_max": float(h_max)},
            "mesh_cells": int(mesh.topology.index_map(mesh.topology.dim).size_local),
            "mesh_vertices": int(mesh.topology.index_map(0).size_local),
            "newton_iterations": len(history),
            "last_relative_update": last_rel,
            "last_linear_residual_norm": last_bnorm,
            "last_alpha": accepted_alpha,
            "history": history,
            "spaces": spaces,
            "facet_tags": facet_tags,
            "omega": omega,
            "psi": psi,
            "residuals": residuals,
        }
        run["common_residuals"] = common_residual_baseline(run)
        run["symmetry"] = baseline_symmetry_metrics(run)
        run["comparison_to_seed_nextgen"] = compare_nextgen_to_baseline(nextgen_seed, run)
        return run
    except Exception as exc:
        return {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()}


def make_plots(summary: dict[str, Any], outdir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    refined = summary.get("refined_runs", [])
    if len(refined) >= 2 and all(run.get("status") == "completed" for run in refined[:2]):
        x = np.linspace(-PHYSICAL.Rdom, PHYSICAL.Rdom, 361)
        y = np.linspace(-PHYSICAL.Rdom, PHYSICAL.Rdom, 361)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
        omega_sym, _ = refined[0]["problem"].sample_cartesian(refined[0]["result"], points[:, 0], points[:, 1])
        omega_bro, _ = refined[1]["problem"].sample_cartesian(refined[1]["result"], points[:, 0], points[:, 1])
        for label, values in (
            ("symmetric_192_576_omega", omega_sym),
            ("broken_m1_192_576_omega", omega_bro),
            ("broken_m1_minus_symmetric_192_576_omega", omega_bro - omega_sym),
        ):
            path = outdir / f"{label}.png"
            fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
            image = ax.imshow(values.reshape(xx.shape).T, origin="lower", extent=(-PHYSICAL.Rdom, PHYSICAL.Rdom, -PHYSICAL.Rdom, PHYSICAL.Rdom), cmap="RdBu_r")
            ax.set_aspect("equal")
            ax.set_title(label)
            fig.colorbar(image, ax=ax)
            fig.savefig(path, dpi=140)
            plt.close(fig)
            created.append(str(path.relative_to(REPO_ROOT)))

    traces = summary.get("trace_runs", {})
    if traces:
        path = outdir / "bifurcation_window_non3_192_576.png"
        fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
        for branch, rows in traces.items():
            sigmas = [run.get("sigma") for run in rows if run.get("status") == "completed"]
            non3 = [run.get("symmetry", {}).get("omega_non_threefold_fraction", np.nan) for run in rows if run.get("status") == "completed"]
            ax.plot(sigmas, non3, "o-", label=branch)
        ax.set_xlabel("sigma")
        ax.set_ylabel("Omega non-3 fraction")
        ax.set_yscale("log")
        ax.set_title("c3_140 bifurcation-window trace on 192/576")
        ax.legend(fontsize=8)
        fig.savefig(path, dpi=140)
        plt.close(fig)
        created.append(str(path.relative_to(REPO_ROOT)))
    return created


def append_registry(summary: dict[str, Any], *, groups: set[str] | None = None) -> None:
    registry = REPO_ROOT / "experiments" / "registry.csv"
    with registry.open("a", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        for group_name, runs in (
            ("refinement", summary.get("refined_runs", [])),
            ("trace_symmetric", summary.get("trace_runs", {}).get("symmetric", [])),
            ("trace_broken_m1", summary.get("trace_runs", {}).get("broken_m1", [])),
            ("baseline", summary.get("baseline_runs", [])),
        ):
            if groups is not None and group_name not in groups:
                continue
            for run in runs:
                sym = run.get("symmetry", {}) or {}
                diag = run.get("diagnostics", {}) or {}
                common = run.get("common_residuals", {}) or {}
                grid = run.get("grid", {})
                mesh = run.get("mesh", {})
                writer.writerow(
                    [
                        RUN_DATE,
                        f"m3-c3-140-refine192-{group_name}-{run.get('label', 'unknown')}",
                        "baseline" if group_name == "baseline" else "nextgen",
                        "experiments/run_m3_c3_140_refine192_fem_crosscheck.py",
                        "{c3=140}",
                        "baseline_fem_disk_Rdom10" if group_name == "baseline" else "nextgen_polar_full_disk_Rdom10",
                        (
                            f"h={mesh.get('h_min')}/{mesh.get('h_max')}"
                            if group_name == "baseline"
                            else f"grid={grid.get('nr')}/{grid.get('ntheta')}"
                        ),
                        "profile_fv_refinement_bifurcation_trace_fem_crosscheck",
                        "M3 c3=140 192/576 refinement, bifurcation window, FEM cross-check; no M4; no eigensolver",
                        run.get("runtime_sec", "n/a"),
                        (
                            f"status={run.get('status')}; sigma={run.get('sigma', 1.0)}; "
                            f"rel={diag.get('relative_residual', 'n/a')}; "
                            f"common={common.get('profile_strong_relative', 'n/a')}; "
                            f"omega_non3={sym.get('omega_non_threefold_fraction', 'n/a')}; "
                            f"last_rel={run.get('last_relative_update', 'n/a')}"
                        ),
                        "n/a",
                        run.get("status", "unknown"),
                        run.get("message", run.get("label", "")),
                    ]
                )


def run_nextgen(summary: dict[str, Any]) -> None:
    state_dir = RAW_ROOT / "states"
    source_sym = load_saved_state("symmetric_160_480")
    source_broken = load_saved_state("broken_m1_160_480")
    target = PolarGrid(radius=PHYSICAL.Rdom, nr=TARGET_GRID[0], ntheta=TARGET_GRID[1])

    refined: list[dict[str, Any]] = []
    for label, source in (("symmetric_192_576", source_sym), ("broken_m1_192_576", source_broken)):
        print(f"[nextgen refine start] {label}", flush=True)
        initial = prolong_state(source, target, sigma=0.0)
        run = solve_grid(TARGET_GRID, label=label, initial_state=initial, sigma=1.0)
        if run["status"] == "completed":
            run["state_path"] = save_state(run, state_dir)
        refined.append(run)
        print(
            f"[nextgen refine done] {label} status={run['status']} "
            f"runtime={run.get('runtime_sec'):.3f} "
            f"rel={run.get('diagnostics', {}).get('relative_residual')} "
            f"non3={run.get('symmetry', {}).get('omega_non_threefold_fraction')}",
            flush=True,
        )
    if all(run["status"] == "completed" for run in refined):
        for run in refined:
            run["branch_distances"] = branch_distance_summary(run, refined[0], refined[1])
            run["branch_class"] = classify_by_distance(run, refined[0], refined[1])

    summary["refined_runs"] = refined
    write_summary(summary)

    traces: dict[str, list[dict[str, Any]]] = {"symmetric": [], "broken_m1": []}
    for branch_name, start in (("symmetric", refined[0]), ("broken_m1", refined[1])):
        if start["status"] != "completed":
            continue
        previous = start
        previous_sigma = 1.0
        for sigma in TRACE_SIGMAS:
            if abs(sigma - 1.0) < 1e-14:
                run = start
            else:
                label = f"trace_{branch_name}_sigma{str(sigma).replace('.', 'p')}_192_576"
                print(f"[trace start] branch={branch_name} sigma={sigma}", flush=True)
                initial = scaled_state_for_sigma(previous, sigma, previous_sigma)
                run = solve_grid(TARGET_GRID, label=label, initial_state=initial, sigma=sigma)
                print(
                    f"[trace done] branch={branch_name} sigma={sigma} status={run['status']} "
                    f"runtime={run.get('runtime_sec'):.3f} "
                    f"rel={run.get('diagnostics', {}).get('relative_residual')} "
                    f"non3={run.get('symmetry', {}).get('omega_non_threefold_fraction')}",
                    flush=True,
                )
            if run["status"] == "completed":
                previous = run
                previous_sigma = float(sigma)
            traces[branch_name].append(run)
            summary["trace_runs"] = traces
            write_summary(summary)
            if run["status"] != "completed":
                break
    summary["trace_runs"] = traces


def run_baseline(summary: dict[str, Any], mesh_spec: tuple[float, float]) -> None:
    baseline_runs: list[dict[str, Any]] = []
    for nextgen_run in summary.get("refined_runs", [])[:2]:
        if nextgen_run.get("status") != "completed":
            continue
        print(f"[baseline start] seed={nextgen_run['label']} mesh={mesh_spec}", flush=True)
        run = run_baseline_seeded_linesearch(nextgen_run, mesh_spec)
        baseline_runs.append(run)
        print(
            f"[baseline done] seed={nextgen_run['label']} status={run.get('status')} "
            f"runtime={run.get('runtime_sec')} common={(run.get('common_residuals') or {}).get('profile_strong_relative')} "
            f"non3={(run.get('symmetry') or {}).get('omega_non_threefold_fraction')}",
            flush=True,
        )
        summary["baseline_runs"] = baseline_runs
        write_summary(summary)
    if len(baseline_runs) >= 2 and all(run.get("status") == "completed" for run in baseline_runs[:2]):
        summary["baseline_pair_distance"] = compare_baseline_pair(baseline_runs[0], baseline_runs[1])


def main() -> int:
    parser = argparse.ArgumentParser(description="M3 c3=140 192/576 refinement and FEM cross-check.")
    parser.add_argument("--stage", choices=("all", "nextgen", "baseline"), default="all")
    parser.add_argument("--h-min", type=float, default=BASELINE_MESH[0])
    parser.add_argument("--h-max", type=float, default=BASELINE_MESH[1])
    args = parser.parse_args()

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "baseline":
        path = SUMMARY_DIR / f"m3_c3_140_refine192_fem_crosscheck_{RUN_DATE}.json"
        loaded = json.loads(path.read_text())
        serialized_refined = {run.get("label"): run for run in loaded.get("refined_runs", [])}
        # Rehydrate nextgen runs from saved states for baseline-only resumes.
        rehydrated = []
        for label in ("symmetric_192_576", "broken_m1_192_576"):
            state = RAW_ROOT / "states" / f"{label}_state.npz"
            if state.exists():
                data = np.load(state)
                problem = make_problem((int(data["grid_nr"]), int(data["grid_ntheta"])))
                run = result_from_arrays(problem, data["omega"], data["psi"], label, sigma=1.0)
                if label in serialized_refined:
                    old = serialized_refined[label]
                    for key in ("runtime_sec", "state_path", "branch_class", "branch_distances"):
                        if key in old:
                            run[key] = old[key]
                    if "diagnostics" in old:
                        run["diagnostics"].update(old["diagnostics"])
                    if "message" in old:
                        run["message"] = old["message"]
                rehydrated.append(run)
        loaded["refined_runs"] = rehydrated
        summary = loaded
    else:
        summary = {
            "status": "running",
            "case": {
                "name": "c3_140_refine192_fem_crosscheck",
                "description": "192/576 branch refinement, bifurcation-window trace, and hmax=0.06 FEM cross-check for c3=140 nonuniqueness",
                "coeffs": COEFFS,
            },
            "input": {
                "target_grid": {"nr": TARGET_GRID[0], "ntheta": TARGET_GRID[1]},
                "source_state_dir": str(SOURCE_STATE_DIR.relative_to(REPO_ROOT)),
                "trace_sigmas": TRACE_SIGMAS,
                "baseline_mesh": {"h_min": float(args.h_min), "h_max": float(args.h_max)},
                "physical": {"mu": PHYSICAL.mu, "R": PHYSICAL.R, "Rdom": PHYSICAL.Rdom, "eps": PHYSICAL.eps},
            },
            "validation_methods": [
                "nextgen branch refinement from saved 160/480 states to 192/576",
                "seeded amplitude trace through sigma 0.65 to 0.70 on 192/576",
                "hmax=0.06 baseline/FEM Newton with residual line search seeded from refined nextgen branches",
                "common Cartesian strong residuals and threefold-symmetry diagnostics",
                "Poisson compatibility diagnostics for nextgen branches",
                "probe comparisons between nextgen and FEM fields",
            ],
            "refined_runs": [],
            "trace_runs": {},
            "baseline_runs": [],
        }
        write_summary(summary)

    if args.stage in {"all", "nextgen"}:
        run_nextgen(summary)

    if args.stage in {"all", "baseline"}:
        run_baseline(summary, (float(args.h_min), float(args.h_max)))

    summary["plot_paths"] = make_plots(summary, RAW_ROOT / "plots")
    summary["status"] = "completed"
    write_summary(summary)
    registry_groups = None
    if args.stage == "nextgen":
        registry_groups = {"refinement", "trace_symmetric", "trace_broken_m1"}
    elif args.stage == "baseline":
        registry_groups = {"baseline"}
    append_registry(summary, groups=registry_groups)
    print(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
