from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-euler")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from nextgen.boundary import FourierBoundaryData
from nextgen.config import PhysicalParams
from nextgen.diagnostics import sampled_strong_profile_residual
from nextgen.polar_profile import PolarGrid, PolarProfileParams, PolarProfileProblem, PolarProfileState


RUN_DATE = "2026-04-29"
PHYSICAL = PhysicalParams(mu=0.8, R=150.0, Rdom=10.0)
COEFFS = {3: 140.0 + 0.0j}
SIGMA_GRID = tuple(float(s) for s in np.linspace(0.0, 1.0, 41))
NESTED_GRIDS = ((72, 216), (96, 288), (128, 384))
SAMPLE_X = np.linspace(-7.0, 7.0, 193)
SAMPLE_Y = np.linspace(-7.0, 7.0, 193)
PROBE_RADII = np.linspace(0.5, 9.5, 91)
PROBE_THETA = np.linspace(-np.pi, np.pi, 720, endpoint=False)
PERTURBATIONS: tuple[dict[str, Any], ...] = (
    {"name": "m1_core_amp0p5", "kind": "mode", "mode": 1, "amplitude": 0.5, "phase": 0.0, "width": 2.2},
    {"name": "m1_core_amp1p5", "kind": "mode", "mode": 1, "amplitude": 1.5, "phase": 0.0, "width": 2.2},
    {"name": "m2_core_amp1p5", "kind": "mode", "mode": 2, "amplitude": 1.5, "phase": 0.37, "width": 2.2},
    {"name": "m4_core_amp1p5", "kind": "mode", "mode": 4, "amplitude": 1.5, "phase": 0.11, "width": 2.2},
    {"name": "mixed_non3_amp1p0", "kind": "mixed", "amplitude": 1.0, "width": 2.4},
)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        blocked = {"problem", "result"}
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


def common_residual(problem: PolarProfileProblem, result: Any) -> dict[str, float]:
    x, y = np.meshgrid(SAMPLE_X, SAMPLE_Y, indexing="ij")
    omega, psi = problem.sample_cartesian(result, x, y)
    return sampled_strong_profile_residual(
        omega,
        psi,
        SAMPLE_X,
        SAMPLE_Y,
        mu=PHYSICAL.mu,
        eps=PHYSICAL.eps,
    )


def solve_grid(
    grid_spec: tuple[int, int],
    *,
    label: str,
    reduced: bool,
    initial_state: PolarProfileState | None = None,
    sigma_grid: tuple[float, ...] = SIGMA_GRID,
    max_iter: int = 72,
) -> dict[str, Any]:
    nr, ntheta = grid_spec
    grid = PolarGrid(radius=PHYSICAL.Rdom, nr=nr, ntheta=ntheta)
    problem = PolarProfileProblem(
        grid,
        PHYSICAL,
        FourierBoundaryData(M=64, coeffs=COEFFS),
        advection_scheme="fv_upwind",
        upwind_blend=1.0,
    )
    result = problem.solve(
        PolarProfileParams(
            sigma_grid=sigma_grid,
            tol_step=1e-8,
            tol_residual=1e-8,
            tol_residual_only=1e-8,
            max_iter=max_iter,
            max_backtracks=30,
            adaptive_continuation=True,
            min_sigma_step=5e-4,
            sigma_step_growth=1.25,
            sigma_step_shrink=0.5,
            least_squares_fallback=False,
            pseudo_transient_fallback=True,
            pseudo_transient_shifts=(1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
            reduced_poisson_newton=reduced,
            reduced_gmres_rtol=1e-8,
            reduced_gmres_maxiter=180,
            reduced_gmres_restart=90,
        ),
        initial_state=initial_state,
    )
    run = {
        "label": label,
        "status": "completed" if result.converged else "nonconverged",
        "grid": {"nr": nr, "ntheta": ntheta, "dr": grid.dr, "dtheta": grid.dtheta},
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


def prolong_state(source: dict[str, Any], target_grid: PolarGrid, *, sigma: float = 0.0) -> PolarProfileState:
    rr, tt = np.meshgrid(target_grid.r, target_grid.theta, indexing="ij")
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)
    omega, psi = source["problem"].sample_cartesian(source["result"], x, y)
    return PolarProfileState(
        omega=np.asarray(omega, dtype=float).reshape(target_grid.shape),
        psi=np.asarray(psi, dtype=float).reshape(target_grid.shape),
        sigma=float(sigma),
        poisson_lambda=float(source["diagnostics"].get("poisson_lambda", 0.0)),
    )


def grid_weights(grid: PolarGrid) -> np.ndarray:
    return grid.quadrature_weights().reshape(grid.shape)


def weighted_norm(values: np.ndarray, grid: PolarGrid) -> float:
    arr = np.asarray(values, dtype=float).reshape(grid.shape)
    return float(np.sqrt(np.sum(arr * arr * grid_weights(grid))))


def symmetry_metrics(problem: PolarProfileProblem, result: Any) -> dict[str, float]:
    grid = problem.grid
    if grid.ntheta % 3 != 0:
        return {"status": "ntheta_not_divisible_by_3"}  # type: ignore[return-value]
    shift = grid.ntheta // 3

    def field_metrics(field: np.ndarray) -> dict[str, float]:
        f = np.asarray(field, dtype=float).reshape(grid.shape)
        norm = max(weighted_norm(f, grid), 1e-30)
        roll1 = np.roll(f, shift, axis=1)
        roll2 = np.roll(f, 2 * shift, axis=1)
        projection = (f + roll1 + roll2) / 3.0
        return {
            "threefold_defect": weighted_norm(f - roll1, grid) / norm,
            "non_threefold_fraction": weighted_norm(f - projection, grid) / norm,
        }

    omega = field_metrics(result.omega)
    psi = field_metrics(result.psi - float(np.sum(result.psi.reshape(-1) * grid.area_weights())))
    return {
        "omega_threefold_defect": omega["threefold_defect"],
        "omega_non_threefold_fraction": omega["non_threefold_fraction"],
        "psi_threefold_defect": psi["threefold_defect"],
        "psi_non_threefold_fraction": psi["non_threefold_fraction"],
    }


def compatibility_metrics(problem: PolarProfileProblem, result: Any) -> dict[str, float]:
    omega = np.asarray(result.omega, dtype=float).reshape(-1)
    omega_int = float(np.sum(omega * problem.grid.quadrature_weights()))
    neumann = problem.psi_neumann_boundary(1.0)
    flux = float(np.sum(neumann) * PHYSICAL.Rdom * problem.grid.dtheta)
    area = np.pi * PHYSICAL.Rdom * PHYSICAL.Rdom
    return {
        "omega_integral": omega_int,
        "omega_mean": omega_int / area,
        "neumann_flux_discrete": flux,
        "omega_minus_flux": omega_int - flux,
        "poisson_lambda": float(result.diagnostics.get("poisson_lambda", float("nan"))),
        "physical_poisson_residual_norm": float(
            result.diagnostics.get("physical_poisson_residual_norm", float("nan"))
        ),
    }


def perturbation_pattern(grid: PolarGrid, spec: dict[str, Any]) -> np.ndarray:
    rr, tt = np.meshgrid(grid.r, grid.theta, indexing="ij")
    width = float(spec.get("width", 2.2))
    envelope = np.exp(-((rr / width) ** 2))
    amplitude = float(spec["amplitude"])
    if spec["kind"] == "mode":
        mode = int(spec["mode"])
        phase = float(spec.get("phase", 0.0))
        return amplitude * envelope * np.cos(mode * tt + phase)
    if spec["kind"] == "mixed":
        return amplitude * envelope * (
            0.70 * np.cos(tt + 0.19)
            - 0.45 * np.sin(2.0 * tt - 0.41)
            + 0.35 * np.cos(4.0 * tt + 0.73)
        )
    raise ValueError(f"Unknown perturbation kind: {spec['kind']}")


def perturbed_state(source: dict[str, Any], spec: dict[str, Any]) -> PolarProfileState:
    problem = source["problem"]
    result = source["result"]
    omega = np.asarray(result.omega, dtype=float).reshape(problem.grid.shape).copy()
    omega += perturbation_pattern(problem.grid, spec)
    psi, lam = problem.solve_poisson(omega, sigma=1.0)
    return PolarProfileState(
        omega=omega,
        psi=psi.reshape(problem.grid.shape),
        poisson_lambda=lam,
        sigma=0.0,
    )


def polar_probe_points() -> np.ndarray:
    rr, tt = np.meshgrid(PROBE_RADII, PROBE_THETA, indexing="ij")
    return np.column_stack((rr.reshape(-1) * np.cos(tt.reshape(-1)), rr.reshape(-1) * np.sin(tt.reshape(-1))))


def compare_runs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float | int]:
    points = polar_probe_points()
    omega_a, psi_a = a["problem"].sample_cartesian(a["result"], points[:, 0], points[:, 1])
    omega_b, psi_b = b["problem"].sample_cartesian(b["result"], points[:, 0], points[:, 1])
    valid = np.isfinite(omega_a) & np.isfinite(psi_a) & np.isfinite(omega_b) & np.isfinite(psi_b)
    omega_diff = omega_a[valid] - omega_b[valid]
    psi_diff = psi_a[valid] - psi_b[valid]
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


def classify_run(run: dict[str, Any], reference: dict[str, Any] | None) -> str:
    if run["status"] != "completed":
        return "nonconverged"
    sym = run.get("symmetry", {})
    non3 = max(
        float(sym.get("omega_non_threefold_fraction", 0.0)),
        float(sym.get("psi_non_threefold_fraction", 0.0)),
    )
    if reference is not None:
        separation = run.get("distance_to_symmetric", {}).get("omega_probe_rms", 0.0)
    else:
        separation = 0.0
    if non3 > 1e-3 and separation > 1e-2:
        return "symmetry_breaking_candidate"
    if separation > 1e-2:
        return "distinct_threefold_candidate"
    return "symmetric_branch"


def save_state(run: dict[str, Any], outdir: Path) -> str | None:
    if run["status"] != "completed":
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{run['label']}_state.npz"
    np.savez_compressed(
        path,
        omega=np.asarray(run["result"].omega, dtype=float),
        psi=np.asarray(run["result"].psi, dtype=float),
        sigma=float(run["diagnostics"].get("final_sigma", 1.0)),
        grid_nr=int(run["grid"]["nr"]),
        grid_ntheta=int(run["grid"]["ntheta"]),
    )
    return str(path.relative_to(REPO_ROOT))


def make_plots(runs: list[dict[str, Any]], outdir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-PHYSICAL.Rdom, PHYSICAL.Rdom, 321)
    y = np.linspace(-PHYSICAL.Rdom, PHYSICAL.Rdom, 321)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    extent = (-PHYSICAL.Rdom, PHYSICAL.Rdom, -PHYSICAL.Rdom, PHYSICAL.Rdom)
    created: list[str] = []

    completed = [run for run in runs if run["status"] == "completed"]
    for run in completed:
        omega, psi = run["problem"].sample_cartesian(run["result"], points[:, 0], points[:, 1])
        omega = omega.reshape(xx.shape)
        psi = psi.reshape(xx.shape)
        label = run["label"]

        path = outdir / f"{label}_omega.png"
        fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
        image = ax.imshow(omega.T, origin="lower", extent=extent, cmap="RdBu_r")
        ax.set_aspect("equal")
        ax.set_title(f"c3_140 {label} Omega")
        fig.colorbar(image, ax=ax)
        fig.savefig(path, dpi=135)
        plt.close(fig)
        created.append(str(path.relative_to(REPO_ROOT)))

        path = outdir / f"{label}_psi_contours.png"
        fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
        ax.contour(xx, yy, psi, levels=50, colors="black", linewidths=0.5)
        ax.set_aspect("equal")
        ax.set_title(f"c3_140 {label} psi")
        fig.savefig(path, dpi=135)
        plt.close(fig)
        created.append(str(path.relative_to(REPO_ROOT)))

    reference = next((run for run in completed if run["label"] == "nested_128_384"), None)
    if reference is not None:
        omega_ref, _ = reference["problem"].sample_cartesian(reference["result"], points[:, 0], points[:, 1])
        omega_ref = omega_ref.reshape(xx.shape)
        for run in completed:
            if run["label"] == reference["label"]:
                continue
            if run.get("classification") == "symmetric_branch":
                continue
            omega, _ = run["problem"].sample_cartesian(run["result"], points[:, 0], points[:, 1])
            diff = omega.reshape(xx.shape) - omega_ref
            path = outdir / f"{run['label']}_omega_minus_nested_128_384.png"
            fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
            image = ax.imshow(diff.T, origin="lower", extent=extent, cmap="RdBu_r")
            ax.set_aspect("equal")
            ax.set_title(f"c3_140 {run['label']} minus nested")
            fig.colorbar(image, ax=ax)
            fig.savefig(path, dpi=135)
            plt.close(fig)
            created.append(str(path.relative_to(REPO_ROOT)))

    path = outdir / "branch_summary.png"
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    labels = [run["label"] for run in completed]
    x_pos = np.arange(len(labels))
    non3 = [run.get("symmetry", {}).get("omega_non_threefold_fraction", np.nan) for run in completed]
    residual = [
        run.get("diagnostics", {}).get("relative_residual", np.nan)
        if run.get("diagnostics") is not None
        else np.nan
        for run in completed
    ]
    ax.bar(x_pos - 0.2, non3, width=0.4, label="Omega non-3 fraction")
    ax2 = ax.twinx()
    ax2.plot(x_pos + 0.2, residual, "o-", color="black", label="native residual")
    ax.set_yscale("log")
    ax2.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("c3_140 branch symmetry summary")
    ax.set_ylabel("non-3 fraction")
    ax2.set_ylabel("native residual")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    created.append(str(path.relative_to(REPO_ROOT)))
    return created


def append_registry(summary: dict[str, Any]) -> None:
    registry = REPO_ROOT / "experiments" / "registry.csv"
    with registry.open("a", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        for run in summary["runs"]:
            grid = run.get("grid", {})
            sym = run.get("symmetry", {})
            dist = run.get("distance_to_symmetric", {})
            common = run.get("common_residuals", {}) or {}
            writer.writerow(
                [
                    RUN_DATE,
                    f"m3-c3-140-symmetry-probe-{run['label']}",
                    "nextgen",
                    "experiments/run_m3_c3_140_symmetry_probe.py",
                    "{c3=140}",
                    "nextgen_polar_full_disk_Rdom10",
                    f"nextgen_nr{grid.get('nr','n/a')}_ntheta{grid.get('ntheta','n/a')}",
                    "profile_continuation_fv_upwind_symmetry_breaking_probe",
                    "M3 c3=140 nonuniqueness/symmetry-breaking probe; no M4; no eigensolver",
                    run.get("runtime_sec", "n/a"),
                    (
                        f"status={run['status']}; class={run.get('classification','n/a')}; "
                        f"rel={run.get('diagnostics', {}).get('relative_residual', 'n/a')}; "
                        f"common={common.get('profile_strong_relative', 'n/a')}; "
                        f"omega_non3={sym.get('omega_non_threefold_fraction', 'n/a')}; "
                        f"omega_rms_to_sym={dist.get('omega_probe_rms', 'n/a')}"
                    ),
                    "n/a",
                    run["status"],
                    run.get("message", ""),
                ]
            )


def main() -> int:
    raw_root = REPO_ROOT / "experiments" / "raw" / "nextgen" / "m3_c3_140_symmetry_probe"
    summary_dir = REPO_ROOT / "experiments" / "summaries"
    state_dir = raw_root / "states"
    plot_dir = raw_root / "plots"
    raw_root.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    print("[case] c3_140 symmetry/nonuniqueness probe", flush=True)
    runs: list[dict[str, Any]] = []

    print("[nested start] grid=72/216", flush=True)
    seed = solve_grid((72, 216), label="nested_72_216", reduced=False)
    runs.append(seed)
    print(
        f"[nested done] label={seed['label']} status={seed['status']} "
        f"runtime={seed.get('runtime_sec'):.3f} "
        f"rel={seed.get('diagnostics', {}).get('relative_residual')}",
        flush=True,
    )
    if seed["status"] != "completed":
        summary = {"status": "partial", "case": {"coeffs": COEFFS}, "runs": runs}
        path = summary_dir / f"m3_c3_140_symmetry_probe_{RUN_DATE}.json"
        path.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True))
        print(json.dumps(sanitize(summary), indent=2, sort_keys=True))
        return 1

    source = seed
    for grid_spec in NESTED_GRIDS[1:]:
        target = PolarGrid(radius=PHYSICAL.Rdom, nr=grid_spec[0], ntheta=grid_spec[1])
        initial = prolong_state(source, target, sigma=0.0)
        label = f"nested_{grid_spec[0]}_{grid_spec[1]}"
        print(f"[nested reduced start] grid={grid_spec[0]}/{grid_spec[1]}", flush=True)
        run = solve_grid(
            grid_spec,
            label=label,
            reduced=True,
            initial_state=initial,
            sigma_grid=(1.0,),
            max_iter=80,
        )
        runs.append(run)
        print(
            f"[nested reduced done] label={label} status={run['status']} "
            f"runtime={run.get('runtime_sec'):.3f} "
            f"rel={run.get('diagnostics', {}).get('relative_residual')}",
            flush=True,
        )
        if run["status"] == "completed":
            source = run

    reference = runs[-1] if runs[-1]["status"] == "completed" else source
    for spec in PERTURBATIONS:
        label = f"perturbed_{spec['name']}_128_384"
        initial = perturbed_state(reference, spec)
        print(f"[perturbed start] label={label}", flush=True)
        run = solve_grid(
            (128, 384),
            label=label,
            reduced=True,
            initial_state=initial,
            sigma_grid=(1.0,),
            max_iter=90,
        )
        if run["status"] == "completed":
            run["perturbation"] = spec
            run["distance_to_symmetric"] = compare_runs(run, reference)
        runs.append(run)
        print(
            f"[perturbed done] label={label} status={run['status']} "
            f"runtime={run.get('runtime_sec'):.3f} "
            f"rel={run.get('diagnostics', {}).get('relative_residual')} "
            f"omega_non3={run.get('symmetry', {}).get('omega_non_threefold_fraction')} "
            f"omega_rms_to_sym={run.get('distance_to_symmetric', {}).get('omega_probe_rms')}",
            flush=True,
        )

    for run in runs:
        run["classification"] = classify_run(run, reference if run is not reference else None)
        state_path = save_state(run, state_dir)
        if state_path is not None:
            run["state_path"] = state_path

    completed = [run for run in runs if run["status"] == "completed"]
    pairwise: dict[str, Any] = {}
    for run in completed:
        if run is reference:
            continue
        pairwise[f"{run['label']}_vs_{reference['label']}"] = compare_runs(run, reference)

    plots = make_plots(runs, plot_dir)
    summary = {
        "status": "completed" if all(run["status"] == "completed" for run in runs[: len(NESTED_GRIDS)]) else "partial",
        "case": {
            "name": "c3_140",
            "description": "large single c3 boundary data; symmetry-breaking/nonuniqueness probe",
            "coeffs": COEFFS,
        },
        "input": {
            "physical": {"mu": PHYSICAL.mu, "R": PHYSICAL.R, "Rdom": PHYSICAL.Rdom, "eps": PHYSICAL.eps},
            "nested_grids": [{"nr": nr, "ntheta": nt} for nr, nt in NESTED_GRIDS],
            "sigma_grid": SIGMA_GRID,
            "perturbations": PERTURBATIONS,
            "symmetry_definition": "weighted L2 defect under theta -> theta + 2*pi/3; non-threefold fraction is distance to threefold projection",
        },
        "runs": runs,
        "reference_branch": reference["label"],
        "pairwise_vs_reference": pairwise,
        "plot_paths": plots,
    }
    append_registry(sanitize(summary))
    path = summary_dir / f"m3_c3_140_symmetry_probe_{RUN_DATE}.json"
    path.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    print(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
