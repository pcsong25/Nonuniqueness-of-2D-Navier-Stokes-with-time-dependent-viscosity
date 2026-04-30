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
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(SRC / "baseline"))

from boundary_data_fourier import FourierBC
from nextgen.boundary import FourierBoundaryData
from nextgen.polar_profile import PolarGrid, PolarProfileParams, PolarProfileProblem, PolarProfileState
from run_m3_c4_c7_high_amplitude_validation import (
    COEFFS,
    PHYSICAL,
    RUN_DATE,
    common_residual_from_sampler,
    compare_samples,
    run_baseline,
    sample_baseline,
    sample_nextgen,
)
from run_m3_c4_c7_seeded_reduced_probe import SIGMA_GRID, prolong_state, solve_grid


BASELINE_MESH = (0.05, 0.08)
NEXTGEN_GRIDS = ((72, 288), (96, 384), (128, 512))
STRUCTURE_RADII = np.linspace(0.5, 9.5, 91)
STRUCTURE_THETA = np.linspace(-np.pi, np.pi, 720, endpoint=False)
MODES = (0, 1, 3, 4, 7, 8, 11, 14)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        blocked = {"problem", "result", "spaces", "rOmega", "rPsi"}
        out: dict[str, Any] = {}
        for k, v in value.items():
            if k in blocked:
                continue
            if k in {"omega", "psi"} and not isinstance(v, (dict, list, tuple, np.ndarray)):
                continue
            out[str(k)] = sanitize(v)
        return out
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


def polar_points(radii: np.ndarray = STRUCTURE_RADII, theta: np.ndarray = STRUCTURE_THETA) -> np.ndarray:
    rr, tt = np.meshgrid(radii, theta, indexing="ij")
    return np.column_stack((rr.reshape(-1) * np.cos(tt.reshape(-1)), rr.reshape(-1) * np.sin(tt.reshape(-1))))


def sample_baseline_polar(baseline: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    points = polar_points()
    omega = sample_baseline(baseline["omega"], baseline["spaces"]["Vomega"], points)
    psi = sample_baseline(baseline["psi"], baseline["spaces"]["Vpsi"], points)
    return omega.reshape(STRUCTURE_RADII.size, STRUCTURE_THETA.size), psi.reshape(
        STRUCTURE_RADII.size, STRUCTURE_THETA.size
    )


def sample_nextgen_polar(run: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    points = polar_points()
    omega, psi = sample_nextgen(run, points)
    return omega.reshape(STRUCTURE_RADII.size, STRUCTURE_THETA.size), psi.reshape(
        STRUCTURE_RADII.size, STRUCTURE_THETA.size
    )


def global_psi_offset(a_psi: np.ndarray, b_psi: np.ndarray) -> float:
    valid = np.isfinite(a_psi) & np.isfinite(b_psi)
    return float(np.mean(a_psi[valid] - b_psi[valid])) if np.any(valid) else float("nan")


def radial_rms(diff: np.ndarray) -> np.ndarray:
    out = np.full(diff.shape[0], np.nan, dtype=float)
    for j in range(diff.shape[0]):
        row = diff[j]
        valid = np.isfinite(row)
        if np.any(valid):
            out[j] = float(np.sqrt(np.mean(row[valid] ** 2)))
    return out


def fourier_coefficients(values: np.ndarray) -> dict[int, np.ndarray]:
    coeffs: dict[int, np.ndarray] = {}
    for mode in MODES:
        phase = np.exp(-1j * mode * STRUCTURE_THETA)
        mode_coeffs = np.full(values.shape[0], np.nan + 0j, dtype=complex)
        for j in range(values.shape[0]):
            row = values[j]
            valid = np.isfinite(row)
            if np.any(valid):
                mode_coeffs[j] = np.mean(row[valid] * phase[valid])
        coeffs[mode] = mode_coeffs
    return coeffs


def summarize_modes(values: np.ndarray) -> dict[str, Any]:
    coeffs = fourier_coefficients(values)
    summary: dict[str, Any] = {}
    for mode, arr in coeffs.items():
        amp = np.abs(arr)
        valid = np.isfinite(amp)
        summary[str(mode)] = {
            "amplitude_rms_over_r": float(np.sqrt(np.mean(amp[valid] ** 2))) if np.any(valid) else float("nan"),
            "max_amplitude": float(np.max(amp[valid])) if np.any(valid) else float("nan"),
            "radius_at_max": float(STRUCTURE_RADII[np.nanargmax(amp)]) if np.any(valid) else float("nan"),
        }
    return summary


def modal_error_summary(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a_coeffs = fourier_coefficients(a)
    b_coeffs = fourier_coefficients(b)
    summary: dict[str, Any] = {}
    for mode in MODES:
        diff = a_coeffs[mode] - b_coeffs[mode]
        amp = np.abs(diff)
        valid = np.isfinite(amp)
        summary[str(mode)] = {
            "coeff_rms_over_r": float(np.sqrt(np.mean(amp[valid] ** 2))) if np.any(valid) else float("nan"),
            "coeff_linf_over_r": float(np.max(amp[valid])) if np.any(valid) else float("nan"),
            "radius_at_linf": float(STRUCTURE_RADII[np.nanargmax(amp)]) if np.any(valid) else float("nan"),
        }
    return summary


def structure_comparison(
    label: str,
    omega: np.ndarray,
    psi: np.ndarray,
    baseline_omega: np.ndarray,
    baseline_psi: np.ndarray,
) -> dict[str, Any]:
    psi_offset = global_psi_offset(psi, baseline_psi)
    omega_diff = omega - baseline_omega
    psi_diff = psi - baseline_psi - psi_offset
    omega_r = radial_rms(omega_diff)
    psi_r = radial_rms(psi_diff)
    flat_compare = compare_samples(
        omega.reshape(-1),
        psi.reshape(-1),
        baseline_omega.reshape(-1),
        baseline_psi.reshape(-1),
    )
    return {
        "label": label,
        "global_probe_errors": flat_compare,
        "radial_error_summary": {
            "omega_rms_max": float(np.nanmax(omega_r)),
            "omega_rms_radius_at_max": float(STRUCTURE_RADII[int(np.nanargmax(omega_r))]),
            "omega_rms_at_r8": float(omega_r[int(np.argmin(np.abs(STRUCTURE_RADII - 8.0)))]),
            "omega_rms_at_r9p5": float(omega_r[int(np.argmin(np.abs(STRUCTURE_RADII - 9.5)))]),
            "psi_offset_removed_rms_max": float(np.nanmax(psi_r)),
            "psi_offset_removed_rms_radius_at_max": float(STRUCTURE_RADII[int(np.nanargmax(psi_r))]),
            "psi_offset_removed_rms_at_r8": float(psi_r[int(np.argmin(np.abs(STRUCTURE_RADII - 8.0)))]),
            "psi_offset_removed_rms_at_r9p5": float(psi_r[int(np.argmin(np.abs(STRUCTURE_RADII - 9.5)))]),
        },
        "modal_error_vs_baseline": {
            "omega": modal_error_summary(omega, baseline_omega),
            "psi": modal_error_summary(psi - psi_offset, baseline_psi),
        },
        "radial_curves": {
            "radii": STRUCTURE_RADII,
            "omega_rms": omega_r,
            "psi_offset_removed_rms": psi_r,
        },
    }


def boundary_convention_errors() -> dict[str, float]:
    theta = np.linspace(-np.pi, np.pi, 4096, endpoint=False)
    nextgen = FourierBoundaryData(M=64, coeffs=COEFFS)
    baseline = FourierBC(64, COEFFS)
    x = PHYSICAL.Rdom * np.cos(theta)
    y = PHYSICAL.Rdom * np.sin(theta)
    omega_next = nextgen.omega_dirichlet(x, y, mu=PHYSICAL.mu, R=PHYSICAL.R, Rdom=PHYSICAL.Rdom, sigma=1.0)
    omega_base = baseline.g_theta(theta, PHYSICAL.mu, PHYSICAL.R)
    neumann_next = nextgen.psi_neumann_circle(theta, mu=PHYSICAL.mu, R=PHYSICAL.R, Rdom=PHYSICAL.Rdom, sigma=1.0)
    neumann_base = baseline.b_theta(theta, PHYSICAL.mu, PHYSICAL.R, PHYSICAL.Rdom, 1.0)
    return {
        "omega_dirichlet_max_abs": float(np.max(np.abs(omega_next - omega_base))),
        "omega_dirichlet_rms": float(np.sqrt(np.mean((omega_next - omega_base) ** 2))),
        "psi_neumann_max_abs": float(np.max(np.abs(neumann_next - neumann_base))),
        "psi_neumann_rms": float(np.sqrt(np.mean((neumann_next - neumann_base) ** 2))),
        "neumann_flux_trapz": float(np.sum(neumann_next) * PHYSICAL.Rdom * (2.0 * np.pi / theta.size)),
    }


def baseline_boundary_error(baseline: dict[str, Any]) -> dict[str, float | int]:
    coords = np.asarray(baseline["spaces"]["Vomega"].tabulate_dof_coordinates())[:, :2]
    radius = np.hypot(coords[:, 0], coords[:, 1])
    mask = np.isclose(radius, PHYSICAL.Rdom, rtol=0.0, atol=1e-8)
    theta = np.arctan2(coords[mask, 1], coords[mask, 0])
    expected = FourierBC(64, COEFFS).g_theta(theta, PHYSICAL.mu, PHYSICAL.R)
    values = np.asarray(baseline["omega"].x.array, dtype=float)[mask]
    diff = values - expected
    return {
        "boundary_dofs": int(np.count_nonzero(mask)),
        "omega_dirichlet_max_abs": float(np.max(np.abs(diff))) if diff.size else float("nan"),
        "omega_dirichlet_rms": float(np.sqrt(np.mean(diff**2))) if diff.size else float("nan"),
    }


def baseline_omega_integral(baseline: dict[str, Any]) -> dict[str, float | str]:
    try:
        from dolfinx import fem
        import ufl

        omega_int = fem.assemble_scalar(fem.form(baseline["omega"] * ufl.dx))
        area = np.pi * PHYSICAL.Rdom * PHYSICAL.Rdom
        return {"omega_integral": float(omega_int), "omega_mean": float(omega_int / area)}
    except Exception as exc:
        return {"status": "failed", "error": repr(exc)}


def nextgen_compatibility(run: dict[str, Any]) -> dict[str, float]:
    problem = run["problem"]
    result = run["result"]
    omega = np.asarray(result.omega, dtype=float).reshape(-1)
    omega_int = float(np.sum(omega * problem.grid.quadrature_weights()))
    neumann = problem.psi_neumann_boundary(1.0)
    neumann_flux = float(np.sum(neumann) * PHYSICAL.Rdom * problem.grid.dtheta)
    area = np.pi * PHYSICAL.Rdom * PHYSICAL.Rdom
    return {
        "omega_integral": omega_int,
        "omega_mean": omega_int / area,
        "neumann_flux_discrete": neumann_flux,
        "omega_minus_flux": omega_int - neumann_flux,
        "poisson_lambda": float(result.diagnostics.get("poisson_lambda", float("nan"))),
        "physical_poisson_residual_norm": float(
            result.diagnostics.get("physical_poisson_residual_norm", float("nan"))
        ),
    }


def grid_points(grid: PolarGrid) -> np.ndarray:
    rr, tt = np.meshgrid(grid.r, grid.theta, indexing="ij")
    return np.column_stack((rr.reshape(-1) * np.cos(tt.reshape(-1)), rr.reshape(-1) * np.sin(tt.reshape(-1))))


def baseline_state_on_grid(baseline: dict[str, Any], grid: PolarGrid) -> PolarProfileState:
    points = grid_points(grid)
    omega = sample_baseline(baseline["omega"], baseline["spaces"]["Vomega"], points).reshape(grid.shape)
    psi = sample_baseline(baseline["psi"], baseline["spaces"]["Vpsi"], points).reshape(grid.shape)
    if np.any(~np.isfinite(omega)) or np.any(~np.isfinite(psi)):
        raise RuntimeError("Baseline interpolation produced nonfinite values on the polar grid.")
    weights = grid.area_weights()
    psi_vec = psi.reshape(-1).copy()
    psi_vec -= float(np.sum(psi_vec * weights))
    return PolarProfileState(omega=omega, psi=psi_vec.reshape(grid.shape), sigma=0.0, poisson_lambda=0.0)


def poisson_isolation(problem: PolarProfileProblem, baseline: dict[str, Any]) -> dict[str, float | int]:
    grid = problem.grid
    points = grid_points(grid)
    omega_base = sample_baseline(baseline["omega"], baseline["spaces"]["Vomega"], points).reshape(grid.shape)
    psi_base = sample_baseline(baseline["psi"], baseline["spaces"]["Vpsi"], points).reshape(grid.shape)
    psi_poisson, lam = problem.solve_poisson(omega_base, sigma=1.0)
    psi_poisson = psi_poisson.reshape(grid.shape)
    valid = np.isfinite(omega_base) & np.isfinite(psi_base) & np.isfinite(psi_poisson)
    diff = psi_poisson[valid] - psi_base[valid]
    offset = float(np.mean(diff)) if diff.size else float("nan")
    diff = diff - offset
    return {
        "valid_grid_points": int(np.count_nonzero(valid)),
        "poisson_lambda_from_baseline_omega": float(lam),
        "psi_rms_offset_removed": float(np.sqrt(np.mean(diff**2))) if diff.size else float("nan"),
        "psi_linf_offset_removed": float(np.max(np.abs(diff))) if diff.size else float("nan"),
    }


def make_plots(
    comparisons: dict[str, dict[str, Any]],
    modal_sources: dict[str, np.ndarray],
    baseline_omega: np.ndarray,
    baseline_psi: np.ndarray,
    outdir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    path = outdir / "radial_error_vs_baseline.png"
    fig, (ax_o, ax_p) = plt.subplots(2, 1, figsize=(7.0, 7.2), sharex=True, constrained_layout=True)
    for label, comparison in comparisons.items():
        curves = comparison["radial_curves"]
        ax_o.plot(curves["radii"], curves["omega_rms"], label=label)
        ax_p.plot(curves["radii"], curves["psi_offset_removed_rms"], label=label)
    ax_o.set_ylabel("Omega RMS")
    ax_p.set_ylabel("psi RMS")
    ax_p.set_xlabel("r")
    ax_o.set_title("c4_100_c7_40 radial error vs fine baseline")
    ax_o.legend(fontsize=8)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    created.append(str(path.relative_to(REPO_ROOT)))

    for mode in (0, 4, 7):
        path = outdir / f"omega_mode_{mode}_amplitude.png"
        fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
        base_coeff = fourier_coefficients(baseline_omega)[mode]
        ax.plot(STRUCTURE_RADII, np.abs(base_coeff), label="baseline_h0p05_0p08")
        for label, omega in modal_sources.items():
            coeff = fourier_coefficients(omega)[mode]
            ax.plot(STRUCTURE_RADII, np.abs(coeff), label=label)
        ax.set_xlabel("r")
        ax.set_ylabel(f"|Omega mode {mode}|")
        ax.set_title(f"c4_100_c7_40 Omega Fourier mode {mode}")
        ax.legend(fontsize=8)
        fig.savefig(path, dpi=140)
        plt.close(fig)
        created.append(str(path.relative_to(REPO_ROOT)))

    if "nextgen_128_512" in modal_sources:
        omega_diff = modal_sources["nextgen_128_512"] - baseline_omega
        psi_placeholder = None
        path = outdir / "nextgen_128_omega_minus_baseline_polar.png"
        fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
        image = ax.imshow(
            omega_diff,
            origin="lower",
            aspect="auto",
            extent=(float(STRUCTURE_THETA[0]), float(STRUCTURE_THETA[-1]), float(STRUCTURE_RADII[0]), float(STRUCTURE_RADII[-1])),
            cmap="RdBu_r",
        )
        ax.set_xlabel("theta")
        ax.set_ylabel("r")
        ax.set_title("nextgen 128/512 Omega minus fine baseline")
        fig.colorbar(image, ax=ax)
        fig.savefig(path, dpi=140)
        plt.close(fig)
        created.append(str(path.relative_to(REPO_ROOT)))
        _ = psi_placeholder, baseline_psi

    return created


def append_registry(summary: dict[str, Any]) -> None:
    registry = REPO_ROOT / "experiments" / "registry.csv"
    fine = summary.get("structure_comparisons", {}).get("nextgen_128_512", {})
    seeded = summary.get("structure_comparisons", {}).get("baseline_seeded_nextgen_128_512", {})
    with registry.open("a", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                RUN_DATE,
                "m3-c4-c7-structure-diagnostics",
                "nextgen_vs_baseline",
                "experiments/run_m3_c4_c7_structure_diagnostics.py",
                "{c4=100,c7=40}",
                "nextgen_polar_full_disk_vs_baseline_disk_Rdom10",
                "nextgen_nested_72_96_128; baseline_h0.05_0.08; baseline_seeded_128",
                "profile_continuation_fv_upwind_nested_reduced_poisson_newton",
                "M3 branch/discretization diagnostic; no M4; no eigensolver",
                summary.get("runtime_summary", "n/a"),
                (
                    f"status={summary['status']}; "
                    f"boundary_omega_err={summary['boundary_convention_errors']['omega_dirichlet_max_abs']:.3e}; "
                    f"boundary_neumann_err={summary['boundary_convention_errors']['psi_neumann_max_abs']:.3e}; "
                    f"nextgen128_omega_rms={fine.get('global_probe_errors', {}).get('omega_probe_rms', 'n/a')}; "
                    f"baseline_seeded128_omega_rms={seeded.get('global_probe_errors', {}).get('omega_probe_rms', 'n/a')}"
                ),
                "n/a",
                summary["status"],
                "Structural c4/c7 diagnostic completed without starting M4 or the eigensolver.",
            ]
        )


def main() -> int:
    raw_root = REPO_ROOT / "experiments" / "raw" / "nextgen" / "m3_c4_c7_structure_diagnostics" / "c4_100_c7_40"
    summary_dir = REPO_ROOT / "experiments" / "summaries"
    raw_root.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    boundary_errors = boundary_convention_errors()
    print(f"[boundary] {boundary_errors}", flush=True)

    nextgen_runs: list[dict[str, Any]] = []
    seed_grid = NEXTGEN_GRIDS[0]
    print(f"[nextgen seed start] grid={seed_grid[0]}/{seed_grid[1]}", flush=True)
    seed = solve_grid(seed_grid, reduced=False, scheme="fv_upwind", upwind_blend=1.0, sigma_grid=SIGMA_GRID)
    nextgen_runs.append(seed)
    print(
        f"[nextgen seed done] grid={seed_grid[0]}/{seed_grid[1]} status={seed['status']} "
        f"runtime={seed.get('runtime_sec'):.3f}",
        flush=True,
    )

    source = seed
    for grid_spec in NEXTGEN_GRIDS[1:]:
        target_grid = PolarGrid(radius=PHYSICAL.Rdom, nr=grid_spec[0], ntheta=grid_spec[1])
        initial = prolong_state(source, target_grid)
        print(f"[nextgen nested reduced start] grid={grid_spec[0]}/{grid_spec[1]}", flush=True)
        run = solve_grid(
            grid_spec,
            reduced=True,
            scheme="fv_upwind",
            upwind_blend=1.0,
            initial_state=initial,
            sigma_grid=(1.0,),
        )
        nextgen_runs.append(run)
        print(
            f"[nextgen nested reduced done] grid={grid_spec[0]}/{grid_spec[1]} "
            f"status={run['status']} runtime={run.get('runtime_sec'):.3f}",
            flush=True,
        )
        if run["status"] == "completed":
            source = run

    print(f"[baseline start] h={BASELINE_MESH[0]}/{BASELINE_MESH[1]}", flush=True)
    baseline = run_baseline(BASELINE_MESH)
    print(
        f"[baseline done] status={baseline['status']} runtime={baseline.get('runtime_sec', 'n/a')}",
        flush=True,
    )
    if baseline["status"] == "completed":
        baseline["common_residuals"] = common_residual_from_sampler(
            lambda points, run=baseline: (
                sample_baseline(run["omega"], run["spaces"]["Vomega"], points),
                sample_baseline(run["psi"], run["spaces"]["Vpsi"], points),
            )
        )

    status = "completed" if baseline["status"] == "completed" and all(run["status"] == "completed" for run in nextgen_runs) else "partial"
    baseline_seeded_run: dict[str, Any] | None = None
    if status == "completed":
        grid = PolarGrid(radius=PHYSICAL.Rdom, nr=128, ntheta=512)
        initial = baseline_state_on_grid(baseline, grid)
        print("[baseline-seeded nextgen start] grid=128/512", flush=True)
        baseline_seeded_run = solve_grid(
            (128, 512),
            reduced=True,
            scheme="fv_upwind",
            upwind_blend=1.0,
            initial_state=initial,
            sigma_grid=(1.0,),
        )
        print(
            f"[baseline-seeded nextgen done] status={baseline_seeded_run['status']} "
            f"runtime={baseline_seeded_run.get('runtime_sec'):.3f}",
            flush=True,
        )
        if baseline_seeded_run["status"] != "completed":
            status = "partial"

    structure_comparisons: dict[str, Any] = {}
    compatibility: dict[str, Any] = {}
    poisson_checks: dict[str, Any] = {}
    mode_sources: dict[str, np.ndarray] = {}
    plot_paths: list[str] = []

    if baseline["status"] == "completed":
        baseline_omega, baseline_psi = sample_baseline_polar(baseline)
        baseline_structure = {
            "omega_modes": summarize_modes(baseline_omega),
            "psi_modes": summarize_modes(baseline_psi),
            "omega_integral": baseline_omega_integral(baseline),
            "boundary_error": baseline_boundary_error(baseline),
            "common_residuals": baseline.get("common_residuals", {}),
        }
        for run in nextgen_runs:
            if run["status"] != "completed":
                continue
            label = f"nextgen_{run['grid']['nr']}_{run['grid']['ntheta']}"
            omega, psi = sample_nextgen_polar(run)
            structure_comparisons[label] = structure_comparison(label, omega, psi, baseline_omega, baseline_psi)
            compatibility[label] = nextgen_compatibility(run)
            mode_sources[label] = omega
        if baseline_seeded_run is not None and baseline_seeded_run["status"] == "completed":
            omega, psi = sample_nextgen_polar(baseline_seeded_run)
            label = "baseline_seeded_nextgen_128_512"
            structure_comparisons[label] = structure_comparison(label, omega, psi, baseline_omega, baseline_psi)
            compatibility[label] = nextgen_compatibility(baseline_seeded_run)
            mode_sources[label] = omega
            if nextgen_runs[-1]["status"] == "completed":
                omega_high, psi_high = sample_nextgen_polar(nextgen_runs[-1])
                structure_comparisons["baseline_seeded_vs_nested_nextgen_128_512"] = structure_comparison(
                    "baseline_seeded_vs_nested_nextgen_128_512",
                    omega,
                    psi,
                    omega_high,
                    psi_high,
                )
        if nextgen_runs[-1]["status"] == "completed":
            poisson_checks["nextgen_128_poisson_from_baseline_omega"] = poisson_isolation(nextgen_runs[-1]["problem"], baseline)
        plot_paths = make_plots(structure_comparisons, mode_sources, baseline_omega, baseline_psi, raw_root)
    else:
        baseline_structure = {}

    runtime_pieces = [f"baseline={baseline.get('runtime_sec', 'n/a')}"]
    for run in nextgen_runs:
        runtime_pieces.append(f"nextgen_{run['grid']['nr']}_{run['grid']['ntheta']}={run.get('runtime_sec', 'n/a')}")
    if baseline_seeded_run is not None:
        runtime_pieces.append(f"baseline_seeded_128_512={baseline_seeded_run.get('runtime_sec', 'n/a')}")

    summary = {
        "status": status,
        "case": {"name": "c4_100_c7_40", "coeffs": COEFFS},
        "input": {
            "physical": {"mu": PHYSICAL.mu, "R": PHYSICAL.R, "Rdom": PHYSICAL.Rdom, "eps": PHYSICAL.eps},
            "baseline_mesh": {"h_min": BASELINE_MESH[0], "h_max": BASELINE_MESH[1]},
            "nextgen_grids": [{"nr": nr, "ntheta": ntheta} for nr, ntheta in NEXTGEN_GRIDS],
            "structure_radii": STRUCTURE_RADII,
            "structure_ntheta": int(STRUCTURE_THETA.size),
            "modes": MODES,
        },
        "boundary_convention_errors": boundary_errors,
        "baseline_structure": baseline_structure,
        "nextgen_runs": nextgen_runs,
        "baseline_run": baseline,
        "baseline_seeded_nextgen_run": baseline_seeded_run,
        "compatibility": compatibility,
        "poisson_checks": poisson_checks,
        "structure_comparisons": structure_comparisons,
        "plot_paths": plot_paths,
        "runtime_summary": "; ".join(runtime_pieces),
    }
    append_registry(summary)
    path = summary_dir / f"m3_c4_c7_structure_diagnostics_{RUN_DATE}.json"
    path.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    print(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
