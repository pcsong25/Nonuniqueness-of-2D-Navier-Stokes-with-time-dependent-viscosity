from __future__ import annotations

import argparse
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

from nextgen.boundary import FourierBoundaryData
from nextgen.config import PhysicalParams
from nextgen.diagnostics import sampled_strong_profile_residual
from nextgen.polar_profile import PolarGrid, PolarProfileProblem, PolarProfileResult
from run_m3_c3_140_symmetry_probe import (
    COEFFS,
    PHYSICAL,
    common_residual,
    compatibility_metrics,
    symmetry_metrics,
)
from run_m3_c3_140_nonuniqueness_validation import (
    baseline_symmetry_metrics,
    compare_nextgen_to_baseline,
    compare_runs,
    common_residual_baseline,
)
from run_m3_c3_140_refine192_fem_crosscheck import (
    compare_baseline_pair,
    run_baseline_seeded_linesearch,
    sample_baseline,
)


RUN_DATE = "2026-04-30"
RAW_ROOT = REPO_ROOT / "experiments" / "raw" / "nextgen" / "m3_c3_140_residual_audit"
SUMMARY_DIR = REPO_ROOT / "experiments" / "summaries"
FEM_CACHE = RAW_ROOT / "fem_cache"
FENICS_CACHE = RAW_ROOT / "fenics_cache"
PLOT_DIR = RAW_ROOT / "plots"
TABLE_DIR = RAW_ROOT / "tables"
FENICS_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(FENICS_CACHE)
SAMPLE_X = np.linspace(-7.0, 7.0, 193)
SAMPLE_Y = np.linspace(-7.0, 7.0, 193)
PLOT_X = np.linspace(-10.0, 10.0, 361)
PLOT_Y = np.linspace(-10.0, 10.0, 361)

NEXTGEN_SOURCES = {
    "160/480": {
        "symmetric": REPO_ROOT
        / "experiments/raw/nextgen/m3_c3_140_nonuniqueness_validation/states/symmetric_160_480_state.npz",
        "broken_m1": REPO_ROOT
        / "experiments/raw/nextgen/m3_c3_140_nonuniqueness_validation/states/broken_m1_160_480_state.npz",
    },
    "192/576": {
        "symmetric": REPO_ROOT
        / "experiments/raw/nextgen/m3_c3_140_refine192_fem_crosscheck/states/symmetric_192_576_state.npz",
        "broken_m1": REPO_ROOT
        / "experiments/raw/nextgen/m3_c3_140_refine192_fem_crosscheck/states/broken_m1_192_576_state.npz",
    },
}


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


def make_problem(grid_spec: tuple[int, int]) -> PolarProfileProblem:
    return PolarProfileProblem(
        PolarGrid(radius=PHYSICAL.Rdom, nr=grid_spec[0], ntheta=grid_spec[1]),
        PHYSICAL,
        FourierBoundaryData(M=64, coeffs=COEFFS),
        advection_scheme="fv_upwind",
        upwind_blend=1.0,
    )


def load_nextgen_state(path: Path, label: str, branch: str, grid_label: str) -> dict[str, Any]:
    data = np.load(path)
    problem = make_problem((int(data["grid_nr"]), int(data["grid_ntheta"])))
    omega = np.asarray(data["omega"], dtype=float).reshape(problem.grid.shape)
    psi = np.asarray(data["psi"], dtype=float).reshape(problem.grid.shape)
    sigma = float(data["sigma"]) if "sigma" in data.files else 1.0
    diagnostics = problem.residual_summary(omega, psi, sigma, 0.0)
    diagnostics.update(
        {
            "final_sigma": sigma,
            "grid_nr": problem.grid.nr,
            "grid_ntheta": problem.grid.ntheta,
            "grid_dr": problem.grid.dr,
            "grid_dtheta": problem.grid.dtheta,
        }
    )
    result = PolarProfileResult(
        omega=omega,
        psi=psi,
        grid=problem.grid,
        converged=True,
        iterations=(),
        runtime_sec=0.0,
        message="loaded_state",
        diagnostics=diagnostics,
    )
    run = {
        "label": label,
        "branch": branch,
        "discretization": "nextgen_fv",
        "grid_label": grid_label,
        "status": "completed",
        "problem": problem,
        "result": result,
        "diagnostics": diagnostics,
        "common_residuals": common_residual(problem, result),
        "compatibility": compatibility_metrics(problem, result),
        "symmetry": symmetry_metrics(problem, result),
        "boundary_defects": {
            "omega_dirichlet_max_abs": 0.0,
            "psi_neumann_max_abs": 0.0,
            "note": "polar FV boundary data are analytic RHS terms, not mutable state dofs",
        },
    }
    return run


def nextgen_plot_samples(run: dict[str, Any], x: np.ndarray = PLOT_X, y: np.ndarray = PLOT_Y) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(x, y, indexing="ij")
    omega, psi = run["problem"].sample_cartesian(run["result"], xx.reshape(-1), yy.reshape(-1))
    return omega.reshape(xx.shape), psi.reshape(xx.shape)


def common_grid_nextgen(run: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(SAMPLE_X, SAMPLE_Y, indexing="ij")
    omega, psi = run["problem"].sample_cartesian(run["result"], xx.reshape(-1), yy.reshape(-1))
    return omega.reshape(xx.shape), psi.reshape(xx.shape)


def fem_cache_path(branch: str) -> Path:
    return FEM_CACHE / f"baseline_h0p06_{branch}_audit.npz"


def load_cached_fem(branch: str) -> dict[str, Any] | None:
    path = fem_cache_path(branch)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "label": str(data["label"].item()),
        "branch": branch,
        "discretization": "baseline_fem",
        "status": str(data["status"].item()),
        "mesh": json.loads(str(data["mesh_json"].item())),
        "runtime_sec": float(data["runtime_sec"]),
        "newton_iterations": int(data["newton_iterations"]),
        "last_relative_update": float(data["last_relative_update"]),
        "last_linear_residual_norm": float(data["last_linear_residual_norm"]),
        "common_residuals": json.loads(str(data["common_residuals_json"].item())),
        "symmetry": json.loads(str(data["symmetry_json"].item())),
        "residuals": json.loads(str(data["residuals_json"].item())),
        "boundary_defects": json.loads(str(data["boundary_defects_json"].item())),
        "compatibility": json.loads(str(data["compatibility_json"].item())),
        "comparison_to_seed_nextgen": json.loads(str(data["comparison_to_seed_nextgen_json"].item())),
        "plot_x": data["plot_x"],
        "plot_y": data["plot_y"],
        "plot_omega": data["plot_omega"],
        "plot_psi": data["plot_psi"],
        "sample_x": data["sample_x"],
        "sample_y": data["sample_y"],
        "sample_omega": data["sample_omega"],
        "sample_psi": data["sample_psi"],
    }


def fem_boundary_defects(run: dict[str, Any]) -> dict[str, float]:
    omega = run["omega"]
    psi = run["psi"]
    spaces = run["spaces"]
    baseline_dir = SRC / "baseline"
    sys.path.insert(0, str(baseline_dir))
    from dolfinx import fem
    from boundary_data_fourier import FourierBC

    fourier_bc = FourierBC(64, COEFFS)
    facet_tags = run["facet_tags"]
    outer_tag = 1
    mesh = spaces["Vomega"].mesh
    bfacets = facet_tags.find(outer_tag)
    omega_bdofs = fem.locate_dofs_topological(spaces["Vomega"], mesh.topology.dim - 1, bfacets)
    xy_o = spaces["Vomega"].tabulate_dof_coordinates()[omega_bdofs, :2]
    theta_o = np.arctan2(xy_o[:, 1], xy_o[:, 0])
    expected_omega = PHYSICAL.R ** (-1.0 / PHYSICAL.mu) * fourier_bc.Omega_bar(theta_o)
    omega_vals = np.asarray(omega.x.array, dtype=float)[omega_bdofs]
    omega_diff = omega_vals - expected_omega

    psi_bdofs = fem.locate_dofs_topological(spaces["Vpsi"], mesh.topology.dim - 1, bfacets)
    xy_p = spaces["Vpsi"].tabulate_dof_coordinates()[psi_bdofs, :2]
    theta_p = np.arctan2(xy_p[:, 1], xy_p[:, 0])
    expected_flux = fourier_bc.b_theta(theta_p, mu=PHYSICAL.mu, R=PHYSICAL.R, Rdom=PHYSICAL.Rdom, sigma=1.0)
    return {
        "omega_dirichlet_max_abs": float(np.max(np.abs(omega_diff))) if omega_diff.size else float("nan"),
        "omega_dirichlet_rms": float(np.sqrt(np.mean(omega_diff**2))) if omega_diff.size else float("nan"),
        "psi_neumann_data_max_abs": float(np.max(np.abs(expected_flux))) if expected_flux.size else float("nan"),
        "note": "FEM Neumann data are natural flux data; max_abs reports prescribed flux magnitude, not a pointwise recovered flux error.",
    }


def fem_compatibility(run: dict[str, Any]) -> dict[str, float]:
    baseline_dir = SRC / "baseline"
    sys.path.insert(0, str(baseline_dir))
    import ufl
    from dolfinx import fem
    from boundary_data_fourier import FourierBC

    mesh = run["spaces"]["Vomega"].mesh
    omega_int = float(fem.assemble_scalar(fem.form(run["omega"] * ufl.dx)))
    fourier_bc = FourierBC(64, COEFFS)
    facet_tags = run["facet_tags"]
    psi_bdofs = fem.locate_dofs_topological(run["spaces"]["Vpsi"], mesh.topology.dim - 1, facet_tags.find(1))
    xy = run["spaces"]["Vpsi"].tabulate_dof_coordinates()[psi_bdofs, :2]
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    g_vals = fourier_bc.b_theta(theta, mu=PHYSICAL.mu, R=PHYSICAL.R, Rdom=PHYSICAL.Rdom, sigma=1.0)
    flux = float(np.sum(g_vals) * (2.0 * np.pi * PHYSICAL.Rdom) / max(theta.size, 1))
    area = np.pi * PHYSICAL.Rdom * PHYSICAL.Rdom
    return {
        "omega_integral": omega_int,
        "omega_mean": omega_int / area,
        "neumann_flux_discrete": flux,
        "omega_minus_flux": omega_int - flux,
    }


def run_or_load_fem(branch: str, seed: dict[str, Any], *, refresh: bool) -> dict[str, Any]:
    if not refresh:
        cached = load_cached_fem(branch)
        if cached is not None:
            return cached
    FEM_CACHE.mkdir(parents=True, exist_ok=True)
    run = run_baseline_seeded_linesearch(seed, (0.06, 0.06))
    if run.get("status") != "completed":
        return {"branch": branch, "discretization": "baseline_fem", **sanitize(run)}
    boundary = {
        "omega_dirichlet_max_abs": float("nan"),
        "omega_dirichlet_rms": float("nan"),
        "psi_neumann_data_max_abs": float("nan"),
        "note": "facet tags unavailable; FEM boundary defects not evaluated",
    }
    compatibility = {"omega_integral": float("nan"), "omega_mean": float("nan"), "neumann_flux_discrete": float("nan"), "omega_minus_flux": float("nan")}
    if run.get("facet_tags") is not None:
        boundary = fem_boundary_defects(run)
        compatibility = fem_compatibility(run)

    xx, yy = np.meshgrid(PLOT_X, PLOT_Y, indexing="ij")
    points_plot = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    plot_omega = sample_baseline(run["omega"], run["spaces"]["Vomega"], points_plot).reshape(xx.shape)
    plot_psi = sample_baseline(run["psi"], run["spaces"]["Vpsi"], points_plot).reshape(xx.shape)
    sx, sy = np.meshgrid(SAMPLE_X, SAMPLE_Y, indexing="ij")
    points_sample = np.column_stack((sx.reshape(-1), sy.reshape(-1)))
    sample_omega = sample_baseline(run["omega"], run["spaces"]["Vomega"], points_sample).reshape(sx.shape)
    sample_psi = sample_baseline(run["psi"], run["spaces"]["Vpsi"], points_sample).reshape(sx.shape)
    cache_payload = {
        "label": run["label"],
        "status": run["status"],
        "mesh_json": json.dumps(run["mesh"]),
        "runtime_sec": run["runtime_sec"],
        "newton_iterations": run["newton_iterations"],
        "last_relative_update": run["last_relative_update"],
        "last_linear_residual_norm": run["last_linear_residual_norm"],
        "common_residuals_json": json.dumps(sanitize(run["common_residuals"])),
        "symmetry_json": json.dumps(sanitize(run["symmetry"])),
        "residuals_json": json.dumps(sanitize(run["residuals"])),
        "boundary_defects_json": json.dumps(boundary),
        "compatibility_json": json.dumps(compatibility),
        "comparison_to_seed_nextgen_json": json.dumps(run["comparison_to_seed_nextgen"]),
        "plot_x": PLOT_X,
        "plot_y": PLOT_Y,
        "plot_omega": plot_omega,
        "plot_psi": plot_psi,
        "sample_x": SAMPLE_X,
        "sample_y": SAMPLE_Y,
        "sample_omega": sample_omega,
        "sample_psi": sample_psi,
    }
    np.savez_compressed(fem_cache_path(branch), **cache_payload)
    return load_cached_fem(branch) or {"branch": branch, "discretization": "baseline_fem", **sanitize(run)}


def separation_from_samples(a_omega: np.ndarray, a_psi: np.ndarray, b_omega: np.ndarray, b_psi: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(a_omega) & np.isfinite(a_psi) & np.isfinite(b_omega) & np.isfinite(b_psi)
    domega = a_omega[valid] - b_omega[valid]
    dpsi = a_psi[valid] - b_psi[valid]
    psi_offset = float(np.mean(dpsi)) if dpsi.size else float("nan")
    if dpsi.size:
        dpsi = dpsi - psi_offset
    return {
        "valid_count": int(np.count_nonzero(valid)),
        "omega_rms": float(np.sqrt(np.mean(domega**2))) if domega.size else float("nan"),
        "omega_linf": float(np.max(np.abs(domega))) if domega.size else float("nan"),
        "psi_mean_offset": psi_offset,
        "psi_rms_offset_removed": float(np.sqrt(np.mean(dpsi**2))) if dpsi.size else float("nan"),
        "psi_linf_offset_removed": float(np.max(np.abs(dpsi))) if dpsi.size else float("nan"),
    }


def audit_row(run: dict[str, Any]) -> dict[str, Any]:
    native = run.get("diagnostics", {})
    weak = run.get("residuals", {}).get("norms", {})
    common = run.get("common_residuals", {})
    sym = run.get("symmetry", {})
    comp = run.get("compatibility", {})
    bdy = run.get("boundary_defects", {})
    return {
        "label": run.get("label"),
        "discretization": run.get("discretization"),
        "grid_or_mesh": run.get("grid_label") or json.dumps(run.get("mesh", {}), sort_keys=True),
        "branch": run.get("branch"),
        "native_relative_residual": native.get("relative_residual"),
        "native_residual_norm": native.get("residual_norm"),
        "fem_weak_omega_l2": weak.get("Omega_l2"),
        "fem_weak_psi_l2": weak.get("Psi_l2"),
        "common_profile_strong_relative": common.get("profile_strong_relative"),
        "common_omega_strong_l2": common.get("omega_strong_l2"),
        "common_poisson_strong_l2": common.get("poisson_strong_l2"),
        "omega_non3": sym.get("omega_non_threefold_fraction"),
        "psi_non3": sym.get("psi_non_threefold_fraction"),
        "omega_integral": comp.get("omega_integral"),
        "neumann_flux_discrete": comp.get("neumann_flux_discrete"),
        "omega_minus_flux": comp.get("omega_minus_flux"),
        "physical_poisson_residual_norm": comp.get("physical_poisson_residual_norm"),
        "omega_boundary_max_abs_error": bdy.get("omega_dirichlet_max_abs"),
        "psi_neumann_boundary_max_abs_error": bdy.get("psi_neumann_max_abs"),
        "runtime_sec": run.get("runtime_sec"),
        "newton_iterations": run.get("newton_iterations"),
        "last_relative_update": run.get("last_relative_update"),
        "status": run.get("status"),
    }


def write_table(rows: list[dict[str, Any]]) -> str:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / "c3_140_residual_audit_table.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path.relative_to(REPO_ROOT))


def make_plots(nextgen_192: dict[str, dict[str, Any]], fem_runs: dict[str, dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    ng_sym_o, _ = nextgen_plot_samples(nextgen_192["symmetric"])
    ng_bro_o, _ = nextgen_plot_samples(nextgen_192["broken_m1"])
    fem_sym_o = fem_runs["symmetric"]["plot_omega"]
    fem_bro_o = fem_runs["broken_m1"]["plot_omega"]
    fields = [
        ("nextgen symmetric Omega", ng_sym_o),
        ("nextgen broken Omega", ng_bro_o),
        ("FEM symmetric Omega", fem_sym_o),
        ("FEM broken Omega", fem_bro_o),
        ("nextgen broken - symmetric", ng_bro_o - ng_sym_o),
        ("FEM broken - symmetric", fem_bro_o - fem_sym_o),
        ("FEM symmetric - nextgen symmetric", fem_sym_o - ng_sym_o),
        ("FEM broken - nextgen broken", fem_bro_o - ng_bro_o),
    ]
    extent = (-PHYSICAL.Rdom, PHYSICAL.Rdom, -PHYSICAL.Rdom, PHYSICAL.Rdom)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), constrained_layout=True)
    for ax, (title, values) in zip(axes.ravel(), fields):
        image = ax.imshow(values.T, origin="lower", extent=extent, cmap="RdBu_r")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    path = PLOT_DIR / "omega_nextgen_fem_branch_comparison.png"
    fig.savefig(path, dpi=145)
    plt.close(fig)
    created.append(str(path.relative_to(REPO_ROOT)))

    # Same comparison with common color scale for the two branch-difference panels.
    diff_ng = ng_bro_o - ng_sym_o
    diff_fem = fem_bro_o - fem_sym_o
    vmax = float(np.nanmax(np.abs(np.concatenate((diff_ng.reshape(-1), diff_fem.reshape(-1))))))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for ax, title, values in (
        (axes[0], "nextgen broken - symmetric", diff_ng),
        (axes[1], "FEM broken - symmetric", diff_fem),
    ):
        image = ax.imshow(values.T, origin="lower", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_aspect("equal")
        ax.set_title(title)
        fig.colorbar(image, ax=ax)
    path = PLOT_DIR / "omega_branch_difference_common_scale.png"
    fig.savefig(path, dpi=145)
    plt.close(fig)
    created.append(str(path.relative_to(REPO_ROOT)))
    return created


def append_registry(summary: dict[str, Any]) -> None:
    registry = REPO_ROOT / "experiments" / "registry.csv"
    with registry.open("a", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                RUN_DATE,
                "m3-c3-140-residual-audit-validation-table",
                "nextgen_vs_baseline",
                "experiments/run_m3_c3_140_residual_audit.py",
                "{c3=140}",
                "nextgen_polar_full_disk_Rdom10_and_baseline_fem_disk_Rdom10",
                "nextgen_160/480; nextgen_192/576; baseline_h0.06",
                "residual_audit_validation_table_and_plots",
                "M3 c3=140 residual audit and branch validation; no M4; no eigensolver",
                summary.get("runtime_sec", "n/a"),
                (
                    f"rows={len(summary.get('audit_rows', []))}; "
                    f"sep_nextgen192={summary.get('branch_separations', {}).get('nextgen_192_576', {}).get('omega_probe_rms')}; "
                    f"sep_fem={summary.get('branch_separations', {}).get('fem_h0p06', {}).get('omega_rms')}; "
                    f"plots={len(summary.get('plot_paths', []))}"
                ),
                "n/a",
                "completed",
                "Residual audit table and comparison plots generated.",
            ]
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit residuals/norms for c3=140 nonuniqueness branches.")
    parser.add_argument("--refresh-fem", action="store_true", help="rerun h=0.06 FEM branch solves even if cached samples exist")
    args = parser.parse_args()

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    nextgen: dict[str, dict[str, dict[str, Any]]] = {}
    for grid_label, branches in NEXTGEN_SOURCES.items():
        nextgen[grid_label] = {}
        for branch, path in branches.items():
            nextgen[grid_label][branch] = load_nextgen_state(path, path.stem.replace("_state", ""), branch, grid_label)

    fem_runs = {
        "symmetric": run_or_load_fem("symmetric", nextgen["192/576"]["symmetric"], refresh=args.refresh_fem),
        "broken_m1": run_or_load_fem("broken_m1", nextgen["192/576"]["broken_m1"], refresh=args.refresh_fem),
    }

    rows = []
    for grid_label in ("160/480", "192/576"):
        for branch in ("symmetric", "broken_m1"):
            rows.append(audit_row(nextgen[grid_label][branch]))
    for branch in ("symmetric", "broken_m1"):
        rows.append(audit_row(fem_runs[branch]))

    branch_separations: dict[str, Any] = {
        "nextgen_160_480": compare_runs(nextgen["160/480"]["symmetric"], nextgen["160/480"]["broken_m1"]),
        "nextgen_192_576": compare_runs(nextgen["192/576"]["symmetric"], nextgen["192/576"]["broken_m1"]),
    }
    if all(fem_runs[b].get("status") == "completed" for b in ("symmetric", "broken_m1")):
        branch_separations["fem_h0p06"] = separation_from_samples(
            fem_runs["symmetric"]["sample_omega"],
            fem_runs["symmetric"]["sample_psi"],
            fem_runs["broken_m1"]["sample_omega"],
            fem_runs["broken_m1"]["sample_psi"],
        )
        for branch in ("symmetric", "broken_m1"):
            ng_o, ng_p = common_grid_nextgen(nextgen["192/576"][branch])
            branch_separations[f"nextgen192_vs_fem_{branch}"] = separation_from_samples(
                ng_o,
                ng_p,
                fem_runs[branch]["sample_omega"],
                fem_runs[branch]["sample_psi"],
            )

    table_path = write_table(rows)
    plot_paths = make_plots(nextgen["192/576"], fem_runs) if all(fem_runs[b].get("status") == "completed" for b in ("symmetric", "broken_m1")) else []
    definitions = {
        "nextgen_native_residual": "weighted discrete FV residual norm sqrt(||R_omega||_w^2 + ||R_poisson||_w^2) divided by max(||Omega||_w, ||Lap psi||_w)",
        "fem_weak_residual": "raw PETSc l2 norms of assembled Galerkin residual vectors R_Omega(w) and R_psi(phi)",
        "common_sampled_strong_residual": "sample Omega, psi on common Cartesian grid and evaluate centered-difference strong residuals R_Omega=v.grad(Omega)-Omega-eps Lap Omega and R_Psi=Lap psi-Omega; relative norm divides by max(||Omega||_L2, ||Lap psi||_L2)",
        "omega_non3": "||Omega - (Omega + rotate_120 Omega + rotate_240 Omega)/3||_weighted / ||Omega||_weighted",
        "compatibility_defect": "integral_disk Omega minus discrete outer Neumann flux for psi",
        "branch_separation": "RMS and Linf differences on common polar or Cartesian probes; psi differences have their mean offset removed",
    }
    summary = {
        "status": "completed",
        "case": {"coeffs": COEFFS, "name": "c3_140_residual_audit"},
        "audit_rows": rows,
        "branch_separations": branch_separations,
        "definitions": definitions,
        "table_path": table_path,
        "plot_paths": plot_paths,
        "fem_cache_dir": str(FEM_CACHE.relative_to(REPO_ROOT)),
        "notes": [
            "FEM field arrays are regenerated and cached because previous h=0.06 summaries intentionally sanitized non-serializable FEM Function objects.",
            "FEM boundary Neumann data are natural boundary data; this audit reports the prescribed flux scale and weak residual, not a pointwise normal-derivative recovery error.",
        ],
    }
    path = SUMMARY_DIR / f"m3_c3_140_residual_audit_{RUN_DATE}.json"
    path.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    append_registry(summary)
    print(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
