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
from scipy.interpolate import RegularGridInterpolator

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-euler")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from run_m3_c3_140_baseline_closer_fv_homotopy import (
    TransportSpec,
    make_problem as make_nextgen_problem,
    result_from_arrays as nextgen_result_from_arrays,
)
from run_m3_c3_140_nonuniqueness_validation import (
    PROBE_RADII,
    PROBE_THETA,
    SAMPLE_X,
    SAMPLE_Y,
    baseline_symmetry_metrics,
    common_residual_baseline,
    compare_nextgen_to_baseline,
    sample_baseline,
)
from run_m3_c3_140_symmetry_probe import COEFFS, PHYSICAL


RUN_DATE = "2026-05-01"
RAW_ROOT = REPO_ROOT / "experiments" / "raw" / "baseline" / "m3_c3_140_fem_branch_discovery"
TABLE_DIR = RAW_ROOT / "tables"
FIELD_DIR = RAW_ROOT / "fields"
PLOT_DIR = RAW_ROOT / "plots"
SUMMARY_DIR = REPO_ROOT / "experiments" / "summaries"

DEFAULT_MESH = (0.06, 0.06)
DEFAULT_CONFIRMATION_MESH = (0.05, 0.05)
DEFAULT_SIGMAS = (1.0, 0.9, 0.8, 0.7, 0.65, 0.64, 0.63, 0.62, 0.6, 0.575, 0.55)
DEFAULT_AMPLITUDES = (0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0)
M1_PHASES = (0.0, float(np.pi / 3.0), float(2.0 * np.pi / 3.0))
SECONDARY_PHASES = (0.0,)
PERTURBATION_WIDTH = 2.2
SYMMETRIC_H0P06_CACHE = (
    REPO_ROOT
    / "experiments/raw/nextgen/m3_c3_140_residual_audit/fem_cache/baseline_h0p06_symmetric_audit.npz"
)

SELECTED_FV_STATES = {
    "symmetric_fv_blend_0p125": {
        "path": REPO_ROOT
        / "experiments/raw/nextgen/m3_c3_140_baseline_closer_fv_homotopy/states/symmetric_fv_blend_0p125_192_576_state.npz",
        "branch": "symmetric",
        "transport": TransportSpec("fv_blend_0p125", "fv_blended", 0.125),
    },
    "broken_m1_fv_blend_0p015625_tight": {
        "path": REPO_ROOT
        / "experiments/raw/nextgen/m3_c3_140_selected_branch_audit/states/broken_m1_fv_blend_0p015625_tight_192_576_state.npz",
        "branch": "broken_m1",
        "transport": TransportSpec("fv_blend_0p015625", "fv_blended", 0.015625),
    },
}


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        blocked = {
            "context",
            "problem",
            "result",
            "spaces",
            "omega",
            "psi",
            "rOmega",
            "rPsi",
            "facet_tags",
            "mesh",
            "poisson",
            "assembler",
            "continuation",
            "fourier_bc",
        }
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


def mesh_label(mesh: tuple[float, float]) -> str:
    return f"h{mesh[0]:g}_{mesh[1]:g}".replace(".", "p")


def sigma_label(sigma: float) -> str:
    return f"sigma{float(sigma):g}".replace(".", "p")


def build_context(
    mesh_spec: tuple[float, float],
    *,
    sigma_grid: tuple[float, ...],
    max_newton: int,
    tol_newton: float,
) -> dict[str, Any]:
    baseline_dir = SRC / "baseline"
    sys.path.insert(0, str(baseline_dir))
    from config import SimulationParams
    from mesh import build_disk
    from spaces import create_spaces
    from boundary_data_fourier import FourierBC
    from poisson_neumann import PoissonNeumann
    from newton_block import NewtonAssembler
    from continuation import ContinuationSolver
    from residuals_quick import residuals_quick

    h_min, h_max = mesh_spec
    params = SimulationParams(
        mu=PHYSICAL.mu,
        R=PHYSICAL.R,
        Rdom=PHYSICAL.Rdom,
        M=64,
        tol_newton=float(tol_newton),
        max_newton=int(max_newton),
        sigma_grid=tuple(float(v) for v in sigma_grid),
        h_min=float(h_min),
        h_max=float(h_max),
    )
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
    return {
        "params": params,
        "mesh": mesh,
        "facet_tags": facet_tags,
        "spaces": spaces,
        "fourier_bc": fourier_bc,
        "poisson": poisson,
        "assembler": assembler,
        "continuation": continuation,
        "residuals_quick": residuals_quick,
    }


def vec_data(function: Any) -> np.ndarray:
    try:
        return function.vector.array
    except AttributeError:
        return function.x.array


def ghost(function: Any) -> None:
    baseline_dir = SRC / "baseline"
    sys.path.insert(0, str(baseline_dir))
    from petsc4py import PETSc

    try:
        function.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    except AttributeError:
        if hasattr(function.x, "scatter_forward"):
            function.x.scatter_forward()


def copy_function(function: Any, space: Any) -> Any:
    baseline_dir = SRC / "baseline"
    sys.path.insert(0, str(baseline_dir))
    from dolfinx import fem

    out = fem.Function(space)
    vec_data(out)[: vec_data(function).size] = vec_data(function)
    ghost(out)
    return out


def fem_boundary_defects(run: dict[str, Any], sigma: float) -> dict[str, Any]:
    baseline_dir = SRC / "baseline"
    sys.path.insert(0, str(baseline_dir))
    from dolfinx import fem

    spaces = run["spaces"]
    facet_tags = run["facet_tags"]
    mesh = spaces["Vomega"].mesh
    bfacets = facet_tags.find(1)
    omega_bdofs = fem.locate_dofs_topological(spaces["Vomega"], mesh.topology.dim - 1, bfacets)
    xy_o = spaces["Vomega"].tabulate_dof_coordinates()[omega_bdofs, :2]
    theta_o = np.arctan2(xy_o[:, 1], xy_o[:, 0])
    expected_omega = (
        PHYSICAL.R ** (-1.0 / PHYSICAL.mu)
        * run["fourier_bc"].Omega_bar(theta_o)
        * float(sigma)
    )
    omega_vals = np.asarray(run["omega"].x.array, dtype=float)[omega_bdofs]
    omega_diff = omega_vals - expected_omega

    psi_bdofs = fem.locate_dofs_topological(spaces["Vpsi"], mesh.topology.dim - 1, bfacets)
    xy_p = spaces["Vpsi"].tabulate_dof_coordinates()[psi_bdofs, :2]
    theta_p = np.arctan2(xy_p[:, 1], xy_p[:, 0])
    expected_flux = run["fourier_bc"].b_theta(
        theta_p,
        mu=PHYSICAL.mu,
        R=PHYSICAL.R,
        Rdom=PHYSICAL.Rdom,
        sigma=float(sigma),
    )
    return {
        "omega_dirichlet_max_abs": float(np.max(np.abs(omega_diff))) if omega_diff.size else float("nan"),
        "omega_dirichlet_rms": float(np.sqrt(np.mean(omega_diff**2))) if omega_diff.size else float("nan"),
        "psi_neumann_data_max_abs": float(np.max(np.abs(expected_flux))) if expected_flux.size else float("nan"),
        "note": "FEM Neumann data are natural flux data; max_abs is prescribed flux magnitude, not recovered pointwise flux error.",
    }


def fem_compatibility(run: dict[str, Any], sigma: float) -> dict[str, float]:
    baseline_dir = SRC / "baseline"
    sys.path.insert(0, str(baseline_dir))
    import ufl
    from dolfinx import fem

    mesh = run["spaces"]["Vomega"].mesh
    omega_int = float(fem.assemble_scalar(fem.form(run["omega"] * ufl.dx)))
    facet_tags = run["facet_tags"]
    psi_bdofs = fem.locate_dofs_topological(run["spaces"]["Vpsi"], mesh.topology.dim - 1, facet_tags.find(1))
    xy = run["spaces"]["Vpsi"].tabulate_dof_coordinates()[psi_bdofs, :2]
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    g_vals = run["fourier_bc"].b_theta(
        theta,
        mu=PHYSICAL.mu,
        R=PHYSICAL.R,
        Rdom=PHYSICAL.Rdom,
        sigma=float(sigma),
    )
    flux = float(np.sum(g_vals) * (2.0 * np.pi * PHYSICAL.Rdom) / max(theta.size, 1))
    area = np.pi * PHYSICAL.Rdom * PHYSICAL.Rdom
    return {
        "omega_integral": omega_int,
        "omega_mean": omega_int / area,
        "neumann_flux_discrete": flux,
        "omega_minus_flux": omega_int - flux,
    }


def finish_fem_run(
    context: dict[str, Any],
    *,
    omega: Any,
    psi: Any,
    label: str,
    branch: str,
    sigma: float,
    status: str,
    runtime_sec: float,
    newton_iterations: int | None,
    last_relative_update: float | None,
    last_linear_residual_norm: float | None,
    history: list[dict[str, Any]] | None = None,
    message: str = "",
    perturbation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = context["params"]
    mesh = context["mesh"]
    spaces = context["spaces"]
    run = {
        "label": label,
        "branch": branch,
        "discretization": "baseline_fem",
        "status": status,
        "runtime_sec": float(runtime_sec),
        "mesh_spec": {"h_min": float(params.h_min), "h_max": float(params.h_max)},
        "mesh_cells": int(mesh.topology.index_map(mesh.topology.dim).size_local),
        "mesh_vertices": int(mesh.topology.index_map(0).size_local),
        "sigma": float(sigma),
        "newton_iterations": newton_iterations,
        "last_relative_update": last_relative_update,
        "last_linear_residual_norm": last_linear_residual_norm,
        "history": history or [],
        "message": message,
        "spaces": spaces,
        "facet_tags": context["facet_tags"],
        "fourier_bc": context["fourier_bc"],
        "omega": omega,
        "psi": psi,
        "perturbation": perturbation or {},
    }
    try:
        run["residuals"] = context["residuals_quick"](
            mesh,
            spaces,
            omega,
            psi,
            mu=params.mu,
            R=params.R,
            Rdom=params.Rdom,
            facet_tags=context["facet_tags"],
            outer_tag=1,
            fourier_bc=context["fourier_bc"],
            sigma=float(sigma),
        )
        run["common_residuals"] = common_residual_baseline(run)
        run["symmetry"] = baseline_symmetry_metrics(run)
        run["boundary_defects"] = fem_boundary_defects(run, sigma)
        run["compatibility"] = fem_compatibility(run, sigma)
    except Exception as exc:
        run["diagnostic_error"] = repr(exc)
        run["diagnostic_traceback"] = traceback.format_exc()
    return run


def continuation_grid(target_sigma: float, max_step: float = 0.025) -> tuple[float, ...]:
    if target_sigma <= 0.0:
        return (0.0,)
    n = max(int(np.ceil(float(target_sigma) / float(max_step))), 1)
    return tuple(float(v) for v in np.linspace(0.0, float(target_sigma), n + 1))


def run_symmetric_root(
    mesh_spec: tuple[float, float],
    sigma: float,
    *,
    max_newton: int,
    tol_newton: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    t0 = time.perf_counter()
    try:
        context = build_context(
            mesh_spec,
            sigma_grid=continuation_grid(float(sigma)),
            max_newton=max_newton,
            tol_newton=tol_newton,
        )
    except Exception as exc:
        return {}, {
            "label": f"fem_symmetric_{mesh_label(mesh_spec)}_{sigma_label(sigma)}",
            "branch": "symmetric",
            "discretization": "baseline_fem",
            "status": "blocked_import_or_setup",
            "runtime_sec": time.perf_counter() - t0,
            "mesh_spec": {"h_min": float(mesh_spec[0]), "h_max": float(mesh_spec[1])},
            "sigma": float(sigma),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    try:
        omega, psi = context["continuation"].run(context["mesh"], context["spaces"])
        runtime = time.perf_counter() - t0
        run = finish_fem_run(
            context,
            omega=omega,
            psi=psi,
            label=f"fem_symmetric_{mesh_label(mesh_spec)}_{sigma_label(sigma)}",
            branch="symmetric",
            sigma=sigma,
            status="completed",
            runtime_sec=runtime,
            newton_iterations=None,
            last_relative_update=None,
            last_linear_residual_norm=None,
            message="zero_start_sigma_continuation",
        )
        return context, run
    except Exception as exc:
        runtime = time.perf_counter() - t0
        run = {
            "label": f"fem_symmetric_{mesh_label(mesh_spec)}_{sigma_label(sigma)}",
            "branch": "symmetric",
            "discretization": "baseline_fem",
            "status": "failed",
            "runtime_sec": runtime,
            "mesh_spec": {"h_min": float(mesh_spec[0]), "h_max": float(mesh_spec[1])},
            "sigma": float(sigma),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        return context, run


def cached_omega_function(context: dict[str, Any], cache_path: Path, sigma: float) -> Any:
    baseline_dir = SRC / "baseline"
    sys.path.insert(0, str(baseline_dir))
    from dolfinx import fem

    data = np.load(cache_path, allow_pickle=True)
    sample_x = np.asarray(data["sample_x"], dtype=float)
    sample_y = np.asarray(data["sample_y"], dtype=float)
    sample_omega = np.asarray(data["sample_omega"], dtype=float)
    interpolator = RegularGridInterpolator(
        (sample_x, sample_y),
        sample_omega,
        bounds_error=False,
        fill_value=np.nan,
    )
    coords_raw = context["spaces"]["Vomega"].tabulate_dof_coordinates()
    coords = coords_raw[:, :2] if coords_raw.ndim > 1 else coords_raw.reshape(-1, 3)[:, :2]
    values = interpolator(coords)
    values = np.where(np.isfinite(values), values, 0.0)
    omega = fem.Function(context["spaces"]["Vomega"])
    vec_data(omega)[: values.size] = values
    ghost(omega)
    context["continuation"]._apply_omega_dirichlet_on_boundary(omega, context["spaces"]["Vomega"], float(sigma))
    return omega


def run_cached_symmetric_root(
    mesh_spec: tuple[float, float],
    sigma: float,
    *,
    max_newton: int,
    tol_newton: float,
    ls_max_steps: int,
    ls_shrink: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    t0 = time.perf_counter()
    try:
        if abs(mesh_spec[0] - 0.06) > 1e-12 or abs(mesh_spec[1] - 0.06) > 1e-12 or abs(float(sigma) - 1.0) > 1e-12:
            raise ValueError("cached_h0p06_symmetric source is only valid for mesh 0.06/0.06 at sigma=1.0")
        if not SYMMETRIC_H0P06_CACHE.exists():
            raise FileNotFoundError(SYMMETRIC_H0P06_CACHE)
        context = build_context(
            mesh_spec,
            sigma_grid=(float(sigma),),
            max_newton=max_newton,
            tol_newton=tol_newton,
        )
        omega = cached_omega_function(context, SYMMETRIC_H0P06_CACHE, sigma)
        psi = context["poisson"].solve_state(omega, sigma=float(sigma))
        run = fixed_sigma_newton(
            context,
            omega=omega,
            psi=psi,
            sigma=sigma,
            label=f"fem_symmetric_cached_h0p06_{sigma_label(sigma)}",
            branch="symmetric",
            max_newton=max_newton,
            tol_newton=tol_newton,
            ls_max_steps=ls_max_steps,
            ls_shrink=ls_shrink,
            perturbation={"source": str(SYMMETRIC_H0P06_CACHE.relative_to(REPO_ROOT)), "type": "cached_fem_symmetric"},
        )
        run["runtime_sec"] = time.perf_counter() - t0
        run["message"] = "cached_h0p06_symmetric_fixed_sigma_polish"
        return context, run
    except Exception as exc:
        return {}, {
            "label": f"fem_symmetric_cached_h0p06_{sigma_label(sigma)}",
            "branch": "symmetric",
            "discretization": "baseline_fem",
            "status": "failed",
            "runtime_sec": time.perf_counter() - t0,
            "mesh_spec": {"h_min": float(mesh_spec[0]), "h_max": float(mesh_spec[1])},
            "sigma": float(sigma),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def perturbation_specs(
    amplitudes: tuple[float, ...],
    *,
    families: tuple[str, ...] = ("m1_core", "m2_core", "mixed_non3"),
    phases: tuple[float, ...] | None = None,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if "m1_core" in families:
        for amp in amplitudes:
            for phase in (M1_PHASES if phases is None else phases):
                specs.append({"family": "m1_core", "mode": 1, "amplitude": float(amp), "phase": float(phase)})
    for family, mode in (("m2_core", 2), ("mixed_non3", None)):
        if family not in families:
            continue
        for amp in amplitudes:
            for phase in (SECONDARY_PHASES if phases is None else phases):
                specs.append({"family": family, "mode": mode, "amplitude": float(amp), "phase": float(phase)})
    return specs


def analytic_perturbation_values(coords: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    x = coords[:, 0]
    y = coords[:, 1]
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    rho = np.clip(r / PHYSICAL.Rdom, 0.0, 1.0)
    envelope = r * np.exp(-((r / PERTURBATION_WIDTH) ** 2)) * (1.0 - rho**2) ** 2
    amp = float(spec["amplitude"])
    phase = float(spec.get("phase", 0.0))
    family = str(spec["family"])
    if family == "mixed_non3":
        angular = (
            np.cos(theta + phase)
            + 0.5 * np.cos(2.0 * theta + 0.37 + phase)
            + 0.25 * np.cos(4.0 * theta + 0.11)
        )
    else:
        mode = int(spec["mode"])
        angular = np.cos(float(mode) * theta + phase)
    return amp * envelope * angular


def perturbed_omega_from_symmetric(context: dict[str, Any], symmetric: dict[str, Any], spec: dict[str, Any], sigma: float) -> Any:
    omega = copy_function(symmetric["omega"], context["spaces"]["Vomega"])
    coords_raw = context["spaces"]["Vomega"].tabulate_dof_coordinates()
    coords = coords_raw[:, :2] if coords_raw.ndim > 1 else coords_raw.reshape(-1, 3)[:, :2]
    values = vec_data(omega)
    delta = analytic_perturbation_values(coords, spec)
    values[: delta.size] += delta
    ghost(omega)
    context["continuation"]._apply_omega_dirichlet_on_boundary(omega, context["spaces"]["Vomega"], float(sigma))
    return omega


def split_block_vec(x: Any, context: dict[str, Any]) -> Any:
    baseline_dir = SRC / "baseline"
    sys.path.insert(0, str(baseline_dir))
    from petsc4py import PETSc
    from continuation import ContinuationSolver

    n_o = ContinuationSolver._owned_size(context["spaces"]["Vomega"])
    n_p = ContinuationSolver._owned_size(context["spaces"]["Vpsi"])
    is_o = PETSc.IS().createStride(n_o, 0, 1)
    is_p = PETSc.IS().createStride(n_p, n_o, 1)
    return (x.getSubVector(is_o), is_o), (x.getSubVector(is_p), is_p)


def subvec_array(v: Any) -> np.ndarray:
    try:
        return v.array.copy()
    except AttributeError:
        with v.localForm() as lf:
            return lf.array.copy()


def fixed_sigma_newton(
    context: dict[str, Any],
    *,
    omega: Any,
    psi: Any,
    sigma: float,
    label: str,
    branch: str,
    max_newton: int,
    tol_newton: float,
    ls_max_steps: int,
    ls_shrink: float,
    perturbation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline_dir = SRC / "baseline"
    sys.path.insert(0, str(baseline_dir))
    from petsc4py import PETSc

    t0 = time.perf_counter()
    history: list[dict[str, Any]] = []
    converged = False
    last_rel = float("nan")
    last_bnorm = float("nan")
    accepted_alpha = float("nan")
    try:
        context["continuation"]._apply_omega_dirichlet_on_boundary(
            omega,
            context["spaces"]["Vomega"],
            float(sigma),
        )
        psi = context["poisson"].solve_state(omega, sigma=float(sigma))
        for it in range(1, int(max_newton) + 1):
            A, b = context["assembler"].assemble(omega, psi, mu=PHYSICAL.mu)
            bnorm = float(b.norm(PETSc.NormType.NORM_2))
            last_bnorm = bnorm
            ksp = PETSc.KSP().create(context["mesh"].comm)
            ksp.setOperators(A)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.setFromOptions()
            x = b.duplicate()
            x.set(0.0)
            ksp.solve(b, x)
            (x1, is1), (x2, is2) = split_block_vec(x, context)
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
                _, b_trial = context["assembler"].assemble(omega, psi, mu=PHYSICAL.mu)
                trial_bnorm = float(b_trial.norm(PETSc.NormType.NORM_2))
                if trial_bnorm < bnorm or alpha <= float(ls_shrink) ** max(int(ls_max_steps) - 1, 1):
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
            last_bnorm = trial_bnorm
            history.append(
                {
                    "iteration": it,
                    "bnorm_before": bnorm,
                    "bnorm_after": trial_bnorm,
                    "rel_omega_step": last_rel,
                    "alpha": accepted_alpha,
                }
            )
            print(
                f"[fem fixed] {label} it={it} bnorm={bnorm:.3e}->{trial_bnorm:.3e} "
                f"rel={last_rel:.3e} alpha={accepted_alpha:.3g}",
                flush=True,
            )
            if last_rel <= float(tol_newton):
                converged = True
                break

        return finish_fem_run(
            context,
            omega=omega,
            psi=psi,
            label=label,
            branch=branch,
            sigma=sigma,
            status="completed" if converged else "nonconverged",
            runtime_sec=time.perf_counter() - t0,
            newton_iterations=len(history),
            last_relative_update=last_rel,
            last_linear_residual_norm=last_bnorm,
            history=history,
            perturbation=perturbation,
            message="fixed_sigma_linesearch_newton",
        )
    except Exception as exc:
        return {
            "label": label,
            "branch": branch,
            "discretization": "baseline_fem",
            "status": "failed",
            "sigma": float(sigma),
            "runtime_sec": time.perf_counter() - t0,
            "mesh_spec": {"h_min": float(context["params"].h_min), "h_max": float(context["params"].h_max)},
            "perturbation": perturbation or {},
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def fem_orbit_distance(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    rr, tt = np.meshgrid(PROBE_RADII, PROBE_THETA, indexing="ij")
    points = np.column_stack((rr.reshape(-1) * np.cos(tt.reshape(-1)), rr.reshape(-1) * np.sin(tt.reshape(-1))))
    ao = sample_baseline(a["omega"], a["spaces"]["Vomega"], points).reshape(PROBE_RADII.size, PROBE_THETA.size)
    ap = sample_baseline(a["psi"], a["spaces"]["Vpsi"], points).reshape(PROBE_RADII.size, PROBE_THETA.size)
    bo = sample_baseline(b["omega"], b["spaces"]["Vomega"], points).reshape(PROBE_RADII.size, PROBE_THETA.size)
    bp = sample_baseline(b["psi"], b["spaces"]["Vpsi"], points).reshape(PROBE_RADII.size, PROBE_THETA.size)
    weights = np.repeat(
        PROBE_RADII * (PROBE_RADII[1] - PROBE_RADII[0]) * (2.0 * np.pi / PROBE_THETA.size),
        PROBE_THETA.size,
    ).reshape(PROBE_RADII.size, PROBE_THETA.size)
    best: dict[str, Any] | None = None
    shift = PROBE_THETA.size // 3
    for reflect in (False, True):
        base_o = bo[:, ::-1] if reflect else bo
        base_p = bp[:, ::-1] if reflect else bp
        for k, s in enumerate((0, shift, 2 * shift)):
            trans_o = np.roll(base_o, s, axis=1)
            trans_p = np.roll(base_p, s, axis=1)
            valid = np.isfinite(ao) & np.isfinite(ap) & np.isfinite(trans_o) & np.isfinite(trans_p)
            if not np.any(valid):
                continue
            w = np.where(valid, weights, 0.0)
            domega = np.where(valid, ao - trans_o, 0.0)
            dpsi = np.where(valid, ap - trans_p, 0.0)
            dpsi -= float(np.sum(dpsi * w) / max(np.sum(w), 1e-30))
            omega_rms = float(np.sqrt(np.sum(domega * domega * w) / max(np.sum(w), 1e-30)))
            psi_rms = float(np.sqrt(np.sum(dpsi * dpsi * w) / max(np.sum(w), 1e-30)))
            row = {
                "omega_orbit_rms": omega_rms,
                "psi_orbit_rms_offset_removed": psi_rms,
                "transform": f"{'reflect_' if reflect else ''}c3_shift_{k}",
                "valid_count": int(np.count_nonzero(valid)),
            }
            if best is None or omega_rms < float(best["omega_orbit_rms"]):
                best = row
    return best or {"valid_count": 0}


def classify_fem_candidate(run: dict[str, Any], symmetric: dict[str, Any] | None) -> str:
    if run.get("status") != "completed":
        return str(run.get("status", "noncompleted"))
    omega_non3 = float(run.get("symmetry", {}).get("omega_non_threefold_fraction", float("nan")))
    if symmetric is None or symmetric.get("status") != "completed":
        return "broken_candidate_uncompared" if omega_non3 >= 1e-3 else "symmetric_like_uncompared"
    distance = fem_orbit_distance(run, symmetric)
    omega_sep = float(distance.get("omega_orbit_rms", float("nan")))
    run["distance_to_symmetric_fem"] = distance
    if omega_non3 >= 1e-3 and omega_sep > 1e-2:
        return "broken_candidate"
    if omega_non3 <= 1e-6:
        return "symmetric"
    return "other_root_candidate"


def scale_omega_for_sigma(context: dict[str, Any], source: dict[str, Any], target_sigma: float, source_sigma: float) -> tuple[Any, Any]:
    baseline_dir = SRC / "baseline"
    sys.path.insert(0, str(baseline_dir))
    from dolfinx import fem

    omega = fem.Function(context["spaces"]["Vomega"])
    coords_raw = context["spaces"]["Vomega"].tabulate_dof_coordinates()
    coords = coords_raw[:, :2] if coords_raw.ndim > 1 else coords_raw.reshape(-1, 3)[:, :2]
    sampled = sample_baseline(source["omega"], source["spaces"]["Vomega"], coords)
    if np.any(~np.isfinite(sampled)):
        sampled = np.where(np.isfinite(sampled), sampled, 0.0)
    ratio = float(target_sigma) / max(float(source_sigma), 1e-30)
    vec_data(omega)[: sampled.size] = ratio * sampled
    ghost(omega)
    context["continuation"]._apply_omega_dirichlet_on_boundary(omega, context["spaces"]["Vomega"], float(target_sigma))
    psi = context["poisson"].solve_state(omega, sigma=float(target_sigma))
    return omega, psi


def select_discovered_candidate(discovery_runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [run for run in discovery_runs if run.get("branch_class") == "broken_candidate"]
    if not candidates:
        return None
    candidates.sort(
        key=lambda run: (
            -float(run.get("sigma", 0.0)),
            float(run.get("perturbation", {}).get("amplitude", float("inf"))),
            str(run.get("perturbation", {}).get("family", "")),
        )
    )
    return candidates[0]


def save_fem_samples(run: dict[str, Any]) -> str | None:
    if run.get("status") != "completed":
        return None
    FIELD_DIR.mkdir(parents=True, exist_ok=True)
    xx, yy = np.meshgrid(SAMPLE_X, SAMPLE_Y, indexing="ij")
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    omega = sample_baseline(run["omega"], run["spaces"]["Vomega"], points).reshape(xx.shape)
    psi = sample_baseline(run["psi"], run["spaces"]["Vpsi"], points).reshape(xx.shape)
    path = FIELD_DIR / f"{run['label']}_samples.npz"
    np.savez_compressed(
        path,
        sample_x=SAMPLE_X,
        sample_y=SAMPLE_Y,
        sample_omega=omega,
        sample_psi=psi,
        sigma=float(run.get("sigma", 1.0)),
        label=run.get("label", ""),
        branch=run.get("branch", ""),
    )
    return str(path.relative_to(REPO_ROOT))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"status": "no_rows"}]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path.relative_to(REPO_ROOT))


def run_table_row(run: dict[str, Any]) -> dict[str, Any]:
    weak = run.get("residuals", {}).get("norms", {})
    sym = run.get("symmetry", {})
    comp = run.get("compatibility", {})
    boundary = run.get("boundary_defects", {})
    distance = run.get("distance_to_symmetric_fem", {})
    perturbation = run.get("perturbation", {})
    return {
        "label": run.get("label"),
        "branch": run.get("branch"),
        "branch_class": run.get("branch_class"),
        "status": run.get("status"),
        "sigma": run.get("sigma"),
        "mesh_h_min": run.get("mesh_spec", {}).get("h_min"),
        "mesh_h_max": run.get("mesh_spec", {}).get("h_max"),
        "perturbation_family": perturbation.get("family"),
        "perturbation_amplitude": perturbation.get("amplitude"),
        "perturbation_phase": perturbation.get("phase"),
        "runtime_sec": run.get("runtime_sec"),
        "newton_iterations": run.get("newton_iterations"),
        "last_relative_update": run.get("last_relative_update"),
        "last_linear_residual_norm": run.get("last_linear_residual_norm"),
        "fem_weak_omega_l2": weak.get("Omega_l2"),
        "fem_weak_psi_l2": weak.get("Psi_l2"),
        "omega_non3": sym.get("omega_non_threefold_fraction"),
        "psi_non3": sym.get("psi_non_threefold_fraction"),
        "omega_orbit_rms_to_symmetric": distance.get("omega_orbit_rms"),
        "psi_orbit_rms_to_symmetric": distance.get("psi_orbit_rms_offset_removed"),
        "compatibility_omega_minus_flux": comp.get("omega_minus_flux"),
        "omega_boundary_max_abs_error": boundary.get("omega_dirichlet_max_abs"),
        "sample_path": run.get("sample_path"),
        "message": run.get("message", run.get("error")),
    }


def separation_row(a: dict[str, Any], b: dict[str, Any], label: str) -> dict[str, Any]:
    distance = fem_orbit_distance(a, b)
    return {
        "pair": label,
        "a_label": a.get("label"),
        "b_label": b.get("label"),
        "sigma": a.get("sigma"),
        "omega_orbit_rms": distance.get("omega_orbit_rms"),
        "psi_orbit_rms_offset_removed": distance.get("psi_orbit_rms_offset_removed"),
        "best_transform": distance.get("transform"),
        "valid_count": distance.get("valid_count"),
        "a_omega_non3": a.get("symmetry", {}).get("omega_non_threefold_fraction"),
        "b_omega_non3": b.get("symmetry", {}).get("omega_non_threefold_fraction"),
    }


def load_selected_nextgen(label: str, spec: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(spec["path"])
    if not path.exists():
        return None
    data = np.load(path)
    grid = (int(data["grid_nr"]), int(data["grid_ntheta"]))
    problem = make_nextgen_problem(grid, spec["transport"])
    sigma = float(data["sigma"]) if "sigma" in data.files else 1.0
    return nextgen_result_from_arrays(
        problem,
        data["omega"],
        data["psi"],
        label=label,
        branch=str(spec["branch"]),
        transport=spec["transport"],
        sigma=sigma,
        poisson_lambda=0.0,
        message=f"loaded_selected_fv:{path.relative_to(REPO_ROOT)}",
    )


def selected_fv_comparisons(fem_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    completed_sigma1 = [run for run in fem_runs if run.get("status") == "completed" and abs(float(run.get("sigma", -1.0)) - 1.0) < 1e-12]
    if not completed_sigma1:
        return comparisons
    selected = {label: load_selected_nextgen(label, spec) for label, spec in SELECTED_FV_STATES.items()}
    for fem_run in completed_sigma1:
        for fv_label, fv_run in selected.items():
            if fv_run is None:
                comparisons.append(
                    {
                        "fem_label": fem_run.get("label"),
                        "fv_label": fv_label,
                        "status": "missing_selected_fv_state",
                    }
                )
                continue
            metrics = compare_nextgen_to_baseline(fv_run, fem_run)
            comparisons.append(
                {
                    "fem_label": fem_run.get("label"),
                    "fem_branch_class": fem_run.get("branch_class"),
                    "fv_label": fv_label,
                    "fv_branch": fv_run.get("branch"),
                    "status": "completed",
                    **metrics,
                }
            )
    return comparisons


def make_plots(summary: dict[str, Any]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    ladder = [run for run in summary.get("sigma_ladder_runs", []) if run.get("status") == "completed"]
    if ladder:
        path = PLOT_DIR / "fem_non3_vs_sigma.png"
        fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
        for branch_class in sorted({str(run.get("branch_class")) for run in ladder}):
            rows = [run for run in ladder if str(run.get("branch_class")) == branch_class]
            ax.plot(
                [run.get("sigma") for run in rows],
                [run.get("symmetry", {}).get("omega_non_threefold_fraction", np.nan) for run in rows],
                "o-",
                label=branch_class,
            )
        ax.set_xlabel("sigma")
        ax.set_ylabel("Omega non-3 fraction")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        fig.savefig(path, dpi=140)
        plt.close(fig)
        created.append(str(path.relative_to(REPO_ROOT)))

    discovery = summary.get("discovery_runs", [])
    if discovery:
        amps = sorted({float(run.get("perturbation", {}).get("amplitude", np.nan)) for run in discovery if run.get("perturbation")})
        phases = sorted({float(run.get("perturbation", {}).get("phase", np.nan)) for run in discovery if run.get("perturbation")})
        m1 = [run for run in discovery if run.get("perturbation", {}).get("family") == "m1_core"]
        if amps and phases and m1:
            value = np.full((len(amps), len(phases)), np.nan)
            for run in m1:
                i = amps.index(float(run["perturbation"]["amplitude"]))
                j = phases.index(float(run["perturbation"]["phase"]))
                value[i, j] = run.get("symmetry", {}).get("omega_non_threefold_fraction", np.nan)
            path = PLOT_DIR / "fem_m1_discovery_heatmap.png"
            fig, ax = plt.subplots(figsize=(6.0, 4.4), constrained_layout=True)
            image = ax.imshow(value, origin="lower", aspect="auto")
            ax.set_xticks(range(len(phases)), [f"{p:.2g}" for p in phases])
            ax.set_yticks(range(len(amps)), [f"{a:g}" for a in amps])
            ax.set_xlabel("phase")
            ax.set_ylabel("amplitude")
            fig.colorbar(image, ax=ax, label="Omega non-3 fraction")
            fig.savefig(path, dpi=140)
            plt.close(fig)
            created.append(str(path.relative_to(REPO_ROOT)))
    return created


def append_registry(summary: dict[str, Any]) -> None:
    registry = REPO_ROOT / "experiments" / "registry.csv"
    best = summary.get("selected_discovered_candidate") or {}
    with registry.open("a", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                RUN_DATE,
                "m3-c3-140-fem-branch-discovery",
                "baseline",
                "experiments/run_m3_c3_140_fem_branch_discovery.py",
                "{c3=140}",
                "baseline_fem_disk_Rdom10",
                f"mesh={summary.get('primary_mesh')}; sigmas={summary.get('sigma_ladder')}",
                "fem_analytic_perturbation_branch_discovery",
                "M3 c3=140 FEM branch discovery; no M4; no eigensolver",
                summary.get("runtime_sec", "n/a"),
                (
                    f"best={best.get('label')}; class={best.get('branch_class')}; "
                    f"omega_non3={best.get('omega_non3')}; status={summary.get('status')}"
                ),
                "n/a",
                summary.get("status", "completed"),
                "Independent FEM branch discovery from analytic perturbations.",
            ]
        )


def run_discovery(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIELD_DIR.mkdir(parents=True, exist_ok=True)

    mesh_spec = (float(args.mesh[0]), float(args.mesh[1]))
    sigmas = tuple(float(v) for v in args.sigma)
    amplitudes = tuple(float(v) for v in args.amplitude)
    specs = perturbation_specs(
        amplitudes,
        families=tuple(str(v) for v in args.family),
        phases=None if args.phase is None else tuple(float(v) for v in args.phase),
    )

    symmetric_roots: dict[float, dict[str, Any]] = {}
    contexts: dict[float, dict[str, Any]] = {}
    discovery_runs: list[dict[str, Any]] = []
    sigma_ladder_runs: list[dict[str, Any]] = []

    for sigma in sigmas:
        print(f"[fem symmetric start] sigma={sigma:g} mesh={mesh_spec}", flush=True)
        if args.symmetric_source == "cached_h0p06":
            context, symmetric = run_cached_symmetric_root(
                mesh_spec,
                sigma,
                max_newton=int(args.max_newton),
                tol_newton=float(args.tol_newton),
                ls_max_steps=int(args.ls_max_steps),
                ls_shrink=float(args.ls_shrink),
            )
        else:
            context, symmetric = run_symmetric_root(
                mesh_spec,
                sigma,
                max_newton=int(args.max_newton),
                tol_newton=float(args.tol_newton),
            )
        contexts[float(sigma)] = context
        symmetric_roots[float(sigma)] = symmetric
        symmetric["branch_class"] = "symmetric_reference" if symmetric.get("status") == "completed" else symmetric.get("status")
        if symmetric.get("status") == "completed":
            symmetric["sample_path"] = save_fem_samples(symmetric)
        sigma_ladder_runs.append(symmetric)
        print(f"[fem symmetric done] sigma={sigma:g} status={symmetric.get('status')}", flush=True)
        if symmetric.get("status") != "completed":
            continue

        for spec in specs:
            label = (
                f"fem_discovery_{mesh_label(mesh_spec)}_{sigma_label(sigma)}_"
                f"{spec['family']}_a{str(spec['amplitude']).replace('.', 'p')}_"
                f"p{str(round(float(spec.get('phase', 0.0)), 6)).replace('.', 'p')}"
            )
            omega = perturbed_omega_from_symmetric(context, symmetric, spec, sigma)
            psi = context["poisson"].solve_state(omega, sigma=float(sigma))
            run = fixed_sigma_newton(
                context,
                omega=omega,
                psi=psi,
                sigma=sigma,
                label=label,
                branch="discovery",
                max_newton=int(args.max_newton),
                tol_newton=float(args.tol_newton),
                ls_max_steps=int(args.ls_max_steps),
                ls_shrink=float(args.ls_shrink),
                perturbation=spec,
            )
            run["branch_class"] = classify_fem_candidate(run, symmetric)
            if run.get("status") == "completed":
                run["sample_path"] = save_fem_samples(run)
            discovery_runs.append(run)
            print(
                f"[fem discovery done] sigma={sigma:g} {spec['family']} "
                f"a={spec['amplitude']:g} status={run.get('status')} class={run.get('branch_class')}",
                flush=True,
            )

    selected = select_discovered_candidate(discovery_runs)
    if selected is not None:
        sigma_ladder_runs.extend(
            continue_discovered_branch(
                selected,
                contexts,
                symmetric_roots,
                sigmas,
                args,
            )
        )

    confirmation_runs: list[dict[str, Any]] = []
    if selected is not None and not bool(args.skip_confirmation):
        confirmation_runs = run_confirmation(selected, args)

    fem_vs_fv = selected_fv_comparisons(sigma_ladder_runs + discovery_runs + confirmation_runs)
    discovery_table = write_csv(TABLE_DIR / "fem_discovery_start_table.csv", [run_table_row(run) for run in discovery_runs])
    ladder_table = write_csv(TABLE_DIR / "fem_sigma_ladder_table.csv", [run_table_row(run) for run in sigma_ladder_runs + confirmation_runs])
    separations = branch_separation_rows(sigma_ladder_runs + discovery_runs + confirmation_runs, symmetric_roots)
    separation_table = write_csv(TABLE_DIR / "fem_branch_separation_table.csv", separations)
    fv_table = write_csv(TABLE_DIR / "fem_vs_selected_fv_table.csv", fem_vs_fv)

    selected_summary = None
    if selected is not None:
        selected_summary = run_table_row(selected)
    summary = {
        "status": "completed",
        "case": {"name": "m3_c3_140_fem_branch_discovery", "coeffs": COEFFS},
        "primary_mesh": {"h_min": mesh_spec[0], "h_max": mesh_spec[1]},
        "confirmation_mesh": None
        if bool(args.skip_confirmation)
        else {"h_min": float(args.confirmation_mesh[0]), "h_max": float(args.confirmation_mesh[1])},
        "sigma_ladder": sigmas,
        "amplitudes": amplitudes,
        "families": tuple(str(v) for v in args.family),
        "phases": None if args.phase is None else tuple(float(v) for v in args.phase),
        "symmetric_source": str(args.symmetric_source),
        "runtime_sec": time.perf_counter() - t0,
        "discovery_runs": discovery_runs,
        "sigma_ladder_runs": sigma_ladder_runs,
        "confirmation_runs": confirmation_runs,
        "selected_discovered_candidate": selected_summary,
        "fem_vs_selected_fv": fem_vs_fv,
        "table_paths": {
            "discovery_starts": discovery_table,
            "sigma_ladder": ladder_table,
            "branch_separation": separation_table,
            "fem_vs_selected_fv": fv_table,
        },
        "plot_paths": [],
        "notes": [
            "M3 only; no M4 operator assembly and no eigensolver.",
            "Selected FV states are loaded only after FEM discovery for comparison; they are not used as FEM initial conditions.",
            "Discovery starts are analytic interior perturbations of FEM symmetric roots with psi recomputed by the FEM Poisson solve.",
        ],
    }
    summary["plot_paths"] = make_plots(summary)
    return summary


def continue_discovered_branch(
    selected: dict[str, Any],
    contexts: dict[float, dict[str, Any]],
    symmetric_roots: dict[float, dict[str, Any]],
    sigmas: tuple[float, ...],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    ordered = sorted(set(float(v) for v in sigmas), reverse=True)
    source_sigma = float(selected["sigma"])
    start_index = ordered.index(source_sigma) if source_sigma in ordered else 0

    def walk(targets: list[float], direction: str) -> None:
        source = selected
        source_sigma = float(selected["sigma"])
        for target_sigma in targets:
            if target_sigma not in contexts or not contexts[target_sigma]:
                break
            context = contexts[target_sigma]
            omega, psi = scale_omega_for_sigma(context, source, target_sigma, source_sigma)
            label = (
                f"fem_discovered_ladder_{direction}_"
                f"{mesh_label((context['params'].h_min, context['params'].h_max))}_{sigma_label(target_sigma)}"
            )
            run = fixed_sigma_newton(
                context,
                omega=omega,
                psi=psi,
                sigma=target_sigma,
                label=label,
                branch="broken_discovered",
                max_newton=int(args.max_newton),
                tol_newton=float(args.tol_newton),
                ls_max_steps=int(args.ls_max_steps),
                ls_shrink=float(args.ls_shrink),
                perturbation={"source": source.get("label"), "type": f"nearest_neighbor_sigma_ladder_{direction}"},
            )
            run["branch_class"] = classify_fem_candidate(run, symmetric_roots.get(target_sigma))
            if run.get("status") == "completed":
                run["sample_path"] = save_fem_samples(run)
                if run.get("branch_class") == "broken_candidate":
                    source = run
                    source_sigma = target_sigma
            runs.append(run)
            if run.get("branch_class") != "broken_candidate":
                break

    walk(ordered[start_index + 1 :], "down")
    walk(list(reversed(ordered[:start_index])), "up")
    return runs


def run_confirmation(selected: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    mesh_spec = (float(args.confirmation_mesh[0]), float(args.confirmation_mesh[1]))
    sigma = float(selected.get("sigma", 1.0))
    spec = dict(selected.get("perturbation", {}))
    if not spec:
        return []
    context, symmetric = run_symmetric_root(
        mesh_spec,
        sigma,
        max_newton=int(args.max_newton),
        tol_newton=float(args.tol_newton),
    )
    symmetric["branch_class"] = "symmetric_reference" if symmetric.get("status") == "completed" else symmetric.get("status")
    runs = [symmetric]
    if symmetric.get("status") != "completed":
        return runs
    omega = perturbed_omega_from_symmetric(context, symmetric, spec, sigma)
    psi = context["poisson"].solve_state(omega, sigma=float(sigma))
    run = fixed_sigma_newton(
        context,
        omega=omega,
        psi=psi,
        sigma=sigma,
        label=f"fem_confirmation_{mesh_label(mesh_spec)}_{sigma_label(sigma)}_{spec.get('family', 'unknown')}",
        branch="confirmation",
        max_newton=int(args.max_newton),
        tol_newton=float(args.tol_newton),
        ls_max_steps=int(args.ls_max_steps),
        ls_shrink=float(args.ls_shrink),
        perturbation=spec,
    )
    run["branch_class"] = classify_fem_candidate(run, symmetric)
    if run.get("status") == "completed":
        run["sample_path"] = save_fem_samples(run)
    runs.append(run)
    return runs


def branch_separation_rows(runs: list[dict[str, Any]], symmetric_roots: dict[float, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        sigma = float(run.get("sigma", -1.0))
        symmetric = symmetric_roots.get(sigma)
        if (
            symmetric is not None
            and symmetric.get("status") == "completed"
            and run.get("status") == "completed"
            and run.get("label") != symmetric.get("label")
        ):
            rows.append(separation_row(run, symmetric, f"{run.get('label')}_vs_symmetric"))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M3 c3=140 independent FEM branch-discovery sigma ladder.")
    parser.add_argument("--mesh", nargs=2, type=float, default=DEFAULT_MESH, metavar=("H_MIN", "H_MAX"))
    parser.add_argument("--confirmation-mesh", nargs=2, type=float, default=DEFAULT_CONFIRMATION_MESH, metavar=("H_MIN", "H_MAX"))
    parser.add_argument("--skip-confirmation", action="store_true")
    parser.add_argument("--sigma", nargs="*", type=float, default=list(DEFAULT_SIGMAS))
    parser.add_argument("--amplitude", nargs="*", type=float, default=list(DEFAULT_AMPLITUDES))
    parser.add_argument("--family", nargs="*", choices=("m1_core", "m2_core", "mixed_non3"), default=["m1_core", "m2_core", "mixed_non3"])
    parser.add_argument("--phase", nargs="*", type=float, default=None)
    parser.add_argument("--symmetric-source", choices=("continuation", "cached_h0p06"), default="continuation")
    parser.add_argument("--max-newton", type=int, default=40)
    parser.add_argument("--tol-newton", type=float, default=1e-9)
    parser.add_argument("--ls-max-steps", type=int, default=10)
    parser.add_argument("--ls-shrink", type=float, default=0.5)
    parser.add_argument("--no-registry", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_discovery(args)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARY_DIR / f"m3_c3_140_fem_branch_discovery_{RUN_DATE}.json"
    path.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    if not bool(args.no_registry):
        append_registry(sanitize(summary))
    print(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
