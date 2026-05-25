#!/usr/bin/env python3
"""Detect a symmetry-breaking profile for mu=2/3 and s cos(4 theta).

Workflow:
1. Compute symmetric FEM profiles for s values starting at s=200.
2. Compute the first 10 linearized eigenvalues.
3. Stop once at least four returned eigenvalues have negative real part.
4. Apply interior-only artificial perturbations that preserve the boundary
   data, then run fixed-boundary Newton to search for a symmetry-breaking
   profile.

The baseline FourierBC convention mirrors positive modes, so exact
``s cos(4 theta)`` is encoded as ``coeffs={4: s/2}``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

BRANCH_DIR = Path(__file__).resolve().parent
MANUSCRIPT_ROOT = BRANCH_DIR.parents[0]
REPO_ROOT = MANUSCRIPT_ROOT.parents[1]
C3_BRANCH_DIR = MANUSCRIPT_ROOT / "c3_spectrum_continuation"
SYMB_DIR = MANUSCRIPT_ROOT / "symmetry_breaking_profiles"
sys.path.insert(0, str(MANUSCRIPT_ROOT))
sys.path.insert(0, str(C3_BRANCH_DIR))
sys.path.insert(0, str(SYMB_DIR))

from common import (  # noqa: E402
    add_baseline_to_path,
    amplitude_label,
    append_registry_row,
    boundary_description,
    coeffs_for_exact_cos_mode,
    mesh_label,
    today,
    write_json,
)
from run_s140_bifurcation_probe import (  # noqa: E402
    boundary_max_abs_diff,
    copy_function,
    ghost_forward,
    linf,
    load_h5_profile,
    rms,
    run_newton_from,
    vec_array,
    write_xdmf_profile,
)
from run_s140_artificial_perturb_probe import artificial_pattern, diff_function  # noqa: E402
from run_spectrum_continuation import build_baseline_context, mesh_stats  # noqa: E402


EIG_COLUMNS = [
    "s",
    "mode",
    "mu",
    "mesh_id",
    "eigen_index",
    "real_lambda",
    "imag_lambda",
    "profile_residual_omega_l2",
    "profile_residual_psi_l2",
    "profile_checkpoint",
    "metadata_checkpoint",
    "status",
]

PERT_COLUMNS = [
    "s",
    "mode",
    "mu",
    "trial_id",
    "status",
    "m_mode",
    "phase",
    "eta",
    "scale",
    "newton_iterations",
    "final_newton_residual",
    "final_relative_step",
    "profile_residual_omega_l2",
    "profile_residual_psi_l2",
    "boundary_max_abs_diff",
    "omega_rms_separation",
    "omega_relative_rms_separation",
    "omega_linf_separation",
    "psi_rms_separation",
    "psi_relative_rms_separation",
    "psi_linf_separation",
    "omega_c4_fraction_symmetric",
    "omega_c4_fraction_candidate",
    "omega_c4_fraction_excess",
    "omega_difference_non4_fraction",
    "psi_c4_fraction_symmetric",
    "psi_c4_fraction_candidate",
    "psi_c4_fraction_excess",
    "psi_difference_non4_fraction",
    "wall_time_sec",
    "profile_checkpoint",
    "plot_path",
    "message",
]


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    return value


def symmetry_fraction_on_probe(
    V: Any,
    fun: Any,
    *,
    symmetry_order: int,
    r_count: int = 50,
    theta_count: int = 240,
    r_max: float = 9.75,
) -> float:
    from scipy.interpolate import LinearNDInterpolator

    raw = V.tabulate_dof_coordinates()
    if raw.ndim == 1:
        raw = raw.reshape(-1, V.mesh.geometry.x.shape[1])
    interp = LinearNDInterpolator(raw[:, :2], vec_array(fun).copy(), fill_value=np.nan)
    radii = np.linspace(0.05, r_max, r_count)
    theta = np.linspace(0.0, 2.0 * np.pi, theta_count, endpoint=False)
    rr, tt = np.meshgrid(radii, theta, indexing="ij")
    samples = []
    for j in range(int(symmetry_order)):
        shift = 2.0 * np.pi * j / float(symmetry_order)
        x = rr * np.cos(tt + shift)
        y = rr * np.sin(tt + shift)
        samples.append(interp(x.reshape(-1), y.reshape(-1)).reshape(rr.shape))
    stacked = np.stack(samples, axis=0)
    valid = np.all(np.isfinite(stacked), axis=0)
    if not np.any(valid):
        return float("nan")
    avg = np.mean(stacked[:, valid], axis=0)
    defect = stacked[0, valid] - avg
    denom = np.sqrt(np.mean(stacked[0, valid] ** 2))
    return float(np.sqrt(np.mean(defect ** 2)) / max(denom, 1e-30))


def triangulation(mesh: Any) -> mtri.Triangulation:
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    cells = mesh.topology.connectivity(tdim, 0).array.reshape(-1, 3)
    xy = mesh.geometry.x[:, :2]
    return mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=cells)


def plot_profile_comparison(path: Path, mesh: Any, omega_sym: Any, omega_candidate: Any, title: str) -> None:
    tri = triangulation(mesh)
    sym = vec_array(omega_sym)
    cand = vec_array(omega_candidate)
    diff = cand - sym
    fields = [sym, cand, diff]
    labels = ["symmetric", "symmetry-breaking", "symmetry-breaking - symmetric"]
    vmax = max(linf(sym), linf(cand), 1e-30)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), dpi=220, constrained_layout=True)
    for ax, values, label in zip(axes, fields, labels):
        local_vmax = linf(values) if label.endswith("symmetric") is False and " - " in label else vmax
        cf = ax.tricontourf(tri, values, levels=64, cmap="RdBu_r", vmin=-local_vmax, vmax=local_vmax)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(label, fontsize=9)
        ax.set_aspect("equal", "box")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def append_rows(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "n/a") for key in columns})


def set_exact_boundary_values(context: dict[str, Any], omega: Any) -> None:
    from dolfinx import fem

    facets = context["facet_tags"].find(1)
    bdofs = fem.locate_dofs_topological(context["spaces"]["Vomega"], context["mesh"].topology.dim - 1, facets)
    coords = context["spaces"]["Vomega"].tabulate_dof_coordinates()
    if coords.ndim == 1:
        coords = coords.reshape(-1, context["mesh"].geometry.x.shape[1])
    xy = coords[bdofs.astype(np.int64), :2]
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    vec_array(omega)[bdofs.astype(np.int64)] = context["fourier_bc"].Omega_bar(theta)
    ghost_forward(omega)


def seeded_profile_solve(args: argparse.Namespace, context: dict[str, Any], target_s: float):
    if args.seed_profile_h5 is None:
        return context["continuation"].run(context["mesh"], context["spaces"])
    if args.seed_s is None:
        raise ValueError("--seed-s is required when --seed-profile-h5 is used")
    omega_seed, _psi_seed, _geom = load_h5_profile(Path(args.seed_profile_h5), context["mesh"], context["spaces"])
    ratio = float(target_s) / float(args.seed_s)
    omega0 = copy_function(context["spaces"]["Vomega"], ratio * vec_array(omega_seed))
    set_exact_boundary_values(context, omega0)
    psi0 = context["poisson"].solve_state(omega0, sigma=1.0)
    omega, psi, history, converged = run_newton_from(
        context,
        omega0,
        psi0,
        max_newton=int(args.max_newton),
        tol=float(args.tol_newton),
        linesearch=not args.no_linesearch,
    )
    if not converged:
        final = history[-1]
        raise RuntimeError(
            "Seeded fixed-boundary Newton did not converge: "
            f"it={final['iteration']}, rel={final['relative_step']}, residual={final['residual_norm']}"
        )
    return omega, psi


def compute_profile_and_eigs(args: argparse.Namespace, s_value: float, out_dirs: dict[str, Path]) -> dict[str, Any]:
    add_baseline_to_path()
    from eigs_linearized_new import ManifoldEigenSolver
    from residuals_quick import residuals_quick

    mesh_id = mesh_label(args.mesh[0], args.mesh[1])
    amp_id = amplitude_label(s_value)
    profile_xdmf = out_dirs["checkpoints"] / f"{amp_id}_{mesh_id}_profile.xdmf"
    metadata_path = out_dirs["checkpoints"] / f"{amp_id}_{mesh_id}_metadata.json"
    metadata: dict[str, Any] = {
        "status": "started",
        "s": float(s_value),
        "mode": int(args.mode),
        "mu": float(args.mu),
        "boundary": boundary_description(s_value, args.mode),
        "mesh": {"h_min": float(args.mesh[0]), "h_max": float(args.mesh[1])},
        "started_at": datetime.now().isoformat(),
        "profile_checkpoint": relpath(profile_xdmf),
    }
    write_json(metadata_path, jsonable(metadata))

    ns = argparse.Namespace(
        mesh=[float(args.mesh[0]), float(args.mesh[1])],
        sigma_steps=int(args.sigma_steps),
        mu=float(args.mu),
        R=float(args.R),
        Rdom=float(args.Rdom),
        M=int(args.M),
        tol_newton=float(args.tol_newton),
        max_newton=int(args.max_newton),
    )
    context = build_baseline_context(ns, coeffs_for_exact_cos_mode(s_value, args.mode))
    stats = mesh_stats(context["mesh"], context["spaces"])
    metadata["mesh_stats"] = stats
    write_json(metadata_path, jsonable(metadata))

    t0 = time.perf_counter()
    profile_t0 = time.perf_counter()
    omega, psi = seeded_profile_solve(args, context, s_value)
    profile_wall = time.perf_counter() - profile_t0
    write_xdmf_profile(profile_xdmf, context["mesh"], omega, psi)
    pinned = getattr(context["poisson"], "gauge_dof", None)
    residuals = residuals_quick(
        context["mesh"],
        context["spaces"],
        omega,
        psi,
        mu=context["params"].mu,
        R=context["params"].R,
        Rdom=context["params"].Rdom,
        facet_tags=context["facet_tags"],
        outer_tag=1,
        fourier_bc=context["fourier_bc"],
        sigma=1.0,
        pinned_psi_dof=pinned,
    )
    omega_res = float(residuals["norms"]["Omega_l2"])
    psi_res = float(residuals["norms"]["Psi_l2"])
    if omega_res > float(args.max_profile_residual_omega) or psi_res > float(args.max_profile_residual_psi):
        metadata.update(
            {
                "status": "failed_profile_residual_gate",
                "profile": {
                    "wall_time_sec": profile_wall,
                    "weak_residuals": {
                        "Omega_l2": omega_res,
                        "Psi_l2": psi_res,
                    },
                    "residual_gate": {
                        "max_profile_residual_omega": float(args.max_profile_residual_omega),
                        "max_profile_residual_psi": float(args.max_profile_residual_psi),
                    },
                },
            }
        )
        write_json(metadata_path, jsonable(metadata))
        raise RuntimeError(
            "Profile residual gate failed before eigensolve: "
            f"Omega_l2={omega_res:.6e}, Psi_l2={psi_res:.6e}"
        )

    eig_t0 = time.perf_counter()
    esolver = ManifoldEigenSolver(
        context["mesh"],
        context["spaces"],
        omega,
        psi,
        context["params"],
        context["facet_tags"],
        outer_tag=1,
    )
    modes = esolver.solve(
        k=int(args.k_eigs),
        sigma=float(args.eig_shift),
        tol=float(args.eig_tol),
        max_it=int(args.eig_max_it),
        verbose=True,
    )
    eig_wall = time.perf_counter() - eig_t0
    eigs = [complex(mode.lam) for mode in modes[: int(args.k_eigs)]]
    unstable_count = int(sum(1 for lam in eigs if lam.real < 0.0))
    rows = [
        {
            "s": float(s_value),
            "mode": int(args.mode),
            "mu": float(args.mu),
            "mesh_id": mesh_id,
            "eigen_index": idx,
            "real_lambda": float(lam.real),
            "imag_lambda": float(lam.imag),
            "profile_residual_omega_l2": residuals["norms"]["Omega_l2"],
            "profile_residual_psi_l2": residuals["norms"]["Psi_l2"],
            "profile_checkpoint": relpath(profile_xdmf),
            "metadata_checkpoint": relpath(metadata_path),
            "status": "completed",
        }
        for idx, lam in enumerate(eigs, start=1)
    ]
    append_rows(out_dirs["data"] / f"{args.run_tag}_eigenvalues.csv", EIG_COLUMNS, rows)

    metadata.update(
        {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "wall_time_sec": time.perf_counter() - t0,
            "profile": {
                "wall_time_sec": profile_wall,
                "weak_residuals": {
                    "Omega_l2": residuals["norms"]["Omega_l2"],
                    "Psi_l2": residuals["norms"]["Psi_l2"],
                },
            },
            "eigs": {
                "wall_time_sec": eig_wall,
                "k_requested": int(args.k_eigs),
                "unstable_count_re_lt_0": unstable_count,
                "eigenvalues": [
                    {"index": idx, "real": float(lam.real), "imag": float(lam.imag)}
                    for idx, lam in enumerate(eigs, start=1)
                ],
            },
        }
    )
    write_json(metadata_path, jsonable(metadata))
    print(f"[spectrum] s={s_value:g} unstable_count={unstable_count} minRe={min(l.real for l in eigs):.6e}")
    return {
        "context": context,
        "omega": omega,
        "psi": psi,
        "residuals": residuals,
        "eigenvalues": eigs,
        "unstable_count": unstable_count,
        "profile_xdmf": profile_xdmf,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }


def make_artificial_initial_guess(context: dict[str, Any], omega_sym: Any, eta: float, m_mode: int, phase: float, r0: float):
    Vomega = context["spaces"]["Vomega"]
    pattern = artificial_pattern(Vomega, m_mode=m_mode, phase=phase, r0=r0, Rdom=context["params"].Rdom)
    scale = float(eta) * rms(vec_array(omega_sym)) / max(rms(pattern), 1e-30)
    omega0 = copy_function(Vomega, vec_array(omega_sym) + scale * pattern)
    ghost_forward(omega0)
    psi0 = context["poisson"].solve_state(omega0, sigma=1.0)
    return omega0, psi0, scale


def perturbation_probe(args: argparse.Namespace, selected: dict[str, Any], out_dirs: dict[str, Path]) -> list[dict[str, Any]]:
    add_baseline_to_path()
    from residuals_quick import residuals_quick

    context = selected["context"]
    omega_sym = selected["omega"]
    psi_sym = selected["psi"]
    s_value = float(selected["metadata"]["s"])
    pinned = getattr(context["poisson"], "gauge_dof", None)
    omega_sym_c4 = symmetry_fraction_on_probe(context["spaces"]["Vomega"], omega_sym, symmetry_order=args.mode)
    psi_sym_c4 = symmetry_fraction_on_probe(context["spaces"]["Vpsi"], psi_sym, symmetry_order=args.mode)

    rows: list[dict[str, Any]] = []
    for m_mode in args.perturb_modes:
        if int(m_mode) % int(args.mode) == 0:
            print(f"[skip] m={m_mode} is compatible with C{args.mode}")
            continue
        for phase in args.phases:
            for eta in args.etas:
                trial_id = f"m{m_mode}_phase{phase:g}_eta{eta:g}".replace(".", "p").replace("-", "m")
                t0 = time.perf_counter()
                try:
                    omega0, psi0, scale = make_artificial_initial_guess(context, omega_sym, eta, m_mode, phase, args.r0)
                    init_bdiff = boundary_max_abs_diff(context["spaces"], context["facet_tags"], omega0, omega_sym)
                    omega_candidate, psi_candidate, history, converged = run_newton_from(
                        context,
                        omega0,
                        psi0,
                        max_newton=args.perturb_max_newton,
                        tol=args.tol_newton,
                        linesearch=not args.no_linesearch,
                    )
                    residuals = residuals_quick(
                        context["mesh"],
                        context["spaces"],
                        omega_candidate,
                        psi_candidate,
                        mu=context["params"].mu,
                        R=context["params"].R,
                        Rdom=context["params"].Rdom,
                        facet_tags=context["facet_tags"],
                        outer_tag=1,
                        fourier_bc=context["fourier_bc"],
                        sigma=1.0,
                        pinned_psi_dof=pinned,
                    )
                    omega_sep = vec_array(omega_candidate) - vec_array(omega_sym)
                    psi_sep = vec_array(psi_candidate) - vec_array(psi_sym)
                    omega_diff_fun = diff_function(context["spaces"]["Vomega"], omega_candidate, omega_sym)
                    psi_diff_fun = diff_function(context["spaces"]["Vpsi"], psi_candidate, psi_sym)
                    omega_c4 = symmetry_fraction_on_probe(context["spaces"]["Vomega"], omega_candidate, symmetry_order=args.mode)
                    psi_c4 = symmetry_fraction_on_probe(context["spaces"]["Vpsi"], psi_candidate, symmetry_order=args.mode)
                    omega_diff_non4 = symmetry_fraction_on_probe(context["spaces"]["Vomega"], omega_diff_fun, symmetry_order=args.mode)
                    psi_diff_non4 = symmetry_fraction_on_probe(context["spaces"]["Vpsi"], psi_diff_fun, symmetry_order=args.mode)
                    rel_omega = rms(omega_sep) / max(rms(vec_array(omega_sym)), 1e-30)
                    rel_psi = rms(psi_sep) / max(rms(vec_array(psi_sym)), 1e-30)
                    final = history[-1]
                    distinct = (
                        converged
                        and rel_omega > float(args.distinct_relative_omega_rms)
                        and omega_diff_non4 > float(args.distinct_difference_non4)
                    )
                    status = "completed_distinct_symmetry_breaking_candidate" if distinct else ("completed" if converged else "nonconverged")
                    profile_path = out_dirs["checkpoints"] / f"{amplitude_label(s_value)}_{trial_id}_profile.xdmf"
                    write_xdmf_profile(profile_path, context["mesh"], omega_candidate, psi_candidate)
                    plot_path = out_dirs["plots"] / f"{amplitude_label(s_value)}_{trial_id}_omega_comparison"
                    plot_profile_comparison(
                        plot_path,
                        context["mesh"],
                        omega_sym,
                        omega_candidate,
                        rf"$\mu=2/3$, $\sigma={s_value:g}$: C4 symmetry-breaking probe",
                    )
                    row = {
                        "s": s_value,
                        "mode": int(args.mode),
                        "mu": float(args.mu),
                        "trial_id": trial_id,
                        "status": status,
                        "m_mode": int(m_mode),
                        "phase": float(phase),
                        "eta": float(eta),
                        "scale": scale,
                        "newton_iterations": int(final["iteration"]),
                        "final_newton_residual": float(final["residual_norm"]),
                        "final_relative_step": float(final["relative_step"]),
                        "profile_residual_omega_l2": residuals["norms"]["Omega_l2"],
                        "profile_residual_psi_l2": residuals["norms"]["Psi_l2"],
                        "boundary_max_abs_diff": boundary_max_abs_diff(context["spaces"], context["facet_tags"], omega_candidate, omega_sym),
                        "omega_rms_separation": rms(omega_sep),
                        "omega_relative_rms_separation": rel_omega,
                        "omega_linf_separation": linf(omega_sep),
                        "psi_rms_separation": rms(psi_sep),
                        "psi_relative_rms_separation": rel_psi,
                        "psi_linf_separation": linf(psi_sep),
                        "omega_c4_fraction_symmetric": omega_sym_c4,
                        "omega_c4_fraction_candidate": omega_c4,
                        "omega_c4_fraction_excess": omega_c4 - omega_sym_c4,
                        "omega_difference_non4_fraction": omega_diff_non4,
                        "psi_c4_fraction_symmetric": psi_sym_c4,
                        "psi_c4_fraction_candidate": psi_c4,
                        "psi_c4_fraction_excess": psi_c4 - psi_sym_c4,
                        "psi_difference_non4_fraction": psi_diff_non4,
                        "wall_time_sec": time.perf_counter() - t0,
                        "profile_checkpoint": relpath(profile_path),
                        "plot_path": relpath(plot_path.with_suffix(".png")),
                        "message": f"initial_boundary_diff={init_bdiff:.3e}",
                    }
                except Exception as exc:
                    row = {
                        "s": s_value,
                        "mode": int(args.mode),
                        "mu": float(args.mu),
                        "trial_id": trial_id,
                        "status": "failed",
                        "m_mode": int(m_mode),
                        "phase": float(phase),
                        "eta": float(eta),
                        "wall_time_sec": time.perf_counter() - t0,
                        "message": repr(exc),
                    }
                    with (out_dirs["logs"] / f"{args.run_tag}_failures.log").open("a") as f:
                        f.write(f"\n[{datetime.now().isoformat()}] {trial_id}\n")
                        f.write(traceback.format_exc())
                rows.append(row)
                append_rows(out_dirs["data"] / f"{args.run_tag}_perturbations.csv", PERT_COLUMNS, [row])
                print(
                    f"[perturb] {trial_id}: {row['status']} "
                    f"relOmega={row.get('omega_relative_rms_separation', 'n/a')} "
                    f"diffNon4={row.get('omega_difference_non4_fraction', 'n/a')}"
                )
                if args.stop_on_distinct and row["status"] == "completed_distinct_symmetry_breaking_candidate":
                    return rows
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-tag", default="branchB_k4_mu2over3_s200_bifurcation_probe")
    parser.add_argument("--mode", type=int, default=4)
    parser.add_argument("--mu", type=float, default=2.0 / 3.0)
    parser.add_argument("--s-start", type=float, default=200.0)
    parser.add_argument("--s-step", type=float, default=25.0)
    parser.add_argument("--s-max", type=float, default=300.0)
    parser.add_argument("--required-unstable", type=int, default=4)
    parser.add_argument("--mesh", nargs=2, type=float, default=[0.06, 0.06])
    parser.add_argument("--k-eigs", type=int, default=10)
    parser.add_argument("--eig-shift", type=float, default=-1.0)
    parser.add_argument("--eig-tol", type=float, default=1e-8)
    parser.add_argument("--eig-max-it", type=int, default=300)
    parser.add_argument("--R", type=float, default=150.0)
    parser.add_argument("--Rdom", type=float, default=10.0)
    parser.add_argument("--M", type=int, default=48)
    parser.add_argument("--sigma-steps", type=int, default=21)
    parser.add_argument("--max-newton", type=int, default=48)
    parser.add_argument("--tol-newton", type=float, default=1e-8)
    parser.add_argument("--max-profile-residual-omega", type=float, default=1e-2)
    parser.add_argument("--max-profile-residual-psi", type=float, default=1e-8)
    parser.add_argument("--seed-s", type=float, default=None)
    parser.add_argument("--seed-profile-h5", type=Path, default=None)
    parser.add_argument("--perturb-modes", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--phases", nargs="+", type=float, default=[0.0])
    parser.add_argument("--etas", nargs="+", type=float, default=[0.25, 0.5])
    parser.add_argument("--r0", type=float, default=2.5)
    parser.add_argument("--perturb-max-newton", type=int, default=24)
    parser.add_argument("--distinct-relative-omega-rms", type=float, default=1e-2)
    parser.add_argument("--distinct-difference-non4", type=float, default=0.1)
    parser.add_argument("--no-linesearch", action="store_true")
    parser.add_argument("--stop-on-distinct", action="store_true")
    parser.add_argument("--no-registry", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dirs = {
        "data": BRANCH_DIR / "data",
        "logs": BRANCH_DIR / "logs",
        "plots": BRANCH_DIR / "plots" / args.run_tag,
        "checkpoints": BRANCH_DIR / "checkpoints" / args.run_tag,
    }
    for directory in out_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dirs["logs"] / f"{args.run_tag}_manifest.json"
    manifest: dict[str, Any] = {
        "status": "dry_run",
        "run_tag": args.run_tag,
        "boundary": boundary_description(args.s_start, args.mode),
        "mu": float(args.mu),
        "mode": int(args.mode),
        "mesh": {"h_min": float(args.mesh[0]), "h_max": float(args.mesh[1])},
        "s_policy": {
            "start": float(args.s_start),
            "step": float(args.s_step),
            "max": float(args.s_max),
            "required_unstable_re_lt_0": int(args.required_unstable),
        },
        "outputs": {key: relpath(value) for key, value in out_dirs.items()},
    }
    write_json(manifest_path, manifest)
    if not args.execute:
        print(f"[dry-run] wrote {manifest_path}")
        return 0

    t_all = time.perf_counter()
    selected = None
    spectra = []
    s_value = float(args.s_start)
    while s_value <= float(args.s_max) + 1e-12:
        result = compute_profile_and_eigs(args, s_value, out_dirs)
        spectra.append(
            {
                "s": float(result["metadata"]["s"]),
                "unstable_count": int(result["unstable_count"]),
                "profile_checkpoint": relpath(result["profile_xdmf"]),
                "metadata_checkpoint": relpath(result["metadata_path"]),
                "profile_residuals": result["metadata"]["profile"]["weak_residuals"],
                "eigenvalues": [
                    {"index": idx, "real": float(lam.real), "imag": float(lam.imag)}
                    for idx, lam in enumerate(result["eigenvalues"], start=1)
                ],
            }
        )
        if int(result["unstable_count"]) >= int(args.required_unstable):
            selected = result
            break
        s_value += float(args.s_step)

    perturb_rows: list[dict[str, Any]] = []
    if selected is not None:
        perturb_rows = perturbation_probe(args, selected, out_dirs)

    status = "completed" if selected is not None else "completed_no_required_instability"
    manifest.update(
        {
            "status": status,
            "completed_at": datetime.now().isoformat(),
            "wall_time_sec": time.perf_counter() - t_all,
            "spectra": jsonable(spectra),
            "selected_s": None if selected is None else float(selected["metadata"]["s"]),
            "perturbation_trials": jsonable(perturb_rows),
            "eigenvalue_csv": relpath(out_dirs["data"] / f"{args.run_tag}_eigenvalues.csv"),
            "perturbation_csv": relpath(out_dirs["data"] / f"{args.run_tag}_perturbations.csv"),
        }
    )
    write_json(manifest_path, jsonable(manifest))

    if not args.no_registry:
        eig_summary = "n/a"
        error_metrics = "n/a"
        if selected is not None:
            eig_summary = (
                f"selected_s={selected['metadata']['s']}; "
                f"unstable_count={selected['unstable_count']}; "
                + "; ".join(f"{lam.real:.6e}{lam.imag:+.6e}j" for lam in selected["eigenvalues"])
            )
            residuals = selected["residuals"]["norms"]
            best = next((row for row in perturb_rows if row.get("status") == "completed_distinct_symmetry_breaking_candidate"), None)
            error_metrics = (
                f"profile_R_Omega={residuals['Omega_l2']}; profile_R_Psi={residuals['Psi_l2']}; "
                f"best_perturb={best['trial_id'] if best else 'none'}"
            )
        append_registry_row(
            {
                "date": today(),
                "tag": f"fem-branchB-{args.run_tag}",
                "codebase": "baseline",
                "script": relpath(Path(__file__)),
                "input_case": f"s_cos{int(args.mode)}theta_mu{float(args.mu):.6g}",
                "domain": f"disk_Rdom{float(args.Rdom):g}",
                "grid_or_mesh": f"h={float(args.mesh[0]):g}/{float(args.mesh[1]):g}",
                "solver": "baseline_profile_eigs_artificial_perturbation_fixed_boundary_newton",
                "parameters": (
                    f"s_start={args.s_start}; s_step={args.s_step}; s_max={args.s_max}; "
                    f"k_eigs={args.k_eigs}; required_unstable={args.required_unstable}; "
                    f"perturb_modes={args.perturb_modes}; etas={args.etas}"
                ),
                "wall_time_sec": f"{manifest['wall_time_sec']:.6f}",
                "error_metrics": error_metrics,
                "eigenvalues_summary": eig_summary,
                "status": status,
                "notes": "C4/mu=2/3 symmetry-breaking probe using exact s cos(4theta) boundary data.",
            }
        )
    print(f"[done] status={status} manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
