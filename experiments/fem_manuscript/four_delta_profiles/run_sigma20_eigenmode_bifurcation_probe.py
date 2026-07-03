#!/usr/bin/env python3
"""Four-delta sigma=20 eigenmode plots and eigenmode perturbation probe."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig-euler")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

MANUSCRIPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MANUSCRIPT_ROOT.parents[1]
C3_BRANCH_DIR = MANUSCRIPT_ROOT / "c3_spectrum_continuation"
DIRAC_DIR = MANUSCRIPT_ROOT / "dirac_pair_profiles"
SYMB_DIR = MANUSCRIPT_ROOT / "symmetry_breaking_profiles"
sys.path.insert(0, str(MANUSCRIPT_ROOT))
sys.path.insert(0, str(C3_BRANCH_DIR))
sys.path.insert(0, str(DIRAC_DIR))
sys.path.insert(0, str(SYMB_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import add_baseline_to_path, append_registry_row, mesh_label, today, write_json  # noqa: E402
from run_dirac_pair_eigenvalues import copy_function, load_h5_profile, vec_array  # noqa: E402
from run_four_delta_sigma_ladder import coeffs_for_four_delta, set_exact_boundary_values  # noqa: E402
from run_s140_bifurcation_probe import boundary_max_abs_diff, linf, rms, run_newton_from, write_xdmf_profile  # noqa: E402
from run_spectrum_continuation import build_baseline_context, jsonable, mesh_stats  # noqa: E402


BRANCH_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE_H5 = (
    BRANCH_DIR
    / "checkpoints/four_delta_eps0p01_M30_h0p06_0p06_sigma_ladder/sigma20_eps0p01_M30_h0p06_0p06_profile.h5"
)

EIG_COLUMNS = [
    "profile_id",
    "sigma",
    "epsilon",
    "truncation_M",
    "h_min",
    "h_max",
    "eigen_index",
    "real_lambda",
    "imag_lambda",
    "profile_residual_omega_l2",
    "profile_residual_psi_l2",
    "profile_checkpoint",
    "metadata_checkpoint",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--sigma", type=float, default=20.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--truncation-M", type=int, default=30)
    parser.add_argument("--mesh", nargs=2, type=float, metavar=("H_MIN", "H_MAX"), default=[0.06, 0.06])
    parser.add_argument("--mu", type=float, default=0.8)
    parser.add_argument("--R", type=float, default=150.0)
    parser.add_argument("--Rdom", type=float, default=10.0)
    parser.add_argument("--profile-h5", type=Path, default=DEFAULT_PROFILE_H5)
    parser.add_argument("--mode-k", type=int, default=2)
    parser.add_argument("--candidate-k", type=int, default=5)
    parser.add_argument("--eig-shift", type=float, default=-1.0)
    parser.add_argument("--eig-tol", type=float, default=1e-8)
    parser.add_argument("--eig-max-it", type=int, default=300)
    parser.add_argument("--perturb-mode", type=int, default=1)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--sign", type=float, default=1.0)
    parser.add_argument("--tol-newton", type=float, default=1e-8)
    parser.add_argument("--max-newton", type=int, default=48)
    parser.add_argument("--no-linesearch", action="store_true")
    parser.add_argument("--run-tag", default="four_delta_sigma20_mode1_bifurcation_probe")
    parser.add_argument("--force-modes", action="store_true")
    parser.add_argument("--force-candidate", action="store_true")
    return parser.parse_args()


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def label_token(text: str) -> str:
    return text.replace("+", "p").replace("-", "m").replace(".", "p")


def append_rows(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "n/a") for key in columns})


def write_failure_log(path: Path, exc: BaseException) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(f"\n[{datetime.now().isoformat()}] failed\n")
        f.write(repr(exc))
        f.write("\n")
        f.write(traceback.format_exc())
        f.write("\n")


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    if scale <= 0.0:
        return values.copy()
    return values / scale


def triangulation_vertices(mesh: Any) -> mtri.Triangulation:
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    cells = mesh.topology.connectivity(tdim, 0).array.reshape(-1, 3)
    xy = mesh.geometry.x[:, :2]
    return mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=cells)


def triangulation_dofs(mesh: Any, V: Any) -> mtri.Triangulation:
    raw = V.tabulate_dof_coordinates()
    if raw.ndim == 1:
        raw = raw.reshape(-1, mesh.geometry.x.shape[1])
    xy = raw[:, :2]
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    tri_dofs = np.array([V.dofmap.cell_dofs(c) for c in range(num_cells)], dtype=np.int32)
    return mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=tri_dofs)


def style_axis(ax: plt.Axes, radius: float) -> None:
    ax.set_aspect("equal", "box")
    ax.set_xlim(-1.02 * radius, 1.02 * radius)
    ax.set_ylim(-1.02 * radius, 1.02 * radius)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("0.25")
    ax.add_patch(plt.Circle((0.0, 0.0), radius, fill=False, lw=0.75, color="0.2", alpha=0.55))


def plot_mode_panel(
    tri: mtri.Triangulation,
    fields: list[np.ndarray],
    lambdas: list[complex],
    out_base: Path,
    *,
    label: str,
    radius: float,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 14,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "figure.dpi": 300,
            "savefig.dpi": 600,
        }
    )
    fig, axes = plt.subplots(1, len(fields), figsize=(4.3 * len(fields), 3.85), constrained_layout=True)
    if len(fields) == 1:
        axes = np.asarray([axes])
    levels = np.linspace(-1.0, 1.0, 512)
    cf = None
    for idx, (ax, values, lam) in enumerate(zip(axes, fields, lambdas), start=1):
        cf = ax.tricontourf(tri, normalize(values), levels=levels, cmap="seismic")
        style_axis(ax, radius)
        ax.set_title(rf"$j={idx}$, $\lambda={lam.real:.4f}{lam.imag:+.2e}i$", fontsize=13, pad=8)
    assert cf is not None
    cb = fig.colorbar(cf, ax=axes, fraction=0.035, pad=0.018)
    cb.set_label(label, fontsize=13)
    cb.ax.tick_params(labelsize=11, width=0.6, length=2.5)
    cb.outline.set_linewidth(0.6)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_profile_comparison(
    tri: mtri.Triangulation,
    omega_sym: Any,
    omega_candidate: Any,
    out_base: Path,
    *,
    radius: float,
) -> None:
    sym = np.asarray(vec_array(omega_sym), dtype=float)
    cand = np.asarray(vec_array(omega_candidate), dtype=float)
    diff = cand - sym
    vmax = max(linf(sym), linf(cand), 1e-14)
    diff_vmax = max(linf(diff), 1e-14)
    fields = [(sym, vmax, "symmetric"), (cand, vmax, "perturbed Newton solution"), (diff, diff_vmax, "difference")]
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 12,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "figure.dpi": 300,
            "savefig.dpi": 600,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.85), constrained_layout=True)
    for ax, (values, local_vmax, title) in zip(axes, fields):
        levels = np.linspace(-local_vmax, local_vmax, 512)
        cf = ax.tricontourf(tri, values, levels=levels, cmap="seismic")
        style_axis(ax, radius)
        ax.set_title(title, pad=7)
        cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.016)
        cb.ax.tick_params(labelsize=9, width=0.6, length=2.5)
        cb.outline.set_linewidth(0.6)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def save_mode_cache(path: Path, modes: list[Any]) -> None:
    payload: dict[str, Any] = {
        "n_modes": np.array([len(modes)], dtype=np.int64),
        "real_lambda": np.array([complex(mode.lam).real for mode in modes], dtype=float),
        "imag_lambda": np.array([complex(mode.lam).imag for mode in modes], dtype=float),
    }
    for idx, mode in enumerate(modes, start=1):
        payload[f"omega_re_{idx}"] = vec_array(mode.omega_re).copy()
        payload[f"omega_im_{idx}"] = vec_array(mode.omega_im).copy()
        payload[f"psi_re_{idx}"] = vec_array(mode.psi_re).copy()
        payload[f"psi_im_{idx}"] = vec_array(mode.psi_im).copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_mode_cache(path: Path, spaces: dict[str, Any], n_modes: int) -> list[Any]:
    from dolfinx import fem
    from run_dirac_pair_eigenvalues import ghost_forward

    modes = []
    with np.load(path) as data:
        n_loaded = min(int(data["n_modes"][0]), int(n_modes))
        for idx in range(1, n_loaded + 1):
            omega_re = fem.Function(spaces["Vomega"])
            omega_im = fem.Function(spaces["Vomega"])
            psi_re = fem.Function(spaces["Vpsi"])
            psi_im = fem.Function(spaces["Vpsi"])
            vec_array(omega_re)[:] = data[f"omega_re_{idx}"]
            vec_array(omega_im)[:] = data[f"omega_im_{idx}"]
            vec_array(psi_re)[:] = data[f"psi_re_{idx}"]
            vec_array(psi_im)[:] = data[f"psi_im_{idx}"]
            for fun in (omega_re, omega_im, psi_re, psi_im):
                ghost_forward(fun)
            modes.append(
                SimpleNamespace(
                    lam=complex(float(data["real_lambda"][idx - 1]), float(data["imag_lambda"][idx - 1])),
                    omega_re=omega_re,
                    omega_im=omega_im,
                    psi_re=psi_re,
                    psi_im=psi_im,
                )
            )
    return modes


def build_context(args: argparse.Namespace) -> dict[str, Any]:
    ns = argparse.Namespace(
        mesh=[float(args.mesh[0]), float(args.mesh[1])],
        mu=float(args.mu),
        R=float(args.R),
        Rdom=float(args.Rdom),
        M=2 * int(args.truncation_M),
        tol_newton=float(args.tol_newton),
        max_newton=int(args.max_newton),
        sigma_steps=21,
    )
    coeffs = coeffs_for_four_delta(args.sigma, args.truncation_M, args.epsilon)
    return build_baseline_context(ns, coeffs)


def write_eig_csv(path: Path, profile_id: str, args: argparse.Namespace, lambdas: list[complex], residuals: dict[str, Any], profile_h5: Path, metadata_path: Path) -> None:
    rows = [
        {
            "profile_id": profile_id,
            "sigma": float(args.sigma),
            "epsilon": float(args.epsilon),
            "truncation_M": int(args.truncation_M),
            "h_min": float(args.mesh[0]),
            "h_max": float(args.mesh[1]),
            "eigen_index": idx,
            "real_lambda": float(lam.real),
            "imag_lambda": float(lam.imag),
            "profile_residual_omega_l2": float(residuals["norms"]["Omega_l2"]),
            "profile_residual_psi_l2": float(residuals["norms"]["Psi_l2"]),
            "profile_checkpoint": relpath(profile_h5),
            "metadata_checkpoint": relpath(metadata_path),
            "status": "completed",
        }
        for idx, lam in enumerate(lambdas, start=1)
    ]
    append_rows(path, EIG_COLUMNS, rows)


def main() -> int:
    args = parse_args()
    data_dir = BRANCH_DIR / "data"
    log_dir = BRANCH_DIR / "logs"
    plot_dir = BRANCH_DIR / "plots" / args.run_tag
    checkpoint_dir = BRANCH_DIR / "checkpoints" / args.run_tag
    for path in (data_dir, log_dir, plot_dir, checkpoint_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = log_dir / f"{args.run_tag}_manifest.json"
    mode_cache = plot_dir / "sigma20_first2_mode_arrays.npz"
    mode_csv = data_dir / f"{args.run_tag}_symmetric_first2_eigenvalues.csv"
    candidate_eig_csv = data_dir / f"{args.run_tag}_candidate_eigenvalues.csv"
    trial_id = f"mode{args.perturb_mode}_eta{label_token(f'{args.eta:g}')}_sign{label_token(f'{args.sign:+g}')}"
    candidate_xdmf = checkpoint_dir / f"{trial_id}_profile.xdmf"
    candidate_metadata = checkpoint_dir / f"{trial_id}_metadata.json"
    candidate_plot = plot_dir / f"{trial_id}_omega_comparison"

    manifest = {
        "status": "initialized" if args.execute else "dry_run",
        "run_tag": args.run_tag,
        "sigma": float(args.sigma),
        "epsilon": float(args.epsilon),
        "truncation_M": int(args.truncation_M),
        "mesh": {"h_min": float(args.mesh[0]), "h_max": float(args.mesh[1])},
        "profile_h5": relpath(args.profile_h5),
        "mode_k": int(args.mode_k),
        "candidate_k": int(args.candidate_k),
        "eig_shift": float(args.eig_shift),
        "eig_tol": float(args.eig_tol),
        "perturbation": {
            "mode": int(args.perturb_mode),
            "eta": float(args.eta),
            "sign": float(args.sign),
            "formula": "Omega0=Omega_sym+sign*eta*rms(Omega_sym)/rms(Re(deltaOmega_j))*Re(deltaOmega_j); psi0 solved from perturbed Omega0.",
            "boundary_policy": "exact four-delta boundary values are reimposed after adding the eigenmode.",
        },
        "outputs": {
            "mode_cache": relpath(mode_cache),
            "mode_omega_plot": relpath(plot_dir / "sigma20_first2_OmegaRe_heatmap_panel.pdf"),
            "mode_psi_plot": relpath(plot_dir / "sigma20_first2_PsiRe_heatmap_panel.pdf"),
            "candidate_profile_h5": relpath(candidate_xdmf.with_suffix(".h5")),
            "candidate_comparison_plot": relpath(candidate_plot.with_suffix(".pdf")),
            "candidate_eigenvalues_csv": relpath(candidate_eig_csv),
        },
    }
    write_json(manifest_path, jsonable(manifest))
    if not args.execute:
        print(f"[dry-run] wrote {manifest_path}")
        return 0

    add_baseline_to_path()
    from eigs_linearized_new import ManifoldEigenSolver
    from residuals_quick import residuals_quick

    t_all = time.perf_counter()
    try:
        context = build_context(args)
        stats = mesh_stats(context["mesh"], context["spaces"])
        omega_sym, psi_sym, geom_diff = load_h5_profile(args.profile_h5, context["mesh"], context["spaces"])
        pinned = getattr(context["poisson"], "gauge_dof", None)
        sym_residuals = residuals_quick(
            context["mesh"],
            context["spaces"],
            omega_sym,
            psi_sym,
            mu=context["params"].mu,
            R=context["params"].R,
            Rdom=context["params"].Rdom,
            facet_tags=context["facet_tags"],
            outer_tag=1,
            fourier_bc=context["fourier_bc"],
            sigma=1.0,
            pinned_psi_dof=pinned,
        )

        if mode_cache.exists() and not args.force_modes:
            modes = load_mode_cache(mode_cache, context["spaces"], args.mode_k)
            mode_wall = 0.0
            print(f"[resume] loaded mode cache {mode_cache}")
        else:
            mode_t0 = time.perf_counter()
            esolver = ManifoldEigenSolver(
                context["mesh"],
                context["spaces"],
                omega_sym,
                psi_sym,
                context["params"],
                context["facet_tags"],
                outer_tag=1,
            )
            modes = esolver.solve(
                k=int(args.mode_k),
                sigma=float(args.eig_shift),
                tol=float(args.eig_tol),
                max_it=int(args.eig_max_it),
                verbose=True,
            )[: int(args.mode_k)]
            mode_wall = time.perf_counter() - mode_t0
            save_mode_cache(mode_cache, modes)

        lambdas = [complex(mode.lam) for mode in modes]
        write_eig_csv(mode_csv, "symmetric", args, lambdas, sym_residuals, args.profile_h5, manifest_path)
        tri_omega = triangulation_vertices(context["mesh"])
        tri_psi = triangulation_dofs(context["mesh"], context["spaces"]["Vpsi"])
        plot_mode_panel(
            tri_omega,
            [vec_array(mode.omega_re).copy() for mode in modes],
            lambdas,
            plot_dir / "sigma20_first2_OmegaRe_heatmap_panel",
            label=r"normalized $\Re(\delta\Omega)$",
            radius=float(args.Rdom),
        )
        plot_mode_panel(
            tri_psi,
            [vec_array(mode.psi_re).copy() for mode in modes],
            lambdas,
            plot_dir / "sigma20_first2_PsiRe_heatmap_panel",
            label=r"normalized $\Re(\delta\psi)$",
            radius=float(args.Rdom),
        )

        if int(args.perturb_mode) < 1 or int(args.perturb_mode) > len(modes):
            raise ValueError(f"perturb-mode must be between 1 and {len(modes)}")
        mode = modes[int(args.perturb_mode) - 1]
        scale = float(args.sign) * float(args.eta) * rms(vec_array(omega_sym)) / max(rms(vec_array(mode.omega_re)), 1e-30)
        omega0 = copy_function(context["spaces"]["Vomega"], vec_array(omega_sym) + scale * vec_array(mode.omega_re))
        set_exact_boundary_values(context, omega0)
        psi0 = context["poisson"].solve_state(omega0, sigma=1.0)
        init_boundary_diff = boundary_max_abs_diff(context["spaces"], context["facet_tags"], omega0, omega_sym)

        candidate_status = "profile_loaded"
        if candidate_metadata.exists() and candidate_xdmf.with_suffix(".h5").exists() and not args.force_candidate:
            omega_candidate, psi_candidate, cand_geom_diff = load_h5_profile(candidate_xdmf.with_suffix(".h5"), context["mesh"], context["spaces"])
            with candidate_metadata.open() as f:
                import json

                cand_meta = json.load(f)
            candidate_wall = 0.0
            history = cand_meta.get("newton_history", [])
            print(f"[resume] loaded candidate profile {candidate_xdmf.with_suffix('.h5')}")
        else:
            candidate_status = "newton_started"
            candidate_t0 = time.perf_counter()
            omega_candidate, psi_candidate, history, converged = run_newton_from(
                context,
                omega0,
                psi0,
                max_newton=int(args.max_newton),
                tol=float(args.tol_newton),
                linesearch=not args.no_linesearch,
            )
            candidate_wall = time.perf_counter() - candidate_t0
            if not converged:
                raise RuntimeError(f"perturbed Newton did not converge after {args.max_newton} iterations")
            cand_geom_diff = 0.0
            write_xdmf_profile(candidate_xdmf, context["mesh"], omega_candidate, psi_candidate)

        cand_residuals = residuals_quick(
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
        boundary_diff = boundary_max_abs_diff(context["spaces"], context["facet_tags"], omega_candidate, omega_sym)
        plot_profile_comparison(tri_omega, omega_sym, omega_candidate, candidate_plot, radius=float(args.Rdom))

        final_history = history[-1] if history else {}
        candidate_profile_manifest = {
            "status": "profile_completed_eigs_started",
            "run_tag": args.run_tag,
            "sigma": float(args.sigma),
            "epsilon": float(args.epsilon),
            "truncation_M": int(args.truncation_M),
            "mesh_stats": stats,
            "profile_checkpoint_geometry_diff": geom_diff,
            "candidate_checkpoint_geometry_diff": cand_geom_diff,
            "source_eigenvalue": {"real": float(complex(mode.lam).real), "imag": float(complex(mode.lam).imag)},
            "perturbation_scale": scale,
            "initial_boundary_max_abs_diff": init_boundary_diff,
            "newton_status": candidate_status,
            "newton_history": history,
            "newton_wall_time_sec": candidate_wall,
            "final_newton_residual": final_history.get("residual_norm", "n/a"),
            "final_relative_step": final_history.get("relative_step", "n/a"),
            "symmetric_profile_residuals": {
                "Omega_l2": float(sym_residuals["norms"]["Omega_l2"]),
                "Psi_l2": float(sym_residuals["norms"]["Psi_l2"]),
            },
            "candidate_profile_residuals": {
                "Omega_l2": float(cand_residuals["norms"]["Omega_l2"]),
                "Psi_l2": float(cand_residuals["norms"]["Psi_l2"]),
            },
            "separation": {
                "omega_rms": rms(omega_sep),
                "omega_relative_rms": rms(omega_sep) / max(rms(vec_array(omega_sym)), 1e-30),
                "omega_linf": linf(omega_sep),
                "psi_rms": rms(psi_sep),
                "psi_relative_rms": rms(psi_sep) / max(rms(vec_array(psi_sym)), 1e-30),
                "psi_linf": linf(psi_sep),
                "boundary_max_abs_diff": boundary_diff,
            },
            "outputs": manifest["outputs"],
        }
        write_json(candidate_metadata, jsonable(candidate_profile_manifest))

        eig_t0 = time.perf_counter()
        cand_solver = ManifoldEigenSolver(
            context["mesh"],
            context["spaces"],
            omega_candidate,
            psi_candidate,
            context["params"],
            context["facet_tags"],
            outer_tag=1,
        )
        cand_modes = cand_solver.solve(
            k=int(args.candidate_k),
            sigma=float(args.eig_shift),
            tol=float(args.eig_tol),
            max_it=int(args.eig_max_it),
            verbose=True,
        )[: int(args.candidate_k)]
        cand_eig_wall = time.perf_counter() - eig_t0
        cand_lambdas = [complex(mode.lam) for mode in cand_modes]
        write_eig_csv(
            candidate_eig_csv,
            trial_id,
            args,
            cand_lambdas,
            cand_residuals,
            candidate_xdmf.with_suffix(".h5"),
            candidate_metadata,
        )

        candidate_manifest = {
            **candidate_profile_manifest,
            "status": "completed",
            "candidate_eigs": {
                "wall_time_sec": cand_eig_wall,
                "k_requested": int(args.candidate_k),
                "shift": float(args.eig_shift),
                "tol": float(args.eig_tol),
                "min_real_lambda": float(min(lam.real for lam in cand_lambdas)),
                "unstable_count_re_lt_0": int(sum(1 for lam in cand_lambdas if lam.real < 0.0)),
                "eigenvalues": [
                    {"index": idx, "real": float(lam.real), "imag": float(lam.imag)}
                    for idx, lam in enumerate(cand_lambdas, start=1)
                ],
            },
        }
        write_json(candidate_metadata, jsonable(candidate_manifest))

        manifest.update(
            {
                "status": "completed",
                "mesh_stats": stats,
                "profile_checkpoint_geometry_diff": geom_diff,
                "mode_wall_time_sec": mode_wall,
                "candidate_metadata": relpath(candidate_metadata),
                "candidate_separation": candidate_manifest["separation"],
                "candidate_eigenvalues": candidate_manifest["candidate_eigs"]["eigenvalues"],
                "wall_time_sec": time.perf_counter() - t_all,
            }
        )
        write_json(manifest_path, jsonable(manifest))
        append_registry_row(
            {
                "date": today(),
                "tag": f"fem-four-delta-{args.run_tag}",
                "codebase": "baseline",
                "script": relpath(Path(__file__)),
                "input_case": "regularized_four_delta_sigma20_eigenmode_probe",
                "domain": f"disk_Rdom{float(args.Rdom):g}",
                "grid_or_mesh": f"h={float(args.mesh[0]):g}/{float(args.mesh[1]):g}",
                "solver": "saved_profile_eigenmode_reconstruction_then_fixed_boundary_newton_and_eigs",
                "parameters": f"sigma={args.sigma}; epsilon={args.epsilon}; M={args.truncation_M}; perturb_mode={args.perturb_mode}; eta={args.eta}; k={args.candidate_k}",
                "wall_time_sec": f"{manifest['wall_time_sec']:.6f}",
                "error_metrics": (
                    f"candidate_R_Omega={candidate_manifest['candidate_profile_residuals']['Omega_l2']}; "
                    f"omega_rel_sep={candidate_manifest['separation']['omega_relative_rms']}"
                ),
                "eigenvalues_summary": "; ".join(f"{lam.real:.6e}{lam.imag:+.6e}j" for lam in cand_lambdas),
                "status": "completed",
                "notes": f"Mode plots and candidate comparison in {relpath(plot_dir)}.",
            }
        )
        print(f"[done] omega_modes={plot_dir / 'sigma20_first2_OmegaRe_heatmap_panel.pdf'}")
        print(f"[done] candidate_profile={candidate_xdmf.with_suffix('.h5')}")
        print(f"[done] candidate_eigs={candidate_eig_csv}")
        for idx, lam in enumerate(cand_lambdas, start=1):
            print(f"candidate #{idx:2d}: lambda = {lam.real:.15e} {lam.imag:+.15e}j")
        return 0
    except Exception as exc:
        manifest.update({"status": "failed", "error": repr(exc), "wall_time_sec": time.perf_counter() - t_all})
        write_json(manifest_path, jsonable(manifest))
        write_failure_log(log_dir / f"{args.run_tag}.log", exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
