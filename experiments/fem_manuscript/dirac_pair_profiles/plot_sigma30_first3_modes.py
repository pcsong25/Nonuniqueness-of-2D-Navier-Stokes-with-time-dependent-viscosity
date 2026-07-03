#!/usr/bin/env python3
"""Plot first three eigenfunctions for the sigma=30 regularized two-Dirac profile."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time
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
sys.path.insert(0, str(MANUSCRIPT_ROOT))
sys.path.insert(0, str(C3_BRANCH_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import add_baseline_to_path, append_registry_row, today, write_json  # noqa: E402
from run_dirac_pair_eigenvalues import load_h5_profile, vec_array  # noqa: E402
from run_dirac_pair_profile import coeffs_for_regularized_dirac_pair  # noqa: E402
from run_spectrum_continuation import build_baseline_context, jsonable, mesh_stats  # noqa: E402


BRANCH_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE_H5 = (
    BRANCH_DIR
    / "checkpoints/dirac_pair_eps0p01_M50_h0p06_0p06_sigma_ladder/sigma30_eps0p01_M50_h0p06_0p06_profile.h5"
)
DEFAULT_OUTPUT_DIR = BRANCH_DIR / "plots/dirac_pair_sigma30_first3_eigenmodes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma", type=float, default=30.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--truncation-M", type=int, default=50)
    parser.add_argument("--mesh", nargs=2, type=float, metavar=("H_MIN", "H_MAX"), default=[0.06, 0.06])
    parser.add_argument("--mu", type=float, default=0.8)
    parser.add_argument("--R", type=float, default=150.0)
    parser.add_argument("--Rdom", type=float, default=10.0)
    parser.add_argument("--tol-newton", type=float, default=1e-8)
    parser.add_argument("--max-newton", type=int, default=48)
    parser.add_argument("--profile-h5", type=Path, default=DEFAULT_PROFILE_H5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--eig-shift", type=float, default=-1.0)
    parser.add_argument("--eig-tol", type=float, default=1e-8)
    parser.add_argument("--eig-max-it", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually run the eigenmode solve; otherwise write a dry-run manifest.")
    return parser.parse_args()


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def normalize(values: np.ndarray) -> np.ndarray:
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


def style_axis(ax: plt.Axes, radius: float = 10.0) -> None:
    ax.set_aspect("equal", "box")
    ax.set_xlim(-1.02 * radius, 1.02 * radius)
    ax.set_ylim(-1.02 * radius, 1.02 * radius)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("0.25")
    ax.add_patch(plt.Circle((0.0, 0.0), radius, fill=False, lw=0.75, color="0.2", alpha=0.55))


def plot_panel(tri: mtri.Triangulation, fields: list[np.ndarray], lambdas: list[complex], out_base: Path, *, label: str) -> None:
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
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.85), constrained_layout=True)
    levels = np.linspace(-1.0, 1.0, 512)
    cf = None
    for idx, (ax, values, lam) in enumerate(zip(axes, fields, lambdas), start=1):
        z = normalize(values)
        cf = ax.tricontourf(tri, z, levels=levels, cmap="seismic")
        style_axis(ax)
        ax.set_title(rf"$j={idx}$, $\lambda={lam.real:.4f}{lam.imag:+.2e}i$", fontsize=13, pad=8)
    assert cf is not None
    cb = fig.colorbar(cf, ax=axes, fraction=0.026, pad=0.015)
    cb.set_label(label, fontsize=13)
    cb.ax.tick_params(labelsize=11, width=0.6, length=2.5)
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
    np.savez_compressed(path, **payload)


def write_eigenvalue_csv(path: Path, lambdas: list[complex]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["eigen_index", "real_lambda", "imag_lambda"])
        writer.writeheader()
        for idx, lam in enumerate(lambdas, start=1):
            writer.writerow({"eigen_index": idx, "real_lambda": float(lam.real), "imag_lambda": float(lam.imag)})


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "sigma30_first3_eigenmodes_manifest.json"
    cache_path = args.output_dir / "sigma30_first3_mode_arrays.npz"
    omega_base = args.output_dir / "sigma30_first3_OmegaRe_heatmap_panel"
    psi_base = args.output_dir / "sigma30_first3_PsiRe_heatmap_panel"
    csv_path = args.output_dir / "sigma30_first3_eigenvalues.csv"

    manifest = {
        "status": "initialized" if args.execute else "dry_run",
        "sigma": float(args.sigma),
        "epsilon": float(args.epsilon),
        "truncation_M": int(args.truncation_M),
        "profile_h5": relpath(args.profile_h5),
        "k": int(args.k),
        "eig_shift": float(args.eig_shift),
        "eig_tol": float(args.eig_tol),
        "eig_max_it": int(args.eig_max_it),
        "output_dir": relpath(args.output_dir),
    }
    write_json(manifest_path, jsonable(manifest))
    if not args.execute:
        print(f"[dry-run] wrote {manifest_path}")
        return 0

    add_baseline_to_path()
    from eigs_linearized_new import ManifoldEigenSolver
    from residuals_quick import residuals_quick

    ns = argparse.Namespace(
        mesh=[float(args.mesh[0]), float(args.mesh[1])],
        mu=float(args.mu),
        R=float(args.R),
        Rdom=float(args.Rdom),
        M=2 * int(args.truncation_M),
        tol_newton=float(args.tol_newton),
        max_newton=int(args.max_newton),
        sigma_steps=2,
    )
    coeffs = coeffs_for_regularized_dirac_pair(args.sigma, args.truncation_M, args.epsilon)
    context = build_baseline_context(ns, coeffs)
    omega, psi, geom_diff = load_h5_profile(args.profile_h5, context["mesh"], context["spaces"])
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
    stats = mesh_stats(context["mesh"], context["spaces"])

    t0 = time.perf_counter()
    esolver = ManifoldEigenSolver(
        context["mesh"],
        context["spaces"],
        omega,
        psi,
        context["params"],
        context["facet_tags"],
        outer_tag=1,
    )
    modes = esolver.solve(k=int(args.k), sigma=float(args.eig_shift), tol=float(args.eig_tol), max_it=int(args.eig_max_it), verbose=True)
    eig_wall = time.perf_counter() - t0
    modes = modes[: int(args.k)]
    lambdas = [complex(mode.lam) for mode in modes]
    save_mode_cache(cache_path, modes)
    write_eigenvalue_csv(csv_path, lambdas)

    tri_omega = triangulation_vertices(context["mesh"])
    tri_psi = triangulation_dofs(context["mesh"], context["spaces"]["Vpsi"])
    plot_panel(
        tri_omega,
        [vec_array(mode.omega_re).copy() for mode in modes],
        lambdas,
        omega_base,
        label=r"normalized $\Re(\delta\Omega)$",
    )
    plot_panel(
        tri_psi,
        [vec_array(mode.psi_re).copy() for mode in modes],
        lambdas,
        psi_base,
        label=r"normalized $\Re(\delta\psi)$",
    )

    manifest.update(
        {
            "status": "completed",
            "mesh_stats": stats,
            "profile_checkpoint_geometry_diff": geom_diff,
            "profile_residuals": {
                "Omega_l2": float(residuals["norms"]["Omega_l2"]),
                "Psi_l2": float(residuals["norms"]["Psi_l2"]),
            },
            "eigs_wall_time_sec": eig_wall,
            "eigenvalues": [
                {"index": idx, "real": float(lam.real), "imag": float(lam.imag)}
                for idx, lam in enumerate(lambdas, start=1)
            ],
            "outputs": {
                "mode_cache": relpath(cache_path),
                "eigenvalue_csv": relpath(csv_path),
                "omega_heatmap_pdf": relpath(omega_base.with_suffix(".pdf")),
                "omega_heatmap_png": relpath(omega_base.with_suffix(".png")),
                "psi_heatmap_pdf": relpath(psi_base.with_suffix(".pdf")),
                "psi_heatmap_png": relpath(psi_base.with_suffix(".png")),
            },
        }
    )
    write_json(manifest_path, jsonable(manifest))
    append_registry_row(
        {
            "date": today(),
            "tag": "fem-dirac-pair-sigma30-first3-eigenmode-plots",
            "codebase": "baseline",
            "script": relpath(Path(__file__)),
            "input_case": "regularized_two_dirac_pair_sigma30_eps0p01_M50",
            "domain": "disk_Rdom10",
            "grid_or_mesh": "h=0.06/0.06",
            "solver": "saved_profile_shift_invert_eigenmode_reconstruction",
            "parameters": f"k={args.k}; eig_shift={args.eig_shift}; eig_tol={args.eig_tol}",
            "wall_time_sec": f"{eig_wall:.6f}",
            "error_metrics": f"profile_weak_R_Omega={residuals['norms']['Omega_l2']}; profile_weak_R_Psi={residuals['norms']['Psi_l2']}",
            "eigenvalues_summary": "; ".join(f"{lam.real:.6e}{lam.imag:+.6e}j" for lam in lambdas),
            "status": "completed",
            "notes": f"Generated first three normalized vorticity and streamfunction eigenmode panels in {relpath(args.output_dir)}.",
        }
    )
    print(f"[done] {omega_base.with_suffix('.pdf')}")
    print(f"[done] {psi_base.with_suffix('.pdf')}")
    for idx, lam in enumerate(lambdas, start=1):
        print(f"#{idx:2d}: lambda = {lam.real:.15e} {lam.imag:+.15e}j")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
