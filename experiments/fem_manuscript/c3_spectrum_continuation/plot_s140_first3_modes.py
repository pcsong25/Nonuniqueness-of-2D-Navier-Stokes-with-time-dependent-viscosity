#!/usr/bin/env python3
"""Plot the first three s=140 FEM eigenmodes for Branch A.

This script avoids recomputing the nonlinear profile. It rebuilds the same
h_min=h_max=0.06 baseline mesh, checks it matches the saved checkpoint mesh,
loads checkpointed Omega/psi arrays, reruns the linearized eigensolver, and
plots the real parts of the first three eigenmodes.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig-euler")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

BRANCH_DIR = Path(__file__).resolve().parent
MANUSCRIPT_ROOT = BRANCH_DIR.parents[0]
REPO_ROOT = MANUSCRIPT_ROOT.parents[1]
sys.path.insert(0, str(MANUSCRIPT_ROOT))
sys.path.insert(0, str(BRANCH_DIR))

from common import add_baseline_to_path, coeffs_for_exact_c3_cos, write_json  # noqa: E402
from run_spectrum_continuation import build_baseline_context, mesh_stats  # noqa: E402


def vec_array(fun: Any) -> np.ndarray:
    try:
        return fun.vector.array
    except AttributeError:
        return fun.x.array


def ghost_forward(fun: Any) -> None:
    from petsc4py import PETSc

    try:
        fun.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    except AttributeError:
        if hasattr(fun.x, "scatter_forward"):
            fun.x.scatter_forward()


def normalize(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    if scale <= 0.0:
        return values.copy()
    return values / scale


def safe_name(value: complex) -> str:
    text = f"{value.real:+.6e}{value.imag:+.6e}i"
    return (
        text.replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
        .replace("i", "i")
    )


def save_figure(fig: Any, path_stem: Path) -> None:
    fig.savefig(str(path_stem) + ".png")
    fig.savefig(str(path_stem) + ".pdf")


def build_triangulations(mesh: Any, Vpsi: Any) -> tuple[mtri.Triangulation, mtri.Triangulation]:
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    conn = mesh.topology.connectivity(tdim, 0)
    cells = conn.array.reshape(-1, 3)
    xy = mesh.geometry.x[:, :2]
    tri_vertices = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=cells)

    raw = Vpsi.tabulate_dof_coordinates()
    if raw.ndim == 1:
        raw = raw.reshape(-1, mesh.geometry.x.shape[1])
    xy_dofs = raw[:, :2]
    num_cells = mesh.topology.index_map(tdim).size_local
    tri_dofs = np.array([Vpsi.dofmap.cell_dofs(c) for c in range(num_cells)], dtype=np.int32)
    tri_psi = mtri.Triangulation(xy_dofs[:, 0], xy_dofs[:, 1], triangles=tri_dofs)
    return tri_vertices, tri_psi


def plot_heatmap(tri: mtri.Triangulation, values: np.ndarray, lam: complex, path: Path, title: str, cbar_label: str) -> None:
    z = normalize(values)
    vmax = max(1.0, float(np.max(np.abs(z))))
    fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=220)
    cf = ax.tricontourf(tri, z, levels=80, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03, label=cbar_label)
    ax.set_aspect("equal", "box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title + "\n" + rf"$\lambda={lam.real:.6f}{lam.imag:+.6f}i$", fontsize=10)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def plot_streamlines(tri: mtri.Triangulation, values: np.ndarray, lam: complex, path: Path, title: str) -> None:
    z = normalize(values)
    vmax = max(1.0, float(np.max(np.abs(z))))
    levels = np.linspace(-vmax, vmax, 33)
    fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=220)
    cf = ax.tricontourf(tri, z, levels=levels, cmap="RdBu_r", alpha=0.86)
    ax.tricontour(tri, z, levels=levels, colors="k", linewidths=0.45, alpha=0.8)
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03, label=r"normalized $\Re(\delta\psi)$")
    ax.set_aspect("equal", "box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title + "\n" + rf"$\lambda={lam.real:.6f}{lam.imag:+.6f}i$", fontsize=10)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def plot_panel(tri: mtri.Triangulation, modes: list[Any], field: str, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.35), dpi=220, constrained_layout=True)
    for j, (ax, mode) in enumerate(zip(axes, modes), start=1):
        lam = complex(mode.lam)
        fun = mode.omega_re if field == "omega" else mode.psi_re
        z = normalize(vec_array(fun))
        levels = np.linspace(-1.0, 1.0, 41)
        cf = ax.tricontourf(tri, z, levels=levels, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        if field == "psi":
            ax.tricontour(tri, z, levels=levels, colors="k", linewidths=0.35, alpha=0.75)
        ax.set_aspect("equal", "box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"$j={j}$, $\lambda={lam.real:.4f}{lam.imag:+.1e}i$", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.colorbar(cf, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    save_figure(fig, path)
    plt.close(fig)


def plot_manuscript_panel(
    tri: mtri.Triangulation,
    modes: list[Any],
    field: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), dpi=300, constrained_layout=True)
    label = r"normalized $\Re(\delta\Omega)$" if field == "omega" else r"normalized $\Re(\delta\psi)$"
    for ax, mode in zip(axes, modes):
        fun = mode.omega_re if field == "omega" else mode.psi_re
        z = normalize(vec_array(fun))
        levels = np.linspace(-1.0, 1.0, 41)
        cf = ax.tricontourf(tri, z, levels=levels, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        if field == "psi":
            ax.tricontour(tri, z, levels=levels, colors="k", linewidths=0.32, alpha=0.78)
        ax.set_aspect("equal", "box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), fraction=0.026, pad=0.018)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    save_figure(fig, path)
    plt.close(fig)


def write_caption_file(path: Path, rows: list[dict[str, float]]) -> None:
    lam_text = "; ".join(
        rf"$\lambda_{int(row['mode_index'])}={row['real_lambda']:.6f}{row['imag_lambda']:+.6f}i$"
        for row in rows
    )
    path.write_text(
        "\n".join(
            [
                r"% LaTeX-ready caption snippets for s=140 eigenmode figures.",
                r"\newcommand{\sOneFortyEigenvalues}{" + lam_text + r"}",
                r"\newcommand{\sOneFortyOmegaCaption}{Real vorticity components of the first three eigenmodes of the linearized operator at $\sigma=140$. Each panel is normalized by its maximum absolute value. \sOneFortyEigenvalues.}",
                r"\newcommand{\sOneFortyPsiCaption}{Real streamfunction components of the first three eigenmodes of the linearized operator at $\sigma=140$. Filled contours and black contour lines show the same normalized field. \sOneFortyEigenvalues.}",
                r"\newcommand{\sOneFortyOmegaLabel}{fig:sigma140-first3-omega}",
                r"\newcommand{\sOneFortyPsiLabel}{fig:sigma140-first3-psi}",
                "",
            ]
        )
    )


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


def load_mode_cache(path: Path, spaces: dict[str, Any], n_plot: int) -> list[Any]:
    from dolfinx import fem

    with np.load(path) as data:
        real = data["real_lambda"]
        imag = data["imag_lambda"]
        n_modes = min(int(data["n_modes"][0]), int(n_plot))
        modes = []
        for idx in range(1, n_modes + 1):
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
                    lam=complex(float(real[idx - 1]), float(imag[idx - 1])),
                    omega_re=omega_re,
                    omega_im=omega_im,
                    psi_re=psi_re,
                    psi_im=psi_im,
                )
            )
    return modes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=float, default=140.0)
    parser.add_argument("--k-solve", type=int, default=10, help="Solve this many modes, then plot first 3 after sorting by Re(lambda).")
    parser.add_argument("--n-plot", type=int, default=3)
    parser.add_argument("--eig-shift", type=float, default=-1.0)
    parser.add_argument("--eig-tol", type=float, default=1e-8)
    parser.add_argument("--eig-max-it", type=int, default=300)
    parser.add_argument("--checkpoint-h5", type=Path, default=BRANCH_DIR / "checkpoints/branchA_h006_refine_s125_140_k10/s140_h0p06_0p06_profile.h5")
    parser.add_argument("--output-dir", type=Path, default=BRANCH_DIR / "plots/s140_first3_eigenmodes")
    parser.add_argument("--reuse-mode-cache", action="store_true", help="Reuse saved eigenmode arrays if available.")
    args = parser.parse_args()

    add_baseline_to_path()
    from dolfinx import fem
    from eigs_linearized_new import ManifoldEigenSolver

    ns = argparse.Namespace(
        mesh=[0.06, 0.06],
        sigma_steps=21,
        mu=0.8,
        R=150.0,
        Rdom=10.0,
        M=48,
        tol_newton=1e-8,
        max_newton=48,
    )

    t0 = time.perf_counter()
    context = build_baseline_context(ns, coeffs_for_exact_c3_cos(args.s))
    mesh = context["mesh"]
    spaces = context["spaces"]
    omega = fem.Function(spaces["Vomega"])
    psi = fem.Function(spaces["Vpsi"])

    with h5py.File(args.checkpoint_h5, "r") as h5:
        saved_geom = h5["/Mesh/mesh/geometry"][:]
        geom_diff = float(np.max(np.abs(mesh.geometry.x[:, :2] - saved_geom)))
        if geom_diff > 1e-12:
            raise RuntimeError(f"checkpoint mesh does not match rebuilt mesh; max geometry diff={geom_diff:.3e}")
        omega_data = np.asarray(h5["/Function/Omega/0"][:]).reshape(-1)
        psi_data = np.asarray(h5["/Function/psi/0"][:]).reshape(-1)

    if len(vec_array(omega)) != omega_data.size or len(vec_array(psi)) != psi_data.size:
        raise RuntimeError("checkpoint function size does not match rebuilt function space")
    vec_array(omega)[:] = omega_data
    vec_array(psi)[:] = psi_data
    ghost_forward(omega)
    ghost_forward(psi)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mode_cache = args.output_dir / "s140_first3_mode_arrays.npz"
    if args.reuse_mode_cache and mode_cache.exists():
        modes = load_mode_cache(mode_cache, spaces, args.n_plot)
    else:
        esolver = ManifoldEigenSolver(
            mesh,
            spaces,
            omega,
            psi,
            context["params"],
            context["facet_tags"],
            outer_tag=1,
        )
        modes = esolver.solve(
            k=int(args.k_solve),
            sigma=float(args.eig_shift),
            tol=float(args.eig_tol),
            max_it=int(args.eig_max_it),
            verbose=True,
        )
        modes = modes[: int(args.n_plot)]
        save_mode_cache(mode_cache, modes)

    tri_omega, tri_psi = build_triangulations(mesh, spaces["Vpsi"])
    rows = []
    for j, mode in enumerate(modes, start=1):
        lam = complex(mode.lam)
        rows.append({"mode_index": j, "real_lambda": lam.real, "imag_lambda": lam.imag})
        stem = args.output_dir / f"s140_mode{j:02d}_lambda_{safe_name(lam)}"
        plot_heatmap(
            tri_omega,
            vec_array(mode.omega_re),
            lam,
            stem.with_name(stem.name + "_OmegaRe_heatmap"),
            rf"$\sigma = 140$, mode {j}: vorticity heat map",
            r"normalized $\Re(\delta\Omega)$",
        )
        plot_streamlines(
            tri_psi,
            vec_array(mode.psi_re),
            lam,
            stem.with_name(stem.name + "_PsiRe_streamlines"),
            rf"$s=140$, mode {j}: streamfunction contours",
        )

    plot_panel(tri_omega, modes, "omega", args.output_dir / "s140_first3_OmegaRe_heatmap_panel", r"$\sigma = 140$: first three vorticity eigenmodes")
    plot_panel(tri_psi, modes, "psi", args.output_dir / "s140_first3_PsiRe_streamline_panel", r"$s=140$: first three streamfunction eigenmodes")
    plot_manuscript_panel(tri_omega, modes, "omega", args.output_dir / "s140_first3_OmegaRe_heatmap_panel_manuscript")
    plot_manuscript_panel(tri_psi, modes, "psi", args.output_dir / "s140_first3_PsiRe_streamline_panel_manuscript")

    csv_path = args.output_dir / "s140_first3_eigenvalues.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode_index", "real_lambda", "imag_lambda"])
        writer.writeheader()
        writer.writerows(rows)
    write_caption_file(args.output_dir / "s140_first3_manuscript_captions.tex", rows)

    manifest = {
        "s": args.s,
        "checkpoint_h5": str(args.checkpoint_h5.relative_to(REPO_ROOT)),
        "output_dir": str(args.output_dir.relative_to(REPO_ROOT)),
        "mode_cache": str(mode_cache.relative_to(REPO_ROOT)),
        "reused_mode_cache": bool(args.reuse_mode_cache and mode_cache.exists()),
        "k_solve": args.k_solve,
        "n_plot": args.n_plot,
        "eig_shift": args.eig_shift,
        "eig_tol": args.eig_tol,
        "eig_max_it": args.eig_max_it,
        "mesh_stats": mesh_stats(mesh, spaces),
        "max_checkpoint_geometry_diff": geom_diff,
        "eigenvalues": rows,
        "wall_time_sec": time.perf_counter() - t0,
        "note": "Real parts of eigenmodes are plotted; eigenvector scaling/sign is arbitrary, so fields are normalized by max absolute value per mode.",
    }
    write_json(args.output_dir / "s140_first3_modes_manifest.json", manifest)
    print(f"output_dir={args.output_dir}")
    for row in rows:
        print(f"mode {row['mode_index']}: lambda={row['real_lambda']:.15g}{row['imag_lambda']:+.15g}i")


if __name__ == "__main__":
    main()
