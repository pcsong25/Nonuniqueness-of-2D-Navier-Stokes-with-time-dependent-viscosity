#!/usr/bin/env python3
"""Manuscript-style plots for the s=140 artificial m1 broken profile."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

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
C3_BRANCH_DIR = MANUSCRIPT_ROOT / "c3_spectrum_continuation"
sys.path.insert(0, str(MANUSCRIPT_ROOT))

from common import append_registry_row, today, write_json  # noqa: E402


DEFAULT_SYM = C3_BRANCH_DIR / "checkpoints/branchA_h006_refine_s125_140_k10/s140_h0p06_0p06_profile.h5"
DEFAULT_BROKEN = BRANCH_DIR / "checkpoints/branchB_s140_artificial_localized_probe/m1_phase0_eta0p25_profile.h5"


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_profile(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        xy = np.asarray(h5["/Mesh/mesh/geometry"][:], dtype=float)
        cells = np.asarray(h5["/Mesh/mesh/topology"][:], dtype=np.int64)
        omega = np.asarray(h5["/Function/Omega/0"][:], dtype=float).reshape(-1)
    return xy, cells, omega


def max_abs_percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(np.abs(np.asarray(values, dtype=float)), percentile))


def add_circle(ax: plt.Axes, radius: float = 10.0) -> None:
    circle = plt.Circle((0.0, 0.0), radius, fill=False, lw=0.75, color="0.2", alpha=0.55)
    ax.add_patch(circle)


def style_axis(ax: plt.Axes, zoom: float | None = None) -> None:
    ax.set_aspect("equal", "box")
    if zoom is None:
        ax.set_xlim(-10.2, 10.2)
        ax.set_ylim(-10.2, 10.2)
    else:
        ax.set_xlim(-zoom, zoom)
        ax.set_ylim(-zoom, zoom)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("0.25")


def draw_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    tri: mtri.Triangulation,
    values: np.ndarray,
    *,
    title: str,
    vmax: float,
    levels: int,
    zoom: float | None = None,
    contour: bool = True,
) -> None:
    filled_levels = np.linspace(-vmax, vmax, int(levels))
    cf = ax.tricontourf(tri, values, levels=filled_levels, cmap="RdBu_r", extend="both")
    if contour:
        line_levels = np.linspace(-vmax, vmax, 9)
        ax.tricontour(tri, values, levels=line_levels, colors="k", linewidths=0.28, alpha=0.42)
    add_circle(ax)
    style_axis(ax, zoom=zoom)
    ax.set_title(title, fontsize=15, pad=8)
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.025)
    cb.ax.tick_params(labelsize=11, width=0.6, length=2.5)
    cb.outline.set_linewidth(0.6)


def draw_panel_with_shared_colorbar(
    ax: plt.Axes,
    tri: mtri.Triangulation,
    values: np.ndarray,
    *,
    title: str,
    vmax: float,
    levels: int,
    cmap: str = "RdBu_r",
) -> object:
    filled_levels = np.linspace(-vmax, vmax, int(levels))
    cf = ax.tricontourf(tri, values, levels=filled_levels, cmap=cmap)
    add_circle(ax)
    style_axis(ax)
    ax.set_title(title, fontsize=15, pad=8)
    return cf


def plot_full(tri: mtri.Triangulation, sym: np.ndarray, broken: np.ndarray, out_base: Path, args: argparse.Namespace) -> None:
    diff = broken - sym
    field_vmax = max(max_abs_percentile(sym, args.field_percentile), max_abs_percentile(broken, args.field_percentile))
    diff_vmax = max(max_abs_percentile(diff, args.diff_percentile), 1e-14)

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "figure.dpi": 240,
            "savefig.dpi": 420,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.25), constrained_layout=True)
    draw_panel(fig, axes[0], tri, sym, title="symmetric", vmax=field_vmax, levels=args.levels)
    draw_panel(fig, axes[1], tri, broken, title="symmetry-breaking", vmax=field_vmax, levels=args.levels)
    draw_panel(fig, axes[2], tri, diff, title="difference", vmax=diff_vmax, levels=args.levels)
    fig.suptitle(r"$\sigma = 140$: bifurcation of vorticity profiles", fontsize=17)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_heatmap_faithful(
    tri: mtri.Triangulation,
    sym: np.ndarray,
    broken: np.ndarray,
    out_base: Path,
    args: argparse.Namespace,
) -> None:
    diff = broken - sym
    shared_vmax = max(float(np.max(np.abs(sym))), float(np.max(np.abs(broken))), float(np.max(np.abs(diff))))
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "figure.dpi": 260,
            "savefig.dpi": 460,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.85), constrained_layout=True)
    cf = draw_panel_with_shared_colorbar(
        axes[0], tri, sym, title="symmetric", vmax=shared_vmax, levels=args.levels, cmap=args.faithful_cmap
    )
    draw_panel_with_shared_colorbar(
        axes[1],
        tri,
        broken,
        title="symmetry-breaking",
        vmax=shared_vmax,
        levels=args.levels,
        cmap=args.faithful_cmap,
    )
    draw_panel_with_shared_colorbar(
        axes[2], tri, diff, title="difference", vmax=shared_vmax, levels=args.levels, cmap=args.faithful_cmap
    )
    cb = fig.colorbar(cf, ax=axes, fraction=0.026, pad=0.015)
    cb.ax.tick_params(labelsize=11, width=0.6, length=2.5)
    cb.outline.set_linewidth(0.6)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_heatmap_faithful_smooth(
    tri: mtri.Triangulation,
    sym: np.ndarray,
    broken: np.ndarray,
    out_base: Path,
    args: argparse.Namespace,
) -> None:
    diff = broken - sym
    shared_vmax = max(float(np.max(np.abs(sym))), float(np.max(np.abs(broken))), float(np.max(np.abs(diff))))
    smooth_levels = int(args.smooth_levels)
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "figure.dpi": 320,
            "savefig.dpi": 720,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.85), constrained_layout=True)
    cf = draw_panel_with_shared_colorbar(
        axes[0], tri, sym, title="symmetric", vmax=shared_vmax, levels=smooth_levels, cmap=args.faithful_cmap
    )
    draw_panel_with_shared_colorbar(
        axes[1],
        tri,
        broken,
        title="symmetry-breaking",
        vmax=shared_vmax,
        levels=smooth_levels,
        cmap=args.faithful_cmap,
    )
    draw_panel_with_shared_colorbar(
        axes[2], tri, diff, title="difference", vmax=shared_vmax, levels=smooth_levels, cmap=args.faithful_cmap
    )
    cb = fig.colorbar(cf, ax=axes, fraction=0.026, pad=0.015)
    cb.ax.tick_params(labelsize=11, width=0.6, length=2.5)
    cb.outline.set_linewidth(0.6)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_heatmap_clean(
    tri: mtri.Triangulation,
    sym: np.ndarray,
    broken: np.ndarray,
    out_base: Path,
    args: argparse.Namespace,
) -> None:
    diff = broken - sym
    shared_vmax = max(max_abs_percentile(sym, args.field_percentile), max_abs_percentile(broken, args.field_percentile))
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "figure.dpi": 240,
            "savefig.dpi": 420,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    draw_panel(fig, axes[0], tri, sym, title="symmetric", vmax=shared_vmax, levels=args.levels, contour=False)
    draw_panel(
        fig,
        axes[1],
        tri,
        broken,
        title="symmetry-breaking",
        vmax=shared_vmax,
        levels=args.levels,
        contour=False,
    )
    draw_panel(fig, axes[2], tri, diff, title="difference", vmax=shared_vmax, levels=args.levels, contour=False)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_full_plus_zoom(
    tri: mtri.Triangulation,
    sym: np.ndarray,
    broken: np.ndarray,
    out_base: Path,
    args: argparse.Namespace,
) -> None:
    diff = broken - sym
    field_vmax = max(max_abs_percentile(sym, args.field_percentile), max_abs_percentile(broken, args.field_percentile))
    diff_vmax = max(max_abs_percentile(diff, args.diff_percentile), 1e-14)

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 14,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "figure.dpi": 240,
            "savefig.dpi": 420,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.5), constrained_layout=True)
    labels = ["symmetric", "symmetry-breaking", "difference"]
    fields = [sym, broken, diff]
    vmaxes = [field_vmax, field_vmax, diff_vmax]
    for j, (label, values, vmax) in enumerate(zip(labels, fields, vmaxes)):
        draw_panel(fig, axes[0, j], tri, values, title=label, vmax=vmax, levels=args.levels, zoom=None)
        draw_panel(fig, axes[1, j], tri, values, title=f"{label} zoom", vmax=vmax, levels=args.levels, zoom=args.zoom_radius)
    fig.suptitle(r"$\sigma = 140$: vorticity profiles and central zoom", fontsize=17)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_difference_detail(
    tri: mtri.Triangulation,
    sym: np.ndarray,
    broken: np.ndarray,
    out_base: Path,
    args: argparse.Namespace,
) -> None:
    diff = broken - sym
    diff_vmax = max(max_abs_percentile(diff, args.diff_percentile), 1e-14)
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "figure.dpi": 260,
            "savefig.dpi": 460,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6), constrained_layout=True)
    draw_panel(
        fig,
        axes[0],
        tri,
        diff,
        title="difference",
        vmax=diff_vmax,
        levels=args.levels,
        zoom=None,
        contour=True,
    )
    draw_panel(
        fig,
        axes[1],
        tri,
        diff,
        title="difference zoom",
        vmax=diff_vmax,
        levels=args.levels,
        zoom=args.zoom_radius,
        contour=True,
    )
    fig.suptitle(r"$\sigma = 140$: symmetry-breaking vorticity component", fontsize=18)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symmetric-h5", type=Path, default=DEFAULT_SYM)
    parser.add_argument("--broken-h5", type=Path, default=DEFAULT_BROKEN)
    parser.add_argument("--run-tag", default="branchB_s140_artificial_localized_probe")
    parser.add_argument("--levels", type=int, default=96)
    parser.add_argument("--field-percentile", type=float, default=99.6)
    parser.add_argument("--diff-percentile", type=float, default=99.2)
    parser.add_argument("--zoom-radius", type=float, default=4.0)
    parser.add_argument("--faithful-cmap", default="seismic")
    parser.add_argument("--smooth-levels", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = BRANCH_DIR / "plots" / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = BRANCH_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    xy_sym, cells_sym, omega_sym = load_profile(args.symmetric_h5)
    xy_broken, cells_broken, omega_broken = load_profile(args.broken_h5)
    if xy_sym.shape != xy_broken.shape or np.max(np.abs(xy_sym - xy_broken)) > 1e-12:
        raise RuntimeError("symmetric and broken profiles are not on the same mesh")
    if cells_sym.shape != cells_broken.shape or np.max(np.abs(cells_sym - cells_broken)) != 0:
        raise RuntimeError("symmetric and broken profile cell topology differs")

    tri = mtri.Triangulation(xy_sym[:, 0], xy_sym[:, 1], triangles=cells_sym)
    full_base = out_dir / "m1_phase0_eta0p25_omega_comparison_manuscript"
    clean_base = out_dir / "m1_phase0_eta0p25_omega_comparison_heatmap_clean"
    faithful_base = out_dir / "m1_phase0_eta0p25_omega_comparison_heatmap_faithful"
    smooth_base = out_dir / "m1_phase0_eta0p25_omega_comparison_heatmap_faithful_smooth"
    zoom_base = out_dir / "m1_phase0_eta0p25_omega_comparison_zoom_manuscript"
    diff_base = out_dir / "m1_phase0_eta0p25_omega_difference_detail_manuscript"
    plot_full(tri, omega_sym, omega_broken, full_base, args)
    plot_heatmap_clean(tri, omega_sym, omega_broken, clean_base, args)
    plot_heatmap_faithful(tri, omega_sym, omega_broken, faithful_base, args)
    plot_heatmap_faithful_smooth(tri, omega_sym, omega_broken, smooth_base, args)
    plot_full_plus_zoom(tri, omega_sym, omega_broken, zoom_base, args)
    plot_difference_detail(tri, omega_sym, omega_broken, diff_base, args)

    manifest = {
        "status": "completed",
        "run_tag": args.run_tag,
        "field_percentile": float(args.field_percentile),
        "diff_percentile": float(args.diff_percentile),
        "zoom_radius": float(args.zoom_radius),
        "outputs": {
            "full_pdf": relpath(full_base.with_suffix(".pdf")),
            "full_png": relpath(full_base.with_suffix(".png")),
            "clean_heatmap_pdf": relpath(clean_base.with_suffix(".pdf")),
            "clean_heatmap_png": relpath(clean_base.with_suffix(".png")),
            "faithful_heatmap_pdf": relpath(faithful_base.with_suffix(".pdf")),
            "faithful_heatmap_png": relpath(faithful_base.with_suffix(".png")),
            "faithful_smooth_heatmap_pdf": relpath(smooth_base.with_suffix(".pdf")),
            "faithful_smooth_heatmap_png": relpath(smooth_base.with_suffix(".png")),
            "zoom_pdf": relpath(zoom_base.with_suffix(".pdf")),
            "zoom_png": relpath(zoom_base.with_suffix(".png")),
            "difference_detail_pdf": relpath(diff_base.with_suffix(".pdf")),
            "difference_detail_png": relpath(diff_base.with_suffix(".png")),
        },
    }
    write_json(logs / "branchB_s140_artificial_manuscript_plot_manifest.json", manifest)
    append_registry_row(
        {
            "date": today(),
            "tag": "fem-branchB-s140-artificial-manuscript-plot",
            "codebase": "baseline",
            "script": relpath(Path(__file__)),
            "input_case": "s_cos3theta_s140",
            "domain": "disk_Rdom10",
            "grid_or_mesh": "h=0.06/0.06",
            "solver": "postprocess_manuscript_profile_plot",
            "parameters": f"field_percentile={args.field_percentile}; diff_percentile={args.diff_percentile}; zoom_radius={args.zoom_radius}",
            "wall_time_sec": "n/a",
            "error_metrics": "n/a",
            "eigenvalues_summary": "n/a",
            "status": "completed",
            "notes": "Regenerated artificial localized symmetry-breaking comparison with larger panels, fonts, contours, and zoom panels.",
        }
    )
    print(f"[done] {full_base.with_suffix('.pdf')}")
    print(f"[done] {clean_base.with_suffix('.pdf')}")
    print(f"[done] {faithful_base.with_suffix('.pdf')}")
    print(f"[done] {smooth_base.with_suffix('.pdf')}")
    print(f"[done] {zoom_base.with_suffix('.pdf')}")
    print(f"[done] {diff_base.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
