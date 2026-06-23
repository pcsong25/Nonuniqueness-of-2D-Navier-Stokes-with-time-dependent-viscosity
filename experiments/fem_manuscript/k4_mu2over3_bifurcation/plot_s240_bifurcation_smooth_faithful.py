#!/usr/bin/env python3
"""Smooth faithful heatmap-only comparison for the k=4, mu=2/3 bifurcation scout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig-euler")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

BRANCH_DIR = Path(__file__).resolve().parent
MANUSCRIPT_ROOT = BRANCH_DIR.parents[0]
REPO_ROOT = MANUSCRIPT_ROOT.parents[1]
sys.path.insert(0, str(MANUSCRIPT_ROOT))

from common import append_registry_row, today, write_json  # noqa: E402


DEFAULT_RUN_TAG = "branchB_k4_mu2over3_h008_s200_320_scout"
DEFAULT_CHECKPOINT_DIR = BRANCH_DIR / "checkpoints" / DEFAULT_RUN_TAG
DEFAULT_PLOT_DIR = BRANCH_DIR / "plots" / DEFAULT_RUN_TAG
DEFAULT_SYMMETRIC = DEFAULT_CHECKPOINT_DIR / "s240_h0p08_0p08_profile.h5"
DEFAULT_BROKEN = DEFAULT_CHECKPOINT_DIR / "s240_m3_phase0_eta0p5_profile.h5"


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
    return xy[:, :2], cells, omega


def style_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal", "box")
    ax.set_xlim(-10.2, 10.2)
    ax.set_ylim(-10.2, 10.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("0.25")
    ax.add_patch(plt.Circle((0.0, 0.0), 10.0, fill=False, lw=0.75, color="0.2", alpha=0.55))


def draw_panel(
    ax: plt.Axes,
    tri: mtri.Triangulation,
    values: np.ndarray,
    *,
    title: str,
    vmax: float,
    levels: int,
    cmap: str,
) -> object:
    filled_levels = np.linspace(-vmax, vmax, int(levels))
    cf = ax.tricontourf(tri, values, levels=filled_levels, cmap=cmap)
    style_axis(ax)
    ax.set_title(title, fontsize=15, pad=8)
    return cf


def plot_comparison(
    tri: mtri.Triangulation,
    sym: np.ndarray,
    broken: np.ndarray,
    out_base: Path,
    *,
    levels: int,
    cmap: str,
) -> float:
    diff = broken - sym
    shared_vmax = max(float(np.max(np.abs(sym))), float(np.max(np.abs(broken))), float(np.max(np.abs(diff))))
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
    cf = draw_panel(axes[0], tri, sym, title="symmetric", vmax=shared_vmax, levels=levels, cmap=cmap)
    draw_panel(axes[1], tri, broken, title="symmetry-breaking", vmax=shared_vmax, levels=levels, cmap=cmap)
    draw_panel(axes[2], tri, diff, title="difference", vmax=shared_vmax, levels=levels, cmap=cmap)
    cb = fig.colorbar(cf, ax=axes, fraction=0.026, pad=0.015)
    cb.ax.tick_params(labelsize=11, width=0.6, length=2.5)
    cb.outline.set_linewidth(0.6)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)
    return shared_vmax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symmetric-h5", type=Path, default=DEFAULT_SYMMETRIC)
    parser.add_argument("--broken-h5", type=Path, default=DEFAULT_BROKEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--output-stem", default="s240_m3_phase0_eta0p5_omega_comparison_heatmap_faithful_smooth")
    parser.add_argument("--levels", type=int, default=1024)
    parser.add_argument("--cmap", default="seismic")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = BRANCH_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    xy_sym, cells_sym, omega_sym = load_profile(args.symmetric_h5)
    xy_broken, cells_broken, omega_broken = load_profile(args.broken_h5)
    if xy_sym.shape != xy_broken.shape or np.max(np.abs(xy_sym - xy_broken)) > 1e-12:
        raise RuntimeError("symmetric and symmetry-breaking profiles are not on the same mesh")
    if cells_sym.shape != cells_broken.shape or np.max(np.abs(cells_sym - cells_broken)) != 0:
        raise RuntimeError("symmetric and symmetry-breaking cell topology differs")

    tri = mtri.Triangulation(xy_sym[:, 0], xy_sym[:, 1], triangles=cells_sym)
    out_base = args.output_dir / args.output_stem
    shared_vmax = plot_comparison(
        tri,
        omega_sym,
        omega_broken,
        out_base,
        levels=args.levels,
        cmap=args.cmap,
    )

    manifest = {
        "status": "completed",
        "symmetric_h5": relpath(args.symmetric_h5),
        "broken_h5": relpath(args.broken_h5),
        "color_scale": "true shared max over symmetric, symmetry-breaking, and difference fields; no percentile clipping or saturation.",
        "shared_vmax": shared_vmax,
        "levels": int(args.levels),
        "cmap": args.cmap,
        "outputs": {"pdf": relpath(out_base.with_suffix(".pdf")), "png": relpath(out_base.with_suffix(".png"))},
    }
    write_json(logs / "s240_k4_mu2over3_smooth_faithful_plot_manifest.json", manifest)
    append_registry_row(
        {
            "date": today(),
            "tag": "fem-branchB-k4-mu2over3-s240-smooth-faithful-plot",
            "codebase": "baseline",
            "script": relpath(Path(__file__)),
            "input_case": "s_cos4theta_mu0.666667_s240",
            "domain": "disk_Rdom10",
            "grid_or_mesh": "h=0.08/0.08",
            "solver": "postprocess_saved_profiles",
            "parameters": f"levels={args.levels}; cmap={args.cmap}; no_suptitle; no_contours",
            "wall_time_sec": "n/a",
            "error_metrics": "n/a",
            "eigenvalues_summary": "n/a",
            "status": "completed",
            "notes": "Generated smooth no-title heatmap-only k=4 scout bifurcation comparison with faithful shared color scale.",
        }
    )
    print(f"[done] {out_base.with_suffix('.pdf')}")
    print(f"[done] {out_base.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
