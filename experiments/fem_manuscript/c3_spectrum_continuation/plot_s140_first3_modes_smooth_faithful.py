#!/usr/bin/env python3
"""Smooth faithful heatmaps for the first three s=140 unstable eigenmodes."""

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


DEFAULT_PROFILE = BRANCH_DIR / "checkpoints/branchA_h006_refine_s125_140_k10/s140_h0p06_0p06_profile.h5"
DEFAULT_MODE_CACHE = BRANCH_DIR / "plots/s140_first3_eigenmodes/s140_first3_mode_arrays.npz"
DEFAULT_OUTPUT_DIR = BRANCH_DIR / "plots/s140_first3_eigenmodes"


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_profile_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        xy = np.asarray(h5["/Mesh/mesh/geometry"][:], dtype=float)
        cells = np.asarray(h5["/Mesh/mesh/topology"][:], dtype=np.int64)
    return xy[:, :2], cells


def normalize(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    if scale <= 0.0:
        return values.copy()
    return values / scale


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


def plot_modes(
    tri: mtri.Triangulation,
    omegas: list[np.ndarray],
    lambdas: list[complex],
    out_base: Path,
    *,
    levels: int,
    cmap: str,
) -> None:
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
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.85), constrained_layout=True)
    filled_levels = np.linspace(-1.0, 1.0, int(levels))
    cf = None
    for idx, (ax, omega, lam) in enumerate(zip(axes, omegas, lambdas), start=1):
        z = normalize(omega)
        cf = ax.tricontourf(tri, z, levels=filled_levels, cmap=cmap)
        style_axis(ax)
        ax.set_title(rf"$j={idx}$, $\lambda={lam.real:.4f}{lam.imag:+.1e}i$", fontsize=14, pad=8)
    assert cf is not None
    cb = fig.colorbar(cf, ax=axes, fraction=0.026, pad=0.015)
    cb.set_label(r"normalized $\Re(\delta\Omega)$", fontsize=13)
    cb.ax.tick_params(labelsize=11, width=0.6, length=2.5)
    cb.outline.set_linewidth(0.6)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-h5", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--mode-cache", type=Path, default=DEFAULT_MODE_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default="s140_first3_OmegaRe_heatmap_panel_faithful_smooth")
    parser.add_argument("--levels", type=int, default=1024)
    parser.add_argument("--cmap", default="seismic")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = BRANCH_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    xy, cells = load_profile_mesh(args.profile_h5)
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=cells)

    cache = np.load(args.mode_cache)
    n_modes = int(cache["n_modes"][0])
    if n_modes < 3:
        raise RuntimeError(f"expected at least 3 cached modes, found {n_modes}")
    lambdas = [complex(float(cache["real_lambda"][j]), float(cache["imag_lambda"][j])) for j in range(3)]
    omegas = [np.asarray(cache[f"omega_re_{j}"], dtype=float).reshape(-1) for j in range(1, 4)]
    if any(omega.shape[0] != xy.shape[0] for omega in omegas):
        raise RuntimeError("cached eigenmode arrays do not match the profile mesh vertex count")

    out_base = args.output_dir / args.output_stem
    plot_modes(tri, omegas, lambdas, out_base, levels=args.levels, cmap=args.cmap)

    manifest = {
        "status": "completed",
        "profile_h5": relpath(args.profile_h5),
        "mode_cache": relpath(args.mode_cache),
        "normalization": "Each eigenmode is divided by max(abs(Re(deltaOmega))) before plotting; eigenvector amplitude is arbitrary.",
        "color_scale": "shared normalized scale [-1, 1], no percentile clipping or saturation.",
        "levels": int(args.levels),
        "cmap": args.cmap,
        "outputs": {"pdf": relpath(out_base.with_suffix(".pdf")), "png": relpath(out_base.with_suffix(".png"))},
    }
    write_json(logs / "s140_first3_modes_smooth_faithful_manifest.json", manifest)
    append_registry_row(
        {
            "date": today(),
            "tag": "fem-branchA-s140-first3-eigenmode-smooth-faithful-plot",
            "codebase": "baseline",
            "script": relpath(Path(__file__)),
            "input_case": "s_cos3theta_s140",
            "domain": "disk_Rdom10",
            "grid_or_mesh": "h=0.06/0.06",
            "solver": "postprocess_cached_eigenmodes",
            "parameters": f"levels={args.levels}; cmap={args.cmap}; no_suptitle; no_contours",
            "wall_time_sec": "n/a",
            "error_metrics": "n/a",
            "eigenvalues_summary": "; ".join(f"{lam.real:.6e}{lam.imag:+.6e}j" for lam in lambdas),
            "status": "completed",
            "notes": "Generated smooth no-title heatmap-only panel for the first three s=140 unstable vorticity eigenmodes.",
        }
    )
    print(f"[done] {out_base.with_suffix('.pdf')}")
    print(f"[done] {out_base.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
