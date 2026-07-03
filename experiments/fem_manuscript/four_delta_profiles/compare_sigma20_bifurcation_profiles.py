#!/usr/bin/env python3
"""Compare sigma=20 four-delta symmetric and eigenmode-perturbed profiles."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig-euler")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

MANUSCRIPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MANUSCRIPT_ROOT.parents[1]
BRANCH_DIR = Path(__file__).resolve().parent

PROFILES = [
    (
        "symmetric",
        "symmetric",
        BRANCH_DIR
        / "checkpoints/four_delta_eps0p01_M30_h0p06_0p06_sigma_ladder/sigma20_eps0p01_M30_h0p06_0p06_profile.h5",
    ),
    (
        "mode1_branch",
        "mode 1 branch",
        BRANCH_DIR / "checkpoints/four_delta_sigma20_mode1_bifurcation_probe/mode1_eta0p5_signp1_profile.h5",
    ),
    (
        "mode2_branch",
        "mode 2 branch",
        BRANCH_DIR / "checkpoints/four_delta_sigma20_mode1_bifurcation_probe/mode2_eta0p5_signp1_profile.h5",
    ),
]

SYMMETRIES = {
    "identity": lambda x, y: (x, y),
    "rotation_pi": lambda x, y: (-x, -y),
    "reflection_x": lambda x, y: (x, -y),
    "reflection_y": lambda x, y: (-x, y),
}


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_profile(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        xy = np.asarray(h5["/Mesh/mesh/geometry"][:], dtype=float)
        topo = np.asarray(h5["/Mesh/mesh/topology"][:], dtype=np.int32)
        omega = np.asarray(h5["/Function/Omega/0"][:], dtype=float).reshape(-1)
        psi = np.asarray(h5["/Function/psi/0"][:], dtype=float).reshape(-1)
    return {"path": path, "xy": xy, "topology": topo, "omega": omega, "psi": psi}


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.nanmean(values * values)))


def linf(values: np.ndarray) -> float:
    return float(np.nanmax(np.abs(np.asarray(values, dtype=float))))


def triangulation(profile: dict[str, Any]) -> mtri.Triangulation:
    xy = profile["xy"]
    return mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=profile["topology"])


def interpolator(tri: mtri.Triangulation, values: np.ndarray) -> mtri.LinearTriInterpolator:
    return mtri.LinearTriInterpolator(tri, np.asarray(values, dtype=float))


def sample(interp: mtri.LinearTriInterpolator, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = interp(x, y)
    if hasattr(out, "filled"):
        return np.asarray(out.filled(np.nan), dtype=float)
    return np.asarray(out, dtype=float)


def probe_grid(radius: float = 9.65, n: int = 260) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(-radius, radius, n)
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    mask = xx * xx + yy * yy <= radius * radius
    return xx, yy, mask


def relative_symmetry_defects(profile: dict[str, Any], xx: np.ndarray, yy: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    tri = triangulation(profile)
    interp = interpolator(tri, profile["omega"])
    base = sample(interp, xx, yy)
    valid0 = mask & np.isfinite(base)
    denom = max(rms(base[valid0]), 1e-30)
    defects = {}
    for name, transform in SYMMETRIES.items():
        tx, ty = transform(xx, yy)
        moved = sample(interp, tx, ty)
        valid = valid0 & np.isfinite(moved)
        defects[f"omega_{name}_rel_rms"] = rms(base[valid] - moved[valid]) / denom
    return defects


def pair_distance(
    a: dict[str, Any],
    b: dict[str, Any],
    xx: np.ndarray,
    yy: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    tri_a = triangulation(a)
    tri_b = triangulation(b)
    interp_a = interpolator(tri_a, a["omega"])
    interp_b = interpolator(tri_b, b["omega"])
    vals_a = sample(interp_a, xx, yy)
    valid_a = mask & np.isfinite(vals_a)
    denom = max(rms(vals_a[valid_a]), 1e-30)
    distances: dict[str, float] = {}
    for name, transform in SYMMETRIES.items():
        tx, ty = transform(xx, yy)
        vals_b = sample(interp_b, tx, ty)
        valid = valid_a & np.isfinite(vals_b)
        distances[name] = rms(vals_a[valid] - vals_b[valid]) / denom
    best_name = min(distances, key=distances.get)
    return {
        "best_symmetry": best_name,
        "best_omega_relative_rms": distances[best_name],
        **{f"omega_relative_rms_{name}": value for name, value in distances.items()},
    }


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


def plot_profile_panel(profiles: list[tuple[str, str, dict[str, Any]]], out_base: Path) -> None:
    vmax = max(linf(item[2]["omega"]) for item in profiles)
    levels = np.linspace(-vmax, vmax, 512)
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
    fig, axes = plt.subplots(1, len(profiles), figsize=(12.8, 3.85), constrained_layout=True)
    cf = None
    for ax, (_pid, label, profile) in zip(axes, profiles):
        cf = ax.tricontourf(triangulation(profile), profile["omega"], levels=levels, cmap="seismic")
        style_axis(ax)
        ax.set_title(label, pad=7)
    assert cf is not None
    cb = fig.colorbar(cf, ax=axes, fraction=0.026, pad=0.015)
    cb.set_label(r"$\Omega$", fontsize=13)
    cb.ax.tick_params(labelsize=11, width=0.6, length=2.5)
    cb.outline.set_linewidth(0.6)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_difference_panel(profiles: dict[str, dict[str, Any]], out_base: Path) -> None:
    pairs = [
        ("mode1_branch - symmetric", profiles["mode1_branch"]["omega"] - profiles["symmetric"]["omega"]),
        ("mode2_branch - symmetric", profiles["mode2_branch"]["omega"] - profiles["symmetric"]["omega"]),
        ("mode2_branch - mode1_branch", profiles["mode2_branch"]["omega"] - profiles["mode1_branch"]["omega"]),
    ]
    vmax = max(linf(values) for _label, values in pairs)
    levels = np.linspace(-vmax, vmax, 512)
    tri = triangulation(profiles["symmetric"])
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.85), constrained_layout=True)
    cf = None
    for ax, (label, values) in zip(axes, pairs):
        cf = ax.tricontourf(tri, values, levels=levels, cmap="seismic")
        style_axis(ax)
        ax.set_title(label, pad=7)
    assert cf is not None
    cb = fig.colorbar(cf, ax=axes, fraction=0.026, pad=0.015)
    cb.set_label(r"$\Delta\Omega$", fontsize=13)
    cb.ax.tick_params(labelsize=11, width=0.6, length=2.5)
    cb.outline.set_linewidth(0.6)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "n/a") for key in columns})


def main() -> int:
    out_dir = BRANCH_DIR / "plots/four_delta_sigma20_profile_symmetry_comparison"
    data_dir = BRANCH_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    loaded = [(pid, label, load_profile(path)) for pid, label, path in PROFILES]
    profiles = {pid: profile for pid, _label, profile in loaded}
    plot_profile_panel(loaded, out_dir / "sigma20_three_profile_omega_panel")
    plot_difference_panel(profiles, out_dir / "sigma20_three_profile_difference_panel")

    xx, yy, mask = probe_grid()
    symmetry_rows = []
    for pid, label, profile in loaded:
        row = {"profile_id": pid, "label": label, "profile_h5": relpath(profile["path"])}
        row.update(relative_symmetry_defects(profile, xx, yy, mask))
        symmetry_rows.append(row)

    pair_rows = []
    pairs = [
        ("symmetric", "mode1_branch"),
        ("symmetric", "mode2_branch"),
        ("mode1_branch", "mode2_branch"),
    ]
    for a_id, b_id in pairs:
        row = {"profile_a": a_id, "profile_b": b_id}
        row.update(pair_distance(profiles[a_id], profiles[b_id], xx, yy, mask))
        pair_rows.append(row)

    symmetry_csv = data_dir / "four_delta_sigma20_profile_symmetry_metrics.csv"
    pair_csv = data_dir / "four_delta_sigma20_profile_pairwise_distances_d2.csv"
    write_csv(symmetry_csv, symmetry_rows)
    write_csv(pair_csv, pair_rows)

    manifest = {
        "status": "completed",
        "profiles": [{"profile_id": pid, "label": label, "path": relpath(profile["path"])} for pid, label, profile in loaded],
        "symmetry_group": list(SYMMETRIES),
        "note": "Symmetry metrics use interpolated Omega samples on the disk r<=9.65. Pairwise distances are relative RMS distances minimized over identity, rotation by pi, reflection_x, and reflection_y.",
        "outputs": {
            "profile_panel": relpath(out_dir / "sigma20_three_profile_omega_panel.pdf"),
            "difference_panel": relpath(out_dir / "sigma20_three_profile_difference_panel.pdf"),
            "symmetry_csv": relpath(symmetry_csv),
            "pairwise_csv": relpath(pair_csv),
        },
        "symmetry_metrics": symmetry_rows,
        "pairwise_distances": pair_rows,
    }
    with (out_dir / "sigma20_profile_symmetry_comparison_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(manifest["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
