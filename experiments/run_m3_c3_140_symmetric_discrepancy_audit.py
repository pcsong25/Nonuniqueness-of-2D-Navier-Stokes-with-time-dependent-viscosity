from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-euler")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from nextgen.boundary import FourierBoundaryData
from run_m3_c3_140_residual_audit import (
    COEFFS,
    FEM_CACHE,
    PHYSICAL,
    SAMPLE_X,
    SAMPLE_Y,
    fem_boundary_defects,
    load_cached_fem,
    load_nextgen_state,
    run_baseline_seeded_linesearch,
    sample_baseline,
    sanitize,
)


RUN_DATE = "2026-04-30"
RAW_ROOT = REPO_ROOT / "experiments" / "raw" / "nextgen" / "m3_c3_140_symmetric_discrepancy_audit"
SUMMARY_DIR = REPO_ROOT / "experiments" / "summaries"
PLOT_DIR = RAW_ROOT / "plots"

STATE_160 = REPO_ROOT / "experiments/raw/nextgen/m3_c3_140_nonuniqueness_validation/states/symmetric_160_480_state.npz"
STATE_192 = REPO_ROOT / "experiments/raw/nextgen/m3_c3_140_refine192_fem_crosscheck/states/symmetric_192_576_state.npz"


def grid_points() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(SAMPLE_X, SAMPLE_Y, indexing="ij")
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    return xx, yy, points


def nextgen_sample(run: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    xx, yy, _ = grid_points()
    omega, psi = run["problem"].sample_cartesian(run["result"], xx.reshape(-1), yy.reshape(-1))
    return omega.reshape(xx.shape), psi.reshape(xx.shape)


def fem_sample_direct(function: Any, space: Any, points: np.ndarray) -> np.ndarray:
    from dolfinx import geometry

    mesh = space.mesh
    points3 = np.column_stack((points[:, 0], points[:, 1], np.zeros(points.shape[0])))
    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points3)
    colliding = geometry.compute_colliding_cells(mesh, candidates, points3)
    values = np.full(points.shape[0], np.nan, dtype=float)
    for i, point in enumerate(points3):
        cells = colliding.links(i)
        if len(cells) == 0:
            continue
        values[i] = float(np.asarray(function.eval(point.reshape(1, 3), np.asarray([cells[0]], dtype=np.int32))).reshape(-1)[0])
    return values


def diff_metrics(a: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray, *, mask: np.ndarray | None = None) -> dict[str, float | int | list[float]]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    valid = np.isfinite(aa) & np.isfinite(bb)
    if mask is not None:
        valid &= mask
    diff = aa[valid] - bb[valid]
    av = aa[valid]
    bv = bb[valid]
    if diff.size == 0:
        return {"valid_count": 0}
    rms = float(np.sqrt(np.mean(diff**2)))
    a_rms = float(np.sqrt(np.mean(av**2)))
    b_rms = float(np.sqrt(np.mean(bv**2)))
    absdiff = np.full(aa.shape, np.nan, dtype=float)
    absdiff[valid] = np.abs(aa[valid] - bb[valid])
    max_ij = np.unravel_index(int(np.nanargmax(absdiff)), aa.shape)
    return {
        "valid_count": int(diff.size),
        "rms": rms,
        "linf": float(np.max(np.abs(diff))),
        "relative_to_a_rms": rms / max(a_rms, 1e-30),
        "relative_to_b_rms": rms / max(b_rms, 1e-30),
        "a_rms": a_rms,
        "b_rms": b_rms,
        "max_abs_location": [float(x[max_ij[0]]), float(y[max_ij[1]])],
        "a_at_max_abs": float(aa[max_ij]),
        "b_at_max_abs": float(bb[max_ij]),
        "diff_at_max_abs": float(aa[max_ij] - bb[max_ij]),
    }


def least_squares_scale(a: np.ndarray, b: np.ndarray, *, mask: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(a) & np.isfinite(b) & mask
    av = np.asarray(a[valid], dtype=float)
    bv = np.asarray(b[valid], dtype=float)
    scale = float(np.dot(av, bv) / max(np.dot(av, av), 1e-30))
    residual = scale * av - bv
    return {
        "scale_a_to_b": scale,
        "rms_after_scaling": float(np.sqrt(np.mean(residual**2))),
        "rms_before_scaling": float(np.sqrt(np.mean((av - bv) ** 2))),
    }


def rotation_sweep(a: np.ndarray, b: np.ndarray, xx: np.ndarray, yy: np.ndarray, *, mask: np.ndarray) -> dict[str, float]:
    interpolator = RegularGridInterpolator((SAMPLE_X, SAMPLE_Y), a, bounds_error=False, fill_value=np.nan)
    coords = np.column_stack((xx[mask], yy[mask]))
    target = b[mask]
    valid_target = np.isfinite(target)
    coords = coords[valid_target]
    target = target[valid_target]
    best = {"angle_deg": 0.0, "rms": float("inf")}
    for angle in np.linspace(-4.0, 4.0, 161):
        c = float(np.cos(np.deg2rad(angle)))
        s = float(np.sin(np.deg2rad(angle)))
        # Rotated field f_angle(x,y)=f(R_{-angle}(x,y)).
        sample_coords = np.column_stack((c * coords[:, 0] + s * coords[:, 1], -s * coords[:, 0] + c * coords[:, 1]))
        values = interpolator(sample_coords)
        valid = np.isfinite(values)
        if not np.any(valid):
            continue
        rms = float(np.sqrt(np.mean((values[valid] - target[valid]) ** 2)))
        if rms < best["rms"]:
            best = {"angle_deg": float(angle), "rms": rms}
    return best


def boundary_convention_check() -> dict[str, float]:
    baseline_dir = SRC / "baseline"
    sys.path.insert(0, str(baseline_dir))
    from boundary_data_fourier import FourierBC

    theta = np.linspace(-np.pi, np.pi, 4096, endpoint=False)
    x = PHYSICAL.Rdom * np.cos(theta)
    y = PHYSICAL.Rdom * np.sin(theta)
    baseline = FourierBC(64, COEFFS)
    nextgen = FourierBoundaryData(M=64, coeffs=COEFFS)
    omega_base = PHYSICAL.R ** (-1.0 / PHYSICAL.mu) * baseline.Omega_bar(theta)
    omega_next = nextgen.omega_dirichlet(x, y, mu=PHYSICAL.mu, R=PHYSICAL.R, Rdom=PHYSICAL.Rdom)
    flux_base = baseline.b_theta(theta, mu=PHYSICAL.mu, R=PHYSICAL.R, Rdom=PHYSICAL.Rdom, sigma=1.0)
    flux_next = nextgen.psi_neumann_circle(theta, mu=PHYSICAL.mu, R=PHYSICAL.R, Rdom=PHYSICAL.Rdom)
    return {
        "omega_boundary_max_abs_difference": float(np.max(np.abs(omega_base - omega_next))),
        "omega_boundary_rms_difference": float(np.sqrt(np.mean((omega_base - omega_next) ** 2))),
        "psi_neumann_max_abs_difference": float(np.max(np.abs(flux_base - flux_next))),
        "psi_neumann_rms_difference": float(np.sqrt(np.mean((flux_base - flux_next) ** 2))),
    }


def mesh_label(h_min: float, h_max: float) -> str:
    return f"h{h_min:g}_{h_max:g}".replace(".", "p")


def write_plots(fields: dict[str, np.ndarray], label: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = PLOT_DIR / label
    plot_dir.mkdir(parents=True, exist_ok=True)
    extent = (float(SAMPLE_X[0]), float(SAMPLE_X[-1]), float(SAMPLE_Y[0]), float(SAMPLE_Y[-1]))
    created: list[str] = []
    panels = [
        ("nextgen 192 symmetric", fields["nextgen_192"]),
        ("FEM direct symmetric", fields["fem_direct"]),
        ("FEM direct - nextgen", fields["fem_direct"] - fields["nextgen_192"]),
        ("FEM LinearND - FEM direct", fields["fem_linear"] - fields["fem_direct"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), constrained_layout=True)
    for ax, (title, values) in zip(axes, panels):
        image = ax.imshow(values.T, origin="lower", extent=extent, cmap="RdBu_r")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    path = plot_dir / "symmetric_direct_fem_vs_nextgen.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    created.append(str(path.relative_to(REPO_ROOT)))

    x_window = (-1.0, 1.0)
    y_window = (3.3, 5.3)
    ix = (SAMPLE_X >= x_window[0]) & (SAMPLE_X <= x_window[1])
    iy = (SAMPLE_Y >= y_window[0]) & (SAMPLE_Y <= y_window[1])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), constrained_layout=True)
    close_panels = [
        ("nextgen 192", fields["nextgen_192"][np.ix_(ix, iy)]),
        ("FEM direct", fields["fem_direct"][np.ix_(ix, iy)]),
        ("FEM direct - nextgen", (fields["fem_direct"] - fields["nextgen_192"])[np.ix_(ix, iy)]),
    ]
    close_extent = (float(SAMPLE_X[ix][0]), float(SAMPLE_X[ix][-1]), float(SAMPLE_Y[iy][0]), float(SAMPLE_Y[iy][-1]))
    for ax, (title, values) in zip(axes, close_panels):
        image = ax.imshow(values.T, origin="lower", extent=close_extent, cmap="RdBu_r")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    path = plot_dir / "symmetric_top_core_closeup.png"
    fig.savefig(path, dpi=150)
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
                "m3-c3-140-symmetric-discrepancy-audit",
                "nextgen_vs_baseline",
                "experiments/run_m3_c3_140_symmetric_discrepancy_audit.py",
                "{c3=140}",
                f"nextgen_192/576_and_baseline_fem_{summary['mesh_label']}",
                "sample_grid=193x193",
                "symmetric_branch_comparison_debug",
                "M3 c3=140 symmetric nextgen/FEM discrepancy audit; no M4; no eigensolver",
                summary.get("runtime_sec", "n/a"),
                (
                    f"mesh={summary['mesh_label']}; "
                    f"direct_vs_nextgen_rms={summary['metrics']['fem_direct_vs_nextgen_192_full']['rms']}; "
                    f"linear_vs_direct_rms={summary['metrics']['fem_linear_vs_fem_direct_full']['rms']}; "
                    f"boundary_omega_max={summary['boundary_convention']['omega_boundary_max_abs_difference']}"
                ),
                "n/a",
                "completed",
                "Symmetric nextgen/FEM comparison checked with direct FEM sampling.",
            ]
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug the c3=140 symmetric nextgen/FEM discrepancy.")
    parser.add_argument("--use-cached-fem", action="store_true", help="skip direct FEM sampling and use cached LinearND samples only")
    parser.add_argument("--h-min", type=float, default=0.06, help="FEM h_min for direct rerun")
    parser.add_argument("--h-max", type=float, default=0.06, help="FEM h_max for direct rerun")
    args = parser.parse_args()

    t0 = time.perf_counter()
    h_label = mesh_label(args.h_min, args.h_max)
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    nextgen_160 = load_nextgen_state(STATE_160, "symmetric_160_480", "symmetric", "160/480")
    nextgen_192 = load_nextgen_state(STATE_192, "symmetric_192_576", "symmetric", "192/576")
    ng160_o, _ = nextgen_sample(nextgen_160)
    ng192_o, _ = nextgen_sample(nextgen_192)
    xx, yy, points = grid_points()
    radius = np.hypot(xx, yy)
    full_mask = radius <= 7.0
    active_mask = full_mask & ((np.abs(ng192_o) > 0.5) | (np.abs(ng160_o) > 0.5))

    fem_cached = load_cached_fem("symmetric")
    if fem_cached is None and args.use_cached_fem:
        raise RuntimeError(f"Missing FEM cache in {FEM_CACHE}; run the residual audit first.")
    fem_linear_o = np.asarray(fem_cached["sample_omega"], dtype=float) if fem_cached is not None else np.full_like(ng192_o, np.nan)
    fem_boundary = {"status": "not_available_from_cache"}
    if args.use_cached_fem:
        if abs(args.h_min - 0.06) > 1e-14 or abs(args.h_max - 0.06) > 1e-14:
            raise ValueError("--use-cached-fem is only valid for the cached h_min=h_max=0.06 data")
        fem_direct_o = fem_linear_o.copy()
        direct_note = "direct FEM sampling skipped; FEM direct field equals cached LinearND field"
    else:
        fem_run = run_baseline_seeded_linesearch(nextgen_192, (float(args.h_min), float(args.h_max)))
        if fem_run.get("status") != "completed":
            raise RuntimeError(f"FEM symmetric rerun did not complete: {fem_run}")
        fem_boundary = fem_boundary_defects(fem_run)
        fem_direct_o = fem_sample_direct(fem_run["omega"], fem_run["spaces"]["Vomega"], points).reshape(xx.shape)
        direct_note = "direct FEM samples evaluated through dolfinx cell lookup on the actual FEM mesh"
        fem_linear_o = sample_baseline(fem_run["omega"], fem_run["spaces"]["Vomega"], points).reshape(xx.shape)

    metrics = {
        "nextgen_160_vs_192_full": diff_metrics(ng160_o, ng192_o, SAMPLE_X, SAMPLE_Y, mask=full_mask),
        "nextgen_160_vs_192_active": diff_metrics(ng160_o, ng192_o, SAMPLE_X, SAMPLE_Y, mask=active_mask),
        "fem_direct_vs_nextgen_192_full": diff_metrics(fem_direct_o, ng192_o, SAMPLE_X, SAMPLE_Y, mask=full_mask),
        "fem_direct_vs_nextgen_192_active": diff_metrics(fem_direct_o, ng192_o, SAMPLE_X, SAMPLE_Y, mask=active_mask),
        "fem_linear_vs_fem_direct_full": diff_metrics(fem_linear_o, fem_direct_o, SAMPLE_X, SAMPLE_Y, mask=full_mask),
        "fem_linear_vs_fem_direct_active": diff_metrics(fem_linear_o, fem_direct_o, SAMPLE_X, SAMPLE_Y, mask=active_mask),
        "least_squares_scale_full": least_squares_scale(ng192_o, fem_direct_o, mask=full_mask),
        "least_squares_scale_active": least_squares_scale(ng192_o, fem_direct_o, mask=active_mask),
        "rotation_sweep_full": rotation_sweep(ng192_o, fem_direct_o, xx, yy, mask=full_mask),
        "rotation_sweep_active": rotation_sweep(ng192_o, fem_direct_o, xx, yy, mask=active_mask),
    }
    plot_paths = write_plots(
        {
            "nextgen_192": ng192_o,
            "fem_direct": fem_direct_o,
            "fem_linear": fem_linear_o,
        },
        h_label,
    )
    summary = {
        "status": "completed",
        "case": {"coeffs": COEFFS, "name": "c3_140_symmetric_discrepancy_audit"},
        "mesh_label": h_label,
        "mesh": {"h_min": float(args.h_min), "h_max": float(args.h_max)},
        "runtime_sec": time.perf_counter() - t0,
        "direct_sampling_note": direct_note,
        "boundary_convention": boundary_convention_check(),
        "fem_boundary_defects": fem_boundary,
        "metrics": metrics,
        "plot_paths": plot_paths,
        "interpretation_notes": [
            "The full-mask comparison uses the 193x193 residual-audit Cartesian sample grid restricted to r <= 7.",
            "The active-mask comparison additionally keeps points where either nextgen 160/480 or nextgen 192/576 has |Omega| > 0.5.",
            "FEM LinearND samples are the old cached generic nodal interpolation; FEM direct samples use DOLFINx cell lookup on the actual mesh.",
        ],
    }
    path = SUMMARY_DIR / f"m3_c3_140_symmetric_discrepancy_audit_{h_label}_{RUN_DATE}.json"
    path.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    append_registry(summary)
    print(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
