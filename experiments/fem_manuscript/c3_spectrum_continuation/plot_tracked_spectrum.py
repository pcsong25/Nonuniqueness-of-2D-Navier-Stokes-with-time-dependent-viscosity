#!/usr/bin/env python3
"""Plot Branch A eigenvalue real parts with complex-plane track matching.

The eigensolver output is sorted by real part independently at each amplitude.
For plotting, this script builds lightweight trajectory assignments using
complex eigenvalue data. The plotted vertical coordinate is still only
Re(lambda). Without saved eigenvectors, these are eigenvalue-only heuristics.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
BRANCH_DIR = ROOT / "experiments/fem_manuscript/c3_spectrum_continuation"
DEFAULT_INPUT = BRANCH_DIR / "data/branchA_h006_combined_s10_160_k10_eigenvalues.csv"
DEFAULT_OUTPUT_DIR = BRANCH_DIR / "plots/branchA_h006_s10_160"
DEFAULT_TRACKED_CSV = BRANCH_DIR / "data/branchA_h006_tracked_s10_160_k10_eigenvalues.csv"


@dataclass(frozen=True)
class EigenPoint:
    sigma: float
    source_index: int
    real: float
    imag: float

    @property
    def z(self) -> complex:
        return complex(self.real, self.imag)


def read_points(path: Path, n_eigs: int) -> dict[float, list[EigenPoint]]:
    by_sigma: dict[float, list[EigenPoint]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["eigen_index"])
            if idx > n_eigs:
                continue
            sigma = float(row["s"])
            by_sigma.setdefault(sigma, []).append(
                EigenPoint(
                    sigma=sigma,
                    source_index=idx,
                    real=float(row["real_lambda"]),
                    imag=float(row["imag_lambda"]),
                )
            )
    for sigma, points in by_sigma.items():
        points.sort(key=lambda p: p.source_index)
        if len(points) != n_eigs:
            raise ValueError(f"sigma={sigma:g} has {len(points)} points, expected {n_eigs}")
    return dict(sorted(by_sigma.items()))


def linear_sum_assignment(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Small fallback assignment solver, using scipy when available."""

    try:
        from scipy.optimize import linear_sum_assignment as scipy_lsa

        rows, cols = scipy_lsa(cost)
        return list(zip([int(r) for r in rows], [int(c) for c in cols]))
    except Exception:
        n = len(cost)
        best_perm: tuple[int, ...] | None = None
        best_cost = float("inf")
        for perm in itertools.permutations(range(n)):
            total = sum(cost[i][perm[i]] for i in range(n))
            if total < best_cost:
                best_cost = total
                best_perm = perm
        if best_perm is None:
            raise RuntimeError("empty assignment")
        return [(i, best_perm[i]) for i in range(n)]


def assign_current_to_tracks(
    tracks: list[list[EigenPoint]],
    current: list[EigenPoint],
    method: str,
) -> list[tuple[int, int]]:
    if method == "nearest" or len(tracks[0]) < 2:
        targets = [track[-1].z for track in tracks]
    elif method == "predictor":
        targets = []
        for track in tracks:
            prev = track[-1]
            prevprev = track[-2]
            denom = prev.sigma - prevprev.sigma
            if abs(denom) < 1e-14:
                targets.append(prev.z)
            else:
                slope = (prev.z - prevprev.z) / denom
                targets.append(prev.z + slope * (current[0].sigma - prev.sigma))
    else:
        raise ValueError(f"unknown tracking method: {method}")

    cost = [[abs(target - point.z) for point in current] for target in targets]
    return linear_sum_assignment(cost)


def track_points(
    by_sigma: dict[float, list[EigenPoint]],
    method: str,
) -> list[list[EigenPoint]]:
    sigmas = list(by_sigma)
    first = by_sigma[sigmas[0]]
    tracks: list[list[EigenPoint]] = [[point] for point in first]

    for sigma in sigmas[1:]:
        current = by_sigma[sigma]
        assignment = assign_current_to_tracks(tracks, current, method)
        next_tracks: list[list[EigenPoint]] = [list(track) for track in tracks]
        for track_idx, point_idx in assignment:
            next_tracks[track_idx].append(current[point_idx])
        tracks = next_tracks

    return tracks


def write_tracked_csv(path: Path, tracks: list[list[EigenPoint]], method: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "track_index",
                "sigma",
                "source_eigen_index",
                "real_lambda",
                "imag_lambda",
                "tracking_method",
            ],
        )
        writer.writeheader()
        for track_idx, track in enumerate(tracks, start=1):
            for point in track:
                writer.writerow(
                    {
                        "track_index": track_idx,
                        "sigma": f"{point.sigma:.12g}",
                        "source_eigen_index": point.source_index,
                        "real_lambda": f"{point.real:.16g}",
                        "imag_lambda": f"{point.imag:.16g}",
                        "tracking_method": method,
                    }
                )


def marker_indices(n_points: int, stride: int) -> list[int]:
    indices = list(range(0, n_points, max(stride, 1)))
    if n_points and (n_points - 1) not in indices:
        indices.append(n_points - 1)
    return indices


def plot_tracks(
    output_dir: Path,
    tracks: list[list[EigenPoint]],
    stem: str,
    marker_stride: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=220)
    colors = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]

    for idx, track in enumerate(tracks, start=1):
        xs = [point.sigma for point in track]
        ys = [point.real for point in track]
        color = colors((idx - 1) % 10)
        ax.plot(
            xs,
            ys,
            linestyle="-",
            linewidth=1.0,
            color=color,
            alpha=0.7,
            zorder=1,
        )
        keep = marker_indices(len(xs), marker_stride)
        ax.scatter(
            [xs[i] for i in keep],
            [ys[i] for i in keep],
            marker=markers[(idx - 1) % len(markers)],
            s=28,
            facecolors="none",
            edgecolors=color,
            linewidths=1.0,
            label=fr"$j={idx}$",
            zorder=2,
        )

    ax.axhline(0.0, color="0.2", linewidth=0.8)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$\Re(\lambda)$")
    ax.set_title("Eigenvalues of the linearized operator")
    ax.set_xlim(0, 165)
    ax.grid(True, which="major", color="0.85", linewidth=0.6)
    ax.legend(ncol=2, fontsize=8, frameon=False, loc="best")
    fig.tight_layout()

    outputs = [
        output_dir / f"{stem}.png",
        output_dir / f"{stem}.pdf",
    ]
    for path in outputs:
        fig.savefig(path)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tracked-csv", type=Path, default=DEFAULT_TRACKED_CSV)
    parser.add_argument("--n-eigs", type=int, default=10)
    parser.add_argument(
        "--marker-stride",
        type=int,
        default=1,
        help="Place hollow markers every N samples, always including the final sample.",
    )
    parser.add_argument(
        "--tracking-method",
        choices=("nearest", "predictor"),
        default="predictor",
        help=(
            "nearest matches to the previous complex eigenvalue; predictor "
            "matches to a linear extrapolation from the previous two samples."
        ),
    )
    parser.add_argument(
        "--stem",
        default="branchA_h006_tracked_Re_lambda_first10_s10_160",
    )
    args = parser.parse_args()

    by_sigma = read_points(args.input, args.n_eigs)
    tracks = track_points(by_sigma, args.tracking_method)
    write_tracked_csv(args.tracked_csv, tracks, args.tracking_method)
    outputs = plot_tracks(args.output_dir, tracks, args.stem, args.marker_stride)

    print(f"tracked_csv={args.tracked_csv}")
    for path in outputs:
        print(f"plot={path}")


if __name__ == "__main__":
    main()
