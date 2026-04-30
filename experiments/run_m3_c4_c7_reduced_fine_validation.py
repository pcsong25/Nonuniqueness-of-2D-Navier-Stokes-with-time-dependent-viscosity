from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from nextgen.polar_profile import PolarGrid
from run_m3_c4_c7_high_amplitude_validation import (
    COEFFS,
    PHYSICAL,
    RUN_DATE,
    common_residual_from_sampler,
    compare_nextgen_pair,
    compare_nextgen_to_baseline,
    make_plots,
    run_baseline,
    sample_baseline,
    sanitize,
)
from run_m3_c4_c7_seeded_reduced_probe import prolong_state, solve_grid


SIGMA_GRID = tuple(float(s) for s in np.linspace(0.0, 1.0, 41))
NEXTGEN_GRIDS = ((72, 288), (96, 384), (128, 512))
BASELINE_MESH = (0.05, 0.08)

def append_registry(summary: dict[str, Any]) -> None:
    registry = REPO_ROOT / "experiments" / "registry.csv"
    baseline = summary["baseline_runs"][0]
    baseline_common = baseline.get("common_residuals", {}).get("profile_strong_relative", "n/a")
    baseline_runtime = baseline.get("runtime_sec", "n/a")
    with registry.open("a", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        for run in summary["nextgen_runs"]:
            grid = run["grid"]
            common = run.get("common_residuals", {})
            probe = run.get("probe_errors_vs_fine_baseline", {})
            writer.writerow(
                [
                    RUN_DATE,
                    f"m3-c4-c7-reduced-newton-nr{grid['nr']}_ntheta{grid['ntheta']}",
                    "nextgen_vs_baseline",
                    "experiments/run_m3_c4_c7_reduced_fine_validation.py",
                    "{c4=100,c7=40}",
                    "nextgen_polar_full_disk_vs_baseline_disk_Rdom10",
                    f"nextgen_nr{grid['nr']}_ntheta{grid['ntheta']}; baseline_ref=h0.05_0.08",
                    "profile_continuation_fv_upwind_nested_reduced_poisson_newton",
                    "M3 large-boundary c4/c7 nested reduced-Newton validation; no M4; no eigensolver",
                    f"nextgen={run.get('runtime_sec'):.3f}; baseline_ref={baseline_runtime}",
                    (
                        f"status={run['status']}; "
                        f"final_sigma={run.get('diagnostics', {}).get('final_sigma', 'n/a')}; "
                        f"nextgen_rel={run.get('diagnostics', {}).get('relative_residual', float('nan')):.3e}; "
                        f"nextgen_common={common.get('profile_strong_relative', 'n/a')}; "
                        f"baseline_common={baseline_common}; "
                        f"omega_probe_rms={probe.get('omega_probe_rms', 'n/a')}; "
                        f"psi_probe_rms_offset_removed={probe.get('psi_probe_rms_offset_removed', 'n/a')}"
                    ),
                    "n/a",
                    run["status"],
                    "Nested high-branch seed plus reduced Poisson Newton made fine c4/c7 grids practical.",
                ]
            )


def main() -> int:
    raw_root = REPO_ROOT / "experiments" / "raw" / "nextgen" / "m3_c4_c7_reduced_fine_validation"
    summary_dir = REPO_ROOT / "experiments" / "summaries"
    raw_root.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    print("[case] c4_100_c7_40 reduced fine validation", flush=True)
    nextgen_runs: list[dict[str, Any]] = []

    seed_grid = NEXTGEN_GRIDS[0]
    print(f"[nextgen seed start] grid={seed_grid[0]}/{seed_grid[1]}", flush=True)
    seed = solve_grid(
        seed_grid,
        reduced=False,
        scheme="fv_upwind",
        upwind_blend=1.0,
        sigma_grid=SIGMA_GRID,
    )
    nextgen_runs.append(seed)
    print(
        "[nextgen seed done] "
        f"grid={seed_grid[0]}/{seed_grid[1]} status={seed['status']} "
        f"runtime={seed['runtime_sec']:.3f} "
        f"final_sigma={seed.get('diagnostics', {}).get('final_sigma')} "
        f"common={seed.get('common_residuals', {}).get('profile_strong_relative')}",
        flush=True,
    )

    source = seed
    for grid in NEXTGEN_GRIDS[1:]:
        target_grid = PolarGrid(radius=PHYSICAL.Rdom, nr=grid[0], ntheta=grid[1])
        initial = prolong_state(source, target_grid)
        print(f"[nextgen nested reduced start] grid={grid[0]}/{grid[1]}", flush=True)
        run = solve_grid(
            grid,
            reduced=True,
            scheme="fv_upwind",
            upwind_blend=1.0,
            initial_state=initial,
            sigma_grid=(1.0,),
        )
        nextgen_runs.append(run)
        print(
            "[nextgen done] "
            f"grid={grid[0]}/{grid[1]} status={run['status']} "
            f"runtime={run['runtime_sec']:.3f} "
            f"final_sigma={run.get('diagnostics', {}).get('final_sigma')} "
            f"common={run['common_residuals'].get('profile_strong_relative')}",
            flush=True,
        )
        if run["status"] == "completed":
            source = run

    print(f"[baseline start] h={BASELINE_MESH[0]}/{BASELINE_MESH[1]}", flush=True)
    baseline = run_baseline(BASELINE_MESH)
    if baseline["status"] == "completed":
        baseline["common_residuals"] = common_residual_from_sampler(
            lambda points, run=baseline: (
                sample_baseline(run["omega"], run["spaces"]["Vomega"], points),
                sample_baseline(run["psi"], run["spaces"]["Vpsi"], points),
            )
        )
    print(
        f"[baseline done] h={BASELINE_MESH[0]}/{BASELINE_MESH[1]} "
        f"status={baseline['status']} runtime={baseline.get('runtime_sec')}",
        flush=True,
    )

    baseline_runs = [baseline]
    if baseline["status"] == "completed":
        for run in nextgen_runs:
            if run["status"] == "completed":
                run["probe_errors_vs_fine_baseline"] = compare_nextgen_to_baseline(run, baseline)
    nextgen_grid_difference = (
        compare_nextgen_pair(nextgen_runs[0], nextgen_runs[1])
        if len(nextgen_runs) > 1 and all(run["status"] == "completed" for run in nextgen_runs)
        else None
    )
    plots = make_plots(nextgen_runs, baseline_runs, raw_root / "c4_100_c7_40")

    summary = {
        "status": "completed"
        if all(run["status"] == "completed" for run in nextgen_runs) and baseline["status"] == "completed"
        else "partial",
        "case": {
            "name": "c4_100_c7_40",
            "description": "large c4/c7 profile validation with reduced Poisson Newton",
            "coeffs": COEFFS,
        },
        "input": {
            "physical": {"mu": PHYSICAL.mu, "R": PHYSICAL.R, "Rdom": PHYSICAL.Rdom, "eps": PHYSICAL.eps},
            "nextgen_grids": [{"nr": nr, "ntheta": ntheta} for nr, ntheta in NEXTGEN_GRIDS],
            "baseline_mesh": {"h_min": BASELINE_MESH[0], "h_max": BASELINE_MESH[1]},
            "sigma_grid": SIGMA_GRID,
            "solver": "nested fv_upwind: monolithic 72/288 seed, reduced_poisson_newton=True on finer grids",
        },
        "nextgen_runs": nextgen_runs,
        "baseline_runs": baseline_runs,
        "nextgen_grid_difference": nextgen_grid_difference,
        "plot_paths": plots,
    }
    append_registry(summary)
    path = summary_dir / f"m3_c4_c7_reduced_fine_validation_{RUN_DATE}.json"
    path.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    print(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
