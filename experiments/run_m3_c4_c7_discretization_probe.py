from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-euler")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from nextgen.boundary import FourierBoundaryData
from nextgen.config import PhysicalParams
from nextgen.diagnostics import sampled_strong_profile_residual
from nextgen.polar_profile import PolarGrid, PolarProfileParams, PolarProfileProblem


RUN_DATE = "2026-04-29"
PHYSICAL = PhysicalParams(mu=0.8, R=150.0, Rdom=10.0)
COEFFS = {4: 100.0 + 0.0j, 7: 40.0 + 0.0j}
SIGMA_GRID = tuple(float(s) for s in np.linspace(0.0, 1.0, 41))
SAMPLE_X = np.linspace(-7.0, 7.0, 193)
SAMPLE_Y = np.linspace(-7.0, 7.0, 193)

CONFIGS = (
    {"label": "fv_upwind_reduced_72_288_sigma41", "nr": 72, "ntheta": 288, "scheme": "fv_upwind", "blend": 1.0},
)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def common_residual(problem: PolarProfileProblem, result: Any) -> dict[str, float]:
    x, y = np.meshgrid(SAMPLE_X, SAMPLE_Y, indexing="ij")
    omega, psi = problem.sample_cartesian(result, x, y)
    return sampled_strong_profile_residual(
        omega,
        psi,
        SAMPLE_X,
        SAMPLE_Y,
        mu=PHYSICAL.mu,
        eps=PHYSICAL.eps,
    )


def run_config(config: dict[str, Any]) -> dict[str, Any]:
    grid = PolarGrid(radius=PHYSICAL.Rdom, nr=int(config["nr"]), ntheta=int(config["ntheta"]))
    problem = PolarProfileProblem(
        grid,
        PHYSICAL,
        FourierBoundaryData(M=64, coeffs=COEFFS),
        advection_scheme=str(config["scheme"]),
        upwind_blend=float(config["blend"]),
    )
    result = problem.solve(
        PolarProfileParams(
            sigma_grid=SIGMA_GRID,
            tol_step=1e-8,
            tol_residual=1e-8,
            tol_residual_only=1e-8,
            max_iter=64,
            max_backtracks=28,
            adaptive_continuation=True,
            min_sigma_step=5e-4,
            sigma_step_growth=1.25,
            sigma_step_shrink=0.5,
            least_squares_fallback=False,
            pseudo_transient_fallback=True,
            pseudo_transient_shifts=(1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
            reduced_poisson_newton=True,
            reduced_gmres_rtol=1e-8,
            reduced_gmres_maxiter=160,
            reduced_gmres_restart=80,
        )
    )
    run = {
        "label": str(config["label"]),
        "status": "completed" if result.converged else "nonconverged",
        "grid": {"nr": grid.nr, "ntheta": grid.ntheta, "dr": grid.dr, "dtheta": grid.dtheta},
        "scheme": str(config["scheme"]),
        "upwind_blend": float(config["blend"]),
        "runtime_sec": result.runtime_sec,
        "message": result.message,
        "diagnostics": result.diagnostics,
    }
    if result.converged:
        run["common_residuals"] = common_residual(problem, result)
    return run


def main() -> int:
    runs: list[dict[str, Any]] = []
    for config in CONFIGS:
        print(f"[probe start] {config['label']}", flush=True)
        run = run_config(config)
        runs.append(run)
        common = run.get("common_residuals", {})
        print(
            "[probe done] "
            f"{run['label']} status={run['status']} runtime={run['runtime_sec']:.3f} "
            f"final_sigma={run['diagnostics'].get('final_sigma')} "
            f"native_rel={run['diagnostics'].get('relative_residual')} "
            f"common={common.get('profile_strong_relative')}",
            flush=True,
        )
    summary = {
        "status": "completed",
        "case": {"name": "c4_100_c7_40", "coeffs": COEFFS},
        "input": {"sigma_grid": SIGMA_GRID, "sample_points_per_axis": int(SAMPLE_X.size)},
        "runs": runs,
    }
    path = REPO_ROOT / "experiments" / "summaries" / f"m3_c4_c7_discretization_probe_{RUN_DATE}.json"
    path.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    print(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
