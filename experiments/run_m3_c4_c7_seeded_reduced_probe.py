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
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from nextgen.boundary import FourierBoundaryData
from nextgen.config import PhysicalParams
from nextgen.diagnostics import sampled_strong_profile_residual
from nextgen.polar_profile import (
    PolarGrid,
    PolarProfileParams,
    PolarProfileProblem,
    PolarProfileState,
)


RUN_DATE = "2026-04-29"
PHYSICAL = PhysicalParams(mu=0.8, R=150.0, Rdom=10.0)
COEFFS = {4: 100.0 + 0.0j, 7: 40.0 + 0.0j}
SIGMA_GRID = tuple(float(s) for s in np.linspace(0.0, 1.0, 41))
SAMPLE_X = np.linspace(-7.0, 7.0, 193)
SAMPLE_Y = np.linspace(-7.0, 7.0, 193)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        blocked = {"problem", "result"}
        return {str(k): sanitize(v) for k, v in value.items() if k not in blocked}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def solve_grid(
    grid_spec: tuple[int, int],
    *,
    reduced: bool,
    scheme: str = "fv_upwind",
    upwind_blend: float = 1.0,
    initial_state: PolarProfileState | None = None,
    sigma_grid: tuple[float, ...] = SIGMA_GRID,
) -> dict[str, Any]:
    nr, ntheta = grid_spec
    grid = PolarGrid(radius=PHYSICAL.Rdom, nr=nr, ntheta=ntheta)
    problem = PolarProfileProblem(
        grid,
        PHYSICAL,
        FourierBoundaryData(M=64, coeffs=COEFFS),
        advection_scheme=scheme,
        upwind_blend=upwind_blend,
    )
    result = problem.solve(
        PolarProfileParams(
            sigma_grid=sigma_grid,
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
            reduced_poisson_newton=reduced,
            reduced_gmres_rtol=1e-8,
            reduced_gmres_maxiter=160,
            reduced_gmres_restart=80,
        ),
        initial_state=initial_state,
    )
    return {
        "status": "completed" if result.converged else "nonconverged",
        "grid": {"nr": nr, "ntheta": ntheta, "dr": grid.dr, "dtheta": grid.dtheta},
        "problem": problem,
        "result": result,
        "runtime_sec": result.runtime_sec,
        "scheme": scheme,
        "upwind_blend": upwind_blend,
        "diagnostics": result.diagnostics,
        "message": result.message,
        "common_residuals": common_residual(problem, result) if result.converged else None,
    }


def prolong_state(source: dict[str, Any], target_grid: PolarGrid) -> PolarProfileState:
    rr, tt = np.meshgrid(target_grid.r, target_grid.theta, indexing="ij")
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)
    omega, psi = source["problem"].sample_cartesian(source["result"], x, y)
    return PolarProfileState(
        omega=np.asarray(omega, dtype=float).reshape(target_grid.shape),
        psi=np.asarray(psi, dtype=float).reshape(target_grid.shape),
        sigma=0.0,
        poisson_lambda=float(source["diagnostics"].get("poisson_lambda", 0.0)),
    )


def main() -> int:
    runs: list[dict[str, Any]] = []

    print("[seed start] monolithic 72/288", flush=True)
    seed = solve_grid((72, 288), reduced=False)
    runs.append({"label": "monolithic_seed_72_288", **seed})
    print(
        f"[seed done] status={seed['status']} runtime={seed['runtime_sec']:.3f} "
        f"omega_l2={seed['diagnostics'].get('omega_field_l2')} "
        f"common={seed.get('common_residuals', {}).get('profile_strong_relative')}",
        flush=True,
    )
    if seed["status"] != "completed":
        return 1

    source = seed
    for grid_spec, scheme, blend in (
        ((96, 384), "fv_upwind", 1.0),
        ((128, 512), "fv_upwind", 1.0),
    ):
        target_grid = PolarGrid(radius=PHYSICAL.Rdom, nr=grid_spec[0], ntheta=grid_spec[1])
        initial = prolong_state(source, target_grid)
        print(
            f"[seeded reduced start] grid={grid_spec[0]}/{grid_spec[1]} "
            f"scheme={scheme} blend={blend}",
            flush=True,
        )
        run = solve_grid(
            grid_spec,
            reduced=True,
            scheme=scheme,
            upwind_blend=blend,
            initial_state=initial,
            sigma_grid=(1.0,),
        )
        blend_label = str(blend).replace(".", "p")
        runs.append({f"label": f"seeded_reduced_{scheme}_{blend_label}_{grid_spec[0]}_{grid_spec[1]}", **run})
        print(
            f"[seeded reduced done] grid={grid_spec[0]}/{grid_spec[1]} "
            f"scheme={scheme} blend={blend} status={run['status']} runtime={run['runtime_sec']:.3f} "
            f"omega_l2={run['diagnostics'].get('omega_field_l2')} "
            f"common={run.get('common_residuals', {}).get('profile_strong_relative')}",
            flush=True,
        )
        if run["status"] == "completed":
            source = run

    summary = {"status": "completed", "case": {"coeffs": COEFFS}, "runs": runs}
    path = REPO_ROOT / "experiments" / "summaries" / f"m3_c4_c7_seeded_reduced_probe_{RUN_DATE}.json"
    path.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    print(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
