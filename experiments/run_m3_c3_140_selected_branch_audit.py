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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-euler")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from nextgen.polar_profile import PolarProfileParams
from run_m3_c3_140_baseline_closer_fv_homotopy import (
    RAW_ROOT as HOMOTOPY_RAW_ROOT,
    SUMMARY_DIR,
    TransportSpec,
    distance_between_runs,
    final_relative_update,
    make_problem,
    reproject_state,
    result_from_arrays,
)
from run_m3_c3_140_residual_audit import sanitize
from run_m3_c3_140_symmetry_probe import (
    common_residual,
    compatibility_metrics,
    save_state,
    symmetry_metrics,
)


RUN_DATE = "2026-05-01"
RAW_ROOT = REPO_ROOT / "experiments" / "raw" / "nextgen" / "m3_c3_140_selected_branch_audit"
STATE_DIR = RAW_ROOT / "states"
TABLE_DIR = RAW_ROOT / "tables"

GRID = (192, 576)
BROKEN_BLEND = 0.015625
SYMMETRIC_BLEND = 0.125
BROKEN_SPEC = TransportSpec("fv_blend_0p015625", "fv_blended", BROKEN_BLEND)
SYMMETRIC_SPEC = TransportSpec("fv_blend_0p125", "fv_blended", SYMMETRIC_BLEND)

BROKEN_STATE = (
    HOMOTOPY_RAW_ROOT / "states" / "broken_m1_fv_blend_0p015625_192_576_state.npz"
)
SYMMETRIC_STATE = HOMOTOPY_RAW_ROOT / "states" / "symmetric_fv_blend_0p125_192_576_state.npz"
BROKEN_SUMMARY = SUMMARY_DIR / "m3_c3_140_baseline_closer_fv_homotopy_broken_m1_2026-04-30.json"
SYMMETRIC_SUMMARY = SUMMARY_DIR / "m3_c3_140_baseline_closer_fv_homotopy_2026-04-30.json"


def load_state(path: Path, branch: str, spec: TransportSpec, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    grid = (int(data["grid_nr"]), int(data["grid_ntheta"]))
    problem = make_problem(grid, spec)
    sigma = float(data["sigma"]) if "sigma" in data.files else 1.0
    return result_from_arrays(
        problem,
        data["omega"],
        data["psi"],
        label=label,
        branch=branch,
        transport=spec,
        sigma=sigma,
        poisson_lambda=0.0,
        message=f"loaded_state:{path.relative_to(REPO_ROOT)}",
    )


def run_row(
    run: dict[str, Any],
    *,
    previous_final_update: float | None = None,
    source: str,
) -> dict[str, Any]:
    diag = run.get("diagnostics", {})
    sym = run.get("symmetry", {})
    comp = run.get("compatibility", {})
    common = run.get("common_residuals", {})
    update = final_relative_update(run)
    if not np.isfinite(update) and previous_final_update is not None:
        update = float(previous_final_update)
    rel = float(diag.get("relative_residual", float("nan")))
    physical_poisson = float(diag.get("physical_poisson_residual_norm", float("nan")))
    compat = float(comp.get("omega_minus_flux", float("nan")))
    omega_non3 = float(sym.get("omega_non_threefold_fraction", float("nan")))
    return {
        "label": run.get("label"),
        "branch": run.get("branch"),
        "source": source,
        "transport_label": run.get("transport", {}).get("label"),
        "upwind_blend": run.get("transport", {}).get("upwind_blend"),
        "status": run.get("status"),
        "message": run.get("message"),
        "runtime_sec": run.get("runtime_sec"),
        "newton_steps": diag.get("newton_steps"),
        "native_relative_residual": rel,
        "native_residual_norm": diag.get("residual_norm"),
        "omega_residual_norm": diag.get("omega_residual_norm"),
        "poisson_residual_norm": diag.get("poisson_residual_norm"),
        "physical_poisson_residual_norm": physical_poisson,
        "final_relative_update": update,
        "omega_non3": omega_non3,
        "psi_non3": sym.get("psi_non_threefold_fraction"),
        "compatibility_omega_minus_flux": compat,
        "common_profile_strong_relative": common.get("profile_strong_relative"),
        "gate1_residual_pass": rel <= 1e-8,
        "gate1_update_pass": np.isfinite(update) and update <= 1e-8,
        "gate1_poisson_pass": physical_poisson <= 1e-8,
        "gate1_compatibility_pass": abs(compat) <= 1e-10,
        "gate1_broken_symmetry_present": omega_non3 >= 1e-3,
    }


def pair_row(label: str, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    distance = distance_between_runs(a, b)
    return {
        "pair": label,
        "a_label": a.get("label"),
        "b_label": b.get("label"),
        "omega_full_rms": distance["omega_full"].get("rms"),
        "omega_full_linf": distance["omega_full"].get("linf"),
        "omega_active_rms": distance["omega_active"].get("rms"),
        "omega_active_linf": distance["omega_active"].get("linf"),
        "psi_offset_removed_rms": distance["psi_full_offset_removed"].get("rms"),
        "psi_offset_removed_linf": distance["psi_full_offset_removed"].get("linf"),
        "valid_count": distance["omega_full"].get("valid_count"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path.relative_to(REPO_ROOT))


def selected_row(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text()).get("selected_blend") or {}


def gate3_rows(tight: dict[str, Any]) -> list[dict[str, Any]]:
    broken_selected = selected_row(BROKEN_SUMMARY)
    symmetric_selected = selected_row(SYMMETRIC_SUMMARY)
    tight_vs_previous = pair_row(
        "tightened_broken_vs_previous_broken",
        tight,
        load_state(BROKEN_STATE, "broken_m1", BROKEN_SPEC, "broken_m1_fv_blend_0p015625_192_576_previous"),
    )
    displacement = float(tight_vs_previous["omega_full_rms"] or float("nan"))
    return [
        {
            "branch": "symmetric",
            "transport_label": symmetric_selected.get("transport_label"),
            "omega_rms_to_fem_h0p06": symmetric_selected.get("omega_rms_to_fem_h0p06"),
            "omega_rms_to_fem_h0p05": symmetric_selected.get("omega_rms_to_fem_h0p05"),
            "fv_to_fem_seed_residual_h0p06": _cross_value(SYMMETRIC_SUMMARY, "0.06", "fv_to_fem", "initial_newton_bnorm"),
            "fv_to_fem_seed_residual_h0p05": _cross_value(SYMMETRIC_SUMMARY, "0.05", "fv_to_fem", "initial_newton_bnorm"),
            "fem_to_fv_projected_residual_h0p06": _cross_value(SYMMETRIC_SUMMARY, "0.06", "fem_to_fv", "initial_projected_relative_residual"),
            "fem_to_fv_projected_residual_h0p05": _cross_value(SYMMETRIC_SUMMARY, "0.05", "fem_to_fv", "initial_projected_relative_residual"),
            "source": str(SYMMETRIC_SUMMARY.relative_to(REPO_ROOT)),
        },
        {
            "branch": "broken_m1",
            "transport_label": broken_selected.get("transport_label"),
            "omega_rms_to_fem_h0p06": broken_selected.get("omega_rms_to_fem_h0p06"),
            "omega_rms_to_fem_h0p05": broken_selected.get("omega_rms_to_fem_h0p05"),
            "fv_to_fem_seed_residual_h0p06": _cross_value(BROKEN_SUMMARY, "0.06", "fv_to_fem", "initial_newton_bnorm"),
            "fv_to_fem_seed_residual_h0p05": _cross_value(BROKEN_SUMMARY, "0.05", "fv_to_fem", "initial_newton_bnorm"),
            "fem_to_fv_projected_residual_h0p06": _cross_value(BROKEN_SUMMARY, "0.06", "fem_to_fv", "initial_projected_relative_residual"),
            "fem_to_fv_projected_residual_h0p05": _cross_value(BROKEN_SUMMARY, "0.05", "fem_to_fv", "initial_projected_relative_residual"),
            "tightened_vs_previous_omega_rms": displacement,
            "note": "Gate 3 values inherited from the completed homotopy summary; the tightened state moved by the listed Omega RMS.",
            "source": str(BROKEN_SUMMARY.relative_to(REPO_ROOT)),
        },
    ]


def _cross_value(summary_path: Path, mesh_label: str, direction: str, key: str) -> Any:
    if not summary_path.exists():
        return None
    cross = json.loads(summary_path.read_text()).get("cross_initialization", {})
    normalized = mesh_label.replace(".", "p")
    if "/" in normalized:
        a, b = normalized.split("/", 1)
    else:
        a = b = normalized
    entry = cross.get(mesh_label) or cross.get(f"h{a}_h{b}") or cross.get(f"h{a}_{b}") or cross.get(f"h{a}")
    return (entry or {}).get(direction, {}).get(key)


def append_registry(summary: dict[str, Any]) -> None:
    registry = REPO_ROOT / "experiments" / "registry.csv"
    selected = summary.get("tightened_broken") or {}
    with registry.open("a", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                RUN_DATE,
                "m3-c3-140-selected-branch-audit",
                "nextgen_vs_baseline",
                "experiments/run_m3_c3_140_selected_branch_audit.py",
                "{c3=140}",
                "nextgen_polar_full_disk_Rdom10",
                "grid=192/576; branch=broken_m1; transport=fv_blend_0p015625",
                "selected_branch_newton_tightening_and_compact_audit",
                "M3 c3=140 selected broken FV branch audit; no M4; no eigensolver",
                summary.get("runtime_sec", "n/a"),
                (
                    f"tight_rel={selected.get('native_relative_residual')}; "
                    f"tight_update={selected.get('final_relative_update')}; "
                    f"omega_non3={selected.get('omega_non3')}"
                ),
                "n/a",
                summary.get("status", "completed"),
                "Tightened selected broken FV blend and wrote compact selected-branch audit.",
            ]
        )


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    previous_broken = load_state(
        BROKEN_STATE,
        "broken_m1",
        BROKEN_SPEC,
        "broken_m1_fv_blend_0p015625_192_576_previous",
    )
    symmetric = load_state(
        SYMMETRIC_STATE,
        "symmetric",
        SYMMETRIC_SPEC,
        "symmetric_fv_blend_0p125_192_576",
    )
    problem = make_problem(GRID, BROKEN_SPEC)
    initial = reproject_state(problem, previous_broken["result"].omega, sigma=1.0)
    solve_t0 = time.perf_counter()
    result = problem.solve(
        PolarProfileParams(
            sigma_grid=(1.0,),
            tol_step=float(args.tol_step),
            tol_residual=float(args.tol_residual),
            tol_residual_only=None,
            max_iter=int(args.max_iter),
            max_backtracks=30,
            adaptive_continuation=True,
            min_sigma_step=5e-4,
            sigma_step_growth=1.2,
            sigma_step_shrink=0.5,
            least_squares_fallback=False,
            pseudo_transient_fallback=True,
            pseudo_transient_shifts=(1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
            reduced_poisson_newton=True,
            reduced_gmres_rtol=float(args.gmres_rtol),
            reduced_gmres_maxiter=int(args.gmres_maxiter),
            reduced_gmres_restart=int(args.gmres_restart),
        ),
        initial_state=initial,
    )
    tightened = {
        "label": "broken_m1_fv_blend_0p015625_tight_192_576",
        "branch": "broken_m1",
        "transport": {
            "label": BROKEN_SPEC.label,
            "advection_scheme": BROKEN_SPEC.advection_scheme,
            "upwind_blend": BROKEN_SPEC.upwind_blend,
        },
        "status": "completed" if result.converged else "nonconverged",
        "grid": {"nr": GRID[0], "ntheta": GRID[1], "dr": problem.grid.dr, "dtheta": problem.grid.dtheta},
        "grid_label": "192/576",
        "problem": problem,
        "result": result,
        "runtime_sec": time.perf_counter() - solve_t0,
        "diagnostics": result.diagnostics,
        "message": result.message,
        "common_residuals": common_residual(problem, result),
        "symmetry": symmetry_metrics(problem, result),
        "compatibility": compatibility_metrics(problem, result),
        "boundary_defects": {
            "omega_dirichlet_max_abs": 0.0,
            "psi_neumann_max_abs": 0.0,
            "note": "polar FV boundary data are analytic RHS terms, not mutable state dofs",
        },
    }
    state_path = save_state(tightened, STATE_DIR)

    previous_summary_row = selected_row(BROKEN_SUMMARY)
    gate1 = [
        run_row(
            symmetric,
            previous_final_update=selected_row(SYMMETRIC_SUMMARY).get("final_relative_update"),
            source="saved_selected_symmetric",
        ),
        run_row(
            previous_broken,
            previous_final_update=previous_summary_row.get("final_relative_update"),
            source="saved_selected_broken_before_tightening",
        ),
        run_row(tightened, source="tightened_selected_broken"),
    ]
    separation = [
        pair_row("previous_broken_vs_symmetric_selected", previous_broken, symmetric),
        pair_row("tightened_broken_vs_symmetric_selected", tightened, symmetric),
        pair_row("tightened_broken_vs_previous_broken", tightened, previous_broken),
    ]
    gate3 = gate3_rows(tightened)

    gate1_path = write_csv(TABLE_DIR / "selected_branch_gate1_table.csv", gate1)
    separation_path = write_csv(TABLE_DIR / "selected_branch_separation_table.csv", separation)
    gate3_path = write_csv(TABLE_DIR / "selected_branch_gate3_table.csv", gate3)

    summary = {
        "status": "completed" if result.converged else "nonconverged",
        "case": {"name": "m3_c3_140_selected_branch_audit", "coeffs": {3: 140.0}},
        "grid": {"nr": GRID[0], "ntheta": GRID[1]},
        "branch": "broken_m1",
        "transport": {
            "label": BROKEN_SPEC.label,
            "advection_scheme": BROKEN_SPEC.advection_scheme,
            "upwind_blend": BROKEN_SPEC.upwind_blend,
        },
        "runtime_sec": time.perf_counter() - t0,
        "tightening_controls": {
            "tol_step": float(args.tol_step),
            "tol_residual": float(args.tol_residual),
            "tol_residual_only": None,
            "max_iter": int(args.max_iter),
            "gmres_rtol": float(args.gmres_rtol),
            "gmres_maxiter": int(args.gmres_maxiter),
            "gmres_restart": int(args.gmres_restart),
        },
        "tightened_broken": gate1[-1],
        "previous_broken": gate1[1],
        "selected_symmetric": gate1[0],
        "branch_separation": separation,
        "gate3": gate3,
        "state_path": state_path,
        "table_paths": {
            "gate1": gate1_path,
            "branch_separation": separation_path,
            "gate3": gate3_path,
        },
        "notes": [
            "This audit stays in M3 and does not assemble or run the eigensolver.",
            "The tightening solve disables residual-only convergence so the selected broken state must satisfy both residual and update gates.",
            "Gate 3 FEM values are inherited from the completed homotopy summary because this run is scoped to selected-branch tightening.",
        ],
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="M3 c3=140 selected broken branch tightening audit.")
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--tol-step", type=float, default=1e-8)
    parser.add_argument("--tol-residual", type=float, default=1e-10)
    parser.add_argument("--gmres-rtol", type=float, default=1e-9)
    parser.add_argument("--gmres-maxiter", type=int, default=540)
    parser.add_argument("--gmres-restart", type=int, default=180)
    parser.add_argument("--no-registry", action="store_true")
    args = parser.parse_args()

    summary = run_audit(args)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARY_DIR / f"m3_c3_140_selected_branch_audit_{RUN_DATE}.json"
    path.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    if not args.no_registry:
        append_registry(summary)
    print(json.dumps(sanitize(summary), indent=2, sort_keys=True))
    return 0 if summary.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
