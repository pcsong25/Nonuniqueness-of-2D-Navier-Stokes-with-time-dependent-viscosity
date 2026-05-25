from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig-euler")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

MANUSCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MANUSCRIPT_ROOT))

from common import (  # noqa: E402
    REPO_ROOT,
    add_baseline_to_path,
    amplitude_label,
    append_registry_row,
    boundary_description,
    coeffs_for_exact_c3_cos,
    ensure_branch_tree,
    mesh_label,
    today,
    write_json,
)


BRANCH_DIR = Path(__file__).resolve().parent
CSV_COLUMNS = [
    "s",
    "mesh_id",
    "h_min",
    "h_max",
    "mesh_vertices",
    "mesh_cells",
    "eigen_index",
    "real_lambda",
    "imag_lambda",
    "eig_residual",
    "profile_residual_omega_l2",
    "profile_residual_psi_l2",
    "profile_wall_time",
    "eigs_wall_time",
    "wall_time",
    "status",
    "profile_checkpoint",
    "metadata_checkpoint",
    "message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Branch A FEM c3 boundary continuation and eigenvalue smoke runner."
    )
    parser.add_argument("--s-values", nargs="+", type=float, default=[10.0])
    parser.add_argument("--mesh", nargs=2, type=float, metavar=("H_MIN", "H_MAX"), default=[0.06, 0.06])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--eig-shift", type=float, default=-1.0)
    parser.add_argument("--eig-tol", type=float, default=1e-8)
    parser.add_argument("--eig-max-it", type=int, default=200)
    parser.add_argument("--mu", type=float, default=0.8)
    parser.add_argument("--R", type=float, default=150.0)
    parser.add_argument("--Rdom", type=float, default=10.0)
    parser.add_argument("--M", type=int, default=48)
    parser.add_argument("--tol-newton", type=float, default=1e-8)
    parser.add_argument("--max-newton", type=int, default=48)
    parser.add_argument("--sigma-steps", type=int, default=21)
    parser.add_argument("--max-profile-residual-omega", type=float, default=1e-2)
    parser.add_argument("--max-profile-residual-psi", type=float, default=1e-8)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--force", action="store_true", help="Recompute even if a completed metadata checkpoint exists.")
    parser.add_argument("--no-registry", action="store_true", help="Do not append an execution row to experiments/registry.csv.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the coarse FEM smoke test. Without it, only a dry-run manifest is written.",
    )
    return parser.parse_args()


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    h_min, h_max = (float(args.mesh[0]), float(args.mesh[1]))
    s_values = [float(s) for s in args.s_values]
    run_tag = args.run_tag or f"branchA_{today()}_{mesh_label(h_min, h_max)}"
    return {
        "status": "initialized" if args.execute else "dry_run",
        "branch": "A_c3_spectrum_continuation",
        "run_tag": run_tag,
        "mesh": {"h_min": h_min, "h_max": h_max},
        "s_values": s_values,
        "k": int(args.k),
        "eig_shift": float(args.eig_shift),
        "eig_tol": float(args.eig_tol),
        "eig_max_it": int(args.eig_max_it),
        "physical": {
            "mu": float(args.mu),
            "R": float(args.R),
            "Rdom": float(args.Rdom),
            "M": int(args.M),
        },
        "newton": {
            "tol_newton": float(args.tol_newton),
            "max_newton": int(args.max_newton),
            "sigma_steps": int(args.sigma_steps),
            "max_profile_residual_omega": float(args.max_profile_residual_omega),
            "max_profile_residual_psi": float(args.max_profile_residual_psi),
        },
        "boundary_cases": {
            amplitude_label(s): boundary_description(s) for s in s_values
        },
        "planned_outputs": {
            "data_csv": str((BRANCH_DIR / "data" / f"{run_tag}_eigenvalues.csv").relative_to(MANUSCRIPT_ROOT.parents[1])),
            "checkpoint_dir": str((BRANCH_DIR / "checkpoints" / run_tag).relative_to(MANUSCRIPT_ROOT.parents[1])),
            "log_file": str((BRANCH_DIR / "logs" / f"{run_tag}.log").relative_to(MANUSCRIPT_ROOT.parents[1])),
            "debug_plot_dir": str((BRANCH_DIR / "plots" / run_tag).relative_to(MANUSCRIPT_ROOT.parents[1])),
        },
        "notes": [
            "Exact s cos(3 theta) uses baseline coefficient c3=s/2.",
            "Default mesh is h_min=h_max=0.06 because this is the tested FEM manuscript scale.",
            "Eigenpair residuals are recorded as n/a in this first smoke runner.",
        ],
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def completed_metadata(path: Path, args: argparse.Namespace, amplitude_s: float) -> bool:
    if not path.exists():
        return False
    try:
        import json

        with path.open() as f:
            data = json.load(f)
        mesh = data.get("mesh", {})
        eigs = data.get("eigs", {})
        return (
            data.get("status") == "completed"
            and abs(float(data.get("s", float("nan"))) - float(amplitude_s)) < 1e-12
            and abs(float(mesh.get("h_min", float("nan"))) - float(args.mesh[0])) < 1e-12
            and abs(float(mesh.get("h_max", float("nan"))) - float(args.mesh[1])) < 1e-12
            and int(eigs.get("k_requested", -1)) == int(args.k)
            and abs(float(eigs.get("shift", float("nan"))) - float(args.eig_shift)) < 1e-12
        )
    except Exception:
        return False


def append_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "n/a") for key in CSV_COLUMNS})


def write_failure_log(path: Path, label: str, exc: BaseException) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(f"\n[{datetime.now().isoformat()}] {label} failed\n")
        f.write(repr(exc))
        f.write("\n")
        f.write(traceback.format_exc())
        f.write("\n")


def build_baseline_context(args: argparse.Namespace, coeffs: dict[int, complex]) -> dict[str, Any]:
    add_baseline_to_path()
    from boundary_data_fourier import FourierBC
    from config import SimulationParams, eps_from_R
    from continuation import ContinuationSolver
    from mesh import build_disk
    from newton_block import NewtonAssembler
    from poisson_neumann import PoissonNeumann
    from spaces import create_spaces

    h_min, h_max = (float(args.mesh[0]), float(args.mesh[1]))
    sigma_steps = max(int(args.sigma_steps), 2)
    params = SimulationParams(
        mu=float(args.mu),
        R=float(args.R),
        Rdom=float(args.Rdom),
        M=int(args.M),
        tol_newton=float(args.tol_newton),
        max_newton=int(args.max_newton),
        sigma_grid=tuple(float(v) for v in np.linspace(0.0, 1.0, sigma_steps)),
        h_min=h_min,
        h_max=h_max,
    )
    mesh, facet_tags = build_disk(params.Rdom, params.h_min, params.h_max)
    spaces = create_spaces(mesh)
    fourier_bc = FourierBC(params.M, coeffs)
    poisson = PoissonNeumann(
        mesh,
        spaces["Vpsi"],
        facet_tags,
        outer_tag=1,
        Rdom=params.Rdom,
        fourier_bc=fourier_bc,
        mu=params.mu,
        R=params.R,
    )
    assembler = NewtonAssembler(
        spaces["Vomega"],
        spaces["Vpsi"],
        facet_tags,
        outer_tag=1,
        eps=eps_from_R(params.R, params.Rdom),
        use_supg=False,
        tau_scale=1.0,
    )
    continuation = ContinuationSolver(
        assembler,
        poisson,
        params,
        facet_tags=facet_tags,
        outer_tag=1,
        fourier_bc=fourier_bc,
    )
    return {
        "params": params,
        "mesh": mesh,
        "facet_tags": facet_tags,
        "spaces": spaces,
        "fourier_bc": fourier_bc,
        "poisson": poisson,
        "assembler": assembler,
        "continuation": continuation,
    }


def mesh_stats(mesh: Any, spaces: dict[str, Any]) -> dict[str, int]:
    try:
        vertices = int(mesh.topology.index_map(0).size_global)
        cells = int(mesh.topology.index_map(mesh.topology.dim).size_global)
        dofs_omega = int(spaces["Vomega"].dofmap.index_map.size_global)
        dofs_psi = int(spaces["Vpsi"].dofmap.index_map.size_global)
    except Exception:
        vertices = cells = dofs_omega = dofs_psi = -1
    return {
        "vertices": vertices,
        "cells": cells,
        "dofs_omega": dofs_omega,
        "dofs_psi": dofs_psi,
    }


def run_one_amplitude(args: argparse.Namespace, manifest: dict[str, Any], amplitude_s: float) -> dict[str, Any]:
    add_baseline_to_path()
    from eigs_linearized_new import ManifoldEigenSolver
    from io_utils import write_xdmf
    from residuals_quick import residuals_quick

    run_tag = manifest["run_tag"]
    h_min, h_max = (float(args.mesh[0]), float(args.mesh[1]))
    mesh_id = mesh_label(h_min, h_max)
    amp_id = amplitude_label(amplitude_s)
    checkpoint_dir = BRANCH_DIR / "checkpoints" / run_tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    profile_xdmf = checkpoint_dir / f"{amp_id}_{mesh_id}_profile.xdmf"
    metadata_path = checkpoint_dir / f"{amp_id}_{mesh_id}_metadata.json"

    if completed_metadata(metadata_path, args, amplitude_s) and not args.force:
        print(f"[resume] skipping completed {amp_id} ({metadata_path})")
        return {"status": "skipped_completed", "metadata_checkpoint": metadata_path}

    coeffs = coeffs_for_exact_c3_cos(amplitude_s)
    t0 = time.perf_counter()
    status = "failed"
    metadata: dict[str, Any] = {
        "status": "started",
        "run_tag": run_tag,
        "s": float(amplitude_s),
        "boundary": boundary_description(amplitude_s),
        "mesh": {"h_min": h_min, "h_max": h_max},
        "started_at": datetime.now().isoformat(),
        "profile_checkpoint": relpath(profile_xdmf),
    }
    write_json(metadata_path, jsonable(metadata))

    context = build_baseline_context(args, coeffs)
    stats = mesh_stats(context["mesh"], context["spaces"])
    metadata["mesh_stats"] = stats
    write_json(metadata_path, jsonable(metadata))

    profile_t0 = time.perf_counter()
    omega, psi = context["continuation"].run(context["mesh"], context["spaces"])
    profile_wall = time.perf_counter() - profile_t0
    write_xdmf(str(profile_xdmf), {"Omega": omega, "psi": psi}, context["mesh"])

    pinned = getattr(context["poisson"], "gauge_dof", None)
    residuals = residuals_quick(
        context["mesh"],
        context["spaces"],
        omega,
        psi,
        mu=context["params"].mu,
        R=context["params"].R,
        Rdom=context["params"].Rdom,
        facet_tags=context["facet_tags"],
        outer_tag=1,
        fourier_bc=context["fourier_bc"],
        sigma=1.0,
        pinned_psi_dof=pinned,
    )
    metadata["profile"] = {
        "status": "completed",
        "wall_time_sec": profile_wall,
        "weak_residuals": {
            "Omega_l2": residuals["norms"]["Omega_l2"],
            "Psi_l2": residuals["norms"]["Psi_l2"],
        },
    }
    write_json(metadata_path, jsonable(metadata))

    omega_res = float(residuals["norms"]["Omega_l2"])
    psi_res = float(residuals["norms"]["Psi_l2"])
    if omega_res > float(args.max_profile_residual_omega) or psi_res > float(args.max_profile_residual_psi):
        metadata["status"] = "failed_profile_residual_gate"
        metadata["profile"]["status"] = "failed_residual_gate"
        metadata["profile"]["residual_gate"] = {
            "max_profile_residual_omega": float(args.max_profile_residual_omega),
            "max_profile_residual_psi": float(args.max_profile_residual_psi),
        }
        write_json(metadata_path, jsonable(metadata))
        raise RuntimeError(
            "Profile residual gate failed: "
            f"Omega_l2={omega_res:.6e}, Psi_l2={psi_res:.6e}"
        )

    eig_t0 = time.perf_counter()
    esolver = ManifoldEigenSolver(
        context["mesh"],
        context["spaces"],
        omega,
        psi,
        context["params"],
        context["facet_tags"],
        outer_tag=1,
    )
    modes = esolver.solve(
        k=int(args.k),
        sigma=float(args.eig_shift),
        tol=float(args.eig_tol),
        max_it=int(args.eig_max_it),
        verbose=True,
    )
    eig_wall = time.perf_counter() - eig_t0
    wall = time.perf_counter() - t0
    status = "completed"

    csv_rows = []
    for idx, mode in enumerate(modes[: int(args.k)], start=1):
        lam = complex(mode.lam)
        csv_rows.append(
            {
                "s": float(amplitude_s),
                "mesh_id": mesh_id,
                "h_min": h_min,
                "h_max": h_max,
                "mesh_vertices": stats["vertices"],
                "mesh_cells": stats["cells"],
                "eigen_index": idx,
                "real_lambda": float(lam.real),
                "imag_lambda": float(lam.imag),
                "eig_residual": "n/a",
                "profile_residual_omega_l2": residuals["norms"]["Omega_l2"],
                "profile_residual_psi_l2": residuals["norms"]["Psi_l2"],
                "profile_wall_time": profile_wall,
                "eigs_wall_time": eig_wall,
                "wall_time": wall,
                "status": status,
                "profile_checkpoint": relpath(profile_xdmf),
                "metadata_checkpoint": relpath(metadata_path),
                "message": "h006_fem_point_completed" if h_max <= 0.06 else "coarse_smoke_completed",
            }
        )

    metadata.update(
        {
            "status": status,
            "completed_at": datetime.now().isoformat(),
            "wall_time_sec": wall,
            "eigs": {
                "wall_time_sec": eig_wall,
                "k_requested": int(args.k),
                "k_returned": len(modes),
                "shift": float(args.eig_shift),
                "tol": float(args.eig_tol),
                "max_it": int(args.eig_max_it),
                "eigenvalues": [
                    {"index": i, "real": float(complex(m.lam).real), "imag": float(complex(m.lam).imag)}
                    for i, m in enumerate(modes[: int(args.k)], start=1)
                ],
                "eig_residual_note": "not computed in the first coarse smoke runner",
            },
        }
    )
    write_json(metadata_path, jsonable(metadata))
    return {
        "status": status,
        "metadata_checkpoint": metadata_path,
        "profile_checkpoint": profile_xdmf,
        "rows": csv_rows,
        "wall_time_sec": wall,
        "profile_residuals": metadata["profile"]["weak_residuals"],
        "eigenvalues": metadata["eigs"]["eigenvalues"],
        "mesh_stats": stats,
    }


def run_execution(args: argparse.Namespace, manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    run_tag = manifest["run_tag"]
    data_csv = BRANCH_DIR / "data" / f"{run_tag}_eigenvalues.csv"
    log_file = BRANCH_DIR / "logs" / f"{run_tag}.log"
    if args.force and data_csv.exists():
        data_csv.unlink()
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    status = "completed"

    for amplitude_s in [float(s) for s in args.s_values]:
        label = amplitude_label(amplitude_s)
        try:
            result = run_one_amplitude(args, manifest, amplitude_s)
            if result.get("rows"):
                append_csv_rows(data_csv, result["rows"])
            results.append({"s": amplitude_s, **{k: v for k, v in result.items() if k != "rows"}})
        except Exception as exc:
            status = "partial_failed"
            write_failure_log(log_file, label, exc)
            results.append(
                {
                    "s": amplitude_s,
                    "status": "failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            break

    wall = time.perf_counter() - t0
    completed = [r for r in results if r.get("status") == "completed"]
    skipped = [r for r in results if r.get("status") == "skipped_completed"]
    failed = [r for r in results if r.get("status") == "failed"]
    if failed:
        status = "partial_failed"
    elif completed or skipped:
        status = "completed"

    manifest.update(
        {
            "status": status,
            "executed_at": datetime.now().isoformat(),
            "wall_time_sec": wall,
            "outputs": {
                "data_csv": relpath(data_csv),
                "log_file": relpath(log_file),
            },
            "runs": jsonable(results),
        }
    )
    write_json(manifest_path, jsonable(manifest))

    if not args.no_registry:
        eig_summary = "n/a"
        error_metrics = "n/a"
        if completed:
            first = completed[0]
            eig_summary = "; ".join(
                f"{ev['real']:.6e}{ev['imag']:+.6e}j" for ev in first.get("eigenvalues", [])
            )
            residuals = first.get("profile_residuals", {})
            error_metrics = (
                f"profile_weak_R_Omega={residuals.get('Omega_l2', 'n/a')}; "
                f"profile_weak_R_Psi={residuals.get('Psi_l2', 'n/a')}"
            )
        append_registry_row(
            {
                "date": today(),
                "tag": f"fem-branchA-h006-{run_tag}" if float(args.mesh[1]) <= 0.06 else f"fem-branchA-smoke-{run_tag}",
                "codebase": "baseline",
                "script": "experiments/fem_manuscript/c3_spectrum_continuation/run_spectrum_continuation.py",
                "input_case": "s_cos3theta",
                "domain": f"disk_Rdom{float(args.Rdom):g}",
                "grid_or_mesh": f"h={float(args.mesh[0]):g}/{float(args.mesh[1]):g}",
                "solver": "baseline_fem_continuation_plus_shift_invert_eigs",
                "parameters": (
                    f"s_values={','.join(str(float(s)) for s in args.s_values)}; "
                    f"k={int(args.k)}; eig_shift={float(args.eig_shift)}; "
                    f"mu={float(args.mu)}; R={float(args.R)}; sigma_steps={int(args.sigma_steps)}"
                ),
                "wall_time_sec": f"{wall:.6f}",
                "error_metrics": error_metrics,
                "eigenvalues_summary": eig_summary,
                "status": status,
                "notes": (
                    "Initial h=0.06 Branch A point for s-step calibration; exact s cos(3theta) encoded as c3=s/2."
                    if float(args.mesh[1]) <= 0.06
                    else "Coarse Branch A smoke test; not manuscript data; exact s cos(3theta) encoded as c3=s/2."
                ),
            }
        )
    return manifest


def main() -> int:
    args = parse_args()
    ensure_branch_tree(BRANCH_DIR)
    manifest = build_manifest(args)
    manifest_path = BRANCH_DIR / "logs" / f"{manifest['run_tag']}_manifest.json"
    write_json(manifest_path, manifest)
    print(f"[manifest] wrote {manifest_path}")

    if args.execute:
        manifest = run_execution(args, manifest, manifest_path)
        print(f"[done] status={manifest['status']} manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
