#!/usr/bin/env python3
"""Branch B first pass: perturb the s=140 symmetric FEM profile.

This script uses the accepted Branch A h=0.06 symmetric profile at s=140 and
the saved first-three eigenmode arrays. It applies interior perturbations that
preserve the boundary data, runs fixed-boundary Newton from each perturbed
state, and compares the converged state against the symmetric profile.

The perturbation for mode j and sign q is

    Omega_0 = Omega_sym + q * eta * ||Omega_sym||_rms / ||deltaOmega_j||_rms
              * deltaOmega_j,
    psi_0   = psi_sym   + q * eta * ||Omega_sym||_rms / ||deltaOmega_j||_rms
              * deltaPsi_j,

where (deltaOmega_j, deltaPsi_j) is the real part of a saved eigenmode. Since
the eigenmode reconstruction has zero omega trace on the boundary, this leaves
the boundary value exactly unchanged up to roundoff.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from petsc4py import PETSc

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig-euler")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

BRANCH_DIR = Path(__file__).resolve().parent
MANUSCRIPT_ROOT = BRANCH_DIR.parents[0]
REPO_ROOT = MANUSCRIPT_ROOT.parents[1]
C3_BRANCH_DIR = MANUSCRIPT_ROOT / "c3_spectrum_continuation"
sys.path.insert(0, str(MANUSCRIPT_ROOT))
sys.path.insert(0, str(C3_BRANCH_DIR))

from common import (  # noqa: E402
    add_baseline_to_path,
    append_registry_row,
    coeffs_for_exact_c3_cos,
    today,
    write_json,
)
from run_spectrum_continuation import build_baseline_context, mesh_stats  # noqa: E402


CSV_COLUMNS = [
    "trial_id",
    "status",
    "mode_index",
    "sign",
    "eta",
    "scale",
    "newton_iterations",
    "final_newton_residual",
    "final_relative_step",
    "profile_residual_omega_l2",
    "profile_residual_psi_l2",
    "boundary_max_abs_diff",
    "omega_rms_separation",
    "omega_linf_separation",
    "psi_rms_separation",
    "psi_linf_separation",
    "omega_c3_fraction_symmetric",
    "omega_c3_fraction_candidate",
    "psi_c3_fraction_symmetric",
    "psi_c3_fraction_candidate",
    "wall_time_sec",
    "profile_checkpoint",
    "plot_path",
    "message",
]


def vec_array(fun: Any) -> np.ndarray:
    try:
        return fun.vector.array
    except AttributeError:
        return fun.x.array


def ghost_forward(fun: Any) -> None:
    try:
        fun.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    except AttributeError:
        if hasattr(fun.x, "scatter_forward"):
            fun.x.scatter_forward()


def owned_size(V: Any) -> int:
    im = V.dofmap.index_map
    bs = getattr(im, "block_size", None) or getattr(im, "bs", None) or getattr(V.dofmap, "index_map_bs", 1)
    return im.size_local * int(bs)


def split_block_vec(x: PETSc.Vec, Vomega: Any, Vpsi: Any):
    n_o = owned_size(Vomega)
    n_p = owned_size(Vpsi)
    is_o = PETSc.IS().createStride(n_o, 0, 1)
    is_p = PETSc.IS().createStride(n_p, n_o, 1)
    return (x.getSubVector(is_o), is_o), (x.getSubVector(is_p), is_p)


def subvec_array(v: PETSc.Vec) -> np.ndarray:
    try:
        return v.array.copy()
    except AttributeError:
        with v.localForm() as lf:
            return lf.array.copy()


def copy_function(template_space: Any, values: np.ndarray):
    from dolfinx import fem

    out = fem.Function(template_space)
    vec_array(out)[:] = values
    ghost_forward(out)
    return out


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def linf(values: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(values, dtype=float))))


def load_h5_profile(path: Path, mesh: Any, spaces: dict[str, Any]):
    with h5py.File(path, "r") as h5:
        saved_geom = h5["/Mesh/mesh/geometry"][:]
        geom_diff = float(np.max(np.abs(mesh.geometry.x[:, :2] - saved_geom)))
        if geom_diff > 1e-12:
            raise RuntimeError(f"profile checkpoint mesh mismatch: max geometry diff={geom_diff:.3e}")
        omega = copy_function(spaces["Vomega"], np.asarray(h5["/Function/Omega/0"][:]).reshape(-1))
        psi = copy_function(spaces["Vpsi"], np.asarray(h5["/Function/psi/0"][:]).reshape(-1))
    return omega, psi, geom_diff


def load_mode_arrays(path: Path, spaces: dict[str, Any], mode_index: int):
    with np.load(path) as data:
        omega = copy_function(spaces["Vomega"], data[f"omega_re_{mode_index}"])
        psi = copy_function(spaces["Vpsi"], data[f"psi_re_{mode_index}"])
        lam = complex(float(data["real_lambda"][mode_index - 1]), float(data["imag_lambda"][mode_index - 1]))
    return omega, psi, lam


def make_initial_guess(spaces: dict[str, Any], omega_sym: Any, psi_sym: Any, domega: Any, dpsi: Any, eta: float, sign: float):
    omega_sym_arr = vec_array(omega_sym)
    domega_arr = vec_array(domega)
    scale = float(sign) * float(eta) * rms(omega_sym_arr) / max(rms(domega_arr), 1e-30)
    omega0 = copy_function(spaces["Vomega"], omega_sym_arr + scale * domega_arr)
    psi0 = copy_function(spaces["Vpsi"], vec_array(psi_sym) + scale * vec_array(dpsi))
    return omega0, psi0, scale


def boundary_max_abs_diff(spaces: dict[str, Any], facet_tags: Any, omega_a: Any, omega_b: Any) -> float:
    from dolfinx import fem

    outer_facets = facet_tags.find(1)
    bdofs = fem.locate_dofs_topological(spaces["Vomega"], spaces["Vomega"].mesh.topology.dim - 1, outer_facets)
    if len(bdofs) == 0:
        return float("nan")
    diff = vec_array(omega_a)[bdofs.astype(np.int64)] - vec_array(omega_b)[bdofs.astype(np.int64)]
    return linf(diff)


def run_newton_from(context: dict[str, Any], omega0: Any, psi0: Any, *, max_newton: int, tol: float, linesearch: bool):
    mesh = context["mesh"]
    spaces = context["spaces"]
    assembler = context["assembler"]
    Vomega, Vpsi = spaces["Vomega"], spaces["Vpsi"]

    omega = copy_function(Vomega, vec_array(omega0).copy())
    psi = copy_function(Vpsi, vec_array(psi0).copy())

    A, b = assembler.assemble(omega, psi, mu=context["params"].mu)
    bnorm = float(b.norm(PETSc.NormType.NORM_2))
    history = [{"iteration": 0, "residual_norm": bnorm, "relative_step": None, "alpha": 0.0}]
    converged = False
    final_rel = float("nan")

    for it in range(1, int(max_newton) + 1):
        ksp = PETSc.KSP().create(mesh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()

        x = b.duplicate()
        x.set(0.0)
        ksp.solve(b, x)
        if int(ksp.getConvergedReason()) < 0:
            raise RuntimeError(f"Newton linear solve failed at iteration {it}: reason={int(ksp.getConvergedReason())}")

        (x1, is1), (x2, is2) = split_block_vec(x, Vomega, Vpsi)
        d1 = subvec_array(x1)
        d2 = subvec_array(x2)
        x.restoreSubVector(is1, x1)
        x.restoreSubVector(is2, x2)

        omega_old = vec_array(omega).copy()
        psi_old = vec_array(psi).copy()
        alpha = 1.0
        accepted = False
        for _ in range(7 if linesearch else 1):
            vec_array(omega)[:] = omega_old + alpha * d1
            vec_array(psi)[:] = psi_old + alpha * d2
            ghost_forward(omega)
            ghost_forward(psi)
            A_trial, b_trial = assembler.assemble(omega, psi, mu=context["params"].mu)
            bnorm_trial = float(b_trial.norm(PETSc.NormType.NORM_2))
            if (not linesearch) or bnorm_trial <= bnorm:
                A, b = A_trial, b_trial
                bnorm = bnorm_trial
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            A, b = A_trial, b_trial
            bnorm = bnorm_trial

        final_rel = rms(alpha * d1) / max(rms(vec_array(omega)), 1e-30)
        history.append({"iteration": it, "residual_norm": bnorm, "relative_step": final_rel, "alpha": alpha})
        print(f"[Newton probe] it={it:02d} ||b||={bnorm:.6e} rel={final_rel:.6e} alpha={alpha:.3g}")
        if final_rel < float(tol):
            converged = True
            break

    return omega, psi, history, converged


def interpolation_values(V: Any, fun: Any):
    raw = V.tabulate_dof_coordinates()
    if raw.ndim == 1:
        raw = raw.reshape(-1, V.mesh.geometry.x.shape[1])
    return raw[:, :2], vec_array(fun).copy()


def c3_fraction_on_probe(V: Any, fun: Any, *, r_count: int = 50, theta_count: int = 240, r_max: float = 9.75) -> float:
    from scipy.interpolate import LinearNDInterpolator

    xy, vals = interpolation_values(V, fun)
    interp = LinearNDInterpolator(xy, vals, fill_value=np.nan)
    radii = np.linspace(0.05, r_max, r_count)
    theta = np.linspace(0.0, 2.0 * np.pi, theta_count, endpoint=False)
    rr, tt = np.meshgrid(radii, theta, indexing="ij")

    samples = []
    for shift in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0):
        x = rr * np.cos(tt + shift)
        y = rr * np.sin(tt + shift)
        samples.append(interp(x.reshape(-1), y.reshape(-1)).reshape(rr.shape))
    stacked = np.stack(samples, axis=0)
    valid = np.all(np.isfinite(stacked), axis=0)
    if not np.any(valid):
        return float("nan")
    avg = np.mean(stacked[:, valid], axis=0)
    defect = stacked[0, valid] - avg
    denom = np.sqrt(np.mean(stacked[0, valid] ** 2))
    return float(np.sqrt(np.mean(defect ** 2)) / max(denom, 1e-30))


def triangulation(mesh: Any) -> mtri.Triangulation:
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    cells = mesh.topology.connectivity(tdim, 0).array.reshape(-1, 3)
    xy = mesh.geometry.x[:, :2]
    return mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=cells)


def plot_comparison(path: Path, mesh: Any, omega_sym: Any, omega_candidate: Any, title: str) -> None:
    tri = triangulation(mesh)
    sym = vec_array(omega_sym)
    cand = vec_array(omega_candidate)
    diff = cand - sym
    fields = [sym, cand, diff]
    labels = ["symmetric", "symmetry-breaking", "symmetry-breaking - symmetric"]
    vmax = max(linf(sym), linf(cand), 1e-30)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), dpi=220, constrained_layout=True)
    for ax, values, label in zip(axes, fields, labels):
        local_vmax = linf(values) if label.startswith("candidate -") else vmax
        cf = ax.tricontourf(tri, values, levels=64, cmap="RdBu_r", vmin=-local_vmax, vmax=local_vmax)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(label, fontsize=9)
        ax.set_aspect("equal", "box")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_xdmf_profile(path: Path, mesh: Any, omega: Any, psi: Any) -> None:
    add_baseline_to_path()
    from io_utils import write_xdmf

    write_xdmf(str(path), {"Omega": omega, "psi": psi}, mesh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--s", type=float, default=140.0)
    parser.add_argument("--modes", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--etas", nargs="+", type=float, default=[0.2, 0.5, 1.0])
    parser.add_argument("--signs", nargs="+", type=float, default=[1.0, -1.0])
    parser.add_argument("--max-newton", type=int, default=24)
    parser.add_argument("--tol-newton", type=float, default=1e-8)
    parser.add_argument("--no-linesearch", action="store_true")
    parser.add_argument("--stop-on-distinct", action="store_true")
    parser.add_argument("--distinct-omega-rms", type=float, default=1e-3)
    parser.add_argument("--distinct-c3-excess", type=float, default=1e-3)
    parser.add_argument("--run-tag", default="branchB_s140_eigenmode_perturb_probe")
    parser.add_argument(
        "--symmetric-profile-h5",
        type=Path,
        default=C3_BRANCH_DIR / "checkpoints/branchA_h006_refine_s125_140_k10/s140_h0p06_0p06_profile.h5",
    )
    parser.add_argument(
        "--mode-cache",
        type=Path,
        default=C3_BRANCH_DIR / "plots/s140_first3_eigenmodes/s140_first3_mode_arrays.npz",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_data = BRANCH_DIR / "data"
    out_logs = BRANCH_DIR / "logs"
    out_plots = BRANCH_DIR / "plots" / args.run_tag
    out_checkpoints = BRANCH_DIR / "checkpoints" / args.run_tag
    for d in (out_data, out_logs, out_plots, out_checkpoints):
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = out_logs / f"{args.run_tag}_manifest.json"
    csv_path = out_data / f"{args.run_tag}_results.csv"
    manifest = {
        "status": "dry_run",
        "run_tag": args.run_tag,
        "s": args.s,
        "modes": args.modes,
        "etas": args.etas,
        "signs": args.signs,
        "perturbation_formula": (
            "Omega0=Omega_sym+sign*eta*rms(Omega_sym)/rms(deltaOmega_j)*deltaOmega_j; "
            "psi0=psi_sym+same_scale*deltaPsi_j; delta fields are real parts of saved eigenmode j."
        ),
        "boundary_policy": "deltaOmega_j has zero boundary trace; Newton increments also impose homogeneous omega boundary.",
        "outputs": {
            "csv": str(csv_path.relative_to(REPO_ROOT)),
            "plots": str(out_plots.relative_to(REPO_ROOT)),
            "checkpoints": str(out_checkpoints.relative_to(REPO_ROOT)),
        },
    }
    write_json(manifest_path, manifest)
    if not args.execute:
        print(f"[dry-run] wrote {manifest_path}")
        return 0

    t_all = time.perf_counter()
    ns = argparse.Namespace(
        mesh=[0.06, 0.06],
        sigma_steps=21,
        mu=0.8,
        R=150.0,
        Rdom=10.0,
        M=48,
        tol_newton=float(args.tol_newton),
        max_newton=int(args.max_newton),
    )
    add_baseline_to_path()
    from residuals_quick import residuals_quick

    context = build_baseline_context(ns, coeffs_for_exact_c3_cos(args.s))
    omega_sym, psi_sym, geom_diff = load_h5_profile(args.symmetric_profile_h5, context["mesh"], context["spaces"])
    pinned = getattr(context["poisson"], "gauge_dof", None)
    sym_res = residuals_quick(
        context["mesh"],
        context["spaces"],
        omega_sym,
        psi_sym,
        mu=context["params"].mu,
        R=context["params"].R,
        Rdom=context["params"].Rdom,
        facet_tags=context["facet_tags"],
        outer_tag=1,
        fourier_bc=context["fourier_bc"],
        sigma=1.0,
        pinned_psi_dof=pinned,
    )
    omega_c3_sym = c3_fraction_on_probe(context["spaces"]["Vomega"], omega_sym)
    psi_c3_sym = c3_fraction_on_probe(context["spaces"]["Vpsi"], psi_sym)

    csv_exists = csv_path.exists()
    rows: list[dict[str, Any]] = []
    stop_reason = None
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not csv_exists:
            writer.writeheader()

        for mode_index in args.modes:
            domega, dpsi, lam = load_mode_arrays(args.mode_cache, context["spaces"], mode_index)
            for eta in args.etas:
                for sign in args.signs:
                    trial_id = f"mode{mode_index}_eta{eta:g}_sign{sign:+g}".replace(".", "p").replace("+", "p").replace("-", "m")
                    trial_t0 = time.perf_counter()
                    omega0, psi0, scale = make_initial_guess(context["spaces"], omega_sym, psi_sym, domega, dpsi, eta, sign)
                    init_bdiff = boundary_max_abs_diff(context["spaces"], context["facet_tags"], omega0, omega_sym)
                    try:
                        omega_candidate, psi_candidate, history, converged = run_newton_from(
                            context,
                            omega0,
                            psi0,
                            max_newton=args.max_newton,
                            tol=args.tol_newton,
                            linesearch=not args.no_linesearch,
                        )
                        profile_path = out_checkpoints / f"{trial_id}_profile.xdmf"
                        write_xdmf_profile(profile_path, context["mesh"], omega_candidate, psi_candidate)
                        residuals = residuals_quick(
                            context["mesh"],
                            context["spaces"],
                            omega_candidate,
                            psi_candidate,
                            mu=context["params"].mu,
                            R=context["params"].R,
                            Rdom=context["params"].Rdom,
                            facet_tags=context["facet_tags"],
                            outer_tag=1,
                            fourier_bc=context["fourier_bc"],
                            sigma=1.0,
                            pinned_psi_dof=pinned,
                        )
                        omega_sep = vec_array(omega_candidate) - vec_array(omega_sym)
                        psi_sep = vec_array(psi_candidate) - vec_array(psi_sym)
                        omega_c3 = c3_fraction_on_probe(context["spaces"]["Vomega"], omega_candidate)
                        psi_c3 = c3_fraction_on_probe(context["spaces"]["Vpsi"], psi_candidate)
                        plot_path = out_plots / f"{trial_id}_omega_comparison"
                        plot_comparison(
                            plot_path,
                            context["mesh"],
                            omega_sym,
                            omega_candidate,
                            f"sigma=140, mode {mode_index}, eta={eta:g}, sign={sign:+g}",
                        )
                        final = history[-1]
                        status = "completed_distinct" if converged else "nonconverged"
                        distinct = (
                            converged
                            and rms(omega_sep) / max(rms(vec_array(omega_sym)), 1e-30) > args.distinct_omega_rms
                            and omega_c3 - omega_c3_sym > args.distinct_c3_excess
                        )
                        if distinct:
                            status = "completed_distinct_symmetry_breaking_candidate"
                        row = {
                            "trial_id": trial_id,
                            "status": status,
                            "mode_index": mode_index,
                            "sign": sign,
                            "eta": eta,
                            "scale": scale,
                            "newton_iterations": int(final["iteration"]),
                            "final_newton_residual": float(final["residual_norm"]),
                            "final_relative_step": float(final["relative_step"]),
                            "profile_residual_omega_l2": residuals["norms"]["Omega_l2"],
                            "profile_residual_psi_l2": residuals["norms"]["Psi_l2"],
                            "boundary_max_abs_diff": boundary_max_abs_diff(context["spaces"], context["facet_tags"], omega_candidate, omega_sym),
                            "omega_rms_separation": rms(omega_sep),
                            "omega_linf_separation": linf(omega_sep),
                            "psi_rms_separation": rms(psi_sep),
                            "psi_linf_separation": linf(psi_sep),
                            "omega_c3_fraction_symmetric": omega_c3_sym,
                            "omega_c3_fraction_candidate": omega_c3,
                            "psi_c3_fraction_symmetric": psi_c3_sym,
                            "psi_c3_fraction_candidate": psi_c3,
                            "wall_time_sec": time.perf_counter() - trial_t0,
                            "profile_checkpoint": str(profile_path.relative_to(REPO_ROOT)),
                            "plot_path": str(plot_path.with_suffix(".png").relative_to(REPO_ROOT)),
                            "message": f"lambda_mode={lam.real:.12g}{lam.imag:+.12g}i; initial_boundary_diff={init_bdiff:.3e}",
                        }
                    except Exception as exc:
                        row = {
                            "trial_id": trial_id,
                            "status": "failed",
                            "mode_index": mode_index,
                            "sign": sign,
                            "eta": eta,
                            "scale": scale if "scale" in locals() else "n/a",
                            "wall_time_sec": time.perf_counter() - trial_t0,
                            "message": repr(exc),
                        }
                    writer.writerow({key: row.get(key, "n/a") for key in CSV_COLUMNS})
                    f.flush()
                    rows.append(row)
                    print(f"[trial] {trial_id}: {row['status']} omega_sep={row.get('omega_rms_separation', 'n/a')} c3={row.get('omega_c3_fraction_candidate', 'n/a')}")
                    if args.stop_on_distinct and row["status"] == "completed_distinct_symmetry_breaking_candidate":
                        stop_reason = f"distinct candidate found: {trial_id}"
                        break
                if stop_reason:
                    break
            if stop_reason:
                break

    manifest.update(
        {
            "status": "completed",
            "mesh_stats": mesh_stats(context["mesh"], context["spaces"]),
            "profile_checkpoint_geometry_diff": geom_diff,
            "symmetric_profile_residuals": sym_res["norms"],
            "symmetric_c3": {"Omega": omega_c3_sym, "psi": psi_c3_sym},
            "num_trials": len(rows),
            "stop_reason": stop_reason,
            "wall_time_sec": time.perf_counter() - t_all,
        }
    )
    write_json(manifest_path, manifest)
    append_registry_row(
        {
            "date": today(),
            "tag": f"fem-branchB-{args.run_tag}",
            "codebase": "baseline",
            "script": str(Path(__file__).relative_to(REPO_ROOT)),
            "input_case": "s_cos3theta_s140",
            "domain": "disk_Rdom10",
            "grid_or_mesh": "h=0.06/0.06",
            "solver": "checkpointed_profile_eigenmode_perturbation_fixed_boundary_newton",
            "parameters": f"modes={args.modes}; etas={args.etas}; signs={args.signs}; max_newton={args.max_newton}",
            "wall_time_sec": f"{manifest['wall_time_sec']:.6f}",
            "error_metrics": f"symmetric_R_Omega={sym_res['norms']['Omega_l2']}; symmetric_c3_Omega={omega_c3_sym}",
            "eigenvalues_summary": f"trials={len(rows)}; stop_reason={stop_reason}",
            "status": "completed",
            "notes": "First Branch B symmetry-breaking perturbation probe; candidate status requires follow-up validation.",
        }
    )
    print(f"[done] results={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
