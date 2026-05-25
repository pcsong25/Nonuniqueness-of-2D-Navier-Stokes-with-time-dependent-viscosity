#!/usr/bin/env python3
"""Track the Branch B broken profile in boundary amplitude sigma.

Seed: the artificial localized perturbation candidate at sigma=140.
Targets: upward to 150 and downward through 130,125,120,115,110.

At each new sigma, the previous broken Omega is scaled by sigma_new/sigma_old,
then its boundary dofs are replaced by the accepted symmetric profile boundary
for the target sigma. Psi is recomputed by the baseline Poisson solver before
fixed-boundary Newton. This preserves the exact boundary data and avoids using
eigenfunctions during continuation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np

BRANCH_DIR = Path(__file__).resolve().parent
MANUSCRIPT_ROOT = BRANCH_DIR.parents[0]
REPO_ROOT = MANUSCRIPT_ROOT.parents[1]
C3_BRANCH_DIR = MANUSCRIPT_ROOT / 'c3_spectrum_continuation'
sys.path.insert(0, str(MANUSCRIPT_ROOT))
sys.path.insert(0, str(BRANCH_DIR))
sys.path.insert(0, str(C3_BRANCH_DIR))

from common import add_baseline_to_path, append_registry_row, coeffs_for_exact_c3_cos, today, write_json  # noqa: E402
from run_s140_bifurcation_probe import (  # noqa: E402
    boundary_max_abs_diff,
    c3_fraction_on_probe,
    copy_function,
    linf,
    load_h5_profile,
    plot_comparison,
    rms,
    run_newton_from,
    vec_array,
    write_xdmf_profile,
)
from run_s140_artificial_perturb_probe import diff_function  # noqa: E402
from run_spectrum_continuation import build_baseline_context, mesh_stats  # noqa: E402

CSV_COLUMNS = [
    'sigma', 'direction', 'status', 'source_sigma', 'newton_iterations',
    'final_newton_residual', 'final_relative_step',
    'profile_residual_omega_l2', 'profile_residual_psi_l2', 'boundary_max_abs_diff',
    'omega_rms_separation', 'omega_relative_rms_separation', 'omega_linf_separation',
    'psi_rms_separation', 'psi_relative_rms_separation', 'psi_linf_separation',
    'omega_c3_fraction_symmetric', 'omega_c3_fraction_broken', 'omega_c3_fraction_excess',
    'omega_difference_non3_fraction', 'psi_c3_fraction_symmetric', 'psi_c3_fraction_broken',
    'psi_c3_fraction_excess', 'psi_difference_non3_fraction', 'wall_time_sec',
    'symmetric_profile_h5', 'broken_profile_checkpoint', 'plot_path', 'message',
]


def sigma_label(sigma: float) -> str:
    return f"s{float(sigma):g}".replace('.', 'p')


def find_symmetric_profile_h5(sigma: float) -> Path:
    label = sigma_label(sigma)
    matches = sorted(C3_BRANCH_DIR.glob(f'checkpoints/**/{label}_h0p06_0p06_profile.h5'))
    if not matches:
        raise FileNotFoundError(f'no accepted symmetric profile checkpoint found for sigma={sigma:g}')
    # Prefer refined-grid and refined interval checkpoints over older failed/partial ladders.
    def priority(p: Path) -> tuple[int, str]:
        text = str(p)
        if 'branchA_h006_refined_s0_120_step5_plus150_k10' in text:
            return (0, text)
        if 'branchA_h006_refine_s125_140_k10' in text:
            return (1, text)
        if 'branchA_h006_ladder_s160_280_k10_cont21' in text:
            return (2, text)
        return (3, text)
    return sorted(matches, key=priority)[0]


def load_profile_any_h5(path: Path, context: dict[str, Any]):
    return load_h5_profile(path, context['mesh'], context['spaces'])[:2]


def set_boundary_from_reference(context: dict[str, Any], omega: Any, omega_ref: Any) -> None:
    from dolfinx import fem
    facets = context['facet_tags'].find(1)
    bdofs = fem.locate_dofs_topological(context['spaces']['Vomega'], context['mesh'].topology.dim - 1, facets)
    vec_array(omega)[bdofs.astype(np.int64)] = vec_array(omega_ref)[bdofs.astype(np.int64)]
    try:
        from run_s140_bifurcation_probe import ghost_forward
        ghost_forward(omega)
    except Exception:
        pass


def make_continuation_initial(context: dict[str, Any], prev_omega: Any, prev_sigma: float, target_sigma: float, omega_sym_target: Any):
    ratio = float(target_sigma) / float(prev_sigma)
    omega0 = copy_function(context['spaces']['Vomega'], ratio * vec_array(prev_omega))
    set_boundary_from_reference(context, omega0, omega_sym_target)
    psi0 = context['poisson'].solve_state(omega0, sigma=1.0)
    return omega0, psi0


def compute_metrics(context: dict[str, Any], omega_sym: Any, psi_sym: Any, omega_broken: Any, psi_broken: Any, residuals: dict[str, Any]) -> dict[str, Any]:
    omega_sep = vec_array(omega_broken) - vec_array(omega_sym)
    psi_sep = vec_array(psi_broken) - vec_array(psi_sym)
    omega_diff_fun = diff_function(context['spaces']['Vomega'], omega_broken, omega_sym)
    psi_diff_fun = diff_function(context['spaces']['Vpsi'], psi_broken, psi_sym)
    omega_c3_sym = c3_fraction_on_probe(context['spaces']['Vomega'], omega_sym)
    psi_c3_sym = c3_fraction_on_probe(context['spaces']['Vpsi'], psi_sym)
    omega_c3 = c3_fraction_on_probe(context['spaces']['Vomega'], omega_broken)
    psi_c3 = c3_fraction_on_probe(context['spaces']['Vpsi'], psi_broken)
    return {
        'profile_residual_omega_l2': residuals['norms']['Omega_l2'],
        'profile_residual_psi_l2': residuals['norms']['Psi_l2'],
        'boundary_max_abs_diff': boundary_max_abs_diff(context['spaces'], context['facet_tags'], omega_broken, omega_sym),
        'omega_rms_separation': rms(omega_sep),
        'omega_relative_rms_separation': rms(omega_sep) / max(rms(vec_array(omega_sym)), 1e-30),
        'omega_linf_separation': linf(omega_sep),
        'psi_rms_separation': rms(psi_sep),
        'psi_relative_rms_separation': rms(psi_sep) / max(rms(vec_array(psi_sym)), 1e-30),
        'psi_linf_separation': linf(psi_sep),
        'omega_c3_fraction_symmetric': omega_c3_sym,
        'omega_c3_fraction_broken': omega_c3,
        'omega_c3_fraction_excess': omega_c3 - omega_c3_sym,
        'omega_difference_non3_fraction': c3_fraction_on_probe(context['spaces']['Vomega'], omega_diff_fun),
        'psi_c3_fraction_symmetric': psi_c3_sym,
        'psi_c3_fraction_broken': psi_c3,
        'psi_c3_fraction_excess': psi_c3 - psi_c3_sym,
        'psi_difference_non3_fraction': c3_fraction_on_probe(context['spaces']['Vpsi'], psi_diff_fun),
    }


def solve_or_record_seed(args: argparse.Namespace, sigma: float, direction: str, out_dirs: dict[str, Path], writer: csv.DictWriter, seed_h5: Path) -> tuple[Any, Any]:
    add_baseline_to_path()
    from residuals_quick import residuals_quick

    ns = argparse.Namespace(mesh=[0.06, 0.06], sigma_steps=21, mu=0.8, R=150.0, Rdom=10.0, M=48, tol_newton=args.tol_newton, max_newton=args.max_newton)
    context = build_baseline_context(ns, coeffs_for_exact_c3_cos(sigma))
    sym_h5 = find_symmetric_profile_h5(sigma)
    omega_sym, psi_sym = load_profile_any_h5(sym_h5, context)
    omega_broken, psi_broken = load_profile_any_h5(seed_h5, context)
    pinned = getattr(context['poisson'], 'gauge_dof', None)
    residuals = residuals_quick(context['mesh'], context['spaces'], omega_broken, psi_broken, mu=context['params'].mu, R=context['params'].R, Rdom=context['params'].Rdom, facet_tags=context['facet_tags'], outer_tag=1, fourier_bc=context['fourier_bc'], sigma=1.0, pinned_psi_dof=pinned)
    profile_path = out_dirs['checkpoints'] / f'{sigma_label(sigma)}_{direction}_profile.xdmf'
    write_xdmf_profile(profile_path, context['mesh'], omega_broken, psi_broken)
    plot_path = out_dirs['plots'] / f'{sigma_label(sigma)}_{direction}_omega_comparison'
    plot_comparison(plot_path, context['mesh'], omega_sym, omega_broken, r'$\sigma = 140$: bifurcation of vorticity profiles')
    row = {'sigma': sigma, 'direction': direction, 'status': 'seed_loaded', 'source_sigma': sigma, 'newton_iterations': 0, 'final_newton_residual': 'n/a', 'final_relative_step': 'n/a', 'symmetric_profile_h5': str(sym_h5.relative_to(REPO_ROOT)), 'broken_profile_checkpoint': str(profile_path.relative_to(REPO_ROOT)), 'plot_path': str(plot_path.with_suffix('.png').relative_to(REPO_ROOT)), 'message': 'seed artificial localized broken candidate'}
    row.update(compute_metrics(context, omega_sym, psi_sym, omega_broken, psi_broken, residuals))
    writer.writerow({k: row.get(k, 'n/a') for k in CSV_COLUMNS})
    return omega_broken, psi_broken


def run_point(args: argparse.Namespace, sigma: float, direction: str, source_sigma: float, prev_omega: Any, out_dirs: dict[str, Path], writer: csv.DictWriter) -> tuple[Any, Any, dict[str, Any]]:
    add_baseline_to_path()
    from residuals_quick import residuals_quick

    t0 = time.perf_counter()
    ns = argparse.Namespace(mesh=[0.06, 0.06], sigma_steps=21, mu=0.8, R=150.0, Rdom=10.0, M=48, tol_newton=args.tol_newton, max_newton=args.max_newton)
    context = build_baseline_context(ns, coeffs_for_exact_c3_cos(sigma))
    sym_h5 = find_symmetric_profile_h5(sigma)
    omega_sym, psi_sym = load_profile_any_h5(sym_h5, context)
    omega0, psi0 = make_continuation_initial(context, prev_omega, source_sigma, sigma, omega_sym)
    omega_broken, psi_broken, history, converged = run_newton_from(context, omega0, psi0, max_newton=args.max_newton, tol=args.tol_newton, linesearch=not args.no_linesearch)
    pinned = getattr(context['poisson'], 'gauge_dof', None)
    residuals = residuals_quick(context['mesh'], context['spaces'], omega_broken, psi_broken, mu=context['params'].mu, R=context['params'].R, Rdom=context['params'].Rdom, facet_tags=context['facet_tags'], outer_tag=1, fourier_bc=context['fourier_bc'], sigma=1.0, pinned_psi_dof=pinned)
    profile_path = out_dirs['checkpoints'] / f'{sigma_label(sigma)}_{direction}_profile.xdmf'
    write_xdmf_profile(profile_path, context['mesh'], omega_broken, psi_broken)
    plot_path = out_dirs['plots'] / f'{sigma_label(sigma)}_{direction}_omega_comparison'
    plot_comparison(plot_path, context['mesh'], omega_sym, omega_broken, rf'$\sigma = {sigma:g}$: bifurcation of vorticity profiles')
    final = history[-1]
    row = {'sigma': sigma, 'direction': direction, 'status': 'completed' if converged else 'nonconverged', 'source_sigma': source_sigma, 'newton_iterations': int(final['iteration']), 'final_newton_residual': float(final['residual_norm']), 'final_relative_step': float(final['relative_step']), 'wall_time_sec': time.perf_counter() - t0, 'symmetric_profile_h5': str(sym_h5.relative_to(REPO_ROOT)), 'broken_profile_checkpoint': str(profile_path.relative_to(REPO_ROOT)), 'plot_path': str(plot_path.with_suffix('.png').relative_to(REPO_ROOT)), 'message': 'scaled previous broken profile; exact boundary reset; psi recomputed by Poisson'}
    row.update(compute_metrics(context, omega_sym, psi_sym, omega_broken, psi_broken, residuals))
    writer.writerow({k: row.get(k, 'n/a') for k in CSV_COLUMNS})
    print(f"[track] sigma={sigma:g} direction={direction} status={row['status']} relOmega={row['omega_relative_rms_separation']:.6e} diffNon3={row['omega_difference_non3_fraction']:.6e}")
    return omega_broken, psi_broken, row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--run-tag', default='branchB_broken_branch_sigma_track_s150_110')
    parser.add_argument('--max-newton', type=int, default=24)
    parser.add_argument('--tol-newton', type=float, default=1e-8)
    parser.add_argument('--no-linesearch', action='store_true')
    parser.add_argument('--upward', nargs='*', type=float, default=[150.0])
    parser.add_argument('--downward', nargs='*', type=float, default=[130.0, 125.0, 120.0, 115.0, 110.0])
    parser.add_argument('--seed-sigma', type=float, default=140.0)
    parser.add_argument('--seed-h5', type=Path, default=BRANCH_DIR / 'checkpoints/branchB_s140_artificial_localized_probe/m1_phase0_eta0p25_profile.h5')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.seed_h5.is_absolute():
        args.seed_h5 = (REPO_ROOT / args.seed_h5).resolve()
    out_dirs = {'data': BRANCH_DIR / 'data', 'logs': BRANCH_DIR / 'logs', 'plots': BRANCH_DIR / 'plots' / args.run_tag, 'checkpoints': BRANCH_DIR / 'checkpoints' / args.run_tag}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    csv_path = out_dirs['data'] / f'{args.run_tag}_results.csv'
    manifest_path = out_dirs['logs'] / f'{args.run_tag}_manifest.json'
    manifest = {'status': 'dry_run', 'run_tag': args.run_tag, 'seed_sigma': args.seed_sigma, 'seed_h5': str(args.seed_h5.relative_to(REPO_ROOT)), 'upward': args.upward, 'downward': args.downward, 'csv': str(csv_path.relative_to(REPO_ROOT))}
    write_json(manifest_path, manifest)
    if not args.execute:
        print(f'[dry-run] wrote {manifest_path}')
        return 0
    t_all = time.perf_counter()
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        seed_omega, seed_psi = solve_or_record_seed(args, args.seed_sigma, 'seed', out_dirs, writer, args.seed_h5)
        prev_omega = seed_omega
        prev_sigma = args.seed_sigma
        for sigma in args.upward:
            prev_omega, _, _ = run_point(args, sigma, 'upward', prev_sigma, prev_omega, out_dirs, writer)
            prev_sigma = sigma
        prev_omega = seed_omega
        prev_sigma = args.seed_sigma
        for sigma in args.downward:
            prev_omega, _, _ = run_point(args, sigma, 'downward', prev_sigma, prev_omega, out_dirs, writer)
            prev_sigma = sigma
    manifest.update({'status': 'completed', 'wall_time_sec': time.perf_counter() - t_all})
    write_json(manifest_path, manifest)
    append_registry_row({'date': today(), 'tag': f'fem-branchB-{args.run_tag}', 'codebase': 'baseline', 'script': str(Path(__file__).relative_to(REPO_ROOT)), 'input_case': 's_cos3theta_broken_branch', 'domain': 'disk_Rdom10', 'grid_or_mesh': 'h=0.06/0.06', 'solver': 'broken_branch_fixed_boundary_newton_continuation', 'parameters': f'seed={args.seed_sigma}; upward={args.upward}; downward={args.downward}; max_newton={args.max_newton}', 'wall_time_sec': f"{manifest['wall_time_sec']:.6f}", 'error_metrics': f'results={csv_path.relative_to(REPO_ROOT)}', 'eigenvalues_summary': 'compare with Branch A eigenvalue crossing table', 'status': 'completed', 'notes': 'Tracked artificial-perturbation broken branch in boundary amplitude sigma.'})
    print(f'[done] results={csv_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
