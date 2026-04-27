from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .grid import SquareGrid


def operator_residual_norm(
    A: sp.spmatrix,
    x: np.ndarray,
    b: np.ndarray,
    *,
    relative: bool = True,
) -> float:
    residual = A @ np.asarray(x) - np.asarray(b)
    norm = float(np.linalg.norm(residual))
    if not relative:
        return norm
    scale = max(float(np.linalg.norm(b)), 1e-30)
    return norm / scale


def grid_l2_norm(values: np.ndarray, grid: SquareGrid, *, interior_only: bool = False) -> float:
    arr = np.asarray(values)
    if arr.shape == grid.shape:
        vals = arr[grid.interior_mask] if interior_only else arr.reshape(-1)
    else:
        vals = arr.reshape(-1)
    return float(np.sqrt(np.sum(np.abs(vals) ** 2) * grid.h * grid.h))


def grid_linf_norm(values: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(values))))


def domain_integral(values: np.ndarray, grid: SquareGrid) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.shape != grid.shape:
        raise ValueError(f"Expected field shape {grid.shape}, got {arr.shape}.")
    inner_y = np.trapz(arr, x=grid.y, axis=1)
    return float(np.trapz(inner_y, x=grid.x))


def boundary_integral(values: np.ndarray, grid: SquareGrid) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.shape != grid.shape:
        raise ValueError(f"Expected field shape {grid.shape}, got {arr.shape}.")
    bottom = np.trapz(arr[:, 0], x=grid.x)
    top = np.trapz(arr[:, -1], x=grid.x)
    left = np.trapz(arr[0, :], x=grid.y)
    right = np.trapz(arr[-1, :], x=grid.y)
    return float(bottom + top + left + right)


def boundary_mismatch(values: np.ndarray, expected: np.ndarray, grid: SquareGrid) -> dict[str, float]:
    vals = np.asarray(values)
    exp = np.asarray(expected)
    if vals.shape != grid.shape or exp.shape != grid.shape:
        raise ValueError("values and expected must both be full grid fields.")
    diff = np.zeros(grid.shape, dtype=float)
    diff[grid.boundary_mask] = np.asarray(vals - exp, dtype=float)[grid.boundary_mask]
    return {
        "max_abs": grid_linf_norm(diff[grid.boundary_mask]),
        "l2_boundary": float(np.sqrt(boundary_integral(diff * diff, grid))),
    }


def poisson_compatibility(omega: np.ndarray, neumann: np.ndarray, grid: SquareGrid) -> float:
    """Return integral(omega) - integral_boundary(d_n psi)."""
    return domain_integral(omega, grid) - boundary_integral(neumann, grid)


def profile_residual_metrics(
    omega_residual: np.ndarray,
    poisson_residual: np.ndarray,
    *,
    scale: float = 1.0,
) -> dict[str, float]:
    omega = np.asarray(omega_residual)
    poisson = np.asarray(poisson_residual)
    total = float(np.sqrt(np.linalg.norm(omega) ** 2 + np.linalg.norm(poisson) ** 2))
    return {
        "omega_residual_l2": float(np.linalg.norm(omega)),
        "poisson_residual_l2": float(np.linalg.norm(poisson)),
        "profile_residual_l2": total,
        "profile_residual_relative": total / max(float(scale), 1e-30),
    }


def sampled_strong_profile_residual(
    omega: np.ndarray,
    psi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    mu: float,
    eps: float,
) -> dict[str, float]:
    """Strong-form residual on a common uniform sampling grid.

    This is intended for cross-code comparisons. It avoids comparing a finite
    element weak residual against a finite-difference algebraic residual by
    evaluating both fields on the same uniform grid and using the same centered
    finite-difference stencil away from that grid's boundary.
    """
    omega_arr = np.asarray(omega, dtype=float)
    psi_arr = np.asarray(psi, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if omega_arr.shape != psi_arr.shape:
        raise ValueError("omega and psi must have the same sampled-grid shape.")
    if omega_arr.shape != (x_arr.size, y_arr.size):
        raise ValueError("sampled fields must have shape (len(x), len(y)).")
    if x_arr.size < 3 or y_arr.size < 3:
        raise ValueError("sampled grid must have at least three points per direction.")
    hx = float(x_arr[1] - x_arr[0])
    hy = float(y_arr[1] - y_arr[0])
    if not np.allclose(np.diff(x_arr), hx) or not np.allclose(np.diff(y_arr), hy):
        raise ValueError("sampled grid must be uniform.")

    omega_c = omega_arr[1:-1, 1:-1]
    psi_c = psi_arr[1:-1, 1:-1]
    omega_x = (omega_arr[2:, 1:-1] - omega_arr[:-2, 1:-1]) / (2.0 * hx)
    omega_y = (omega_arr[1:-1, 2:] - omega_arr[1:-1, :-2]) / (2.0 * hy)
    psi_x = (psi_arr[2:, 1:-1] - psi_arr[:-2, 1:-1]) / (2.0 * hx)
    psi_y = (psi_arr[1:-1, 2:] - psi_arr[1:-1, :-2]) / (2.0 * hy)
    lap_omega = (
        (omega_arr[2:, 1:-1] - 2.0 * omega_c + omega_arr[:-2, 1:-1]) / (hx * hx)
        + (omega_arr[1:-1, 2:] - 2.0 * omega_c + omega_arr[1:-1, :-2]) / (hy * hy)
    )
    lap_psi = (
        (psi_arr[2:, 1:-1] - 2.0 * psi_c + psi_arr[:-2, 1:-1]) / (hx * hx)
        + (psi_arr[1:-1, 2:] - 2.0 * psi_c + psi_arr[1:-1, :-2]) / (hy * hy)
    )

    X, Y = np.meshgrid(x_arr[1:-1], y_arr[1:-1], indexing="ij")
    vx = -psi_y - float(mu) * X
    vy = psi_x - float(mu) * Y
    omega_residual = vx * omega_x + vy * omega_y - omega_c - float(eps) * lap_omega
    poisson_residual = lap_psi - omega_c

    cell_area = abs(hx * hy)

    def grid_l2(values: np.ndarray) -> float:
        return float(np.sqrt(np.sum(np.asarray(values) ** 2) * cell_area))

    omega_norm = grid_l2(omega_c)
    psi_lap_norm = grid_l2(lap_psi)
    omega_residual_norm = grid_l2(omega_residual)
    poisson_residual_norm = grid_l2(poisson_residual)
    total = float(np.sqrt(omega_residual_norm**2 + poisson_residual_norm**2))
    scale = max(omega_norm, psi_lap_norm, 1e-30)
    return {
        "omega_strong_l2": omega_residual_norm,
        "poisson_strong_l2": poisson_residual_norm,
        "profile_strong_l2": total,
        "profile_strong_relative": total / scale,
        "omega_field_l2": omega_norm,
        "lap_psi_l2": psi_lap_norm,
        "sample_hx": abs(hx),
        "sample_hy": abs(hy),
        "sample_points": int(omega_c.size),
    }
