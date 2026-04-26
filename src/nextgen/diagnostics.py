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

