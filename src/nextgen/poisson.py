from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .grid import SquareGrid


@dataclass(frozen=True)
class NeumannPoissonSystem:
    """Sparse strong-form system for Delta psi = omega with Neumann data."""

    matrix: sp.csr_matrix
    rhs: np.ndarray
    gauge_ij: tuple[int, int]
    gauge_value: float


def default_gauge_ij(grid: SquareGrid) -> tuple[int, int]:
    """Use the center grid point as the default gauge when available."""
    return (grid.n // 2, grid.n // 2)


def _check_field(grid: SquareGrid, values: np.ndarray, name: str) -> np.ndarray:
    field = np.asarray(values, dtype=float)
    if field.shape != grid.shape:
        raise ValueError(f"{name} must have shape {grid.shape}, got {field.shape}.")
    return field


def _is_corner(grid: SquareGrid, i: int, j: int) -> bool:
    return (i in (0, grid.n - 1)) and (j in (0, grid.n - 1))


def _boundary_inward_neighbor(grid: SquareGrid, i: int, j: int) -> tuple[int, int, float]:
    """Return adjacent inward node and distance for the boundary derivative row."""
    if _is_corner(grid, i, j):
        ni = 1 if i == 0 else grid.n - 2
        nj = 1 if j == 0 else grid.n - 2
        return ni, nj, np.sqrt(2.0) * grid.h
    if i == 0:
        return 1, j, grid.h
    if i == grid.n - 1:
        return grid.n - 2, j, grid.h
    if j == 0:
        return i, 1, grid.h
    if j == grid.n - 1:
        return i, grid.n - 2, grid.h
    raise ValueError("Point is not on the boundary.")


def assemble_neumann_poisson(
    grid: SquareGrid,
    omega: np.ndarray,
    neumann: np.ndarray,
    *,
    gauge_ij: tuple[int, int] | None = None,
    gauge_value: float = 0.0,
) -> NeumannPoissonSystem:
    """Assemble a sparse finite-difference Neumann Poisson system.

    Unknowns live on all grid points. Interior rows impose the five-point
    equation Delta psi = omega. Boundary rows impose

        psi_boundary - psi_inward = distance_to_inward * d_n psi.

    At corners, `psi_inward` is the diagonal inward neighbor and the distance is
    `sqrt(2) * h`, matching the grid's normalized diagonal corner normal.

    A single row is replaced by `psi[gauge_ij] = gauge_value` to remove the
    Neumann nullspace. The default gauge point is the grid center.
    """
    omega_field = _check_field(grid, omega, "omega")
    neumann_field = _check_field(grid, neumann, "neumann")
    gauge = default_gauge_ij(grid) if gauge_ij is None else (int(gauge_ij[0]), int(gauge_ij[1]))
    if not (0 <= gauge[0] < grid.n and 0 <= gauge[1] < grid.n):
        raise ValueError("gauge_ij is outside the grid.")

    size = grid.n * grid.n
    A = sp.lil_matrix((size, size), dtype=float)
    b = np.zeros(size, dtype=float)
    h2 = grid.h * grid.h

    for i in range(grid.n):
        for j in range(grid.n):
            row = grid.flat_index(i, j)
            if grid.interior_mask[i, j]:
                A[row, grid.flat_index(i, j)] = -4.0 / h2
                A[row, grid.flat_index(i - 1, j)] = 1.0 / h2
                A[row, grid.flat_index(i + 1, j)] = 1.0 / h2
                A[row, grid.flat_index(i, j - 1)] = 1.0 / h2
                A[row, grid.flat_index(i, j + 1)] = 1.0 / h2
                b[row] = omega_field[i, j]
            else:
                ni, nj, distance = _boundary_inward_neighbor(grid, i, j)
                A[row, grid.flat_index(i, j)] = 1.0
                A[row, grid.flat_index(ni, nj)] = -1.0
                b[row] = distance * neumann_field[i, j]

    gauge_row = grid.flat_index(*gauge)
    A.rows[gauge_row] = [gauge_row]
    A.data[gauge_row] = [1.0]
    b[gauge_row] = float(gauge_value)

    return NeumannPoissonSystem(
        matrix=A.tocsr(),
        rhs=b,
        gauge_ij=gauge,
        gauge_value=float(gauge_value),
    )


def solve_neumann_poisson(
    grid: SquareGrid,
    omega: np.ndarray,
    neumann: np.ndarray,
    *,
    gauge_ij: tuple[int, int] | None = None,
    gauge_value: float = 0.0,
) -> np.ndarray:
    """Solve the gauged Neumann Poisson system and return a full grid field."""
    system = assemble_neumann_poisson(
        grid,
        omega,
        neumann,
        gauge_ij=gauge_ij,
        gauge_value=gauge_value,
    )
    solution = spla.spsolve(system.matrix, system.rhs)
    return np.asarray(solution).reshape(grid.shape)


def poisson_system_residual(
    grid: SquareGrid,
    psi: np.ndarray,
    omega: np.ndarray,
    neumann: np.ndarray,
    *,
    gauge_ij: tuple[int, int] | None = None,
    gauge_value: float = 0.0,
) -> float:
    """Return the relative residual of the assembled gauged system."""
    system = assemble_neumann_poisson(
        grid,
        omega,
        neumann,
        gauge_ij=gauge_ij,
        gauge_value=gauge_value,
    )
    vec = np.asarray(psi, dtype=float).reshape(-1)
    residual = system.matrix @ vec - system.rhs
    scale = max(float(np.linalg.norm(system.rhs)), 1e-30)
    return float(np.linalg.norm(residual) / scale)
