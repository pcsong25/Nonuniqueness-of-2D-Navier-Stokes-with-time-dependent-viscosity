from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .grid import SquareGrid


@dataclass(frozen=True)
class InteriorOperators:
    laplacian: sp.csr_matrix
    grad_x: sp.csr_matrix
    grad_y: sp.csr_matrix


def _check_field(grid: SquareGrid, values: np.ndarray) -> np.ndarray:
    field = np.asarray(values)
    if field.shape != grid.shape:
        raise ValueError(f"Expected field shape {grid.shape}, got {field.shape}.")
    return field


def _interior_column(grid: SquareGrid, i: int, j: int) -> int | None:
    col = int(grid.interior_id[i, j])
    return col if col >= 0 else None


def interior_laplacian_matrix(grid: SquareGrid) -> sp.csr_matrix:
    """Standard five-point Laplacian on interior unknowns.

    Boundary values are eliminated. Use
    `dirichlet_laplacian_boundary_contribution` to add nonhomogeneous
    Dirichlet boundary contributions.
    """
    h2 = grid.h * grid.h
    A = sp.lil_matrix((grid.n_interior, grid.n_interior), dtype=float)
    for i, j in grid.interior_ij():
        row = int(grid.interior_id[i, j])
        A[row, row] = -4.0 / h2
        for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            col = _interior_column(grid, int(ni), int(nj))
            if col is not None:
                A[row, col] += 1.0 / h2
    return A.tocsr()


def dirichlet_laplacian_boundary_contribution(
    grid: SquareGrid,
    boundary_values: np.ndarray,
) -> np.ndarray:
    field = _check_field(grid, boundary_values)
    h2 = grid.h * grid.h
    out = np.zeros(grid.n_interior, dtype=float)
    for i, j in grid.interior_ij():
        row = int(grid.interior_id[i, j])
        for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if grid.boundary_mask[int(ni), int(nj)]:
                out[row] += float(field[int(ni), int(nj)]) / h2
    return out


def interior_gradient_matrices(grid: SquareGrid) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Centered first-derivative matrices on interior unknowns."""
    scale = 1.0 / (2.0 * grid.h)
    Dx = sp.lil_matrix((grid.n_interior, grid.n_interior), dtype=float)
    Dy = sp.lil_matrix((grid.n_interior, grid.n_interior), dtype=float)
    for i, j in grid.interior_ij():
        row = int(grid.interior_id[i, j])
        left = _interior_column(grid, int(i - 1), int(j))
        right = _interior_column(grid, int(i + 1), int(j))
        down = _interior_column(grid, int(i), int(j - 1))
        up = _interior_column(grid, int(i), int(j + 1))
        if left is not None:
            Dx[row, left] -= scale
        if right is not None:
            Dx[row, right] += scale
        if down is not None:
            Dy[row, down] -= scale
        if up is not None:
            Dy[row, up] += scale
    return Dx.tocsr(), Dy.tocsr()


def dirichlet_gradient_boundary_contribution(
    grid: SquareGrid,
    boundary_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    field = _check_field(grid, boundary_values)
    scale = 1.0 / (2.0 * grid.h)
    bx = np.zeros(grid.n_interior, dtype=float)
    by = np.zeros(grid.n_interior, dtype=float)
    for i, j in grid.interior_ij():
        row = int(grid.interior_id[i, j])
        if grid.boundary_mask[int(i - 1), int(j)]:
            bx[row] -= scale * float(field[int(i - 1), int(j)])
        if grid.boundary_mask[int(i + 1), int(j)]:
            bx[row] += scale * float(field[int(i + 1), int(j)])
        if grid.boundary_mask[int(i), int(j - 1)]:
            by[row] -= scale * float(field[int(i), int(j - 1)])
        if grid.boundary_mask[int(i), int(j + 1)]:
            by[row] += scale * float(field[int(i), int(j + 1)])
    return bx, by


def interior_operators(grid: SquareGrid) -> InteriorOperators:
    Dx, Dy = interior_gradient_matrices(grid)
    return InteriorOperators(
        laplacian=interior_laplacian_matrix(grid),
        grad_x=Dx,
        grad_y=Dy,
    )


def advection_matrix(grid: SquareGrid, vx: np.ndarray, vy: np.ndarray) -> sp.csr_matrix:
    """Centered advection matrix vx*d_x + vy*d_y on interior unknowns."""
    Dx, Dy = interior_gradient_matrices(grid)

    vx_arr = np.asarray(vx)
    vy_arr = np.asarray(vy)
    if vx_arr.shape == grid.shape:
        vx_vec = grid.field_to_interior(vx_arr)
    else:
        vx_vec = vx_arr.reshape(-1)
    if vy_arr.shape == grid.shape:
        vy_vec = grid.field_to_interior(vy_arr)
    else:
        vy_vec = vy_arr.reshape(-1)

    if vx_vec.size != grid.n_interior or vy_vec.size != grid.n_interior:
        raise ValueError("Velocity arrays must be full grid fields or interior vectors.")

    return sp.diags(vx_vec, format="csr") @ Dx + sp.diags(vy_vec, format="csr") @ Dy

