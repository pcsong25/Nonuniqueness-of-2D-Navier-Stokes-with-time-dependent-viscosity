from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nextgen.boundary import FourierBoundaryData
from nextgen.diagnostics import boundary_mismatch, operator_residual_norm
from nextgen.grid import SquareGrid
from nextgen.operators import (
    dirichlet_gradient_boundary_contribution,
    dirichlet_laplacian_boundary_contribution,
    interior_gradient_matrices,
    interior_laplacian_matrix,
)
from nextgen.poisson import poisson_system_residual, solve_neumann_poisson


class FourierBoundaryTests(unittest.TestCase):
    def test_circle_neumann_matches_normal_projection(self) -> None:
        data = FourierBoundaryData(M=8, coeffs={4: 50.0 + 0.0j})
        mu = 0.8
        R = 150.0
        Rdom = 10.0
        theta = np.linspace(-np.pi, np.pi, 21)
        x = Rdom * np.cos(theta)
        y = Rdom * np.sin(theta)
        nx = np.cos(theta)
        ny = np.sin(theta)

        direct = data.psi_neumann_circle(theta, mu=mu, R=R, Rdom=Rdom)
        projected = data.psi_neumann_normal(x, y, nx, ny, mu=mu, R=R, Rdom=Rdom)
        assert_allclose(projected, direct, rtol=1e-12, atol=1e-12)

    def test_circle_neumann_matches_baseline_fourier_formula(self) -> None:
        coeffs = {2: 3.0 - 1.0j, 4: 50.0 + 0.0j}
        data = FourierBoundaryData(M=8, coeffs=coeffs)
        mu = 0.8
        R = 150.0
        Rdom = 10.0
        sigma = 0.7
        theta = np.linspace(-np.pi, np.pi, 37)
        alpha = 2.0 - 1.0 / mu

        baseline = np.zeros(theta.shape, dtype=complex)
        for mode, coeff in data.coeffs_full.items():
            denom = alpha * alpha - float(mode * mode)
            baseline += (
                alpha
                * Rdom
                / denom
                * R ** (-1.0 / mu)
                * coeff
                * np.exp(1j * mode * theta)
            )
        baseline = sigma * np.real(baseline)

        direct = data.psi_neumann_circle(theta, mu=mu, R=R, Rdom=Rdom, sigma=sigma)
        assert_allclose(direct, baseline, rtol=1e-12, atol=1e-12)

    def test_scaled_omega_dirichlet_matches_baseline_on_circle(self) -> None:
        data = FourierBoundaryData(M=8, coeffs={3: 60.0 + 0.0j})
        mu = 0.8
        R = 150.0
        Rdom = 10.0
        theta = np.linspace(-np.pi, np.pi, 29)
        x = Rdom * np.cos(theta)
        y = Rdom * np.sin(theta)

        legacy = data.omega_dirichlet(x, y, mu=mu, R=R)
        scaled = data.omega_dirichlet(x, y, mu=mu, R=R, Rdom=Rdom)
        assert_allclose(scaled, legacy, rtol=1e-12, atol=1e-12)

    def test_scaled_omega_dirichlet_decays_off_circle(self) -> None:
        data = FourierBoundaryData(M=8, coeffs={3: 60.0 + 0.0j})
        mu = 0.8
        R = 150.0
        Rdom = 10.0
        theta = np.array([0.2, 1.1, 2.0])
        radius = 2.0 * Rdom
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        legacy = data.omega_dirichlet(x, y, mu=mu, R=R)
        scaled = data.omega_dirichlet(x, y, mu=mu, R=R, Rdom=Rdom)
        assert_allclose(scaled, legacy * 2.0 ** (-1.0 / mu), rtol=1e-12, atol=1e-12)

    def test_real_mode_mirroring(self) -> None:
        data = FourierBoundaryData(M=4, coeffs={2: 3.0 + 2.0j})
        theta = np.array([0.1, 1.2, 2.4])
        values = data.omega_bar(theta)
        expected = 2.0 * np.real((3.0 + 2.0j) * np.exp(2j * theta))
        assert_allclose(values, expected)


class GridAndOperatorTests(unittest.TestCase):
    def test_grid_indexing_and_normals(self) -> None:
        grid = SquareGrid(half_width=2.0, n=5)
        self.assertEqual(grid.n_interior, 9)
        self.assertEqual(grid.n_boundary, 16)
        nx, ny = grid.boundary_normals()
        assert_allclose(nx[-1, 2], 1.0)
        assert_allclose(ny[2, 0], -1.0)
        assert_allclose(np.hypot(nx[0, 0], ny[0, 0]), 1.0)

    def test_laplacian_reproduces_quadratic_with_boundary_contribution(self) -> None:
        grid = SquareGrid(half_width=1.0, n=7)
        field = grid.X * grid.X + grid.Y * grid.Y
        interior = grid.field_to_interior(field)
        L = interior_laplacian_matrix(grid)
        bdy = dirichlet_laplacian_boundary_contribution(grid, field)
        assert_allclose(L @ interior + bdy, np.full(grid.n_interior, 4.0), atol=1e-12)

    def test_gradient_reproduces_linear_with_boundary_contribution(self) -> None:
        grid = SquareGrid(half_width=1.0, n=7)
        field = 2.0 * grid.X - 3.0 * grid.Y
        interior = grid.field_to_interior(field)
        Dx, Dy = interior_gradient_matrices(grid)
        bx, by = dirichlet_gradient_boundary_contribution(grid, field)
        assert_allclose(Dx @ interior + bx, np.full(grid.n_interior, 2.0), atol=1e-12)
        assert_allclose(Dy @ interior + by, np.full(grid.n_interior, -3.0), atol=1e-12)


class DiagnosticsTests(unittest.TestCase):
    def test_operator_residual_norm_zero_for_exact_system(self) -> None:
        grid = SquareGrid(half_width=1.0, n=5)
        A = interior_laplacian_matrix(grid)
        x = np.ones(grid.n_interior)
        b = A @ x
        self.assertEqual(operator_residual_norm(A, x, b), 0.0)

    def test_boundary_mismatch_reports_boundary_only(self) -> None:
        grid = SquareGrid(half_width=1.0, n=5)
        values = np.zeros(grid.shape)
        expected = np.zeros(grid.shape)
        values[2, 2] = 100.0
        self.assertEqual(boundary_mismatch(values, expected, grid)["max_abs"], 0.0)
        values[0, 2] = 3.0
        self.assertEqual(boundary_mismatch(values, expected, grid)["max_abs"], 3.0)


class PoissonTests(unittest.TestCase):
    def test_neumann_poisson_zero_solution_with_gauge(self) -> None:
        grid = SquareGrid(half_width=1.0, n=7)
        omega = np.zeros(grid.shape)
        neumann = np.zeros(grid.shape)
        psi = solve_neumann_poisson(grid, omega, neumann)
        assert_allclose(psi, np.zeros(grid.shape), atol=1e-12)
        self.assertLess(poisson_system_residual(grid, psi, omega, neumann), 1e-12)

    def test_neumann_poisson_reproduces_linear_solution_and_corner_rows(self) -> None:
        grid = SquareGrid(half_width=1.0, n=9)
        psi_exact = 2.0 * grid.X - 3.0 * grid.Y + 0.5
        omega = np.zeros(grid.shape)
        nx, ny = grid.boundary_normals()
        neumann = 2.0 * nx - 3.0 * ny
        gauge = (grid.n // 2, grid.n // 2)

        psi = solve_neumann_poisson(
            grid,
            omega,
            neumann,
            gauge_ij=gauge,
            gauge_value=float(psi_exact[gauge]),
        )

        assert_allclose(psi, psi_exact, rtol=1e-12, atol=1e-12)
        self.assertLess(
            poisson_system_residual(
                grid,
                psi,
                omega,
                neumann,
                gauge_ij=gauge,
                gauge_value=float(psi_exact[gauge]),
            ),
            1e-12,
        )


if __name__ == "__main__":
    unittest.main()
