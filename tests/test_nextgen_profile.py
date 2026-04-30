from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nextgen.boundary import FourierBoundaryData
from nextgen.config import PhysicalParams
from nextgen.grid import SquareGrid
from nextgen.profile import (
    ProfileNewtonParams,
    ProfileProblem,
    full_grid_gradient_matrices,
)
from nextgen.polar_profile import PolarGrid, PolarProfileParams, PolarProfileProblem


class ProfileOperatorTests(unittest.TestCase):
    def test_full_grid_gradient_reproduces_linear_field(self) -> None:
        grid = SquareGrid(half_width=1.0, n=7)
        gx, gy = full_grid_gradient_matrices(grid)
        field = 2.0 * grid.X - 3.0 * grid.Y + 0.25
        vec = field.reshape(-1)
        assert_allclose(gx @ vec, np.full(grid.n_interior, 2.0), atol=1e-12)
        assert_allclose(gy @ vec, np.full(grid.n_interior, -3.0), atol=1e-12)

    def test_profile_jacobian_matches_finite_difference(self) -> None:
        grid = SquareGrid(half_width=2.0, n=7)
        physical = PhysicalParams(mu=0.8, R=150.0, Rdom=2.0)
        data = FourierBoundaryData(M=6, coeffs={3: 0.2 + 0.0j, 4: 0.1 + 0.0j})
        problem = ProfileProblem(grid, physical, data)
        rng = np.random.default_rng(1234)

        omega = 0.01 * rng.standard_normal(grid.n_interior)
        psi = 0.01 * rng.standard_normal(grid.n * grid.n)
        delta = 0.01 * rng.standard_normal(problem.unknown_size)
        sigma = 0.4
        eps = 1e-7

        residual = problem.residual(omega, psi, sigma)
        jac_delta = problem.jacobian(omega, psi, sigma) @ delta
        perturbed = problem.residual(
            omega + eps * delta[: grid.n_interior],
            psi + eps * delta[grid.n_interior :],
            sigma,
        )

        assert_allclose((perturbed - residual) / eps, jac_delta, rtol=2e-5, atol=2e-7)


class ProfileSolveTests(unittest.TestCase):
    def test_zero_boundary_profile_stays_zero(self) -> None:
        grid = SquareGrid(half_width=1.0, n=9)
        physical = PhysicalParams(mu=0.8, R=150.0, Rdom=1.0)
        data = FourierBoundaryData(M=4, coeffs={})
        problem = ProfileProblem(grid, physical, data)
        result = problem.solve(
            ProfileNewtonParams(
                sigma_grid=(0.0, 1.0),
                tol_step=1e-11,
                tol_residual=1e-11,
                max_newton=4,
            )
        )

        self.assertTrue(result.converged, result.message)
        assert_allclose(result.omega, np.zeros(grid.shape), atol=1e-12)
        assert_allclose(result.psi, np.zeros(grid.shape), atol=1e-12)
        self.assertLess(result.diagnostics["relative_residual"], 1e-11)

    def test_small_continuation_reduces_residual_and_preserves_boundary(self) -> None:
        grid = SquareGrid(half_width=2.0, n=17)
        physical = PhysicalParams(mu=0.8, R=150.0, Rdom=2.0)
        data = FourierBoundaryData(M=8, coeffs={3: 0.06 + 0.0j, 4: 0.05 + 0.0j})
        problem = ProfileProblem(grid, physical, data)
        result = problem.solve(
            ProfileNewtonParams(
                sigma_grid=(0.0, 0.5, 1.0),
                tol_step=1e-8,
                tol_residual=1e-8,
                max_newton=16,
            )
        )

        self.assertTrue(result.converged, result.message)
        positives = [it.relative_residual for it in result.iterations if it.sigma == 1.0]
        self.assertGreater(len(positives), 0)
        self.assertLess(positives[-1], positives[0])

        expected_boundary, _ = problem.boundary_fields(sigma=1.0)
        assert_allclose(
            result.omega[grid.boundary_mask],
            expected_boundary[grid.boundary_mask],
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertLess(result.diagnostics["relative_residual"], 1e-8)


class PolarProfileTests(unittest.TestCase):
    def test_polar_zero_boundary_profile_stays_zero(self) -> None:
        grid = PolarGrid(radius=2.0, nr=8, ntheta=9, symmetry=3)
        physical = PhysicalParams(mu=0.8, R=150.0, Rdom=2.0)
        data = FourierBoundaryData(M=6, coeffs={})
        problem = PolarProfileProblem(grid, physical, data)
        result = problem.solve(
            PolarProfileParams(
                sigma_grid=(0.0, 1.0),
                tol_step=1e-11,
                tol_residual=1e-11,
                max_iter=4,
            )
        )

        self.assertTrue(result.converged, result.message)
        assert_allclose(result.omega, np.zeros(grid.shape), atol=1e-12)
        assert_allclose(result.psi, np.zeros(grid.shape), atol=1e-12)
        self.assertLess(result.diagnostics["relative_residual"], 1e-11)

    def test_polar_profile_jacobian_matches_finite_difference(self) -> None:
        grid = PolarGrid(radius=2.0, nr=5, ntheta=9, symmetry=3)
        physical = PhysicalParams(mu=0.8, R=150.0, Rdom=2.0)
        data = FourierBoundaryData(M=9, coeffs={3: 0.2 + 0.0j})
        problem = PolarProfileProblem(grid, physical, data)
        rng = np.random.default_rng(2345)

        omega = 0.01 * rng.standard_normal(grid.size)
        psi = 0.01 * rng.standard_normal(grid.size)
        delta = 0.01 * rng.standard_normal(2 * grid.size + 1)
        sigma = 0.4
        eps = 1e-7

        lambda_value = 0.01 * rng.standard_normal()
        residual = problem.residual_vector(omega, psi, sigma, lambda_value)
        jac_delta = problem.jacobian(omega, psi, sigma) @ delta
        perturbed = problem.residual_vector(
            omega + eps * delta[: grid.size],
            psi + eps * delta[grid.size : 2 * grid.size],
            sigma,
            lambda_value + eps * delta[-1],
        )

        assert_allclose((perturbed - residual) / eps, jac_delta, rtol=2e-5, atol=2e-7)

    def test_polar_fv_upwind_jacobian_matches_finite_difference(self) -> None:
        grid = PolarGrid(radius=2.0, nr=5, ntheta=12)
        physical = PhysicalParams(mu=0.8, R=150.0, Rdom=2.0)
        data = FourierBoundaryData(M=12, coeffs={2: 0.2 + 0.03j, 5: -0.08 + 0.02j})
        problem = PolarProfileProblem(grid, physical, data, advection_scheme="fv_upwind")
        rng = np.random.default_rng(3456)

        rr, tt = np.meshgrid(grid.r, grid.theta, indexing="ij")
        omega = 0.05 * rr + 0.01 * np.cos(tt) + 0.002 * rng.standard_normal(grid.shape)
        omega[0, :] = float(np.mean(omega[0, :]))
        psi = 0.2 * rr**2 + 0.001 * np.sin(tt) + 0.0002 * rng.standard_normal(grid.shape)
        delta = 0.01 * rng.standard_normal(2 * grid.size + 1)
        sigma = 0.4
        eps = 1e-7

        lambda_value = 0.01 * rng.standard_normal()
        residual = problem.residual_vector(omega, psi, sigma, lambda_value)
        jac_delta = problem.jacobian(omega, psi, sigma) @ delta
        perturbed = problem.residual_vector(
            omega.reshape(-1) + eps * delta[: grid.size],
            psi.reshape(-1) + eps * delta[grid.size : 2 * grid.size],
            sigma,
            lambda_value + eps * delta[-1],
        )

        assert_allclose((perturbed - residual) / eps, jac_delta, rtol=3e-5, atol=5e-7)

    def test_polar_fv_blended_jacobian_matches_finite_difference(self) -> None:
        grid = PolarGrid(radius=2.0, nr=5, ntheta=12)
        physical = PhysicalParams(mu=0.8, R=150.0, Rdom=2.0)
        data = FourierBoundaryData(M=12, coeffs={2: 0.2 + 0.03j, 5: -0.08 + 0.02j})
        rng = np.random.default_rng(4567)

        rr, tt = np.meshgrid(grid.r, grid.theta, indexing="ij")
        omega = 0.05 * rr + 0.01 * np.cos(tt) + 0.002 * rng.standard_normal(grid.shape)
        omega[0, :] = float(np.mean(omega[0, :]))
        psi = 0.2 * rr**2 + 0.001 * np.sin(tt) + 0.0002 * rng.standard_normal(grid.shape)
        delta = 0.01 * rng.standard_normal(2 * grid.size + 1)
        sigma = 0.4
        eps = 1e-7

        for scheme, blend in (("fv_centered", 0.0), ("fv_blended", 0.2)):
            with self.subTest(scheme=scheme):
                problem = PolarProfileProblem(
                    grid,
                    physical,
                    data,
                    advection_scheme=scheme,
                    upwind_blend=blend,
                )
                lambda_value = 0.01 * rng.standard_normal()
                residual = problem.residual_vector(omega, psi, sigma, lambda_value)
                jac_delta = problem.jacobian(omega, psi, sigma) @ delta
                perturbed = problem.residual_vector(
                    omega.reshape(-1) + eps * delta[: grid.size],
                    psi.reshape(-1) + eps * delta[grid.size : 2 * grid.size],
                    sigma,
                    lambda_value + eps * delta[-1],
                )

                assert_allclose((perturbed - residual) / eps, jac_delta, rtol=3e-5, atol=5e-7)

    def test_polar_full_disk_c3_data_preserves_threefold_symmetry(self) -> None:
        grid = PolarGrid(radius=2.0, nr=10, ntheta=36)
        physical = PhysicalParams(mu=0.8, R=150.0, Rdom=2.0)
        data = FourierBoundaryData(M=9, coeffs={3: 0.4 + 0.0j})
        problem = PolarProfileProblem(grid, physical, data, advection_scheme="blended", upwind_blend=0.01)
        result = problem.solve(
            PolarProfileParams(
                sigma_grid=(0.0, 0.5, 1.0),
                tol_step=1e-9,
                tol_residual=1e-9,
                max_iter=16,
            )
        )

        self.assertTrue(result.converged, result.message)
        radii = np.linspace(0.3, 1.6, 8)
        theta = np.linspace(-np.pi, np.pi, 31, endpoint=False)
        rr, tt = np.meshgrid(radii, theta, indexing="ij")
        x = rr * np.cos(tt)
        y = rr * np.sin(tt)
        angle = 2.0 * np.pi / 3.0
        xr = x * np.cos(angle) - y * np.sin(angle)
        yr = x * np.sin(angle) + y * np.cos(angle)
        omega, psi = problem.sample_cartesian(result, x, y)
        omega_rot, psi_rot = problem.sample_cartesian(result, xr, yr)

        assert_allclose(omega, omega_rot, rtol=1e-10, atol=1e-10)
        assert_allclose(psi, psi_rot, rtol=1e-10, atol=1e-10)

    def test_polar_full_disk_handles_mixed_boundary_modes(self) -> None:
        grid = PolarGrid(radius=2.0, nr=10, ntheta=40)
        physical = PhysicalParams(mu=0.8, R=150.0, Rdom=2.0)
        data = FourierBoundaryData(
            M=10,
            coeffs={
                2: 0.18 + 0.04j,
                3: 0.12 + 0.0j,
                5: -0.08 + 0.03j,
            },
        )
        problem = PolarProfileProblem(grid, physical, data)
        result = problem.solve(
            PolarProfileParams(
                sigma_grid=(0.0, 0.5, 1.0),
                tol_step=1e-8,
                tol_residual=1e-8,
                max_iter=20,
            )
        )

        self.assertTrue(result.converged, result.message)
        self.assertLess(result.diagnostics["relative_residual"], 1e-8)

    def test_polar_fv_upwind_handles_small_mixed_boundary_modes(self) -> None:
        grid = PolarGrid(radius=2.0, nr=8, ntheta=32)
        physical = PhysicalParams(mu=0.8, R=150.0, Rdom=2.0)
        data = FourierBoundaryData(
            M=10,
            coeffs={
                2: 0.18 + 0.04j,
                5: -0.08 + 0.03j,
            },
        )
        problem = PolarProfileProblem(grid, physical, data, advection_scheme="fv_upwind")
        result = problem.solve(
            PolarProfileParams(
                sigma_grid=(0.0, 0.5, 1.0),
                tol_step=1e-8,
                tol_residual=1e-8,
                tol_residual_only=1e-8,
                max_iter=24,
                max_backtracks=16,
            )
        )

        self.assertTrue(result.converged, result.message)
        self.assertEqual(result.diagnostics["advection_scheme"], "fv_upwind")
        self.assertLess(result.diagnostics["relative_residual"], 1e-8)

    def test_polar_fv_upwind_reduced_poisson_newton_handles_small_mixed_boundary_modes(self) -> None:
        grid = PolarGrid(radius=2.0, nr=8, ntheta=32)
        physical = PhysicalParams(mu=0.8, R=150.0, Rdom=2.0)
        data = FourierBoundaryData(
            M=10,
            coeffs={
                2: 0.18 + 0.04j,
                5: -0.08 + 0.03j,
            },
        )
        problem = PolarProfileProblem(grid, physical, data, advection_scheme="fv_upwind")
        result = problem.solve(
            PolarProfileParams(
                sigma_grid=(0.0, 0.5, 1.0),
                tol_step=1e-8,
                tol_residual=1e-8,
                tol_residual_only=1e-8,
                max_iter=24,
                max_backtracks=16,
                reduced_poisson_newton=True,
            )
        )

        self.assertTrue(result.converged, result.message)
        self.assertTrue(result.diagnostics["reduced_poisson_newton"])
        self.assertLess(result.diagnostics["relative_residual"], 1e-8)

    def test_polar_blended_advection_handles_moderate_mixed_modes(self) -> None:
        grid = PolarGrid(radius=4.0, nr=12, ntheta=48)
        physical = PhysicalParams(mu=0.8, R=150.0, Rdom=4.0)
        data = FourierBoundaryData(
            M=10,
            coeffs={
                2: 6.0 + 3.0j,
                5: -5.0 + 1.0j,
                8: 2.0 - 1.5j,
            },
        )
        problem = PolarProfileProblem(grid, physical, data, advection_scheme="blended", upwind_blend=0.01)
        result = problem.solve(
            PolarProfileParams(
                sigma_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
                tol_step=1e-8,
                tol_residual=1e-8,
                tol_residual_only=1e-8,
                max_iter=32,
                max_backtracks=16,
            )
        )

        self.assertTrue(result.converged, result.message)
        self.assertEqual(result.diagnostics["advection_scheme"], "blended")
        self.assertLess(result.diagnostics["relative_residual"], 1e-8)


if __name__ == "__main__":
    unittest.main()
