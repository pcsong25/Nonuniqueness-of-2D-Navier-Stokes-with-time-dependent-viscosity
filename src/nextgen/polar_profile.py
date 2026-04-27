from __future__ import annotations

from dataclasses import dataclass, field
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator

from .boundary import FourierBoundaryData
from .config import PhysicalParams
from .profile import ProfileIteration


@dataclass(frozen=True)
class PolarGrid:
    """Cell-centered polar grid on a disk.

    The `symmetry` option is only for diagnostic reduced-sector experiments.
    Production M3 validation uses the default full disk (`symmetry=1`) so the
    mesh is not customized to any boundary-data symmetry.
    """

    radius: float
    nr: int
    ntheta: int
    symmetry: int = 1

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("radius must be positive.")
        if self.nr < 4:
            raise ValueError("nr must be at least 4.")
        if self.ntheta < 8:
            raise ValueError("ntheta must be at least 8.")
        if self.symmetry < 1:
            raise ValueError("symmetry must be at least 1.")

    @property
    def theta_period(self) -> float:
        return 2.0 * np.pi / float(self.symmetry)

    @property
    def dr(self) -> float:
        return float(self.radius) / float(self.nr)

    @property
    def dtheta(self) -> float:
        return self.theta_period / float(self.ntheta)

    @property
    def r(self) -> np.ndarray:
        return (np.arange(self.nr, dtype=float) + 0.5) * self.dr

    @property
    def theta(self) -> np.ndarray:
        return np.arange(self.ntheta, dtype=float) * self.dtheta

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nr, self.ntheta)

    @property
    def size(self) -> int:
        return self.nr * self.ntheta

    def index(self, j: int, k: int) -> int:
        return int(j) * self.ntheta + (int(k) % self.ntheta)

    def area_weights(self) -> np.ndarray:
        weights = self.quadrature_weights()
        return weights / float(np.sum(weights))

    def quadrature_weights(self) -> np.ndarray:
        # Scale sector weights to the full disk so absolute L2 diagnostics are
        # comparable across symmetry-reduced and full-disk runs.
        return np.repeat(
            float(self.symmetry) * self.r * self.dr * self.dtheta,
            self.ntheta,
        )


@dataclass(frozen=True)
class PolarProfileParams:
    sigma_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    tol_step: float = 1e-9
    tol_residual: float = 1e-9
    max_iter: int = 32
    line_search_shrink: float = 0.5
    max_backtracks: int = 8
    initial_damping: float = 1.0


@dataclass(frozen=True)
class PolarProfileResult:
    omega: np.ndarray
    psi: np.ndarray
    grid: PolarGrid
    converged: bool
    iterations: tuple[ProfileIteration, ...]
    runtime_sec: float
    message: str
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def newton_steps(self) -> int:
        return len(self.iterations)


def _periodic_first_matrix(ntheta: int, dtheta: float) -> sp.csr_matrix:
    mat = sp.lil_matrix((ntheta, ntheta), dtype=float)
    scale = 1.0 / (2.0 * dtheta)
    for k in range(ntheta):
        mat[k, (k - 1) % ntheta] = -scale
        mat[k, (k + 1) % ntheta] = scale
    return mat.tocsr()


def _periodic_second_matrix(ntheta: int, dtheta: float) -> sp.csr_matrix:
    mat = sp.lil_matrix((ntheta, ntheta), dtype=float)
    scale = 1.0 / (dtheta * dtheta)
    for k in range(ntheta):
        mat[k, k] = -2.0 * scale
        mat[k, (k - 1) % ntheta] = scale
        mat[k, (k + 1) % ntheta] = scale
    return mat.tocsr()


class PolarProfileProblem:
    """Newton profile solve on a circular polar finite-difference grid."""

    def __init__(
        self,
        grid: PolarGrid,
        physical: PhysicalParams,
        boundary_data: FourierBoundaryData,
    ):
        if not np.isclose(grid.radius, physical.Rdom):
            raise ValueError("PolarGrid.radius must match PhysicalParams.Rdom for the M3 disk solve.")
        self.grid = grid
        self.physical = physical
        self.boundary_data = boundary_data
        self.dtheta = _periodic_first_matrix(grid.ntheta, grid.dtheta)
        self.dtheta2 = _periodic_second_matrix(grid.ntheta, grid.dtheta)
        self.dtheta_full = sp.kron(sp.eye(grid.nr, format="csr"), self.dtheta, format="csr")
        self.dtheta2_full = sp.kron(sp.eye(grid.nr, format="csr"), self.dtheta2, format="csr")
        self.radial_derivative = self._radial_derivative_matrix(boundary="dirichlet")
        self.psi_radial_derivative_matrix = self._radial_derivative_matrix(boundary="interior")
        self.laplacian_dirichlet = self._laplacian_matrix(boundary="dirichlet")
        self.laplacian_neumann = self._laplacian_matrix(boundary="neumann")
        self.mean_weights = self.grid.area_weights()
        self.poisson_matrix = self._bordered_poisson_matrix()
        self.poisson_omega_jacobian = self._poisson_omega_jacobian()

    def omega_boundary(self, sigma: float) -> np.ndarray:
        theta = self.grid.theta
        return self.boundary_data.omega_dirichlet(
            self.physical.Rdom * np.cos(theta),
            self.physical.Rdom * np.sin(theta),
            mu=self.physical.mu,
            R=self.physical.R,
            Rdom=self.physical.Rdom,
            sigma=float(sigma),
        )

    def psi_neumann_boundary(self, sigma: float) -> np.ndarray:
        return self.boundary_data.psi_neumann_circle(
            self.grid.theta,
            mu=self.physical.mu,
            R=self.physical.R,
            Rdom=self.physical.Rdom,
            sigma=float(sigma),
        )

    def solve(self, params: PolarProfileParams | None = None) -> PolarProfileResult:
        controls = PolarProfileParams() if params is None else params
        t0 = time.perf_counter()
        omega = np.zeros(self.grid.size, dtype=float)
        psi = np.zeros(self.grid.size, dtype=float)
        poisson_lambda = 0.0
        iterations: list[ProfileIteration] = []
        sigmas = [float(sigma) for sigma in controls.sigma_grid if float(sigma) > 0.0]
        if not sigmas:
            sigmas = [0.0]
        sigma_prev: float | None = None
        current_sigma = sigmas[0]
        converged = True
        message = "converged"

        for sigma in sigmas:
            current_sigma = sigma
            if sigma_prev not in (None, 0.0):
                omega = omega * (sigma / sigma_prev)
            psi, poisson_lambda = self.solve_poisson(omega, sigma)
            sigma_converged = False
            last_relative_step = 0.0
            for iteration in range(controls.max_iter + 1):
                summary = self.residual_summary(omega, psi, sigma, poisson_lambda)
                if iteration == 0:
                    iterations.append(
                        ProfileIteration(
                            sigma=sigma,
                            iteration=iteration,
                            residual_norm=summary["residual_norm"],
                            relative_residual=summary["relative_residual"],
                            omega_residual_norm=summary["omega_residual_norm"],
                            poisson_residual_norm=summary["poisson_residual_norm"],
                            relative_omega_step=0.0,
                            alpha=0.0,
                            linear_residual_norm=0.0,
                            accepted=True,
                        )
                    )
                if (
                    summary["relative_residual"] <= controls.tol_residual
                    and last_relative_step <= controls.tol_step
                ):
                    sigma_converged = True
                    break
                if iteration == controls.max_iter:
                    break

                jac = self.jacobian(omega, psi, sigma)
                residual = self.residual_vector(omega, psi, sigma, poisson_lambda)
                delta = np.asarray(spla.spsolve(jac, -residual), dtype=float)
                linear_residual = float(
                    np.linalg.norm(jac @ delta + residual)
                    / max(summary["residual_norm"], 1e-30)
                )
                delta_omega = delta[: self.grid.size]
                delta_psi = delta[self.grid.size : 2 * self.grid.size]
                delta_lambda = float(delta[-1])

                accepted = False
                best: tuple[float, np.ndarray, np.ndarray, dict[str, float]] | None = None
                alpha = float(controls.initial_damping)
                for _ in range(controls.max_backtracks + 1):
                    trial_omega = omega + alpha * delta_omega
                    trial_psi = psi + alpha * delta_psi
                    trial_lambda = poisson_lambda + alpha * delta_lambda
                    trial_summary = self.residual_summary(
                        trial_omega,
                        trial_psi,
                        sigma,
                        trial_lambda,
                    )
                    if best is None or trial_summary["residual_norm"] < best[3]["residual_norm"]:
                        best = (alpha, trial_omega, trial_psi, trial_summary)
                    if (
                        trial_summary["relative_residual"] <= controls.tol_residual
                        or trial_summary["residual_norm"] < summary["residual_norm"]
                    ):
                        accepted = True
                        best = (alpha, trial_omega, trial_psi, trial_summary)
                        break
                    alpha *= controls.line_search_shrink

                assert best is not None
                alpha_used, trial_omega, trial_psi, new_summary = best
                old_omega = omega
                omega = trial_omega
                psi = trial_psi
                poisson_lambda = poisson_lambda + alpha_used * delta_lambda
                step = float(
                    np.linalg.norm(alpha_used * delta_omega)
                    / max(np.linalg.norm(old_omega), 1.0)
                )
                iterations.append(
                    ProfileIteration(
                        sigma=sigma,
                        iteration=iteration + 1,
                        residual_norm=new_summary["residual_norm"],
                        relative_residual=new_summary["relative_residual"],
                        omega_residual_norm=new_summary["omega_residual_norm"],
                        poisson_residual_norm=new_summary["poisson_residual_norm"],
                        relative_omega_step=step,
                        alpha=alpha_used,
                        linear_residual_norm=linear_residual,
                        accepted=accepted,
                    )
                )
                last_relative_step = step
                if not accepted:
                    converged = False
                    message = (
                        f"polar Newton line search failed at sigma={sigma:.6g}, "
                        f"iteration={iteration}"
                    )
                    break
                if step <= controls.tol_step and new_summary["relative_residual"] <= controls.tol_residual:
                    sigma_converged = True
                    break
            if not sigma_converged:
                converged = False
                if message == "converged":
                    message = f"polar Newton solve did not converge at sigma={sigma:.6g}"
                break
            sigma_prev = sigma

        runtime = time.perf_counter() - t0
        diagnostics = self.residual_summary(omega, psi, current_sigma, poisson_lambda)
        diagnostics["runtime_sec"] = runtime
        diagnostics["newton_steps"] = len(iterations)
        diagnostics["final_sigma"] = current_sigma
        diagnostics["grid_nr"] = self.grid.nr
        diagnostics["grid_ntheta"] = self.grid.ntheta
        diagnostics["grid_dr"] = self.grid.dr
        diagnostics["grid_dtheta"] = self.grid.dtheta
        diagnostics["poisson_lambda"] = poisson_lambda
        return PolarProfileResult(
            omega=omega.reshape(self.grid.shape),
            psi=psi.reshape(self.grid.shape),
            grid=self.grid,
            converged=converged,
            iterations=tuple(iterations),
            runtime_sec=runtime,
            message=message,
            diagnostics=diagnostics,
        )

    def solve_poisson(self, omega: np.ndarray, sigma: float) -> tuple[np.ndarray, float]:
        rhs = self.poisson_rhs(omega, sigma)
        solution = np.asarray(spla.spsolve(self.poisson_matrix, rhs), dtype=float)
        return solution[: self.grid.size], float(solution[-1])

    def poisson_rhs(self, omega: np.ndarray, sigma: float) -> np.ndarray:
        rhs = np.asarray(omega, dtype=float).reshape(-1) - self._neumann_laplacian_bc(
            self.psi_neumann_boundary(sigma)
        )
        return np.concatenate((rhs, [0.0]))

    def solve_linear_omega(self, psi: np.ndarray, sigma: float) -> np.ndarray:
        psi_vec = np.asarray(psi, dtype=float).reshape(-1)
        omega_boundary = self.omega_boundary(sigma)
        neumann = self.psi_neumann_boundary(sigma)
        psi_theta = self.dtheta_full @ psi_vec
        psi_r = self._psi_radial_derivative(psi_vec, neumann)
        inv_r = np.repeat(1.0 / self.grid.r, self.grid.ntheta)
        r = np.repeat(self.grid.r, self.grid.ntheta)
        vr = -psi_theta * inv_r - self.physical.mu * r
        vtheta_over_r = psi_r * inv_r

        omega_dr, omega_dr_bc = self._radial_advection_matrix_and_bc(vr, omega_boundary)
        omega_dtheta = self._angular_advection_matrix(vtheta_over_r)
        omega_lap_bc = self._dirichlet_laplacian_bc(omega_boundary)
        identity = sp.eye(self.grid.size, format="csr")
        operator = (
            sp.diags(vr, format="csr") @ omega_dr
            + sp.diags(vtheta_over_r, format="csr") @ omega_dtheta
            - identity
            - self.physical.eps * self.laplacian_dirichlet
        )
        rhs = self.physical.eps * omega_lap_bc - vr * omega_dr_bc
        return np.asarray(spla.spsolve(operator.tocsr(), rhs), dtype=float)

    def residual(
        self,
        omega: np.ndarray,
        psi: np.ndarray,
        sigma: float,
        poisson_lambda: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        omega_vec = np.asarray(omega, dtype=float).reshape(-1)
        psi_vec = np.asarray(psi, dtype=float).reshape(-1)
        omega_boundary = self.omega_boundary(sigma)
        neumann = self.psi_neumann_boundary(sigma)
        inv_r = np.repeat(1.0 / self.grid.r, self.grid.ntheta)
        r = np.repeat(self.grid.r, self.grid.ntheta)
        lap_omega = self.laplacian_dirichlet @ omega_vec + self._dirichlet_laplacian_bc(omega_boundary)
        psi_theta = self.dtheta_full @ psi_vec
        psi_r = self._psi_radial_derivative(psi_vec, neumann)
        vr = -psi_theta * inv_r - self.physical.mu * r
        vtheta_over_r = psi_r * inv_r
        omega_dr, omega_dr_bc = self._radial_advection_matrix_and_bc(vr, omega_boundary)
        omega_dtheta = self._angular_advection_matrix(vtheta_over_r)
        omega_r = omega_dr @ omega_vec + omega_dr_bc
        omega_theta = omega_dtheta @ omega_vec
        omega_residual = vr * omega_r + vtheta_over_r * omega_theta - omega_vec - self.physical.eps * lap_omega
        poisson_unknown = np.concatenate((psi_vec, [float(poisson_lambda)]))
        poisson_residual = self.poisson_matrix @ poisson_unknown - self.poisson_rhs(omega_vec, sigma)
        return omega_residual, poisson_residual

    def residual_vector(
        self,
        omega: np.ndarray,
        psi: np.ndarray,
        sigma: float,
        poisson_lambda: float = 0.0,
    ) -> np.ndarray:
        omega_residual, poisson_residual = self.residual(omega, psi, sigma, poisson_lambda)
        return np.concatenate((omega_residual, poisson_residual))

    def physical_poisson_residual(self, omega: np.ndarray, psi: np.ndarray, sigma: float) -> np.ndarray:
        omega_vec = np.asarray(omega, dtype=float).reshape(-1)
        psi_vec = np.asarray(psi, dtype=float).reshape(-1)
        neumann = self.psi_neumann_boundary(sigma)
        return self.laplacian_neumann @ psi_vec + self._neumann_laplacian_bc(neumann) - omega_vec

    def jacobian(self, omega: np.ndarray, psi: np.ndarray, sigma: float) -> sp.csr_matrix:
        omega_vec = np.asarray(omega, dtype=float).reshape(-1)
        psi_vec = np.asarray(psi, dtype=float).reshape(-1)
        omega_boundary = self.omega_boundary(sigma)
        neumann = self.psi_neumann_boundary(sigma)
        inv_r = np.repeat(1.0 / self.grid.r, self.grid.ntheta)
        r = np.repeat(self.grid.r, self.grid.ntheta)

        psi_theta = self.dtheta_full @ psi_vec
        psi_r = self._psi_radial_derivative(psi_vec, neumann)
        vr = -psi_theta * inv_r - self.physical.mu * r
        vtheta_over_r = psi_r * inv_r
        omega_dr, omega_dr_bc = self._radial_advection_matrix_and_bc(vr, omega_boundary)
        omega_dtheta = self._angular_advection_matrix(vtheta_over_r)
        omega_r = omega_dr @ omega_vec + omega_dr_bc
        omega_theta = omega_dtheta @ omega_vec

        identity = sp.eye(self.grid.size, format="csr")
        j_oo = (
            sp.diags(vr, format="csr") @ omega_dr
            + sp.diags(vtheta_over_r, format="csr") @ omega_dtheta
            - identity
            - self.physical.eps * self.laplacian_dirichlet
        )
        j_op = (
            -sp.diags(omega_r * inv_r, format="csr") @ self.dtheta_full
            + sp.diags(omega_theta * inv_r, format="csr") @ self.psi_radial_derivative_matrix
        )
        j_op_augmented = sp.hstack(
            (j_op, sp.csr_matrix((self.grid.size, 1), dtype=float)),
            format="csr",
        )
        return sp.bmat(
            [
                [j_oo, j_op_augmented],
                [self.poisson_omega_jacobian, self.poisson_matrix],
            ],
            format="csr",
        )

    def residual_summary(
        self,
        omega: np.ndarray,
        psi: np.ndarray,
        sigma: float,
        poisson_lambda: float = 0.0,
    ) -> dict[str, float]:
        omega_residual, poisson_residual = self.residual(omega, psi, sigma, poisson_lambda)
        physical_poisson_residual = self.physical_poisson_residual(omega, psi, sigma)
        weights = self.grid.quadrature_weights()
        omega_norm = self._weighted_norm(np.asarray(omega).reshape(-1), weights)
        lap_norm = self._weighted_norm(physical_poisson_residual + np.asarray(omega).reshape(-1), weights)
        omega_residual_norm = self._weighted_norm(omega_residual, weights)
        poisson_residual_norm = float(np.sqrt(self._weighted_norm(poisson_residual[:-1], weights) ** 2 + poisson_residual[-1] ** 2))
        physical_poisson_residual_norm = self._weighted_norm(physical_poisson_residual, weights)
        total = float(np.sqrt(omega_residual_norm**2 + poisson_residual_norm**2))
        scale = max(omega_norm, lap_norm, 1e-30)
        return {
            "residual_norm": total,
            "relative_residual": total / scale,
            "omega_residual_norm": omega_residual_norm,
            "poisson_residual_norm": poisson_residual_norm,
            "physical_poisson_residual_norm": physical_poisson_residual_norm,
            "omega_field_l2": omega_norm,
            "lap_psi_l2": lap_norm,
        }

    def sample_cartesian(
        self,
        result: PolarProfileResult,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        theta = np.arctan2(y, x)
        theta_mod = np.mod(theta, self.grid.theta_period)
        radius = np.hypot(x, y)
        omega = self._sample_periodic(result.omega, radius, theta_mod)
        psi = self._sample_periodic(result.psi, radius, theta_mod)
        omega = np.where(radius <= self.grid.radius, omega, np.nan)
        psi = np.where(radius <= self.grid.radius, psi, np.nan)
        return omega, psi

    def _sample_periodic(self, values: np.ndarray, radius: np.ndarray, theta: np.ndarray) -> np.ndarray:
        theta_ext = np.concatenate((self.grid.theta, [self.grid.theta_period]))
        vals_ext = np.concatenate((values, values[:, :1]), axis=1)
        radius_eval = np.clip(radius, self.grid.r[0], self.grid.r[-1])
        interp = RegularGridInterpolator(
            (self.grid.r, theta_ext),
            vals_ext,
            bounds_error=False,
            fill_value=np.nan,
        )
        return np.asarray(
            interp(np.column_stack((radius_eval.reshape(-1), theta.reshape(-1))))
        ).reshape(radius.shape)

    def _radial_derivative_matrix(self, *, boundary: str) -> sp.csr_matrix:
        mat = sp.lil_matrix((self.grid.size, self.grid.size), dtype=float)
        dr = self.grid.dr
        for j in range(self.grid.nr):
            for k in range(self.grid.ntheta):
                row = self.grid.index(j, k)
                if j == 0:
                    continue
                if j == self.grid.nr - 1:
                    if boundary == "interior":
                        mat[row, row] = 1.0 / dr
                        mat[row, self.grid.index(j - 1, k)] = -1.0 / dr
                        continue
                    mat[row, self.grid.index(j - 1, k)] = -1.0 / (1.5 * dr)
                else:
                    mat[row, self.grid.index(j - 1, k)] = -1.0 / (2.0 * dr)
                    mat[row, self.grid.index(j + 1, k)] = 1.0 / (2.0 * dr)
        return mat.tocsr()

    def _psi_radial_derivative(self, psi: np.ndarray, neumann: np.ndarray) -> np.ndarray:
        del neumann
        return self.psi_radial_derivative_matrix @ np.asarray(psi, dtype=float).reshape(-1)

    def _radial_advection_matrix_and_bc(
        self,
        velocity: np.ndarray,
        boundary_values: np.ndarray,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        mat = sp.lil_matrix((self.grid.size, self.grid.size), dtype=float)
        bc = np.zeros(self.grid.size, dtype=float)
        dr = self.grid.dr
        vel = np.asarray(velocity, dtype=float).reshape(-1)
        for j in range(self.grid.nr):
            for k in range(self.grid.ntheta):
                row = self.grid.index(j, k)
                if vel[row] >= 0.0:
                    if j == 0:
                        continue
                    mat[row, row] = 1.0 / dr
                    mat[row, self.grid.index(j - 1, k)] = -1.0 / dr
                else:
                    if j == self.grid.nr - 1:
                        scale = 1.0 / (0.5 * dr)
                        mat[row, row] = -scale
                        bc[row] = scale * float(boundary_values[k])
                    else:
                        mat[row, self.grid.index(j + 1, k)] = 1.0 / dr
                        mat[row, row] = -1.0 / dr
        return mat.tocsr(), bc

    def _angular_advection_matrix(self, velocity_over_r: np.ndarray) -> sp.csr_matrix:
        mat = sp.lil_matrix((self.grid.size, self.grid.size), dtype=float)
        vel = np.asarray(velocity_over_r, dtype=float).reshape(-1)
        scale = 1.0 / self.grid.dtheta
        for j in range(self.grid.nr):
            for k in range(self.grid.ntheta):
                row = self.grid.index(j, k)
                if vel[row] >= 0.0:
                    mat[row, row] = scale
                    mat[row, self.grid.index(j, k - 1)] = -scale
                else:
                    mat[row, self.grid.index(j, k + 1)] = scale
                    mat[row, row] = -scale
        return mat.tocsr()

    def _radial_dirichlet_derivative_bc(self, boundary_values: np.ndarray) -> np.ndarray:
        out = np.zeros(self.grid.size, dtype=float)
        scale = 1.0 / (1.5 * self.grid.dr)
        j = self.grid.nr - 1
        for k in range(self.grid.ntheta):
            out[self.grid.index(j, k)] = scale * boundary_values[k]
        return out

    def _radial_neumann_derivative_bc(self, neumann: np.ndarray) -> np.ndarray:
        out = np.zeros(self.grid.size, dtype=float)
        j = self.grid.nr - 1
        for k in range(self.grid.ntheta):
            row = self.grid.index(j, k)
            # Boundary value inferred from psi_b = psi_j + (dr/2) * g.
            out[row] = (0.5 * self.grid.dr * neumann[k]) / (1.5 * self.grid.dr)
        return out

    def _bordered_poisson_matrix(self) -> sp.csr_matrix:
        weights = sp.csr_matrix(self.mean_weights.reshape(-1, 1))
        top = sp.hstack((self.laplacian_neumann, weights), format="csr")
        bottom = sp.hstack(
            (
                sp.csr_matrix(self.mean_weights.reshape(1, -1)),
                sp.csr_matrix((1, 1), dtype=float),
            ),
            format="csr",
        )
        return sp.vstack((top, bottom), format="csr")

    def _poisson_omega_jacobian(self) -> sp.csr_matrix:
        mat = sp.eye(self.grid.size, format="lil")
        mat *= -1.0
        return sp.vstack((mat.tocsr(), sp.csr_matrix((1, self.grid.size), dtype=float)), format="csr")

    def _laplacian_matrix(self, *, boundary: str) -> sp.csr_matrix:
        mat = sp.lil_matrix((self.grid.size, self.grid.size), dtype=float)
        dr = self.grid.dr
        for j, rj in enumerate(self.grid.r):
            rin = j * dr
            rout = (j + 1) * dr
            for k in range(self.grid.ntheta):
                row = self.grid.index(j, k)
                if j > 0:
                    coeff = rin / (rj * dr * dr)
                    mat[row, self.grid.index(j - 1, k)] += coeff
                    mat[row, row] -= coeff
                if j < self.grid.nr - 1:
                    coeff = rout / (rj * dr * dr)
                    mat[row, self.grid.index(j + 1, k)] += coeff
                    mat[row, row] -= coeff
                elif boundary == "dirichlet":
                    coeff = self.grid.radius / (rj * dr * (0.5 * dr))
                    mat[row, row] -= coeff
                # For Neumann, the outer flux is a RHS contribution.
                angular_scale = 1.0 / (rj * rj)
                for kk in (k, (k - 1) % self.grid.ntheta, (k + 1) % self.grid.ntheta):
                    mat[row, self.grid.index(j, kk)] += angular_scale * self.dtheta2[k, kk]
        return mat.tocsr()

    def _dirichlet_laplacian_bc(self, boundary_values: np.ndarray) -> np.ndarray:
        out = np.zeros(self.grid.size, dtype=float)
        dr = self.grid.dr
        j = self.grid.nr - 1
        rj = self.grid.r[j]
        coeff = self.grid.radius / (rj * dr * (0.5 * dr))
        for k in range(self.grid.ntheta):
            out[self.grid.index(j, k)] = coeff * boundary_values[k]
        return out

    def _neumann_laplacian_bc(self, neumann: np.ndarray) -> np.ndarray:
        out = np.zeros(self.grid.size, dtype=float)
        j = self.grid.nr - 1
        rj = self.grid.r[j]
        coeff = self.grid.radius / (rj * self.grid.dr)
        for k in range(self.grid.ntheta):
            out[self.grid.index(j, k)] = coeff * neumann[k]
        return out

    @staticmethod
    def _weighted_norm(values: np.ndarray, weights: np.ndarray) -> float:
        return float(np.sqrt(np.sum(np.asarray(values, dtype=float) ** 2 * weights)))


def save_polar_profile_debug_plots(
    problem: PolarProfileProblem,
    result: PolarProfileResult,
    outdir: str | Path,
    *,
    n_cart: int = 241,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-problem.grid.radius, problem.grid.radius, n_cart)
    y = np.linspace(-problem.grid.radius, problem.grid.radius, n_cart)
    X, Y = np.meshgrid(x, y, indexing="ij")
    omega, psi = problem.sample_cartesian(result, X, Y)
    created: list[Path] = []

    path = out / "nextgen_polar_vorticity_heatmap.png"
    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    image = ax.imshow(
        omega.T,
        origin="lower",
        extent=(-problem.grid.radius, problem.grid.radius, -problem.grid.radius, problem.grid.radius),
        cmap="RdBu_r",
    )
    ax.set_aspect("equal")
    ax.set_title("nextgen polar Omega")
    fig.colorbar(image, ax=ax)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    created.append(path)

    path = out / "nextgen_polar_streamlines.png"
    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    ax.contour(X, Y, psi, levels=42, colors="black", linewidths=0.55)
    ax.set_aspect("equal")
    ax.set_title("nextgen polar streamfunction contours")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    created.append(path)
    return created
