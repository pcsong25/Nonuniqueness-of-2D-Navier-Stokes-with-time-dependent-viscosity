from __future__ import annotations

from dataclasses import dataclass, field
import time
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .boundary import FourierBoundaryData
from .config import PhysicalParams
from .diagnostics import boundary_mismatch, poisson_compatibility
from .grid import SquareGrid
from .operators import (
    dirichlet_gradient_boundary_contribution,
    dirichlet_laplacian_boundary_contribution,
    interior_gradient_matrices,
    interior_laplacian_matrix,
)
from .poisson import assemble_neumann_poisson, default_gauge_ij


@dataclass(frozen=True)
class ProfileNewtonParams:
    """Controls for sigma-continuation and damped Newton profile solves."""

    sigma_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    tol_step: float = 1e-9
    tol_residual: float = 1e-9
    max_newton: int = 32
    line_search_shrink: float = 0.5
    max_backtracks: int = 8
    gauge_ij: tuple[int, int] | None = None
    gauge_value: float = 0.0


@dataclass(frozen=True)
class ProfileIteration:
    sigma: float
    iteration: int
    residual_norm: float
    relative_residual: float
    omega_residual_norm: float
    poisson_residual_norm: float
    relative_omega_step: float
    alpha: float
    linear_residual_norm: float
    accepted: bool


@dataclass(frozen=True)
class ProfileResult:
    omega: np.ndarray
    psi: np.ndarray
    converged: bool
    iterations: tuple[ProfileIteration, ...]
    runtime_sec: float
    message: str
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def newton_steps(self) -> int:
        return len(self.iterations)


@dataclass(frozen=True)
class ResidualSummary:
    total_norm: float
    relative_norm: float
    omega_norm: float
    poisson_norm: float
    scale: float


def full_grid_gradient_matrices(grid: SquareGrid) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Centered derivatives at interior nodes with all psi grid values as columns."""
    scale = 1.0 / (2.0 * grid.h)
    gx = sp.lil_matrix((grid.n_interior, grid.n * grid.n), dtype=float)
    gy = sp.lil_matrix((grid.n_interior, grid.n * grid.n), dtype=float)
    for i, j in grid.interior_ij():
        row = int(grid.interior_id[i, j])
        gx[row, grid.flat_index(int(i - 1), int(j))] -= scale
        gx[row, grid.flat_index(int(i + 1), int(j))] += scale
        gy[row, grid.flat_index(int(i), int(j - 1))] -= scale
        gy[row, grid.flat_index(int(i), int(j + 1))] += scale
    return gx.tocsr(), gy.tocsr()


def _poisson_omega_rhs_matrix(grid: SquareGrid, gauge_ij: tuple[int, int]) -> sp.csr_matrix:
    """Map Omega interior unknowns into non-gauge Poisson RHS rows."""
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    gauge_row = grid.flat_index(*gauge_ij)
    for i, j in grid.interior_ij():
        row = grid.flat_index(int(i), int(j))
        if row == gauge_row:
            continue
        rows.append(row)
        cols.append(int(grid.interior_id[i, j]))
        data.append(1.0)
    return sp.csr_matrix((data, (rows, cols)), shape=(grid.n * grid.n, grid.n_interior))


class ProfileProblem:
    """Finite-difference nonlinear profile problem on a square grid."""

    def __init__(
        self,
        grid: SquareGrid,
        physical: PhysicalParams,
        boundary_data: FourierBoundaryData,
        *,
        gauge_ij: tuple[int, int] | None = None,
        gauge_value: float = 0.0,
    ):
        self.grid = grid
        self.physical = physical
        self.boundary_data = boundary_data
        self.gauge_ij = default_gauge_ij(grid) if gauge_ij is None else gauge_ij
        self.gauge_value = float(gauge_value)

        self.laplacian = interior_laplacian_matrix(grid)
        self.grad_x, self.grad_y = interior_gradient_matrices(grid)
        self.psi_grad_x, self.psi_grad_y = full_grid_gradient_matrices(grid)
        self.poisson_omega_rhs = _poisson_omega_rhs_matrix(grid, self.gauge_ij)

        zeros = np.zeros(grid.shape, dtype=float)
        self.poisson_matrix = assemble_neumann_poisson(
            grid,
            zeros,
            zeros,
            gauge_ij=self.gauge_ij,
            gauge_value=self.gauge_value,
        ).matrix

        self.x_interior = grid.field_to_interior(grid.X)
        self.y_interior = grid.field_to_interior(grid.Y)

    @property
    def unknown_size(self) -> int:
        return self.grid.n_interior + self.grid.n * self.grid.n

    def boundary_fields(self, sigma: float) -> tuple[np.ndarray, np.ndarray]:
        """Return full-grid Omega Dirichlet and psi Neumann boundary fields."""
        grid = self.grid
        phys = self.physical
        omega = np.zeros(grid.shape, dtype=float)
        neumann = np.zeros(grid.shape, dtype=float)

        mask = grid.boundary_mask
        omega[mask] = self.boundary_data.omega_dirichlet(
            grid.X[mask],
            grid.Y[mask],
            mu=phys.mu,
            R=phys.R,
            Rdom=phys.Rdom,
            sigma=float(sigma),
        )

        nx, ny = grid.boundary_normals()
        neumann[mask] = self.boundary_data.psi_neumann_normal(
            grid.X[mask],
            grid.Y[mask],
            nx[mask],
            ny[mask],
            mu=phys.mu,
            R=phys.R,
            Rdom=phys.Rdom,
            sigma=float(sigma),
        )
        return omega, neumann

    def omega_full(self, omega_interior: np.ndarray, sigma: float) -> np.ndarray:
        omega_vec = self._omega_vector(omega_interior)
        omega_boundary, _ = self.boundary_fields(sigma)
        out = omega_boundary.copy()
        out[self.grid.interior_mask] = omega_vec
        return out

    def poisson_rhs(self, omega_interior: np.ndarray, sigma: float) -> np.ndarray:
        omega_vec = self._omega_vector(omega_interior)
        _, neumann = self.boundary_fields(sigma)
        zeros = np.zeros(self.grid.shape, dtype=float)
        rhs = assemble_neumann_poisson(
            self.grid,
            zeros,
            neumann,
            gauge_ij=self.gauge_ij,
            gauge_value=self.gauge_value,
        ).rhs
        return rhs + self.poisson_omega_rhs @ omega_vec

    def solve_poisson_for_omega(self, omega_interior: np.ndarray, sigma: float) -> np.ndarray:
        rhs = self.poisson_rhs(omega_interior, sigma)
        psi_vec = spla.spsolve(self.poisson_matrix, rhs)
        return np.asarray(psi_vec, dtype=float).reshape(self.grid.shape)

    def residual(
        self,
        omega_interior: np.ndarray,
        psi: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        omega_vec = self._omega_vector(omega_interior)
        psi_vec = self._psi_vector(psi)
        return np.concatenate(
            (
                self.omega_residual(omega_vec, psi_vec, sigma),
                self.poisson_matrix @ psi_vec - self.poisson_rhs(omega_vec, sigma),
            )
        )

    def omega_residual(
        self,
        omega_interior: np.ndarray,
        psi: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        omega_vec = self._omega_vector(omega_interior)
        psi_vec = self._psi_vector(psi)
        omega_boundary, _ = self.boundary_fields(sigma)

        lap_bc = dirichlet_laplacian_boundary_contribution(self.grid, omega_boundary)
        grad_bc_x, grad_bc_y = dirichlet_gradient_boundary_contribution(self.grid, omega_boundary)

        lap_omega = self.laplacian @ omega_vec + lap_bc
        omega_x = self.grad_x @ omega_vec + grad_bc_x
        omega_y = self.grad_y @ omega_vec + grad_bc_y

        psi_x = self.psi_grad_x @ psi_vec
        psi_y = self.psi_grad_y @ psi_vec
        vx = -psi_y - self.physical.mu * self.x_interior
        vy = psi_x - self.physical.mu * self.y_interior

        return vx * omega_x + vy * omega_y - omega_vec - self.physical.eps * lap_omega

    def jacobian(
        self,
        omega_interior: np.ndarray,
        psi: np.ndarray,
        sigma: float,
    ) -> sp.csr_matrix:
        omega_vec = self._omega_vector(omega_interior)
        psi_vec = self._psi_vector(psi)
        omega_boundary, _ = self.boundary_fields(sigma)

        grad_bc_x, grad_bc_y = dirichlet_gradient_boundary_contribution(self.grid, omega_boundary)
        omega_x = self.grad_x @ omega_vec + grad_bc_x
        omega_y = self.grad_y @ omega_vec + grad_bc_y

        psi_x = self.psi_grad_x @ psi_vec
        psi_y = self.psi_grad_y @ psi_vec
        vx = -psi_y - self.physical.mu * self.x_interior
        vy = psi_x - self.physical.mu * self.y_interior

        identity = sp.eye(self.grid.n_interior, format="csr")
        j_oo = (
            sp.diags(vx, format="csr") @ self.grad_x
            + sp.diags(vy, format="csr") @ self.grad_y
            - identity
            - self.physical.eps * self.laplacian
        )
        j_op = (
            sp.diags(omega_y, format="csr") @ self.psi_grad_x
            - sp.diags(omega_x, format="csr") @ self.psi_grad_y
        )
        j_po = -self.poisson_omega_rhs
        j_pp = self.poisson_matrix
        return sp.bmat([[j_oo, j_op], [j_po, j_pp]], format="csr")

    def residual_summary(
        self,
        omega_interior: np.ndarray,
        psi: np.ndarray,
        sigma: float,
    ) -> ResidualSummary:
        omega_vec = self._omega_vector(omega_interior)
        residual = self.residual(omega_vec, psi, sigma)
        omega_res = residual[: self.grid.n_interior]
        poisson_res = residual[self.grid.n_interior :]
        total = float(np.linalg.norm(residual))
        scale = max(
            float(np.linalg.norm(self.poisson_rhs(omega_vec, sigma))),
            float(np.linalg.norm(omega_vec)),
            1.0,
        )
        return ResidualSummary(
            total_norm=total,
            relative_norm=total / scale,
            omega_norm=float(np.linalg.norm(omega_res)),
            poisson_norm=float(np.linalg.norm(poisson_res)),
            scale=scale,
        )

    def solve(
        self,
        params: ProfileNewtonParams | None = None,
        *,
        initial_omega: np.ndarray | None = None,
        initial_psi: np.ndarray | None = None,
    ) -> ProfileResult:
        controls = ProfileNewtonParams() if params is None else params
        t0 = time.perf_counter()

        if controls.gauge_ij is not None and tuple(controls.gauge_ij) != tuple(self.gauge_ij):
            raise ValueError("ProfileNewtonParams.gauge_ij must match the ProfileProblem gauge.")
        if float(controls.gauge_value) != self.gauge_value:
            raise ValueError("ProfileNewtonParams.gauge_value must match the ProfileProblem gauge.")

        omega_vec = (
            np.zeros(self.grid.n_interior, dtype=float)
            if initial_omega is None
            else self._omega_vector(initial_omega).copy()
        )
        psi_vec = (
            np.zeros(self.grid.n * self.grid.n, dtype=float)
            if initial_psi is None
            else self._psi_vector(initial_psi).copy()
        )

        sigmas = [float(s) for s in controls.sigma_grid if float(s) > 0.0]
        if not sigmas:
            sigmas = [0.0]

        iterations: list[ProfileIteration] = []
        sigma_prev: float | None = None
        current_sigma = sigmas[0]
        converged = True
        message = "converged"

        for sigma in sigmas:
            current_sigma = sigma
            if sigma_prev is not None and sigma_prev != 0.0:
                omega_vec = omega_vec * (sigma / sigma_prev)
            psi_vec = self.solve_poisson_for_omega(omega_vec, sigma).reshape(-1)

            sigma_converged = False
            last_relative_step = 0.0
            for iteration in range(controls.max_newton + 1):
                summary = self.residual_summary(omega_vec, psi_vec, sigma)
                if iteration == 0:
                    iterations.append(
                        ProfileIteration(
                            sigma=sigma,
                            iteration=iteration,
                            residual_norm=summary.total_norm,
                            relative_residual=summary.relative_norm,
                            omega_residual_norm=summary.omega_norm,
                            poisson_residual_norm=summary.poisson_norm,
                            relative_omega_step=0.0,
                            alpha=0.0,
                            linear_residual_norm=0.0,
                            accepted=True,
                        )
                    )
                if (
                    summary.relative_norm <= controls.tol_residual
                    and last_relative_step <= controls.tol_step
                ):
                    sigma_converged = True
                    break

                if iteration == controls.max_newton:
                    break

                jac = self.jacobian(omega_vec, psi_vec, sigma)
                residual = self.residual(omega_vec, psi_vec, sigma)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", spla.MatrixRankWarning)
                    delta = spla.spsolve(jac, -residual)
                delta = np.asarray(delta, dtype=float)
                linear_residual = float(
                    np.linalg.norm(jac @ delta + residual) / max(summary.total_norm, 1e-30)
                )

                delta_omega = delta[: self.grid.n_interior]
                delta_psi = delta[self.grid.n_interior :]

                accepted = False
                best: tuple[float, np.ndarray, np.ndarray, ResidualSummary] | None = None
                alpha = 1.0
                for _ in range(controls.max_backtracks + 1):
                    trial_omega = omega_vec + alpha * delta_omega
                    trial_psi = psi_vec + alpha * delta_psi
                    trial_summary = self.residual_summary(trial_omega, trial_psi, sigma)
                    if best is None or trial_summary.total_norm < best[3].total_norm:
                        best = (alpha, trial_omega, trial_psi, trial_summary)
                    if (
                        trial_summary.relative_norm <= controls.tol_residual
                        or trial_summary.total_norm < summary.total_norm
                    ):
                        accepted = True
                        best = (alpha, trial_omega, trial_psi, trial_summary)
                        break
                    alpha *= controls.line_search_shrink

                assert best is not None
                alpha_used, trial_omega, trial_psi, new_summary = best
                if not accepted:
                    message = (
                        f"line search failed to reduce residual at sigma={sigma:.6g}, "
                        f"iteration={iteration}"
                    )
                    converged = False
                    iterations.append(
                        self._iteration_record(
                            sigma,
                            iteration + 1,
                            new_summary,
                            alpha_used,
                            delta_omega,
                            omega_vec,
                            linear_residual,
                            accepted=False,
                        )
                    )
                    break

                old_omega = omega_vec
                omega_vec = trial_omega
                psi_vec = trial_psi
                record = self._iteration_record(
                    sigma,
                    iteration + 1,
                    new_summary,
                    alpha_used,
                    delta_omega,
                    old_omega,
                    linear_residual,
                    accepted=True,
                )
                iterations.append(record)
                last_relative_step = record.relative_omega_step

                if (
                    record.relative_omega_step <= controls.tol_step
                    and new_summary.relative_norm <= controls.tol_residual
                ):
                    sigma_converged = True
                    break

            if not sigma_converged:
                converged = False
                if message == "converged":
                    message = f"Newton did not converge at sigma={sigma:.6g}"
                break
            sigma_prev = sigma

        runtime = time.perf_counter() - t0
        final_sigma = current_sigma
        omega = self.omega_full(omega_vec, final_sigma)
        psi = psi_vec.reshape(self.grid.shape)
        diagnostics = self.profile_diagnostics(omega_vec, psi_vec, final_sigma)
        diagnostics["runtime_sec"] = runtime
        diagnostics["newton_steps"] = len(iterations)
        diagnostics["final_sigma"] = final_sigma
        return ProfileResult(
            omega=omega,
            psi=psi,
            converged=converged,
            iterations=tuple(iterations),
            runtime_sec=runtime,
            message=message,
            diagnostics=diagnostics,
        )

    def profile_diagnostics(
        self,
        omega_interior: np.ndarray,
        psi: np.ndarray,
        sigma: float,
    ) -> dict[str, float | int | str]:
        omega_vec = self._omega_vector(omega_interior)
        psi_vec = self._psi_vector(psi)
        summary = self.residual_summary(omega_vec, psi_vec, sigma)
        omega = self.omega_full(omega_vec, sigma)
        omega_boundary, neumann = self.boundary_fields(sigma)
        bmatch = boundary_mismatch(omega, omega_boundary, self.grid)
        return {
            "residual_norm": summary.total_norm,
            "relative_residual": summary.relative_norm,
            "omega_residual_norm": summary.omega_norm,
            "poisson_residual_norm": summary.poisson_norm,
            "poisson_compatibility": poisson_compatibility(omega, neumann, self.grid),
            "omega_boundary_max_abs": bmatch["max_abs"],
            "omega_boundary_l2": bmatch["l2_boundary"],
            "psi_gauge_value": float(psi_vec[self.grid.flat_index(*self.gauge_ij)]),
            "grid_n": self.grid.n,
            "grid_h": self.grid.h,
        }

    def _iteration_record(
        self,
        sigma: float,
        iteration: int,
        summary: ResidualSummary,
        alpha: float,
        delta_omega: np.ndarray,
        omega_before: np.ndarray,
        linear_residual: float,
        *,
        accepted: bool,
    ) -> ProfileIteration:
        step = float(alpha) * np.asarray(delta_omega, dtype=float)
        rel_step = float(np.linalg.norm(step) / max(np.linalg.norm(omega_before), 1.0))
        return ProfileIteration(
            sigma=sigma,
            iteration=iteration,
            residual_norm=summary.total_norm,
            relative_residual=summary.relative_norm,
            omega_residual_norm=summary.omega_norm,
            poisson_residual_norm=summary.poisson_norm,
            relative_omega_step=rel_step,
            alpha=float(alpha),
            linear_residual_norm=linear_residual,
            accepted=accepted,
        )

    def _omega_vector(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.shape == self.grid.shape:
            return self.grid.field_to_interior(arr)
        vec = arr.reshape(-1)
        if vec.size != self.grid.n_interior:
            raise ValueError(f"Expected {self.grid.n_interior} Omega unknowns, got {vec.size}.")
        return vec

    def _psi_vector(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.shape == self.grid.shape:
            return arr.reshape(-1)
        vec = arr.reshape(-1)
        if vec.size != self.grid.n * self.grid.n:
            raise ValueError(f"Expected {self.grid.n * self.grid.n} psi unknowns, got {vec.size}.")
        return vec


def solve_profile(
    grid: SquareGrid,
    physical: PhysicalParams,
    boundary_data: FourierBoundaryData,
    params: ProfileNewtonParams | None = None,
) -> ProfileResult:
    problem = ProfileProblem(grid, physical, boundary_data)
    return problem.solve(params)


def save_profile_debug_plots(result: ProfileResult, outdir: str | Path) -> list[Path]:
    """Write simple profile debug plots and return created paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    fields = (("omega", result.omega), ("psi", result.psi))
    for name, field_values in fields:
        path = out / f"{name}.png"
        fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
        image = ax.imshow(np.asarray(field_values).T, origin="lower", cmap="viridis")
        ax.set_title(name)
        fig.colorbar(image, ax=ax)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        created.append(path)

    if result.iterations:
        path = out / "newton_residual_history.png"
        fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
        ax.semilogy([it.relative_residual for it in result.iterations], marker="o")
        ax.set_xlabel("recorded iteration")
        ax.set_ylabel("relative residual")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        created.append(path)

    return created


def iteration_table(iterations: Iterable[ProfileIteration]) -> list[dict[str, float | int | bool]]:
    return [
        {
            "sigma": it.sigma,
            "iteration": it.iteration,
            "relative_residual": it.relative_residual,
            "relative_omega_step": it.relative_omega_step,
            "alpha": it.alpha,
            "accepted": it.accepted,
        }
        for it in iterations
    ]
