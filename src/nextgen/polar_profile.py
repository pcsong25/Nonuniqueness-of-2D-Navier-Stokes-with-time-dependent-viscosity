from __future__ import annotations

from dataclasses import dataclass, field
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator

from .boundary import AffineFourierBoundaryData, FourierBoundaryData
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
    tol_residual_only: float | None = None
    max_iter: int = 32
    line_search_shrink: float = 0.5
    max_backtracks: int = 8
    initial_damping: float = 1.0
    adaptive_continuation: bool = True
    min_sigma_step: float = 1e-3
    sigma_step_growth: float = 1.4
    sigma_step_shrink: float = 0.5
    picard_predictor_steps: int = 0
    secant_predictor: bool = True
    secant_predictor_max_factor: float = 2.0
    least_squares_fallback: bool = True
    least_squares_damps: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1e-1, 1.0)
    least_squares_maxiter: int = 300
    anchored_continuation: bool = False
    anchor_betas: tuple[float, ...] = (10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.0)
    diffusion_continuation: bool = False
    diffusion_scales: tuple[float, ...] = (8.0, 4.0, 2.0, 1.0)


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


@dataclass(frozen=True)
class PolarProfileState:
    omega: np.ndarray
    psi: np.ndarray
    poisson_lambda: float = 0.0
    sigma: float = 0.0


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
        *,
        poisson_gauge: str = "mean",
        gauge_index: int | None = None,
        advection_scheme: str = "upwind",
        upwind_blend: float = 0.01,
    ):
        if not np.isclose(grid.radius, physical.Rdom):
            raise ValueError("PolarGrid.radius must match PhysicalParams.Rdom for the M3 disk solve.")
        if poisson_gauge not in {"mean", "pin"}:
            raise ValueError("poisson_gauge must be 'mean' or 'pin'.")
        if advection_scheme not in {"upwind", "centered", "blended"}:
            raise ValueError("advection_scheme must be 'upwind', 'centered', or 'blended'.")
        if not 0.0 <= float(upwind_blend) <= 1.0:
            raise ValueError("upwind_blend must be between 0 and 1.")
        self.grid = grid
        self.physical = physical
        self.boundary_data = boundary_data
        self.poisson_gauge = poisson_gauge
        self.advection_scheme = advection_scheme
        self.upwind_blend = float(upwind_blend)
        self.gauge_index = 0 if gauge_index is None else int(gauge_index)
        if not 0 <= self.gauge_index < grid.size:
            raise ValueError("gauge_index must be a valid polar grid unknown index.")
        self.dtheta = _periodic_first_matrix(grid.ntheta, grid.dtheta)
        self.dtheta2 = _periodic_second_matrix(grid.ntheta, grid.dtheta)
        self.dtheta_full = sp.kron(sp.eye(grid.nr, format="csr"), self.dtheta, format="csr")
        self.dtheta2_full = sp.kron(sp.eye(grid.nr, format="csr"), self.dtheta2, format="csr")
        self.radial_derivative = self._radial_derivative_matrix(boundary="dirichlet")
        self.psi_radial_derivative_matrix = self._radial_derivative_matrix(boundary="interior")
        (
            self.radial_upwind_positive,
            self.radial_upwind_negative,
            self.radial_upwind_outer_rows,
            self.radial_upwind_outer_scale,
        ) = self._radial_upwind_matrices()
        self.angular_upwind_positive, self.angular_upwind_negative = self._angular_upwind_matrices()
        self.laplacian_dirichlet = self._laplacian_matrix(boundary="dirichlet")
        self.laplacian_neumann = self._laplacian_matrix(boundary="neumann")
        self.mean_weights = self.grid.area_weights()
        self.poisson_matrix = self._poisson_matrix()
        self.poisson_lu = spla.splu(self.poisson_matrix.tocsc())
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

    def solve(
        self,
        params: PolarProfileParams | None = None,
        *,
        initial_state: PolarProfileState | None = None,
    ) -> PolarProfileResult:
        controls = PolarProfileParams() if params is None else params
        t0 = time.perf_counter()
        if initial_state is None:
            omega = np.zeros(self.grid.size, dtype=float)
            psi = np.zeros(self.grid.size, dtype=float)
            poisson_lambda = 0.0
            accepted_sigma = 0.0
        else:
            omega = np.asarray(initial_state.omega, dtype=float).reshape(self.grid.size).copy()
            psi = np.asarray(initial_state.psi, dtype=float).reshape(self.grid.size).copy()
            poisson_lambda = float(initial_state.poisson_lambda)
            accepted_sigma = float(initial_state.sigma)
        iterations: list[ProfileIteration] = []
        targets = sorted({float(sigma) for sigma in controls.sigma_grid if float(sigma) > accepted_sigma})
        if not targets:
            targets = [accepted_sigma]
        current_sigma = targets[0]
        converged = True
        message = "converged"
        adaptive_retries = 0
        completed_sigma_steps = 0
        newton_linear_solve_time = 0.0
        anchored_attempts = 0
        anchored_successes = 0
        anchored_linear_solve_time = 0.0
        diffusion_attempts = 0
        diffusion_successes = 0
        diffusion_linear_solve_time = 0.0
        previous_sigma: float | None = None
        previous_omega: np.ndarray | None = None
        previous_psi: np.ndarray | None = None
        previous_lambda: float | None = None

        for target in targets:
            if target <= accepted_sigma:
                continue
            sigma_step = target - accepted_sigma
            while accepted_sigma < target - 1e-14:
                trial_sigma = min(target, accepted_sigma + sigma_step)
                current_sigma = trial_sigma
                base_omega = omega.copy()
                base_psi = psi.copy()
                base_lambda = poisson_lambda
                guess_omega, guess_psi, guess_lambda = self._initial_guess_for_sigma(
                    omega,
                    psi,
                    poisson_lambda,
                    accepted_sigma,
                    trial_sigma,
                    controls,
                    previous_omega=previous_omega,
                    previous_psi=previous_psi,
                    previous_lambda=previous_lambda,
                    previous_sigma=previous_sigma,
                )
                (
                    trial_converged,
                    omega_candidate,
                    psi_candidate,
                    lambda_candidate,
                    solve_time,
                    failure_message,
                ) = self._newton_solve_sigma(
                    guess_omega,
                    guess_psi,
                    trial_sigma,
                    guess_lambda,
                    controls,
                    iterations,
                )
                newton_linear_solve_time += solve_time
                if (not trial_converged) and controls.anchored_continuation:
                    anchored_attempts += 1
                    (
                        trial_converged,
                        omega_candidate,
                        psi_candidate,
                        lambda_candidate,
                        solve_time,
                        failure_message,
                    ) = self._anchored_solve_sigma(
                        guess_omega,
                        guess_psi,
                        trial_sigma,
                        guess_lambda,
                        controls,
                        iterations,
                    )
                    anchored_linear_solve_time += solve_time
                    if trial_converged:
                        anchored_successes += 1
                if (not trial_converged) and controls.diffusion_continuation:
                    diffusion_attempts += 1
                    (
                        trial_converged,
                        omega_candidate,
                        psi_candidate,
                        lambda_candidate,
                        solve_time,
                        failure_message,
                    ) = self._diffusion_solve_sigma(
                        guess_omega,
                        guess_psi,
                        trial_sigma,
                        guess_lambda,
                        controls,
                        iterations,
                    )
                    diffusion_linear_solve_time += solve_time
                    if trial_converged:
                        diffusion_successes += 1
                if trial_converged:
                    previous_sigma = accepted_sigma
                    previous_omega = base_omega
                    previous_psi = base_psi
                    previous_lambda = base_lambda
                    omega = omega_candidate
                    psi = psi_candidate
                    poisson_lambda = lambda_candidate
                    accepted_sigma = trial_sigma
                    completed_sigma_steps += 1
                    remaining = target - accepted_sigma
                    if remaining > 1e-14:
                        sigma_step = min(
                            remaining,
                            max(controls.min_sigma_step, sigma_step * controls.sigma_step_growth),
                        )
                    continue

                omega = base_omega
                psi = base_psi
                poisson_lambda = base_lambda
                adaptive_retries += 1
                if not controls.adaptive_continuation:
                    converged = False
                    message = failure_message
                    break
                sigma_step *= controls.sigma_step_shrink
                if sigma_step < controls.min_sigma_step:
                    converged = False
                    message = (
                        f"{failure_message}; adaptive sigma step fell below "
                        f"{controls.min_sigma_step:.3e} after accepted sigma={accepted_sigma:.6g}"
                    )
                    break
            if not converged:
                break

        runtime = time.perf_counter() - t0
        diagnostics = self.residual_summary(omega, psi, accepted_sigma, poisson_lambda)
        diagnostics["runtime_sec"] = runtime
        diagnostics["newton_steps"] = len(iterations)
        diagnostics["final_sigma"] = accepted_sigma
        diagnostics["attempted_sigma"] = current_sigma
        diagnostics["adaptive_retries"] = adaptive_retries
        diagnostics["completed_sigma_steps"] = completed_sigma_steps
        diagnostics["newton_linear_solve_time_sec"] = newton_linear_solve_time
        diagnostics["anchored_attempts"] = anchored_attempts
        diagnostics["anchored_successes"] = anchored_successes
        diagnostics["anchored_linear_solve_time_sec"] = anchored_linear_solve_time
        diagnostics["diffusion_attempts"] = diffusion_attempts
        diagnostics["diffusion_successes"] = diffusion_successes
        diagnostics["diffusion_linear_solve_time_sec"] = diffusion_linear_solve_time
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

    def _initial_guess_for_sigma(
        self,
        omega: np.ndarray,
        psi: np.ndarray,
        poisson_lambda: float,
        sigma_prev: float,
        sigma: float,
        controls: PolarProfileParams,
        *,
        previous_omega: np.ndarray | None = None,
        previous_psi: np.ndarray | None = None,
        previous_lambda: float | None = None,
        previous_sigma: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        omega_vec = np.asarray(omega, dtype=float).reshape(-1)
        psi_vec = np.asarray(psi, dtype=float).reshape(-1)
        lambda_value = float(poisson_lambda)
        omega_guess: np.ndarray
        psi_guess: np.ndarray
        lambda_guess: float
        if (
            controls.secant_predictor
            and previous_sigma is not None
            and previous_omega is not None
            and previous_psi is not None
            and previous_lambda is not None
            and sigma_prev > previous_sigma
        ):
            ratio = (float(sigma) - float(sigma_prev)) / max(float(sigma_prev) - float(previous_sigma), 1e-30)
            max_factor = max(float(controls.secant_predictor_max_factor), 0.0)
            ratio = float(np.clip(ratio, -max_factor, max_factor))
            omega_guess = omega_vec + ratio * (omega_vec - np.asarray(previous_omega, dtype=float).reshape(-1))
            psi_guess = psi_vec + ratio * (psi_vec - np.asarray(previous_psi, dtype=float).reshape(-1))
            lambda_guess = lambda_value + ratio * (lambda_value - float(previous_lambda))
            psi_guess, lambda_guess = self.solve_poisson(omega_guess, sigma)
        else:
            if sigma_prev > 0.0:
                omega_guess = omega_vec * (sigma / sigma_prev)
            else:
                omega_guess = omega_vec.copy()
            psi_guess, lambda_guess = self.solve_poisson(omega_guess, sigma)
        for _ in range(max(int(controls.picard_predictor_steps), 0)):
            omega_guess = self.solve_linear_omega(psi_guess, sigma)
            psi_guess, lambda_guess = self.solve_poisson(omega_guess, sigma)
        return omega_guess, psi_guess, lambda_guess

    def _newton_solve_sigma(
        self,
        omega: np.ndarray,
        psi: np.ndarray,
        sigma: float,
        poisson_lambda: float,
        controls: PolarProfileParams,
        iterations: list[ProfileIteration],
        *,
        anchor_omega: np.ndarray | None = None,
        anchor_beta: float = 0.0,
        diffusion_scale: float = 1.0,
    ) -> tuple[bool, np.ndarray, np.ndarray, float, float, str]:
        linear_solve_time = 0.0
        last_relative_step = 0.0
        omega_current = np.asarray(omega, dtype=float).reshape(-1).copy()
        psi_current = np.asarray(psi, dtype=float).reshape(-1).copy()
        lambda_current = float(poisson_lambda)
        anchor_vec = None if anchor_omega is None else np.asarray(anchor_omega, dtype=float).reshape(-1)
        beta = float(anchor_beta)
        eps_scale = float(diffusion_scale)

        for iteration in range(controls.max_iter + 1):
            summary = self._residual_summary_with_anchor(
                omega_current,
                psi_current,
                sigma,
                lambda_current,
                anchor_vec,
                beta,
                eps_scale,
            )
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
                return True, omega_current, psi_current, lambda_current, linear_solve_time, "converged"
            if (
                controls.tol_residual_only is not None
                and summary["relative_residual"] <= controls.tol_residual_only
            ):
                return True, omega_current, psi_current, lambda_current, linear_solve_time, "converged_residual_only"
            if iteration == controls.max_iter:
                break

            jac = self._jacobian_with_anchor(omega_current, psi_current, sigma, anchor_vec, beta, eps_scale)
            residual = self._residual_vector_with_anchor(
                omega_current,
                psi_current,
                sigma,
                lambda_current,
                anchor_vec,
                beta,
                eps_scale,
            )
            t_linear = time.perf_counter()
            factor = spla.splu(jac.tocsc())
            delta = np.asarray(factor.solve(-residual), dtype=float)
            linear_solve_time += time.perf_counter() - t_linear
            linear_residual = float(
                np.linalg.norm(jac @ delta + residual)
                / max(summary["residual_norm"], 1e-30)
            )
            delta_omega = delta[: self.grid.size]
            delta_psi = delta[self.grid.size : 2 * self.grid.size]
            delta_lambda = float(delta[-1])

            accepted = False
            best: tuple[float, np.ndarray, np.ndarray, float, dict[str, float]] | None = None
            alpha = float(controls.initial_damping)
            for _ in range(controls.max_backtracks + 1):
                trial_omega = omega_current + alpha * delta_omega
                trial_psi = psi_current + alpha * delta_psi
                trial_lambda = lambda_current + alpha * delta_lambda
                trial_summary = self._residual_summary_with_anchor(
                    trial_omega,
                    trial_psi,
                    sigma,
                    trial_lambda,
                    anchor_vec,
                    beta,
                    eps_scale,
                )
                if best is None or trial_summary["residual_norm"] < best[4]["residual_norm"]:
                    best = (alpha, trial_omega, trial_psi, trial_lambda, trial_summary)
                if (
                    trial_summary["relative_residual"] <= controls.tol_residual
                    or trial_summary["residual_norm"] < summary["residual_norm"]
                ):
                    accepted = True
                    best = (alpha, trial_omega, trial_psi, trial_lambda, trial_summary)
                    break
                alpha *= controls.line_search_shrink

            assert best is not None
            alpha_used, trial_omega, trial_psi, trial_lambda, new_summary = best
            step_delta_omega = delta_omega
            if not accepted and controls.least_squares_fallback:
                for damp in controls.least_squares_damps:
                    t_linear = time.perf_counter()
                    ls_result = spla.lsmr(
                        jac,
                        -residual,
                        damp=float(damp),
                        maxiter=max(int(controls.least_squares_maxiter), 1),
                    )
                    ls_delta = np.asarray(ls_result[0], dtype=float)
                    linear_solve_time += time.perf_counter() - t_linear
                    ls_delta_omega = ls_delta[: self.grid.size]
                    ls_delta_psi = ls_delta[self.grid.size : 2 * self.grid.size]
                    ls_delta_lambda = float(ls_delta[-1])
                    ls_alpha = float(controls.initial_damping)
                    for _ in range(controls.max_backtracks + 1):
                        ls_omega = omega_current + ls_alpha * ls_delta_omega
                        ls_psi = psi_current + ls_alpha * ls_delta_psi
                        ls_lambda = lambda_current + ls_alpha * ls_delta_lambda
                        ls_summary = self._residual_summary_with_anchor(
                            ls_omega,
                            ls_psi,
                            sigma,
                            ls_lambda,
                            anchor_vec,
                            beta,
                            eps_scale,
                        )
                        if (
                            ls_summary["relative_residual"] <= controls.tol_residual
                            or ls_summary["residual_norm"] < summary["residual_norm"]
                        ):
                            accepted = True
                            alpha_used = ls_alpha
                            trial_omega = ls_omega
                            trial_psi = ls_psi
                            trial_lambda = ls_lambda
                            new_summary = ls_summary
                            step_delta_omega = ls_delta_omega
                            linear_residual = float(
                                np.linalg.norm(jac @ ls_delta + residual)
                                / max(summary["residual_norm"], 1e-30)
                            )
                            break
                        ls_alpha *= controls.line_search_shrink
                    if accepted:
                        break
            step = float(
                np.linalg.norm(alpha_used * step_delta_omega)
                / max(np.linalg.norm(omega_current), 1.0)
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
            if not accepted:
                return (
                    False,
                    omega_current,
                    psi_current,
                    lambda_current,
                    linear_solve_time,
                    f"polar Newton line search failed at sigma={sigma:.6g}, iteration={iteration}",
                )
            omega_current = trial_omega
            psi_current = trial_psi
            lambda_current = trial_lambda
            last_relative_step = step
            if step <= controls.tol_step and new_summary["relative_residual"] <= controls.tol_residual:
                return True, omega_current, psi_current, lambda_current, linear_solve_time, "converged"
            if (
                controls.tol_residual_only is not None
                and new_summary["relative_residual"] <= controls.tol_residual_only
            ):
                return True, omega_current, psi_current, lambda_current, linear_solve_time, "converged_residual_only"

        return (
            False,
            omega_current,
            psi_current,
            lambda_current,
            linear_solve_time,
            f"polar Newton solve did not converge at sigma={sigma:.6g}",
        )

    def _anchored_solve_sigma(
        self,
        omega: np.ndarray,
        psi: np.ndarray,
        sigma: float,
        poisson_lambda: float,
        controls: PolarProfileParams,
        iterations: list[ProfileIteration],
    ) -> tuple[bool, np.ndarray, np.ndarray, float, float, str]:
        omega_current = np.asarray(omega, dtype=float).reshape(-1).copy()
        psi_current = np.asarray(psi, dtype=float).reshape(-1).copy()
        lambda_current = float(poisson_lambda)
        anchor = omega_current.copy()
        total_linear_time = 0.0
        last_message = "anchored continuation was not attempted"
        betas = tuple(float(beta) for beta in controls.anchor_betas)
        if not betas or betas[-1] != 0.0:
            betas = betas + (0.0,)
        for beta in betas:
            (
                converged,
                omega_current,
                psi_current,
                lambda_current,
                linear_time,
                last_message,
            ) = self._newton_solve_sigma(
                omega_current,
                psi_current,
                sigma,
                lambda_current,
                controls,
                iterations,
                anchor_omega=anchor,
                anchor_beta=beta,
            )
            total_linear_time += linear_time
            if not converged:
                return False, omega_current, psi_current, lambda_current, total_linear_time, last_message
            psi_current, lambda_current = self.solve_poisson(omega_current, sigma)
        return True, omega_current, psi_current, lambda_current, total_linear_time, "converged"

    def _diffusion_solve_sigma(
        self,
        omega: np.ndarray,
        psi: np.ndarray,
        sigma: float,
        poisson_lambda: float,
        controls: PolarProfileParams,
        iterations: list[ProfileIteration],
    ) -> tuple[bool, np.ndarray, np.ndarray, float, float, str]:
        omega_current = np.asarray(omega, dtype=float).reshape(-1).copy()
        psi_current = np.asarray(psi, dtype=float).reshape(-1).copy()
        lambda_current = float(poisson_lambda)
        total_linear_time = 0.0
        last_message = "diffusion continuation was not attempted"
        scales = tuple(float(scale) for scale in controls.diffusion_scales if float(scale) > 0.0)
        if not scales or scales[-1] != 1.0:
            scales = scales + (1.0,)
        for scale in scales:
            (
                converged,
                omega_current,
                psi_current,
                lambda_current,
                linear_time,
                last_message,
            ) = self._newton_solve_sigma(
                omega_current,
                psi_current,
                sigma,
                lambda_current,
                controls,
                iterations,
                diffusion_scale=scale,
            )
            total_linear_time += linear_time
            if not converged:
                return False, omega_current, psi_current, lambda_current, total_linear_time, last_message
            psi_current, lambda_current = self.solve_poisson(omega_current, sigma)
        return True, omega_current, psi_current, lambda_current, total_linear_time, "converged"

    def _residual_vector_with_anchor(
        self,
        omega: np.ndarray,
        psi: np.ndarray,
        sigma: float,
        poisson_lambda: float,
        anchor_omega: np.ndarray | None,
        anchor_beta: float,
        diffusion_scale: float = 1.0,
    ) -> np.ndarray:
        residual = self.residual_vector(omega, psi, sigma, poisson_lambda, diffusion_scale=diffusion_scale)
        if anchor_omega is not None and anchor_beta != 0.0:
            residual = residual.copy()
            residual[: self.grid.size] += float(anchor_beta) * (
                np.asarray(omega, dtype=float).reshape(-1) - np.asarray(anchor_omega, dtype=float).reshape(-1)
            )
        return residual

    def _jacobian_with_anchor(
        self,
        omega: np.ndarray,
        psi: np.ndarray,
        sigma: float,
        anchor_omega: np.ndarray | None,
        anchor_beta: float,
        diffusion_scale: float = 1.0,
    ) -> sp.csr_matrix:
        jac = self.jacobian(omega, psi, sigma, diffusion_scale=diffusion_scale)
        if anchor_omega is not None and anchor_beta != 0.0:
            diagonal = np.zeros(2 * self.grid.size + 1, dtype=float)
            diagonal[: self.grid.size] = float(anchor_beta)
            jac = jac + sp.diags(diagonal, format="csr")
        return jac.tocsr()

    def _residual_summary_with_anchor(
        self,
        omega: np.ndarray,
        psi: np.ndarray,
        sigma: float,
        poisson_lambda: float,
        anchor_omega: np.ndarray | None,
        anchor_beta: float,
        diffusion_scale: float = 1.0,
    ) -> dict[str, float]:
        if anchor_omega is None or anchor_beta == 0.0:
            return self.residual_summary(omega, psi, sigma, poisson_lambda, diffusion_scale=diffusion_scale)
        omega_residual, poisson_residual = self.residual(
            omega,
            psi,
            sigma,
            poisson_lambda,
            diffusion_scale=diffusion_scale,
        )
        omega_vec = np.asarray(omega, dtype=float).reshape(-1)
        omega_residual = omega_residual + float(anchor_beta) * (
            omega_vec - np.asarray(anchor_omega, dtype=float).reshape(-1)
        )
        physical_poisson_residual = self.physical_poisson_residual(omega, psi, sigma)
        weights = self.grid.quadrature_weights()
        omega_norm = self._weighted_norm(omega_vec, weights)
        lap_norm = self._weighted_norm(physical_poisson_residual + omega_vec, weights)
        omega_residual_norm = self._weighted_norm(omega_residual, weights)
        poisson_residual_norm = float(
            np.sqrt(self._weighted_norm(poisson_residual[:-1], weights) ** 2 + poisson_residual[-1] ** 2)
        )
        total = float(np.sqrt(omega_residual_norm**2 + poisson_residual_norm**2))
        scale = max(omega_norm, lap_norm, 1e-30)
        return {
            "residual_norm": total,
            "relative_residual": total / scale,
            "omega_residual_norm": omega_residual_norm,
            "poisson_residual_norm": poisson_residual_norm,
            "physical_poisson_residual_norm": self._weighted_norm(physical_poisson_residual, weights),
            "omega_field_l2": omega_norm,
            "lap_psi_l2": lap_norm,
        }

    def solve_poisson(self, omega: np.ndarray, sigma: float) -> tuple[np.ndarray, float]:
        rhs = self.poisson_rhs(omega, sigma)
        solution = np.asarray(self.poisson_lu.solve(rhs), dtype=float)
        return solution[: self.grid.size], float(solution[-1])

    def poisson_rhs(self, omega: np.ndarray, sigma: float) -> np.ndarray:
        rhs = np.asarray(omega, dtype=float).reshape(-1) - self._neumann_laplacian_bc(
            self.psi_neumann_boundary(sigma)
        )
        if self.poisson_gauge == "pin":
            rhs = rhs.copy()
            rhs[self.gauge_index] = 0.0
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

        omega_dr, omega_dr_bc, omega_dtheta = self._omega_advection_operators(
            vr,
            vtheta_over_r,
            omega_boundary,
        )
        omega_lap_bc = self._dirichlet_laplacian_bc(omega_boundary)
        identity = sp.eye(self.grid.size, format="csr")
        operator = (
            sp.diags(vr, format="csr") @ omega_dr
            + sp.diags(vtheta_over_r, format="csr") @ omega_dtheta
            - identity
            - self.physical.eps * self.laplacian_dirichlet
        )
        rhs = self.physical.eps * omega_lap_bc - vr * omega_dr_bc
        return np.asarray(spla.splu(operator.tocsc()).solve(rhs), dtype=float)

    def residual(
        self,
        omega: np.ndarray,
        psi: np.ndarray,
        sigma: float,
        poisson_lambda: float = 0.0,
        *,
        diffusion_scale: float = 1.0,
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
        omega_dr, omega_dr_bc, omega_dtheta = self._omega_advection_operators(
            vr,
            vtheta_over_r,
            omega_boundary,
        )
        omega_r = omega_dr @ omega_vec + omega_dr_bc
        omega_theta = omega_dtheta @ omega_vec
        eps = self.physical.eps * float(diffusion_scale)
        omega_residual = vr * omega_r + vtheta_over_r * omega_theta - omega_vec - eps * lap_omega
        poisson_unknown = np.concatenate((psi_vec, [float(poisson_lambda)]))
        poisson_residual = self.poisson_matrix @ poisson_unknown - self.poisson_rhs(omega_vec, sigma)
        return omega_residual, poisson_residual

    def residual_vector(
        self,
        omega: np.ndarray,
        psi: np.ndarray,
        sigma: float,
        poisson_lambda: float = 0.0,
        *,
        diffusion_scale: float = 1.0,
    ) -> np.ndarray:
        omega_residual, poisson_residual = self.residual(
            omega,
            psi,
            sigma,
            poisson_lambda,
            diffusion_scale=diffusion_scale,
        )
        return np.concatenate((omega_residual, poisson_residual))

    def physical_poisson_residual(self, omega: np.ndarray, psi: np.ndarray, sigma: float) -> np.ndarray:
        omega_vec = np.asarray(omega, dtype=float).reshape(-1)
        psi_vec = np.asarray(psi, dtype=float).reshape(-1)
        neumann = self.psi_neumann_boundary(sigma)
        return self.laplacian_neumann @ psi_vec + self._neumann_laplacian_bc(neumann) - omega_vec

    def jacobian(
        self,
        omega: np.ndarray,
        psi: np.ndarray,
        sigma: float,
        *,
        diffusion_scale: float = 1.0,
    ) -> sp.csr_matrix:
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
        omega_dr, omega_dr_bc, omega_dtheta = self._omega_advection_operators(
            vr,
            vtheta_over_r,
            omega_boundary,
        )
        omega_r = omega_dr @ omega_vec + omega_dr_bc
        omega_theta = omega_dtheta @ omega_vec

        identity = sp.eye(self.grid.size, format="csr")
        j_oo = (
            sp.diags(vr, format="csr") @ omega_dr
            + sp.diags(vtheta_over_r, format="csr") @ omega_dtheta
            - identity
            - (self.physical.eps * float(diffusion_scale)) * self.laplacian_dirichlet
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
        *,
        diffusion_scale: float = 1.0,
    ) -> dict[str, float]:
        omega_residual, poisson_residual = self.residual(
            omega,
            psi,
            sigma,
            poisson_lambda,
            diffusion_scale=diffusion_scale,
        )
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
            "poisson_gauge": self.poisson_gauge,
            "advection_scheme": self.advection_scheme,
            "upwind_blend": self.upwind_blend,
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
        vel = np.asarray(velocity, dtype=float).reshape(-1)
        positive = vel >= 0.0
        mat = (
            sp.diags(positive.astype(float), format="csr") @ self.radial_upwind_positive
            + sp.diags((~positive).astype(float), format="csr") @ self.radial_upwind_negative
        )
        bc = np.zeros(self.grid.size, dtype=float)
        outer_negative = ~positive[self.radial_upwind_outer_rows]
        bc[self.radial_upwind_outer_rows] = (
            outer_negative.astype(float) * self.radial_upwind_outer_scale * np.asarray(boundary_values, dtype=float)
        )
        return mat.tocsr(), bc

    def _angular_advection_matrix(self, velocity_over_r: np.ndarray) -> sp.csr_matrix:
        vel = np.asarray(velocity_over_r, dtype=float).reshape(-1)
        positive = vel >= 0.0
        return (
            sp.diags(positive.astype(float), format="csr") @ self.angular_upwind_positive
            + sp.diags((~positive).astype(float), format="csr") @ self.angular_upwind_negative
        ).tocsr()

    def _omega_advection_operators(
        self,
        radial_speed: np.ndarray,
        angular_speed_over_r: np.ndarray,
        boundary_values: np.ndarray,
    ) -> tuple[sp.csr_matrix, np.ndarray, sp.csr_matrix]:
        centered_radial = self.radial_derivative
        centered_radial_bc = self._radial_dirichlet_derivative_bc(boundary_values)
        centered_angular = self.dtheta_full
        if self.advection_scheme == "centered":
            return centered_radial, centered_radial_bc, centered_angular
        radial_matrix, radial_bc = self._radial_advection_matrix_and_bc(radial_speed, boundary_values)
        angular_matrix = self._angular_advection_matrix(angular_speed_over_r)
        if self.advection_scheme == "upwind":
            return radial_matrix, radial_bc, angular_matrix
        blend = self.upwind_blend
        return (
            ((1.0 - blend) * centered_radial + blend * radial_matrix).tocsr(),
            (1.0 - blend) * centered_radial_bc + blend * radial_bc,
            ((1.0 - blend) * centered_angular + blend * angular_matrix).tocsr(),
        )

    def _radial_upwind_matrices(self) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, float]:
        positive = sp.lil_matrix((self.grid.size, self.grid.size), dtype=float)
        negative = sp.lil_matrix((self.grid.size, self.grid.size), dtype=float)
        dr = self.grid.dr
        for j in range(self.grid.nr):
            for k in range(self.grid.ntheta):
                row = self.grid.index(j, k)
                if j > 0:
                    positive[row, row] = 1.0 / dr
                    positive[row, self.grid.index(j - 1, k)] = -1.0 / dr
                if j == self.grid.nr - 1:
                    negative[row, row] = -1.0 / (0.5 * dr)
                else:
                    negative[row, self.grid.index(j + 1, k)] = 1.0 / dr
                    negative[row, row] = -1.0 / dr
        outer_rows = np.asarray(
            [self.grid.index(self.grid.nr - 1, k) for k in range(self.grid.ntheta)],
            dtype=int,
        )
        return positive.tocsr(), negative.tocsr(), outer_rows, 1.0 / (0.5 * dr)

    def _angular_upwind_matrices(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        positive = sp.lil_matrix((self.grid.size, self.grid.size), dtype=float)
        negative = sp.lil_matrix((self.grid.size, self.grid.size), dtype=float)
        scale = 1.0 / self.grid.dtheta
        for j in range(self.grid.nr):
            for k in range(self.grid.ntheta):
                row = self.grid.index(j, k)
                positive[row, row] = scale
                positive[row, self.grid.index(j, k - 1)] = -scale
                negative[row, self.grid.index(j, k + 1)] = scale
                negative[row, row] = -scale
        return positive.tocsr(), negative.tocsr()

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

    def _poisson_matrix(self) -> sp.csr_matrix:
        if self.poisson_gauge == "pin":
            return self._augmented_pinned_poisson_matrix()
        return self._bordered_poisson_matrix()

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

    def _pinned_poisson_core(self) -> sp.csr_matrix:
        mat = self.laplacian_neumann.tolil()
        mat.rows[self.gauge_index] = [self.gauge_index]
        mat.data[self.gauge_index] = [1.0]
        return mat.tocsr()

    def _augmented_pinned_poisson_matrix(self) -> sp.csr_matrix:
        core = self._pinned_poisson_core()
        top = sp.hstack(
            (core, sp.csr_matrix((self.grid.size, 1), dtype=float)),
            format="csr",
        )
        bottom = sp.hstack(
            (
                sp.csr_matrix((1, self.grid.size), dtype=float),
                sp.csr_matrix([[1.0]], dtype=float),
            ),
            format="csr",
        )
        return sp.vstack((top, bottom), format="csr")

    def _poisson_omega_jacobian(self) -> sp.csr_matrix:
        mat = sp.eye(self.grid.size, format="lil")
        mat *= -1.0
        if self.poisson_gauge == "pin":
            mat.rows[self.gauge_index] = []
            mat.data[self.gauge_index] = []
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


def solve_coefficient_homotopy(
    grid: PolarGrid,
    physical: PhysicalParams,
    boundary_data: FourierBoundaryData,
    params: PolarProfileParams | None = None,
    *,
    poisson_gauge: str = "mean",
    advection_scheme: str = "upwind",
    upwind_blend: float = 0.1,
    mode_order: tuple[int, ...] | None = None,
) -> PolarProfileResult:
    """Solve a profile by adding Fourier modes through affine continuation.

    This is a nonlinear-continuation helper, not a symmetry reduction.  It uses
    the same full-disk residual/Jacobian at every stage, but changes the path
    through coefficient space when scalar amplitude continuation stalls.
    """

    controls = PolarProfileParams() if params is None else params
    coeffs = {
        int(mode): complex(coeff)
        for mode, coeff in boundary_data.coeffs.items()
        if abs(complex(coeff)) > 0.0
    }
    if mode_order is None:
        ordered_modes = tuple(sorted(coeffs, key=lambda mode: (abs(mode), mode)))
    else:
        ordered_modes = tuple(int(mode) for mode in mode_order if int(mode) in coeffs)
        ordered_modes = ordered_modes + tuple(mode for mode in sorted(coeffs) if mode not in ordered_modes)

    current_coeffs: dict[int, complex] = {}
    state = PolarProfileState(
        omega=np.zeros(grid.size, dtype=float),
        psi=np.zeros(grid.size, dtype=float),
        poisson_lambda=0.0,
        sigma=0.0,
    )
    all_iterations: list[ProfileIteration] = []
    stage_messages: list[str] = []
    total_runtime = 0.0
    result: PolarProfileResult | None = None

    if not ordered_modes:
        problem = PolarProfileProblem(
            grid,
            physical,
            boundary_data,
            poisson_gauge=poisson_gauge,
            advection_scheme=advection_scheme,
            upwind_blend=upwind_blend,
        )
        return problem.solve(controls)

    for stage_index, mode in enumerate(ordered_modes, start=1):
        base = FourierBoundaryData(M=boundary_data.M, coeffs=current_coeffs)
        increment = FourierBoundaryData(M=boundary_data.M, coeffs={mode: coeffs[mode]})
        stage_boundary = AffineFourierBoundaryData(base=base, increment=increment)
        problem = PolarProfileProblem(
            grid,
            physical,
            stage_boundary,
            poisson_gauge=poisson_gauge,
            advection_scheme=advection_scheme,
            upwind_blend=upwind_blend,
        )
        result = problem.solve(controls, initial_state=state)
        total_runtime += result.runtime_sec
        all_iterations.extend(result.iterations)
        stage_messages.append(
            (
                f"stage={stage_index}, mode={mode}, converged={result.converged}, "
                f"final_sigma={result.diagnostics.get('final_sigma')}, "
                f"relative_residual={result.diagnostics.get('relative_residual')}"
            )
        )
        if not result.converged:
            break
        current_coeffs[mode] = coeffs[mode]
        state = PolarProfileState(
            omega=result.omega.reshape(-1),
            psi=result.psi.reshape(-1),
            poisson_lambda=float(result.diagnostics.get("poisson_lambda", 0.0)),
            sigma=0.0,
        )

    assert result is not None
    diagnostics = dict(result.diagnostics)
    diagnostics["runtime_sec"] = total_runtime
    diagnostics["newton_steps"] = len(all_iterations)
    diagnostics["coefficient_homotopy"] = "mode_by_mode"
    diagnostics["coefficient_homotopy_stages"] = len(stage_messages)
    diagnostics["coefficient_homotopy_modes"] = ",".join(str(mode) for mode in ordered_modes)
    diagnostics["coefficient_homotopy_stage_messages"] = " | ".join(stage_messages)
    return PolarProfileResult(
        omega=result.omega,
        psi=result.psi,
        grid=result.grid,
        converged=result.converged,
        iterations=tuple(all_iterations),
        runtime_sec=total_runtime,
        message=result.message,
        diagnostics=diagnostics,
    )

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
