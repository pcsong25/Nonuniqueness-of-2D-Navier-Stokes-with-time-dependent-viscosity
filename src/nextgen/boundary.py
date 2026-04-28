from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class FourierBoundaryData:
    """Fourier representation of the far-field vorticity boundary data.

    The baseline accepts nonnegative modes and mirrors them for real-valued data.
    This class follows the same convention and also exposes the derived
    streamfunction angular coefficients.
    """

    M: int
    coeffs: Mapping[int, complex]

    def __post_init__(self) -> None:
        full = {m: 0.0j for m in range(-int(self.M), int(self.M) + 1)}
        for mode, coeff in self.coeffs.items():
            m = int(mode)
            if abs(m) > self.M:
                continue
            c = complex(coeff)
            full[m] = c
            if m != 0:
                full[-m] = np.conj(c)
        object.__setattr__(self, "_coeffs_full", full)

    @property
    def coeffs_full(self) -> dict[int, complex]:
        return dict(self._coeffs_full)

    def _mode_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        modes = np.arange(-self.M, self.M + 1, dtype=int)
        coeffs = np.array([self._coeffs_full[int(m)] for m in modes], dtype=complex)
        return modes, coeffs

    @staticmethod
    def _evaluate(theta: np.ndarray, modes: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=float)
        values = np.exp(1j * np.outer(theta_arr.reshape(-1), modes)) @ coeffs
        return np.real(values).reshape(theta_arr.shape)

    def omega_bar(self, theta: np.ndarray) -> np.ndarray:
        modes, coeffs = self._mode_arrays()
        return self._evaluate(theta, modes, coeffs)

    def psi_bar_coeffs(self, mu: float) -> dict[int, complex]:
        alpha = 2.0 - 1.0 / float(mu)
        out: dict[int, complex] = {}
        for m, coeff in self._coeffs_full.items():
            denom = alpha * alpha - float(m * m)
            if abs(denom) < 1e-12:
                raise ValueError("Resonant Fourier mode: alpha^2 equals m^2.")
            out[m] = coeff / denom
        return out

    def psi_bar(self, theta: np.ndarray, mu: float) -> np.ndarray:
        modes = np.arange(-self.M, self.M + 1, dtype=int)
        coeff_map = self.psi_bar_coeffs(mu)
        coeffs = np.array([coeff_map[int(m)] for m in modes], dtype=complex)
        return self._evaluate(theta, modes, coeffs)

    def psi_bar_theta(self, theta: np.ndarray, mu: float) -> np.ndarray:
        modes = np.arange(-self.M, self.M + 1, dtype=int)
        coeff_map = self.psi_bar_coeffs(mu)
        coeffs = np.array([1j * int(m) * coeff_map[int(m)] for m in modes], dtype=complex)
        return self._evaluate(theta, modes, coeffs)

    def omega_dirichlet(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        mu: float,
        R: float,
        Rdom: float | None = None,
        sigma: float = 1.0,
    ) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        theta = np.arctan2(y_arr, x_arr)
        if Rdom is None:
            scale = float(R) ** (-1.0 / float(mu))
        else:
            radius = np.hypot(x_arr, y_arr)
            if np.any(radius <= 0.0):
                raise ValueError("Far-field vorticity data is undefined at the origin.")
            scale = (float(Rdom) / float(R)) ** (1.0 / float(mu)) * radius ** (
                -1.0 / float(mu)
            )
        return float(sigma) * scale * self.omega_bar(theta)

    def psi_neumann_circle(
        self,
        theta: np.ndarray,
        *,
        mu: float,
        R: float,
        Rdom: float,
        sigma: float = 1.0,
    ) -> np.ndarray:
        alpha = 2.0 - 1.0 / float(mu)
        return (
            float(sigma)
            * alpha
            * float(Rdom)
            * float(R) ** (-1.0 / float(mu))
            * self.psi_bar(theta, mu)
        )

    def psi_neumann_normal(
        self,
        x: np.ndarray,
        y: np.ndarray,
        nx: np.ndarray,
        ny: np.ndarray,
        *,
        mu: float,
        R: float,
        Rdom: float,
        sigma: float = 1.0,
    ) -> np.ndarray:
        """Normal derivative of the scaled far-field streamfunction on any boundary.

        For a square domain this evaluates grad(Psi_far) dot n, where
        Psi_far(z) = (Rdom/R)^(1/mu) |z|^(2-1/mu) Psi_bar(theta).
        On a circular boundary of radius Rdom with radial normals, this reduces
        to the baseline Neumann formula.
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        nx_arr = np.asarray(nx, dtype=float)
        ny_arr = np.asarray(ny, dtype=float)

        radius = np.hypot(x_arr, y_arr)
        if np.any(radius <= 0.0):
            raise ValueError("Normal far-field data is undefined at the origin.")

        normal_len = np.hypot(nx_arr, ny_arr)
        if np.any(normal_len <= 0.0):
            raise ValueError("Boundary normals must be nonzero.")
        nx_unit = nx_arr / normal_len
        ny_unit = ny_arr / normal_len

        theta = np.arctan2(y_arr, x_arr)
        alpha = 2.0 - 1.0 / float(mu)
        psi = self.psi_bar(theta, mu)
        psi_theta = self.psi_bar_theta(theta, mu)

        er_x = x_arr / radius
        er_y = y_arr / radius
        et_x = -y_arr / radius
        et_y = x_arr / radius

        scale = (
            float(sigma)
            * (float(Rdom) / float(R)) ** (1.0 / float(mu))
            * radius ** (alpha - 1.0)
        )
        grad_x = scale * (alpha * psi * er_x + psi_theta * et_x)
        grad_y = scale * (alpha * psi * er_y + psi_theta * et_y)
        return grad_x * nx_unit + grad_y * ny_unit


@dataclass(frozen=True)
class AffineFourierBoundaryData:
    """Boundary homotopy from a base Fourier state toward an increment.

    The supplied ``sigma`` parameter scales only ``increment``.  This supports
    coefficient continuation paths that are not equivalent to scaling all
    target modes from zero at once.
    """

    base: FourierBoundaryData
    increment: FourierBoundaryData

    @property
    def M(self) -> int:
        return max(int(self.base.M), int(self.increment.M))

    def omega_dirichlet(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        mu: float,
        R: float,
        Rdom: float | None = None,
        sigma: float = 1.0,
    ) -> np.ndarray:
        return self.base.omega_dirichlet(x, y, mu=mu, R=R, Rdom=Rdom, sigma=1.0) + self.increment.omega_dirichlet(
            x,
            y,
            mu=mu,
            R=R,
            Rdom=Rdom,
            sigma=sigma,
        )

    def psi_neumann_circle(
        self,
        theta: np.ndarray,
        *,
        mu: float,
        R: float,
        Rdom: float,
        sigma: float = 1.0,
    ) -> np.ndarray:
        return self.base.psi_neumann_circle(theta, mu=mu, R=R, Rdom=Rdom, sigma=1.0) + self.increment.psi_neumann_circle(
            theta,
            mu=mu,
            R=R,
            Rdom=Rdom,
            sigma=sigma,
        )

    def psi_neumann_normal(
        self,
        x: np.ndarray,
        y: np.ndarray,
        nx: np.ndarray,
        ny: np.ndarray,
        *,
        mu: float,
        R: float,
        Rdom: float,
        sigma: float = 1.0,
    ) -> np.ndarray:
        return self.base.psi_neumann_normal(
            x,
            y,
            nx,
            ny,
            mu=mu,
            R=R,
            Rdom=Rdom,
            sigma=1.0,
        ) + self.increment.psi_neumann_normal(
            x,
            y,
            nx,
            ny,
            mu=mu,
            R=R,
            Rdom=Rdom,
            sigma=sigma,
        )
