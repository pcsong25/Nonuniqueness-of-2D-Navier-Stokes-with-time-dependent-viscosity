from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


def current_baseline_coefficients() -> dict[int, complex]:
    """Fourier coefficients matching the current active baseline driver case."""
    return {
        0: 0.0 + 0.0j,
        1: 0.0 + 0.0j,
        2: 0.0 + 0.0j,
        3: 0.0 + 0.0j,
        4: 50.0 + 0.0j,
        5: 0.0 + 0.0j,
        6: 0.0 + 0.0j,
        10: 0.0 + 0.0j,
        12: 0.0 + 0.0j,
        24: 0.0 + 0.0j,
        36: 0.0 + 0.0j,
    }


@dataclass(frozen=True)
class PhysicalParams:
    """Physical/scaling parameters shared with the baseline workflow."""

    mu: float = 0.8
    R: float = 150.0
    Rdom: float = 10.0

    @property
    def eps(self) -> float:
        return (self.Rdom * self.Rdom) / (self.R * self.R)

    @property
    def alpha(self) -> float:
        return 2.0 - 1.0 / self.mu


@dataclass(frozen=True)
class BoundaryParams:
    """Fourier boundary data for Omega_bar(theta)."""

    M: int = 48
    sigma: float = 1.0
    coeffs: Mapping[int, complex] = field(default_factory=current_baseline_coefficients)


@dataclass(frozen=True)
class SquareGridParams:
    """Uniform square grid parameters for direct finite differences."""

    half_width: float = 10.0
    n: int = 129

    def validate(self) -> None:
        if self.half_width <= 0.0:
            raise ValueError("half_width must be positive.")
        if self.n < 3:
            raise ValueError("n must be at least 3.")
        if self.n % 2 == 0:
            raise ValueError("n should be odd so the origin is a grid point.")


@dataclass(frozen=True)
class NextgenCase:
    """Complete foundational configuration for a nextgen reference run."""

    physical: PhysicalParams = field(default_factory=PhysicalParams)
    boundary: BoundaryParams = field(default_factory=BoundaryParams)
    grid: SquareGridParams = field(default_factory=SquareGridParams)

