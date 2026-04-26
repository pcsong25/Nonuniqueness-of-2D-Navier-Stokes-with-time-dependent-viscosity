from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SquareGrid:
    """Uniform square grid with explicit interior/boundary indexing."""

    half_width: float
    n: int
    x: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)
    X: np.ndarray = field(init=False, repr=False)
    Y: np.ndarray = field(init=False, repr=False)
    interior_mask: np.ndarray = field(init=False, repr=False)
    boundary_mask: np.ndarray = field(init=False, repr=False)
    interior_id: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.half_width <= 0.0:
            raise ValueError("half_width must be positive.")
        if self.n < 3:
            raise ValueError("n must be at least 3.")

        x = np.linspace(-float(self.half_width), float(self.half_width), int(self.n))
        y = np.linspace(-float(self.half_width), float(self.half_width), int(self.n))
        X, Y = np.meshgrid(x, y, indexing="ij")

        boundary = np.zeros((self.n, self.n), dtype=bool)
        boundary[0, :] = True
        boundary[-1, :] = True
        boundary[:, 0] = True
        boundary[:, -1] = True
        interior = ~boundary

        interior_id = -np.ones((self.n, self.n), dtype=int)
        interior_id[interior] = np.arange(int(np.count_nonzero(interior)), dtype=int)

        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "Y", Y)
        object.__setattr__(self, "boundary_mask", boundary)
        object.__setattr__(self, "interior_mask", interior)
        object.__setattr__(self, "interior_id", interior_id)

    @property
    def h(self) -> float:
        return 2.0 * float(self.half_width) / float(self.n - 1)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n, self.n)

    @property
    def n_interior(self) -> int:
        return int(np.count_nonzero(self.interior_mask))

    @property
    def n_boundary(self) -> int:
        return int(np.count_nonzero(self.boundary_mask))

    def flat_index(self, i: int, j: int) -> int:
        return int(i) * self.n + int(j)

    def ij_from_flat(self, k: int) -> tuple[int, int]:
        return divmod(int(k), self.n)

    def interior_ij(self) -> np.ndarray:
        return np.argwhere(self.interior_mask)

    def boundary_ij(self) -> np.ndarray:
        return np.argwhere(self.boundary_mask)

    def field_to_interior(self, values: np.ndarray) -> np.ndarray:
        field = np.asarray(values)
        if field.shape != self.shape:
            raise ValueError(f"Expected field shape {self.shape}, got {field.shape}.")
        return np.asarray(field[self.interior_mask])

    def interior_to_field(self, values: np.ndarray, boundary_value: float = 0.0) -> np.ndarray:
        vec = np.asarray(values)
        if vec.size != self.n_interior:
            raise ValueError(f"Expected {self.n_interior} interior values, got {vec.size}.")
        field = np.full(self.shape, float(boundary_value), dtype=vec.dtype)
        field[self.interior_mask] = vec
        return field

    def boundary_normals(self) -> tuple[np.ndarray, np.ndarray]:
        """Outward unit normals on the square boundary.

        Corners use normalized diagonal normals. They are useful for diagnostics
        and plotting; later PDE boundary closures may choose corner-specific
        treatment.
        """
        nx = np.zeros(self.shape, dtype=float)
        ny = np.zeros(self.shape, dtype=float)
        nx[0, :] = -1.0
        nx[-1, :] = 1.0
        ny[:, 0] = -1.0
        ny[:, -1] = 1.0
        length = np.hypot(nx, ny)
        mask = length > 0.0
        nx[mask] /= length[mask]
        ny[mask] /= length[mask]
        return nx, ny

