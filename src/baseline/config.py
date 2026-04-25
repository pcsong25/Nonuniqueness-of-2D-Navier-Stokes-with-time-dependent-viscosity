from dataclasses import dataclass

@dataclass
class SimulationParams:
    mu: float = 0.6
    R: float = 100.0
    Rdom: float = 10.0
    # Fourier truncation
    M: int = 32
    # Newton controls
    tol_newton: float = 1e-10
    max_newton: int = 12
    # σ grid for continuation (monotone increasing, ends with 1.0)
    sigma_grid: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    # Mesh sizing
    h_min: float = 0.12  
    h_max: float = 0.30


def eps_from_R(R: float, R_dom: float) -> float:
    return (R_dom ** 2) / (R ** 2)