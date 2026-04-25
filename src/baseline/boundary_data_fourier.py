import numpy as np

class FourierBC:
    """Fourier coefficients interface for \barΩ and Neumann load b_σ.

    coeffs_complex: dict[m] = c_m for m in [-M..M], real data implies
    c_{-m} = conjugate(c_m). If only m>=0 provided, we mirror.
    """
    def __init__(self, M: int, coeffs_complex: dict[int, complex]):
        self.M = int(M)
        #self.coeffs = dict(coeffs_complex)  # {m: complex c_m}
        # Build full symmetric dict
        self.cm = {m: 0j for m in range(-self.M, self.M + 1)}
        for m, c in coeffs_complex.items():
            if abs(m) > self.M:
                continue
            self.cm[m] = complex(c)
            if m != 0:
                self.cm[-m] = np.conj(c)

    def g_theta(self, theta: np.ndarray, mu: float, R: float) -> np.ndarray:
        """g(θ) = R^{-1/μ} * \barΩ(θ) evaluated at θ-array."""
        scale = R ** (-1.0 / mu)
        mvals = np.arange(-self.M, self.M + 1)
        coeffs = np.array([self.cm[m] for m in mvals])
        E = np.exp(1j * np.outer(theta, mvals))  # (Nθ, 2M+1)
        return scale * np.real(E @ coeffs)

    def b_modes(self, mu: float, R: float, Rdom: float) -> dict[int, complex]:
        """Return complex b_m for Neumann load from exterior source only:
        b_m = [α Rdom /(α^2 - m^2)] * R^{-1/μ} * \barΩ_m, α=2-1/μ (non-resonant).
        """
        alpha = 2.0 - 1.0 / mu
        b = {}
        for m in range(-self.M, self.M + 1):
            denom = alpha * alpha - (m * m)
            if abs(denom) < 1e-12:
                raise ValueError("Resonant case α^2 = m^2 encountered; adjust μ or handle branch.")
            b[m] = (alpha * Rdom / denom) * (R ** (-1.0 / mu)) * self.cm[m]
        return b

    def b_theta(self, theta: np.ndarray, mu: float, R: float, Rdom: float, sigma: float) -> np.ndarray:
        """Evaluate b_σ(θ) from complex modes (vectorized)."""
        mvals = np.arange(-self.M, self.M + 1)
        bm = self.b_modes(mu, R, Rdom)
        coeffs = np.array([bm[m] for m in mvals])
        E = np.exp(1j * np.outer(theta, mvals))
        return np.real(E @ coeffs) * float(sigma)
    
    def Omega_bar(self, theta):
        """
        Return \barΩ(θ) as Re[ Σ_m c_m e^{i m θ} ] using self.coeffs.
        theta: array-like of angles (radians)
        """
        th = np.asarray(theta, dtype=float)
        # vectorized accumulation
        out = np.zeros_like(th, dtype=float)
        for m, c in self.cm.items():
            out += np.real(c * np.exp(1j * m * th))
        return out