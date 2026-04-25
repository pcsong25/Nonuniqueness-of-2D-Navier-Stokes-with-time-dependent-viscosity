# --- file: warmstart_picard.py ---
import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc

try:
    from dolfinx.fem.petsc import assemble_matrix, assemble_vector, set_bc
except Exception:
    from dolfinx.fem import assemble_matrix, assemble_vector, set_bc


class PicardWarmStart:
    """
    Few under-relaxed Picard sweeps:
      1) v_k = ∇^⊥ψ_k − μ x
      2) Solve: ε ΔΩ_{k+1} − v_k·∇Ω_{k+1} − Ω_{k+1} = 0 in B,  Ω|∂B = R^{-1/μ} \barΩ(θ)
      3) Solve: Δψ_{k+1} = Ω_{k+1} with your Neumann flux b_σ(θ)
    Stop after max_sweeps or when weak residual drops by factor target_drop.

    Use as a preprocessor to Newton.
    """

    def __init__(self, mesh, spaces, poisson, facet_tags, outer_tag,
                 fourier_bc, mu: float, R: float, Rdom: float,
                 alpha: float = 0.6, max_sweeps: int = 6, target_drop: float = 1e-2,
                 use_SUPG: bool = False):
        self.mesh = mesh
        self.Vo = spaces["Vomega"]
        self.Vp = spaces["Vpsi"]
        self.poisson = poisson
        self.ftags = facet_tags
        self.outer_tag = int(outer_tag)
        self.fourier_bc = fourier_bc
        self.mu = float(mu)
        self.R = float(R)
        self.Rdom = float(Rdom)
        self.eps = (self.Rdom * self.Rdom) / (self.R * self.R)
        self.alpha = float(alpha)
        self.max_sweeps = int(max_sweeps)
        self.target_drop = float(target_drop)
        self.use_SUPG = bool(use_SUPG)

        # Boundary dofs for Ω Dirichlet
        bfacets = self.ftags.find(self.outer_tag)
        self.omega_bdofs = fem.locate_dofs_topological(self.Vo, self.mesh.topology.dim - 1, bfacets)
        xy_b = self.Vo.tabulate_dof_coordinates()[self.omega_bdofs]
        theta_b = np.arctan2(xy_b[:, 1], xy_b[:, 0])
        self.g_boundary_vals = (self.R ** (-1.0 / self.mu)) * self._Omega_bar(theta_b)

        # Prepare a Function to carry non-constant Dirichlet values
        self.g_fun = fem.Function(self.Vo)
        # Assign only on boundary dofs; interior values won't be used by BC application
        arr = self._vec(self.g_fun)
        arr[self.omega_bdofs.astype(np.int64)] = self.g_boundary_vals
        self._ghost(self.g_fun)
        self.bc_dirichlet = fem.dirichletbc(self.g_fun, self.omega_bdofs)

    # ---------- helpers ----------
    @staticmethod
    def _vec(fn: fem.Function):
        try:
            return fn.vector.array
        except AttributeError:
            return fn.x.array

    @staticmethod
    def _ghost(fn: fem.Function):
        try:
            fn.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        except AttributeError:
            if hasattr(fn.x, "scatter_forward"):
                fn.x.scatter_forward()

    def _Omega_bar(self, theta: np.ndarray) -> np.ndarray:
        if self.fourier_bc is not None and hasattr(self.fourier_bc, "Omega_bar"):
            return np.asarray(self.fourier_bc.Omega_bar(theta), dtype=float)
        if callable(self.fourier_bc):
            return np.asarray(self.fourier_bc(theta), dtype=float)
        raise RuntimeError("FourierBC must provide Omega_bar(theta).")

    def _v_field(self, psi):
        x = ufl.SpatialCoordinate(self.mesh)
        g = ufl.grad(psi)
        rot = ufl.as_vector((-g[1], g[0]))
        return rot - self.mu * ufl.as_vector((x[0], x[1]))

    # ---------- one linear advec-diff solve for Ω with Dirichlet trace ----------
    def _solve_advec_diff(self, v):
        u = ufl.TrialFunction(self.Vo)
        w = ufl.TestFunction(self.Vo)

        aG = (
            self.eps * ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx
            + ufl.inner(ufl.grad(u), v) * w * ufl.dx
            - u * w * ufl.dx
        )

        if self.use_SUPG:
            h = ufl.CellDiameter(self.mesh)
            vmag = ufl.sqrt(ufl.inner(v, v) + 1e-16)
            tau = 1.0 / ufl.sqrt((2 * vmag / h) ** 2 + (4 * self.eps / (h * h)) ** 2 + 1.0)
            r_lin = ufl.inner(ufl.grad(u), v) - u
            aG += tau * (ufl.inner(ufl.grad(w), v)) * r_lin * ufl.dx

        A = assemble_matrix(fem.form(aG), bcs=[self.bc_dirichlet]); A.assemble()
        b = A.createVecRight()
        b.set(0.0)
        set_bc(b, [self.bc_dirichlet])

        ksp = PETSc.KSP().create(self.mesh.comm)
        ksp.setOperators(A); ksp.setType("preonly"); ksp.getPC().setType("lu")
        ksp.setFromOptions()
        x = b.duplicate(); x.set(0.0); ksp.solve(b, x)

        Om = fem.Function(self.Vo)
        try:
            Om.vector.array[:] = x.array
            Om.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        except AttributeError:
            Om.x.array[:] = x.array
            if hasattr(Om.x, "scatter_forward"):
                Om.x.scatter_forward()
        return Om

    # ---------- public API ----------
    def run(self, k_pre: int | None = None):
        # Start with v0 = -μ x  (ψ0 := 0), solve linear core for Ω
        psi = fem.Function(self.Vp)  # zero
        v = self._v_field(psi)       # = -μ x
        Omega = self._solve_advec_diff(v)
        psi = self.poisson.solve_state(Omega, sigma=1.0)

        sweeps = self.max_sweeps if k_pre is None else int(k_pre)
        # Baseline residual (quick)
        base = self._quick_residual(Omega, psi)

        for it in range(sweeps):
            v = self._v_field(psi)
            Om_new = self._solve_advec_diff(v)
            # under-relax
            arr = self._vec(Omega); arr_new = self._vec(Om_new)
            arr[:] = (1.0 - self.alpha) * arr + self.alpha * arr_new
            self._ghost(Omega)

            psi = self.poisson.solve_state(Omega, sigma=1.0)
            cur = self._quick_residual(Omega, psi)
            if cur <= self.target_drop * base:
                break

        return Omega, psi

    # --- tiny weak residual (same formula as residuals_quick, but inline) ---
    def _quick_residual(self, Omega, psi):
        w = ufl.TestFunction(self.Vo)
        v = self._v_field(psi)
        L = fem.form(
            self.eps * ufl.inner(ufl.grad(Omega), ufl.grad(w)) * ufl.dx
            + ufl.inner(ufl.grad(Omega), v) * w * ufl.dx
            - Omega * w * ufl.dx
        )
        r = assemble_vector(L); r.assemble()
        return r.norm(PETSc.NormType.NORM_2)
