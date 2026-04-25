import numpy as np
from petsc4py import PETSc
from dolfinx import fem
import ufl

# Version‑robust assembly imports (support both layouts)
try:
    from dolfinx.fem.petsc import assemble_matrix, assemble_vector
except Exception:
    from dolfinx.fem import assemble_matrix, assemble_vector

class PoissonNeumann:
    """Δψ = Ω in B_Rdom with inhomogeneous Neumann: ∂nψ|∂B = b_σ(θ).
    We pin one ψ dof to fix the constant nullspace (mean(ψ)=0 gauge).
    """
    def __init__(self, mesh, Vpsi, facet_tags, outer_tag: int, Rdom: float, fourier_bc, mu: float, R: float):
        self.mesh = mesh
        self.Vpsi = Vpsi
        self.facet_tags = facet_tags
        self.outer_tag = int(outer_tag)
        self.Rdom = float(Rdom)
        self.fourier_bc = fourier_bc
        self.mu = float(mu)
        self.R = float(R)

        # Bilinear form a(ψ, v) = ∫ ∇ψ·∇v dx
        u = ufl.TrialFunction(Vpsi)
        v = ufl.TestFunction(Vpsi)
        self.vtest = v  # keep the test function to build RHS with current Ω
        self.a_form = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)

        # Boundary sampling (outer circle facets)
        self.boundary_facets = facet_tags.find(self.outer_tag)
        self.boundary_dofs = fem.locate_dofs_topological(Vpsi, mesh.topology.dim - 1, self.boundary_facets)
        xy_b = Vpsi.tabulate_dof_coordinates()[self.boundary_dofs]
        self.theta_b = np.arctan2(xy_b[:, 1], xy_b[:, 0])

        # Gauge: choose one dof to pin
        self.gauge_dof = int(self.boundary_dofs[0]) if self.boundary_dofs.size > 0 else 0

    def _assemble_rhs(self, Omega: fem.Function, sigma: float) -> PETSc.Vec:
        # Volume part: -∫ Ω v dx  (build with the *current* Ω; no coeffs= path)
        L_vol = fem.form(-Omega * self.vtest * ufl.dx)
        b = assemble_vector(L_vol)

        # Boundary Neumann load via nodal sampling of b_σ(θ)
        g_vals = self.fourier_bc.b_theta(self.theta_b, mu=self.mu, R=self.R, Rdom=self.Rdom, sigma=sigma)
        Nb = self.theta_b.size
        if Nb > 0:
            w_uniform = (2.0 * np.pi * self.Rdom) / Nb
            b.setValues(self.boundary_dofs, g_vals * w_uniform, addv=PETSc.InsertMode.ADD_VALUES)
        b.assemble()
        return b

    def solve_state(self, Omega: fem.Function, sigma: float) -> fem.Function:
        A = assemble_matrix(self.a_form); A.assemble()
        b = self._assemble_rhs(Omega, sigma)

        # Pin gauge dof (remove Neumann nullspace)
        A.zeroRows([self.gauge_dof], diag=1.0)
        b.setValue(self.gauge_dof, 0.0); b.assemble()

        # Solve A x = b
        ksp = PETSc.KSP().create(self.mesh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        x = b.duplicate(); x.set(0.0); ksp.solve(b, x)

        psi = fem.Function(self.Vpsi)
        # Version‑robust assignment (vector vs x)
        try:
            psi.vector.array[:] = x.array
            psi.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        except AttributeError:
            psi.x.array[:] = x.array
            if hasattr(psi.x, "scatter_forward"):
                psi.x.scatter_forward()
        return psi