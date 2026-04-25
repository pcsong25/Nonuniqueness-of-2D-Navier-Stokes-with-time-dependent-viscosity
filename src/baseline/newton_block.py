# --- file: newton_block.py ---
import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc

# Version-robust imports for block assembly and BC ops
try:
    from dolfinx.fem.petsc import (
        assemble_matrix_block,
        assemble_vector_block,
        set_bc,
    )
except Exception:  # older layout
    from dolfinx.fem import (
        assemble_matrix_block,
        assemble_vector_block,
        set_bc,
    )


class NewtonAssembler:
    """
    Newton 2×2 block for (δΩ, δψ) with Δψ = Ω constraint (manifold Newton).

    Convention: Δψ = Ω (after integration by parts).

    Jacobian blocks:
      a11: (+ε Δ − v^k · ∇ − 1) δΩ   (Galerkin)  [+ optional SUPG term]
      a12:  (∇^⊥ δψ) · ∇Ω^k           [+ optional SUPG-consistent term]
      a21: − δΩ                        (−M)
      a22: + Δ δψ                      (+K)

    Residuals:
      L1 = −F1(Ω_k, ψ_k) [ + SUPG part if enabled ],  F1 = εΔΩ + v·∇Ω − Ω
      L2 ≡ 0  (we work on the manifold Δψ = Ω; homogeneous Neumann increment)

    Linearized boundary conditions:
      δω|∂B = 0   (preserves Ω BC set once per σ)
      ∂n(δψ)|∂B = 0  (natural), plus pin one δψ dof (gauge).
    """

    def __init__(self, Vomega, Vpsi, facet_tags, outer_tag: int, eps: float,
                 use_supg: bool = False, tau_scale: float = 1.0):
        self.Vo = Vomega
        self.Vp = Vpsi
        self.mesh = Vomega.mesh
        self.eps = float(eps)
        self.use_supg = bool(use_supg)
        self.tau_scale = float(tau_scale)

        # Boundary dofs on the outer boundary
        outer_facets = facet_tags.find(int(outer_tag))
        self.bc_o_dofs = fem.locate_dofs_topological(
            self.Vo, self.mesh.topology.dim - 1, outer_facets
        )
        psi_bdofs = fem.locate_dofs_topological(
            self.Vp, self.mesh.topology.dim - 1, outer_facets
        )
        if len(psi_bdofs) == 0:
            raise RuntimeError("No δψ boundary dofs found to pin the Neumann gauge.")
        pin = np.array([int(psi_bdofs[0])], dtype=np.int32)

        # Dirichlet BC objects (flat list expected by many builds)
        self.bc_o = fem.dirichletbc(PETSc.ScalarType(0), self.bc_o_dofs, self.Vo)  # δω|∂B=0
        self.bc_p = fem.dirichletbc(PETSc.ScalarType(0), pin, self.Vp)             # δψ(pin)=0
        self._bcs_flat = [self.bc_o, self.bc_p]

    def assemble(self, Omega_k: fem.Function, psi_k: fem.Function, mu: float):
        Vo, Vp, mesh = self.Vo, self.Vp, self.mesh

        # Trial / test
        w = ufl.TestFunction(Vo)      # test for δω
        u = ufl.TrialFunction(Vo)     # trial δω
        phi_t = ufl.TestFunction(Vp)  # test for δψ
        phi = ufl.TrialFunction(Vp)   # trial δψ

        # Frozen data at iterate
        x = ufl.SpatialCoordinate(mesh)
        gpsi = ufl.grad(psi_k)
        v = ufl.as_vector((-gpsi[1], gpsi[0])) - mu * ufl.as_vector((x[0], x[1]))
        grad_Ok = ufl.grad(Omega_k)

        eps = self.eps

        # --- Galerkin core (always present) ---
        # a11_G: (-ε Δ + v·∇ − 1) δΩ
        a11_G = (
            eps * ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx
            + ufl.inner(ufl.grad(u), v) * w * ufl.dx
            - ufl.inner(u, w) * ufl.dx
        )
        # a12_G: (∇^⊥ δψ)·∇Ω^k
        gphi = ufl.grad(phi)
        rot_phi = ufl.as_vector((-gphi[1], gphi[0]))
        a12_G = ufl.inner(rot_phi, grad_Ok) * w * ufl.dx

        # Row-1 Galerkin residual at (Ω_k, ψ_k)
        r1_core = (
            eps * ufl.inner(ufl.grad(Omega_k), ufl.grad(w)) * ufl.dx
            + ufl.inner(ufl.grad(Omega_k), v) * w * ufl.dx
            - ufl.inner(Omega_k, w) * ufl.dx
        )

        # --- Optional SUPG (guarded; never build 0 * dx) ---
        if self.use_supg:
            h   = ufl.CellDiameter(mesh)
            vmag = ufl.sqrt(ufl.inner(v, v) + 1e-16)
            tau  = self.tau_scale / ufl.sqrt((2*vmag/h)**2 + (4*eps/h**2)**2 + 1.0)

            # Jacobian SUPG contributions
            r11 = ufl.inner(ufl.grad(u), v) - u                  # linearized strong residual in δΩ
            a11_SUPG = tau * (ufl.inner(ufl.grad(w), v)) * r11 * ufl.dx
            a12_SUPG = tau * (ufl.inner(ufl.grad(w), v)) * (ufl.inner(rot_phi, grad_Ok)) * ufl.dx

            a11 = fem.form(a11_G + a11_SUPG)
            a12 = fem.form(a12_G + a12_SUPG)

            # Residual SUPG term (use advection–reaction part of strong residual)
            r1_SUPG_core = (ufl.inner(ufl.grad(Omega_k), v) - Omega_k)
            r1_SUPG = -tau * (ufl.inner(ufl.grad(w), v)) * r1_SUPG_core * ufl.dx
            L1 = fem.form(-(r1_core) + r1_SUPG)
        else:
            a11 = fem.form(a11_G)
            a12 = fem.form(a12_G)
            L1  = fem.form(-(r1_core))

        # --- Constraint row for Δ(δψ) − δΩ = 0  ---
        # Test by φ_t, integrate by parts with homogeneous Neumann for δψ:
        #   -∫ ∇δψ·∇φ_t dx  − ∫ δΩ φ_t dx = 0
        # ⇒ a21 = −M, a22 = +K, L2 ≡ 0
        a21 = fem.form((u * phi_t) * ufl.dx)
        a22 = fem.form( ufl.inner(ufl.grad(phi), ufl.grad(phi_t)) * ufl.dx )

        c0 = fem.Constant(mesh, PETSc.ScalarType(0))
        L2 = fem.form(c0 * phi_t * ufl.dx)

        # ---- Assemble monolithic AIJ via block helpers ----
        a_blocks = [[a11, a12], [a21, a22]]
        L_blocks = [L1, L2]

        # Matrix with Dirichlet BCs
        A = assemble_matrix_block(a_blocks, bcs=self._bcs_flat)
        A.assemble()

        # Vector — different dolfinx versions permute arguments
        try:
            b = assemble_vector_block(L_blocks, a_blocks, bcs=self._bcs_flat)
        except TypeError:
            b = assemble_vector_block(a_blocks, L_blocks, bcs=self._bcs_flat)

        # Enforce Dirichlet values in RHS
        set_bc(b, self._bcs_flat)

        return A, b






