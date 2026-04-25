# --- file: residuals_quick.py ---
import numpy as np
from petsc4py import PETSc
from dolfinx import fem
import ufl

# Version-robust assembly imports
try:
    from dolfinx.fem.petsc import assemble_vector
except Exception:
    from dolfinx.fem import assemble_vector


def _v_field(mesh, psi, mu: float):
    """v = ∇^⊥ psi - mu * x  (as UFL expression)."""
    x = ufl.SpatialCoordinate(mesh)
    g = ufl.grad(psi)
    rot = ufl.as_vector((-g[1], g[0]))
    return rot - mu * ufl.as_vector((x[0], x[1]))


def residuals_quick(mesh, spaces, Omega, psi, *,
                    mu: float, R: float, Rdom: float,
                    facet_tags, outer_tag: int,
                    fourier_bc, sigma: float,
                    pinned_psi_dof: int | None = None):
    """
    Fast weak residuals (raw PETSc vector norms):

      R_Omega(w) = ∫ [ eps ∇Ω·∇w + (∇Ω·v) w - Ω w ] dx
      R_psi(φ)   = ∫ ∇ψ·∇φ dx + ∫ Ω φ dx - ∫_∂B b_σ(θ) φ ds

    Returns: dict with l2 norms and the residual vectors.
    """
    Vomega, Vpsi = spaces["Vomega"], spaces["Vpsi"]
    eps = (Rdom ** 2) / (R * R)

    # ---- R_Omega ----
    w = ufl.TestFunction(Vomega)
    v = _v_field(mesh, psi, mu)
    L_O = fem.form(
        eps * ufl.inner(ufl.grad(Omega), ufl.grad(w)) * ufl.dx
        + ufl.inner(ufl.grad(Omega), v) * w * ufl.dx
        - Omega * w * ufl.dx
    )
    rOmega = assemble_vector(L_O)
    rOmega.assemble()

    # ---- R_psi (volume) ----
    phi = ufl.TestFunction(Vpsi)
    L_P_vol = fem.form(ufl.inner(ufl.grad(psi), ufl.grad(phi)) * ufl.dx + Omega * phi * ufl.dx)
    rPsi = assemble_vector(L_P_vol)

    # ---- R_psi boundary load: -∫_∂B bσ φ ds via nodal sampling (cheap & consistent)
    bfacets = facet_tags.find(int(outer_tag))
    psi_bdofs = fem.locate_dofs_topological(Vpsi, mesh.topology.dim - 1, bfacets)
    if psi_bdofs.size > 0:
        xy_b = Vpsi.tabulate_dof_coordinates()[psi_bdofs]
        theta_b = np.arctan2(xy_b[:, 1], xy_b[:, 0])
        g_vals = fourier_bc.b_theta(theta_b, mu=mu, R=R, Rdom=Rdom, sigma=float(sigma))
        w_uniform = (2.0 * np.pi * Rdom) / theta_b.size  # uniform arc-length weight
        rPsi.setValues(psi_bdofs, -g_vals * w_uniform, addv=PETSc.InsertMode.ADD_VALUES)

    # Optional: zero the pinned Neumann gauge dof (so it doesn't pollute the report)
    if pinned_psi_dof is not None:
        rPsi.setValue(int(pinned_psi_dof), 0.0)

    rPsi.assemble()

    # ---- simple norms ----
    n_rO = rOmega.norm(PETSc.NormType.NORM_2)
    n_rP = rPsi.norm(PETSc.NormType.NORM_2)

    return {
        "rOmega": rOmega,
        "rPsi": rPsi,
        "norms": {"Omega_l2": float(n_rO), "Psi_l2": float(n_rP)},
        "meta": {"eps": eps, "R": R, "mu": mu, "sigma": float(sigma), "Rdom": Rdom}
    }
