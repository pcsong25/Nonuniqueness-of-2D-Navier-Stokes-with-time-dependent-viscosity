from dolfinx import fem
import ufl

def l2_step(delta_vec, func_ref) -> float:
    """Relative L2 norm of step δ against current function (for Ω)."""
    # Build a temporary Function with δ values to integrate (same space)
  # AFTER (version-robust)
    tmp = fem.Function(func_ref.function_space)
    try:
        tmp.vector.array[:] = delta_vec.array
    except AttributeError:
        tmp.x.array[:] = delta_vec.array

    num = fem.assemble_scalar(fem.form(ufl.inner(tmp, tmp) * ufl.dx)) ** 0.5
    den = fem.assemble_scalar(fem.form(ufl.inner(func_ref, func_ref) * ufl.dx)) ** 0.5
    return float(num / max(den, 1e-30))


def compatibility(Omega, psi, facet_tags, outer_tag: int) -> float:
    ds = ufl.ds(domain=Omega.function_space.mesh, subdomain_data=facet_tags)
    flux = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(psi), ufl.FacetNormal(Omega.function_space.mesh)) * ds(outer_tag)))
    mass = fem.assemble_scalar(fem.form(Omega * ufl.dx))
    return float(mass - flux)