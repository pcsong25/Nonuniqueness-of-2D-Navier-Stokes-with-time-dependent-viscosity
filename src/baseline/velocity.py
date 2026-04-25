import ufl
from dolfinx import fem

def u_from_psi(psi):
    g = ufl.grad(psi)
    return ufl.as_vector((-g[1], g[0]))

def v_from_psi_mu(psi, mu, mesh):
    x = ufl.SpatialCoordinate(mesh)
    g = ufl.grad(psi)
    u = ufl.as_vector((-g[1], g[0]))
    return u - mu * ufl.as_vector((x[0], x[1]))
