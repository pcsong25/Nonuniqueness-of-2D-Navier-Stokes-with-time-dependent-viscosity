# --- spaces.py ---
from dolfinx import fem

def create_spaces(mesh):
    """
    Version-robust CG1 spaces:
    - Scalars: prefer fem.functionspace (newer), fall back to fem.FunctionSpace.
    - No vector space here (we'll build velocity components as scalars in viz).
    """
    try:
        Vpsi   = fem.functionspace(mesh, ("Lagrange", 1))
        Vomega = fem.functionspace(mesh, ("Lagrange", 1))
    except AttributeError:
        Vpsi   = fem.FunctionSpace(mesh, ("Lagrange", 1))
        Vomega = fem.FunctionSpace(mesh, ("Lagrange", 1))
    return {"Vpsi": Vpsi, "Vomega": Vomega}








