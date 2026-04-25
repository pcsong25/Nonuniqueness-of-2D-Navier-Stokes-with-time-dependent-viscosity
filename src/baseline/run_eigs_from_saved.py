# --- file: run_eigs_from_saved.py ---
import numpy as np
from pathlib import Path
from xml.etree import ElementTree as ET
import h5py
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh
from dolfinx.io import XDMFFile
from dolfinx.mesh import meshtags, locate_entities_boundary
from spaces import create_spaces
from eigs_linearized import ManifoldEigenSolver
from config import SimulationParams

comm = MPI.COMM_WORLD
rank = comm.rank

# >>>>> CHANGE THIS to the absolute path of the XDMF you want to use (the one OUTSIDE the NS folder)
IN_XDMF = Path("/Users/caltech/Peicong/results.xdmf").expanduser().resolve()
print(f"[LOAD] XDMF: {IN_XDMF}")

def load_function_from_xdmf(xdmf_path: Path, func_name: str, V: fem.FunctionSpace) -> fem.Function:
    """
    Minimal loader for CG1 fields written by dolfinx XDMFFile.write_function,
    for dolfinx builds without XDMFFile.read_function.
    """
    root = ET.parse(xdmf_path).getroot()
    # Find Grid with Name=func_name
    tgt = None
    for g in root.findall(".//Grid"):
        if g.attrib.get("Name") == func_name:
            tgt = g
            break
    if tgt is None:
        raise RuntimeError(f"Grid Name='{func_name}' not found in {xdmf_path}")
    dataitem = tgt.find(".//Attribute/DataItem[@Format='HDF']")
    if dataitem is None:
        dataitem = tgt.find(".//DataItem[@Format='HDF']")
    if dataitem is None:
        raise RuntimeError(f"HDF5 DataItem for '{func_name}' not found")

    h5ref = dataitem.text.strip()  # e.g. "results.h5:/VisualisationVector"
    h5file, h5path = h5ref.split(":", 1)
    if not h5file:  # relative path → use same folder and same stem as xdmf
        h5file = xdmf_path.with_suffix(".h5")
    else:
        h5file = (xdmf_path.parent / h5file)
    h5file = h5file.resolve()

    # Everyone reads full dataset; we’ll slice to local size on assign
    with h5py.File(h5file, "r") as H:
        vec_all = np.array(H[h5path]).reshape(-1)

    u = fem.Function(V)
    try:
        coeffs = u.vector.array
        coeffs[:] = vec_all[:coeffs.size]
        u.vector.ghostUpdate(addv=fem.petsc.PETSc.InsertMode.INSERT,
                             mode=fem.petsc.PETSc.ScatterMode.FORWARD)
    except AttributeError:
        coeffs = u.x.array
        coeffs[:] = vec_all[:coeffs.size]
        if hasattr(u.x, "scatter_forward"):
            u.x.scatter_forward()
    return u

# --- read mesh (robust to grid name differences) ---
with XDMFFile(comm, str(IN_XDMF), "r") as xf:
    try:
        mesh = xf.read_mesh(name="mesh")  # your newer files use "mesh"
    except Exception:
        try:
            mesh = xf.read_mesh()  # first grid
        except Exception:
            mesh = xf.read_mesh(xpath="/Xdmf/Domain/Grid[1]")

spaces = create_spaces(mesh)

# --- load Omega / psi by name from the same XDMF ---
Omega = load_function_from_xdmf(IN_XDMF, "Omega", spaces["Vomega"])
psi   = load_function_from_xdmf(IN_XDMF, "psi",   spaces["Vpsi"])

# --- rebuild facet tags (outer circle as tag=1). We auto-detect Rdom ---
x = mesh.geometry.x
Rdom = 10
fdim = mesh.topology.dim - 1
def on_outer(xx):
    r = np.sqrt(xx[0]**2 + xx[1]**2)
    return np.isclose(r, Rdom, atol=1e-6 * Rdom)
outer = locate_entities_boundary(mesh, fdim, on_outer)
facet_tags = meshtags(mesh, fdim, outer, np.ones(outer.size, dtype=np.int32))

# --- params: ensure mu and R match the solve that produced the XDMF ---
params = SimulationParams(mu=0.8, R=200.0, Rdom=10.0, M=32,
                              tol_newton=1e-10, max_newton=16,
                              sigma_grid=tuple(np.linspace(0.0, 1.0, 21)),
                              h_min=0.01, h_max=0.02)

# --- run eigen-solver ---
esolver = ManifoldEigenSolver(mesh, spaces, Omega, psi, params, facet_tags, outer_tag=1)

lams, modes_omega, modes_psi = esolver.solve(
        k=3, which="TARGET_REAL", target_real=-1.0,
        ncv=None, max_it=2000, tol=1e-8,
    )

print("Eigenvalues (smallest real part first):")
for j, lam in enumerate(lams):
    print(f"  #{j}: λ = {lam}")

# Optional: save modes for Paraview
# from dolfinx.io import XDMFFile as XOut
# out_file = IN_XDMF.with_name(IN_XDMF.stem + "_eigmodes.xdmf")
# if rank == 0:
#     print(f"[SAVE] modes → {out_file}")
# with XOut(comm, str(out_file), "w") as xf:
#     xf.write_mesh(mesh)
#     for j, omg in enumerate(modes_omega):
#         xf.write_function(omg, 0.0, name=f"deltaOmega_{j}")
#     for j, psj in enumerate(modes_psi):
#         xf.write_function(psj, 0.0, name=f"deltaPsi_{j}")
