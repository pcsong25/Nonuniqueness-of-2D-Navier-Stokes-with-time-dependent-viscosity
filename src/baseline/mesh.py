# --- file: mesh.py ---
import gmsh
from mpi4py import MPI
from dolfinx.io import gmshio

def build_disk(radius: float, h_min: float, h_max: float, *, alg: int = 6):
    """Uniform triangular mesh on a disk. h_max controls target element size."""
    gmsh.initialize()
    gmsh.model.add("disk")

    # Geometry: full circle (no end points to size individually)
    c = gmsh.model.occ.addCircle(0.0, 0.0, 0.0, float(radius))
    loop = gmsh.model.occ.addCurveLoop([c])
    s = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.synchronize()

    # Meshing algorithm
    gmsh.option.setNumber("Mesh.Algorithm", alg)  # 6 = Frontal-Delaunay

    # Global size controls (set both legacy and new names for portability)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(h_min))
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(h_max))
    gmsh.option.setNumber("Mesh.MeshSizeMin",                float(h_min))
    gmsh.option.setNumber("Mesh.MeshSizeMax",                float(h_max))

    # Do not infer size from points/curvature (circle has no endpoints)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints",     0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature",  0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints",                 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",              0)

    # Propagate size from boundary inward
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary",              1)

    # Physical groups
    gmsh.model.addPhysicalGroup(2, [s], 1)
    gmsh.model.setPhysicalName(2, 1, "domain")
    gmsh.model.addPhysicalGroup(1, [c], 1)
    gmsh.model.setPhysicalName(1, 1, "outer")

    # Generate mesh
    gmsh.model.mesh.generate(2)

    # Convert to dolfinx
    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=2
    )

    # Stats (verify refinement changed)
    tdim = mesh.topology.dim
    n_cells = mesh.topology.index_map(tdim).size_local
    n_verts = mesh.topology.index_map(0).size_local
    print(f"[mesh stats] cells={n_cells}, vertices={n_verts}, h_max={h_max}")

    gmsh.finalize()
    return mesh, facet_tags

