# --- file: viz.py ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import matplotlib.tri as mtri
from dolfinx import fem
import ufl



def _func_array(u):
    """Return a writable 1D numpy array of a dolfinx Function, across versions."""
    try:
        return u.vector.array           # newer dolfinx
    except AttributeError:
        return u.x.array                # older dolfinx


def save_vorticity_heatmap(mesh, Omega, filename="vorticity.png"):
    """
    Triangular heatmap of Ω on the mesh (assumes CG1).
    Works for 2D meshes embedded in 2D or 3D (x,y,(z=0)).
    """
    # Ensure cell-vertex connectivity exists
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    conn = mesh.topology.connectivity(tdim, 0)

    # Triangles (gmsh disk -> 2D simplices)
    cells = conn.array.reshape(-1, 3)

    # Vertex coordinates; handle 2D/3D embedding
    X = mesh.geometry.x
    xcoord = X[:, 0]
    ycoord = X[:, 1]

    vals = _func_array(Omega)  # CG1 dof values align with vertices

    tri = mtri.Triangulation(xcoord, ycoord, triangles=cells)
    plt.figure()
    tcf = plt.tricontourf(tri, vals, levels=64)
    plt.colorbar(tcf, label=r"$\Omega$")
    plt.gca().set_aspect("equal", "box")
    plt.title("Vorticity $\\Omega$")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def save_velocity_quiver(mesh, Vscalar, psi, filename="velocity.png", stride=4):
    """
    Plot u = ∇^⊥ψ as a quiver using two scalar CG1 fields (no vector space needed).
    Robust to 2D/3D embedding of the mesh coordinates.
    """
    # Components as scalar expressions: u = (-∂y ψ, ∂x ψ)
    g = ufl.grad(psi)
    u_x_expr = -g[1]
    u_y_expr =  g[0]

    # Interpolate into the provided scalar space
    u_x = fem.Function(Vscalar)
    u_y = fem.Function(Vscalar)
    ex_x = fem.Expression(u_x_expr, Vscalar.element.interpolation_points())
    ex_y = fem.Expression(u_y_expr, Vscalar.element.interpolation_points())
    u_x.interpolate(ex_x)
    u_y.interpolate(ex_y)

    # Dof coordinates; handle flattened/2D arrays and 2D/3D embedding
    raw = Vscalar.tabulate_dof_coordinates()
    if raw.ndim == 1:
        # Infer geometry dimension from mesh coordinates
        gdim = mesh.geometry.x.shape[1]  # 2 or 3
        xy = raw.reshape(-1, gdim)
    else:
        xy = raw
        gdim = xy.shape[1]

    # Use x,y only (ignore z if present)
    xcoord = xy[:, 0]
    ycoord = xy[:, 1]

    Ux = _func_array(u_x)
    Uy = _func_array(u_y)

    # Subsample for readability
    step = max(1, int(stride))
    idx = np.arange(0, xy.shape[0], step)

    plt.figure()
    plt.quiver(xcoord[idx], ycoord[idx], Ux[idx], Uy[idx],
               angles="xy", scale_units="xy", scale=1.0, width=0.003)
    plt.gca().set_aspect("equal", "box")
    plt.title(r"Velocity $\nabla^\perp \psi$")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def save_mesh_plot(mesh, filename="mesh.png", linewidth=0.6):
        """
        Plot the triangular mesh (edges only) and save to a PNG.
        Works for 2D meshes embedded in 2D or 3D.
        """
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        import numpy as np

        # Cell -> vertex connectivity for triangles
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim, 0)
        conn = mesh.topology.connectivity(tdim, 0)
        cells = conn.array.reshape(-1, 3)  # triangles from gmsh

        # Vertex coordinates (use x,y even if embedded in 3D)
        X = mesh.geometry.x
        x = X[:, 0]
        y = X[:, 1]

        tri = mtri.Triangulation(x, y, triangles=cells)
        plt.figure()
        plt.triplot(tri, linewidth=linewidth)
        plt.gca().set_aspect("equal", "box")
        plt.title(f"Mesh: {cells.shape[0]} triangles, {X.shape[0]} vertices")
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()    



def save_streamfunction_contours(mesh, Vpsi, psi, filename="streamlines.png",
                                 n_levels=40, linewidth=1.0, cmap="viridis"):
    """
    Plot streamlines as contours of the streamfunction ψ on the CG1 mesh.
    This preserves symmetry well and avoids the clutter of quiver arrows.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
    Vpsi : fem.FunctionSpace  (scalar CG1 space used for ψ)
    psi  : fem.Function       (streamfunction)
    n_levels : int            Number of contour levels
    linewidth : float         Line width for contour lines
    cmap : str                Matplotlib colormap for filled background (optional)
    """
    # DOF coordinates for CG1 (nodal) basis
    xy = Vpsi.tabulate_dof_coordinates()
    if xy.ndim == 1:  # old layout: flat array
        gdim = 3 if xy.size % 3 == 0 else 2
        xy = xy.reshape(-1, gdim)[:, :2]
    else:
        xy = xy[:, :2]

    # Cell -> dof connectivity (triangles) in global dof indices
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    # Build once; loop is fine here (only for plotting)
    tri_dofs = np.array([Vpsi.dofmap.cell_dofs(c) for c in range(num_cells)], dtype=np.int32)

    # Values of ψ at dofs
    try:
        psi_vals = psi.vector.array
    except AttributeError:
        psi_vals = psi.x.array

    tri = Triangulation(xy[:, 0], xy[:, 1], triangles=tri_dofs)

    plt.figure(figsize=(6, 6), dpi=160)
    # Soft filled background to give structure
    cf = plt.tricontourf(tri, psi_vals, levels=n_levels, cmap=cmap, alpha=0.75)
    # Streamlines (ψ contours)
    cs = plt.tricontour(tri, psi_vals, levels=n_levels, colors="k", linewidths=linewidth, alpha=0.9)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()



