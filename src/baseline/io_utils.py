import os
from dolfinx.io import XDMFFile

def write_xdmf(path, fields: dict, mesh):
    """
    Write mesh and a dict of dolfinx Functions to XDMF, force-removing any
    existing XDMF/HDF5 so HDF5 doesn’t complain “name already exists”.
    """
    # Remove previous files with same basename (XDMF + heavy-data HDF5)
    base, ext = os.path.splitext(path)
    h5_path = base + ".h5"
    for p in (path, h5_path):
        try:
            # Only rank 0 actually removes; other ranks ignore (works for COMM_SELF/serial)
            if mesh.comm.rank == 0 and os.path.exists(p):
                os.remove(p)
        except OSError:
            pass
    # Small barrier so ranks agree files are gone if you ever run with MPI
    try:
        mesh.comm.barrier()
    except Exception:
        pass

    # Write fresh file
    with XDMFFile(mesh.comm, path, "w") as xdmf:
        xdmf.write_mesh(mesh)
        for name, f in fields.items():
            # Make sure the dataset name is unique
            try:
                f.name = name
            except Exception:
                pass
            xdmf.write_function(f)
