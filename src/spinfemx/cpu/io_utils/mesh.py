from pathlib import Path
from mpi4py import MPI
import dolfinx


def load_mesh_xdmf(mesh_path, comm=MPI.COMM_WORLD, name="Grid"):
    """
    Load a DOLFINx mesh from XDMF and store the serial mesh path
    for later FEM/BEM or serial reconstruction routines.

    Parameters
    ----------
    mesh_path : str or pathlib.Path
        Path to the XDMF mesh file.
    comm : MPI.Comm
        MPI communicator.
    name : str
        Mesh name inside the XDMF file.

    Returns
    -------
    mesh : dolfinx.mesh.Mesh
        Loaded DOLFINx mesh.
    """
    mesh_path = Path(mesh_path).resolve()

    with dolfinx.io.XDMFFile(comm, str(mesh_path), "r") as xdmf:
        mesh = xdmf.read_mesh(name=name)

    mesh._serial_mesh_path = mesh_path

    return mesh