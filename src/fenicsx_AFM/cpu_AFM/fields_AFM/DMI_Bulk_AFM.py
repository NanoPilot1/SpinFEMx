import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI


class DMIBulkField:
    """
    Bulk Dzyaloshinskii-Moriya interaction for one AFM sublattice.

    Energy density
    --------------
        e_DMI = D * m dot curl(m)

    ``D`` is given in J/m^2. The same class is instantiated independently
    for m1 and m2. For a compensated AFM with the same microscopic chirality
    on both sublattices, use D1 = D2. Do not flip the sign merely because
    m2 ~= -m1: the DMI energy is invariant under m -> -m.

    The weak form mirrors the FM implementation and includes the natural
    boundary term.
    """

    def __init__(self, mesh, V, D, Ms, Nodal_V, coordinate_scale_m=1e-9):
        self.mesh = mesh
        self.comm = mesh.comm
        self.V = V
        self.D = float(D)
        self.M_s = float(Ms)
        self.mu0 = 4.0 * np.pi * 1e-7
        self.coordinate_scale_m = float(coordinate_scale_m)

        if self.M_s <= 0.0:
            raise ValueError("Ms must be positive for each AFM sublattice.")
        if self.coordinate_scale_m <= 0.0:
            raise ValueError("coordinate_scale_m must be positive.")

        v = ufl.TestFunction(V)
        u = ufl.TrialFunction(V)
        normal = ufl.FacetNormal(mesh)

        Kform = (
            -ufl.inner(ufl.curl(u), v) * ufl.dx
            + 0.5 * ufl.inner(ufl.cross(normal, u), v) * ufl.ds
        )
        self.K = fem.petsc.assemble_matrix(fem.form(Kform), bcs=[])
        self.K.assemble()

        nodal_volume = np.asarray(Nodal_V, dtype=np.float64).reshape(-1)
        prefactor = fem.Function(V)
        if nodal_volume.size != prefactor.x.array.size:
            raise ValueError(
                f"Nodal_V has size {nodal_volume.size}, expected "
                f"{prefactor.x.array.size}."
            )
        if np.any(nodal_volume <= 0.0):
            raise ValueError("All lumped nodal volumes must be positive.")

        # K / V_node contributes 1/L_mesh. Divide by coordinate_scale_m
        # to express the derivative in 1/m.
        prefactor.x.array[:] = (
            2.0 * self.D
            / (self.mu0 * self.M_s)
            / self.coordinate_scale_m
            / nodal_volume
        )
        prefactor.x.scatter_forward()
        self.K.diagonalScale(prefactor.x.petsc_vec, None)

        self.H_DMI = fem.Function(V, name="H_dmi_bulk")

    def compute(self, m, out=None):
        """Compute H_DMI(m) into ``out`` and return the output Function."""
        if out is None:
            out = self.H_DMI
        self.K.mult(m.x.petsc_vec, out.x.petsc_vec)
        out.x.scatter_forward()
        return out

    def Energy(self, m, reduce=False, recompute=True):
        """Return the bulk-DMI energy in J."""
        if recompute:
            self.compute(m, out=self.H_DMI)

        local = fem.assemble_scalar(
            fem.form(ufl.inner(m, self.H_DMI) * ufl.dx)
        )
        local = (
            -0.5
            * self.mu0
            * self.M_s
            * float(local)
            * self.coordinate_scale_m**3
        )
        if reduce:
            return float(self.comm.allreduce(local, op=MPI.SUM))
        return float(local)

    def Energy_global(self, m, recompute=True):
        return self.Energy(m, reduce=True, recompute=recompute)

    def diagonal_array(self, owned_only=True):
        """Return the local matrix diagonal used by the block-Jacobi PC."""
        diag = self.K.getDiagonal().getArray(readonly=True).copy()
        return diag


__all__ = ["DMIBulkField"]
