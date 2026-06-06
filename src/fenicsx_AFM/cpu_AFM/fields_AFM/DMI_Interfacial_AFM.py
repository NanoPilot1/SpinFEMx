import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI


class DMIInterfacialField:
    """
    Interfacial Dzyaloshinskii-Moriya interaction for one AFM sublattice.

    D is given in J/m^2. n0 is the local interface-normal field in
    the same local/ghost ordering as a three-component P1 Function.

    For a flat film, n0 is normally constant. For a curved finite-thickness
    shell, n0 may vary nodally.
    """

    def __init__(
        self,
        mesh,
        V,
        D,
        n0,
        Ms,
        Nodal_V,
        coordinate_scale_m=1e-9,
        normalize_normal=True,
    ):
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

        self.n = fem.Function(V, name="interface_normal")
        n0_arr = np.asarray(n0, dtype=np.float64).reshape(-1)
        if n0_arr.size != self.n.x.array.size:
            raise ValueError(
                f"n0 has size {n0_arr.size}, expected {self.n.x.array.size}."
            )
        self.n.x.array[:] = n0_arr

        if normalize_normal:
            n_arr = self.n.x.array.reshape((-1, 3))
            norms = np.linalg.norm(n_arr, axis=1)
            if np.any(norms <= 1e-300):
                raise ValueError("n0 contains a zero-length interface normal.")
            n_arr[:, :] /= norms[:, None]
        self.n.x.scatter_forward()

        v = ufl.TestFunction(V)
        u = ufl.TrialFunction(V)

        div_u = ufl.div(u)
        n_dot_u = ufl.dot(self.n, u)
        H_D = div_u * self.n - ufl.grad(n_dot_u)

        facet_normal = ufl.FacetNormal(mesh)
        boundary_term = ufl.cross(ufl.cross(self.n, facet_normal), u)

        Kform = (
            ufl.inner(H_D, v) * ufl.dx
            + 0.5 * ufl.inner(boundary_term, v) * ufl.ds
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

        prefactor.x.array[:] = (
            -2.0 * self.D
            / (self.mu0 * self.M_s)
            / self.coordinate_scale_m
            / nodal_volume
        )
        prefactor.x.scatter_forward()
        self.K.diagonalScale(prefactor.x.petsc_vec, None)

        self.H_DMI = fem.Function(V, name="H_dmi_interfacial")

    def compute(self, m, out=None):
        """Compute H_DMI(m) into out and return the output Function."""
        if out is None:
            out = self.H_DMI
        self.K.mult(m.x.petsc_vec, out.x.petsc_vec)
        out.x.scatter_forward()
        return out

    def Energy(self, m, reduce=False, recompute=True):
        """Return the interfacial-DMI energy in J."""
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


__all__ = ["DMIInterfacialField"]
