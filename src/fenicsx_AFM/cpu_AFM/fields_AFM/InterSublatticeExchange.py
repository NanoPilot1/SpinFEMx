import numpy as np
from dolfinx import fem
from mpi4py import MPI


class InterSublatticeExchangeField:
    """
    Standalone local antiferromagnetic coupling between two continuum
    sublattices.

    This auxiliary operator is useful when local coupling is assembled as a
    separate field.  

    Energy density
    --------------
        e_AF = J_af * (1 + m1 dot m2),  J_af > 0

    Effective fields
    ----------------
        H1_AF = -J_af / (mu0 Ms1) * m2
        H2_AF = -J_af / (mu0 Ms2) * m1

    J_af is an energy density in J/m^3. It is not the intralattice
    exchange stiffness A measured in J/m.
    """

    def __init__(self, mesh, V, J_af, Ms1, Ms2, VolN, coordinate_scale_m=1e-9):
        self.mesh = mesh
        self.comm = mesh.comm
        self.V = V
        self.J_af = float(J_af)
        self.Ms1 = float(Ms1)
        self.Ms2 = float(Ms2)
        self.mu_0 = 4.0 * np.pi * 1e-7
        self.coordinate_scale_m = float(coordinate_scale_m)

        if self.J_af <= 0.0:
            raise ValueError("J_af must be positive for AFM coupling.")
        if self.Ms1 <= 0.0 or self.Ms2 <= 0.0:
            raise ValueError("Ms1 and Ms2 must be positive.")

        self.c1 = self.J_af / (self.mu_0 * self.Ms1)
        self.c2 = self.J_af / (self.mu_0 * self.Ms2)

        self.H1_af = fem.Function(V, name="H1_inter_sublattice")
        self.H2_af = fem.Function(V, name="H2_inter_sublattice")

        imap = V.dofmap.index_map
        bs = V.dofmap.index_map_bs
        self.owned_scalar_size = int(bs * imap.size_local)
        if self.owned_scalar_size % 3 != 0:
            raise ValueError("Expected a three-component blocked vector space.")
        self.owned_nodes = self.owned_scalar_size // 3

        vol = np.asarray(VolN, dtype=np.float64).reshape(-1)
        if vol.size != self.H1_af.x.array.size:
            raise ValueError(
                f"VolN has size {vol.size}, expected {self.H1_af.x.array.size}."
            )
        vol3 = vol.reshape((-1, 3))
        if np.any(vol3 <= 0.0):
            raise ValueError("All lumped nodal volumes must be positive.")
        self.vol_node_m3 = vol3.mean(axis=1) * self.coordinate_scale_m**3

    def compute(self, m1, m2, out1=None, out2=None):
        if out1 is None:
            out1 = self.H1_af
        if out2 is None:
            out2 = self.H2_af

        out1.x.array[:] = -self.c1 * m2.x.array
        out2.x.array[:] = -self.c2 * m1.x.array
        out1.x.scatter_forward()
        out2.x.scatter_forward()
        return out1, out2

    def Energy(self, m1, m2, include_offset=True, reduce=False):
        m1_owned = m1.x.array[: self.owned_scalar_size].reshape((-1, 3))
        m2_owned = m2.x.array[: self.owned_scalar_size].reshape((-1, 3))
        dot12 = np.einsum("ij,ij->i", m1_owned, m2_owned)
        if include_offset:
            local = self.J_af * np.sum(self.vol_node_m3[: self.owned_nodes] * (1.0 + dot12))
        else:
            local = self.J_af * np.sum(self.vol_node_m3[: self.owned_nodes] * dot12)
        local = float(local)
        if reduce:
            return float(self.comm.allreduce(local, op=MPI.SUM))
        return local

    def Energy_global(self, m1, m2, include_offset=True):
        return self.Energy(m1, m2, include_offset=include_offset, reduce=True)


__all__ = ["InterSublatticeExchangeField"]
