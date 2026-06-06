import numpy as np
from dolfinx import fem
from mpi4py import MPI


class AnisotropyField:
    """
    Nodal/lumped uniaxial anisotropy for one AFM sublattice.

    Field
    -----
        H_ani = 2 Ku / (mu0 Ms) * (m dot n) n

    The easy-axis field ``n`` may vary from node to node.
    """

    def __init__(
        self,
        mesh,
        V,
        Ku,
        Ms,
        AniVec,
        VolN,
        normalize_axis=True,
        energy_with_offset=False,
        coordinate_scale_m=1e-9,
    ):
        self.mesh = mesh
        self.comm = mesh.comm
        self.V = V
        self.Ku = float(Ku)
        self.M_s = float(Ms)
        self.mu_0 = 4.0 * np.pi * 1e-7
        self.coordinate_scale_m = float(coordinate_scale_m)
        self.prefactor = 2.0 * self.Ku / (self.mu_0 * self.M_s)
        self.energy_with_offset = bool(energy_with_offset)

        if self.M_s <= 0.0:
            raise ValueError("Ms must be positive for each AFM sublattice.")

        self.H_anis = fem.Function(self.V, name="H_anisotropy")
        self.n = fem.Function(self.V, name="anisotropy_axis")

        ani = np.asarray(AniVec, dtype=np.float64).reshape(-1)
        if ani.size != self.n.x.array.size:
            raise ValueError(
                f"AniVec has size {ani.size}, expected {self.n.x.array.size}."
            )
        self.n.x.array[:] = ani

        n_arr = self.n.x.array.reshape((-1, 3))
        if normalize_axis:
            nrm = np.linalg.norm(n_arr, axis=1)
            if np.any(nrm <= 1e-300):
                raise ValueError("AniVec contains a zero-length easy axis.")
            n_arr[:, :] /= nrm[:, None]
        self.n.x.scatter_forward()

        imap = self.V.dofmap.index_map
        bs = self.V.dofmap.index_map_bs
        self.owned_scalar_size = int(bs * imap.size_local)
        if self.owned_scalar_size % 3 != 0:
            raise ValueError("Expected a three-component blocked vector space.")
        self.owned_nodes = self.owned_scalar_size // 3

        self.vol_node_mesh3 = self._extract_nodal_volumes(VolN)
        self.vol_node_m3 = self.vol_node_mesh3 * self.coordinate_scale_m**3

    def _extract_nodal_volumes(self, VolN):
        vol = np.asarray(VolN, dtype=np.float64).reshape(-1)
        if vol.size != self.n.x.array.size:
            raise ValueError(
                f"VolN has size {vol.size}, expected {self.n.x.array.size}."
            )
        vol3 = vol.reshape((-1, 3))
        if np.any(vol3 <= 0.0):
            raise ValueError("All lumped nodal volumes must be positive.")
        return vol3.mean(axis=1).copy()

    def compute(self, m, out=None):
        """Compute the anisotropy field into ``out`` and return it."""
        if out is None:
            out = self.H_anis

        m_arr = m.x.array.reshape((-1, 3))
        n_arr = self.n.x.array.reshape((-1, 3))
        h_arr = out.x.array.reshape((-1, 3))

        mdotn = np.einsum("ij,ij->i", m_arr, n_arr)
        h_arr[:, :] = self.prefactor * mdotn[:, None] * n_arr
        out.x.scatter_forward()
        return out

    def apply_array(self, m_flat, out_flat=None, owned_only=False):
        """Apply the linear anisotropy map directly to a NumPy array."""
        m_arr = np.asarray(m_flat, dtype=np.float64).reshape((-1, 3))
        n_arr = self.n.x.array.reshape((-1, 3))[: m_arr.shape[0]]

        if out_flat is None:
            out_arr = np.empty_like(m_arr)
        else:
            out_arr = np.asarray(out_flat, dtype=np.float64).reshape((-1, 3))

        mdotn = np.einsum("ij,ij->i", m_arr, n_arr)
        out_arr[:, :] = self.prefactor * mdotn[:, None] * n_arr

        flat = out_arr.reshape(-1)
        if owned_only:
            return flat[: self.owned_scalar_size]
        return flat

    def Energy(self, m, include_offset=None, reduce=False):
        """Return uniaxial anisotropy energy in J."""
        if include_offset is None:
            include_offset = self.energy_with_offset

        m_owned = m.x.array[: self.owned_scalar_size].reshape((-1, 3))
        n_owned = self.n.x.array[: self.owned_scalar_size].reshape((-1, 3))
        mdotn = np.einsum("ij,ij->i", m_owned, n_owned)
        vol = self.vol_node_m3[: self.owned_nodes]

        if include_offset:
            local = self.Ku * np.sum(vol * (1.0 - mdotn * mdotn))
        else:
            local = -self.Ku * np.sum(vol * mdotn * mdotn)

        local = float(local)
        if reduce:
            return float(self.comm.allreduce(local, op=MPI.SUM))
        return local

    def Energy_global(self, m, include_offset=None):
        return self.Energy(m, include_offset=include_offset, reduce=True)

    def Energy_lumped(self, m, include_offset=None, reduce=False):
        return self.Energy(m, include_offset=include_offset, reduce=reduce)

    def Energy_lumped_global(self, m, include_offset=None):
        return self.Energy(m, include_offset=include_offset, reduce=True)

    def diagonal_array(self, owned_only=True):
        """Diagonal of dH_ani/dm, mainly for diagnostics."""
        n_arr = self.n.x.array.reshape((-1, 3))
        diag = (self.prefactor * n_arr * n_arr).reshape(-1).copy()
        if owned_only:
            return diag[: self.owned_scalar_size]
        return diag

    def axis_owned(self):
        """Owned easy-axis vectors used by the local 6x6 AFM preconditioner."""
        return self.n.x.array[: self.owned_scalar_size].reshape((-1, 3)).copy()


__all__ = ["AnisotropyField"]
