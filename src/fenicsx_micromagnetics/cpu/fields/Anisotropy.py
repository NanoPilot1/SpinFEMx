import numpy as np
from dolfinx import fem
from mpi4py import MPI


class AnisotropyField:
    """
    Nodal/lumped uniaxial anisotropy field for finite-element micromagnetics.

        H_ani_i = 2 Ku/(mu0 Ms) * (m_i cdot n_i) n_i,

    instead of assembling a consistent FEM mass-like anisotropy matrix.

    """

    def __init__(
        self,
        mesh,
        V,
        Ku,
        Ms,
        AniVec,
        VolN,
        normalize_axis: bool = True,
        energy_with_offset: bool = False,
    ):
        self.mesh = mesh
        self.comm = mesh.comm
        self.V = V
        self.Ku = float(Ku)
        self.M_s = float(Ms)
        self.mu_0 = 4.0 * np.pi * 1e-7
        self.prefactor = 2.0 * self.Ku / (self.mu_0 * self.M_s)
        self.energy_with_offset = bool(energy_with_offset)

        self.H_anis = fem.Function(self.V)
        self.n = fem.Function(self.V)

        # Store easy-axis field in the same local/ghost layout as V.
        ani = np.asarray(AniVec, dtype=np.float64).reshape(-1).copy()
        if ani.size != self.n.x.array.size:
            raise ValueError(
                f"AniVec has size {ani.size}, but V local array has size "
                f"{self.n.x.array.size}. Pass AniVec in the same ordering as V.x.array."
            )
        self.n.x.array[:] = ani

        n_arr = self.n.x.array.reshape((-1, 3))
        if normalize_axis:
            nrm = np.linalg.norm(n_arr, axis=1)
            nrm = np.maximum(nrm, 1e-300)
            n_arr[:, :] = n_arr / nrm[:, None]
        self.n.x.scatter_forward()

        # Owned scalar/vector sizes. For vector P1 in the ordering used here,
        # reshape((-1, 3)) gives nodal vectors.
        imap = self.V.dofmap.index_map
        bs = self.V.dofmap.index_map_bs
        self.owned_scalar_size = int(bs * imap.size_local)
        if self.owned_scalar_size % 3 != 0:
            raise ValueError(
                "Expected a 3-component vector function space with scalar size divisible by 3."
            )
        self.owned_nodes = self.owned_scalar_size // 3

        self.vol_node_nm3 = self._extract_nodal_volumes(VolN)
        self.vol_node_m3 = self.vol_node_nm3 * 1e-27

    def _extract_nodal_volumes(self, VolN):
        vol = np.asarray(VolN, dtype=np.float64).reshape(-1)
        ndofs = self.n.x.array.size
        if vol.size != ndofs:
            raise ValueError(
                f"VolN has size {vol.size}, but V local array has size {ndofs}. "
                "Pass the vector-valued lumped volume array in the same ordering as V.x.array."
            )
        vol3 = vol.reshape((-1, 3))

        # In the usual construction, the three component entries are identical.
        # Average them to be robust to tiny roundoff differences.
        return vol3.mean(axis=1).copy()

    def compute(self, m):
        """
        Compute the nodal anisotropy field for a host-side dolfinx Function m.

        Returns
        -------
        dolfinx.fem.Function
            H_anis in A/m.
        """
        m_arr = m.x.array.reshape((-1, 3))
        n_arr = self.n.x.array.reshape((-1, 3))
        h_arr = self.H_anis.x.array.reshape((-1, 3))

        mdotn = np.einsum("ij,ij->i", m_arr, n_arr)
        h_arr[:, :] = self.prefactor * mdotn[:, None] * n_arr

        self.H_anis.x.scatter_forward()
        return self.H_anis

    def Energy(self, m, include_offset=None, reduce: bool = False):
        """
        Compute nodal/lumped uniaxial anisotropy energy in Joule.

        
        include_offset:
            If False, return the no-offset minimization energy

                E = -Ku sum_i V_i (m_i cdot n_i)^2.

            If True, return the physical offset form

                E = Ku sum_i V_i [1 - (m_i cdot  n_i)^2].

        """
        if include_offset is None:
            include_offset = self.energy_with_offset

        m_owned = m.x.array[: self.owned_scalar_size].reshape((-1, 3))
        n_owned = self.n.x.array[: self.owned_scalar_size].reshape((-1, 3))
        vol = self.vol_node_m3[: self.owned_nodes]

        mdotn = np.einsum("ij,ij->i", m_owned, n_owned)
        if include_offset:
            local_E = self.Ku * np.sum(vol * (1.0 - mdotn * mdotn))
        else:
            local_E = -self.Ku * np.sum(vol * mdotn * mdotn)

        local_E = float(local_E)
        if reduce:
            return float(self.comm.allreduce(local_E, op=MPI.SUM))
        return local_E

    def Energy_global(self, m, include_offset=None):
        """Return the global MPI sum of the nodal anisotropy energy."""
        return self.Energy(m, include_offset=include_offset, reduce=True)

    def Energy_lumped(self, m, include_offset=None, reduce: bool = False):
        """Alias for Energy(m), kept for compatibility with minimizers."""
        return self.Energy(m, include_offset=include_offset, reduce=reduce)

    def Energy_lumped_global(self, m, include_offset=None):
        """Return the global MPI sum of the lumped anisotropy energy."""
        return self.Energy(m, include_offset=include_offset, reduce=True)

    def diagonal_array(self, owned_only: bool = True):
        """
        Return the diagonal of the nodal linear map dH_ani/dm.

        For H_i = prefactor * (n_i n_i^T) m_i, the diagonal entries are
        prefactor*(n_x^2, n_y^2, n_z^2).
        """
        n_arr = self.n.x.array.reshape((-1, 3))
        diag = self.prefactor * n_arr * n_arr
        flat = diag.reshape(-1).copy()
        if owned_only:
            return flat[: self.owned_scalar_size]
        return flat

    def apply_array(self, m_flat, out_flat=None, owned_only: bool = False):

        m_arr = np.asarray(m_flat, dtype=np.float64).reshape((-1, 3))
        n_arr = self.n.x.array.reshape((-1, 3))[: m_arr.shape[0]]

        if out_flat is None:
            out_arr = np.empty_like(m_arr)
        else:
            out_arr = np.asarray(out_flat).reshape((-1, 3))

        mdotn = np.einsum("ij,ij->i", m_arr, n_arr)
        out_arr[:, :] = self.prefactor * mdotn[:, None] * n_arr

        flat = out_arr.reshape(-1)
        if owned_only:
            return flat[: self.owned_scalar_size]
        return flat


__all__ = ["AnisotropyField"]

