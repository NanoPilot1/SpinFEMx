import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix
from mpi4py import MPI


class ExchangeField:
    """
    Full two-sublattice AFM exchange operator for a bipartite  AFM. The continuum exchange model follows the two-sublattice formulation
described by Moreels et al., "mumax+: extensible GPU-accelerated micromagnetics and beyond", npj Computational Materials 12, 71 (2026).

    Energy

        E_ex = A11_1 * int |grad(m1)|^2 dV
             + A11_2 * int |grad(m2)|^2 dV
             + A12   * int grad(m1):grad(m2) dV
             + J_af  * int (1 + m1 dot m2) dV

    with J_af > 0 for antiferromagnetic local coupling.  Here ``J_af`` is
    the homogeneous intersublattice coupling energy density in J/m^3.  

    Effective fields

        H1 =  2*A11_1/(mu0*Ms1) * laplacian(m1)
            + A12/(mu0*Ms1)     * laplacian(m2)
            - J_af/(mu0*Ms1)    * m2

        H2 =  2*A11_2/(mu0*Ms2) * laplacian(m2)
            + A12/(mu0*Ms2)     * laplacian(m1)
            - J_af/(mu0*Ms2)    * m1

    The weak FEM formulation, combined with the tangent-space projection
    in the LLG torque, is consistent with the natural exchange boundary
    conditions for unit-length magnetization fields:

        P_perp(m1) [2*A11_1*d_n(m1) + A12*d_n(m2)] = 0
        P_perp(m2) [2*A11_2*d_n(m2) + A12*d_n(m1)] = 0

    where P_perp(m) = I - m ⊗ m.
    
    Parameters:

    mesh, V:
        DOLFINx mesh and a three-component blocked vector function space.
    A11_1, A11_2, A12:
        Exchange stiffnesses in J/m.
    J_af:
        Positive local AFM coupling density in J/m^3.
    Ms1, Ms2:
        Sublattice saturation magnetizations in A/m.
    nodal_volume:
        Vector-valued lumped nodal volumes in mesh-coordinate units cubed.
    coordinate_scale_m:
        Conversion factor from one mesh-coordinate unit to meters.
    """

    def __init__(
        self,
        mesh,
        V,
        A11_1,
        A11_2,
        A12,
        J_af,
        Ms1,
        Ms2,
        nodal_volume,
        coordinate_scale_m=1e-9,
    ):
        self.mesh = mesh
        self.comm = mesh.comm
        self.V = V
        self.A11_1 = float(A11_1)
        self.A11_2 = float(A11_2)
        self.A12 = float(A12)
        self.J_af = float(J_af)
        self.Ms1 = float(Ms1)
        self.Ms2 = float(Ms2)
        self.mu0 = 4.0 * np.pi * 1e-7
        self.coordinate_scale_m = float(coordinate_scale_m)

        if self.Ms1 <= 0.0 or self.Ms2 <= 0.0:
            raise ValueError("Ms1 and Ms2 must be positive.")
        if self.J_af <= 0.0:
            raise ValueError("J_af must be positive for AFM local coupling.")
        if self.coordinate_scale_m <= 0.0:
            raise ValueError("coordinate_scale_m must be positive.")

        v = ufl.TestFunction(V)
        u = ufl.TrialFunction(V)
        form_grad = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        self.K_grad = assemble_matrix(fem.form(form_grad))
        self.K_grad.assemble()

        vol = np.asarray(nodal_volume, dtype=np.float64).reshape(-1)
        probe = fem.Function(V)
        if vol.size != probe.x.array.size:
            raise ValueError(
                f"nodal_volume has size {vol.size}, expected {probe.x.array.size}."
            )
        if np.any(vol <= 0.0):
            raise ValueError("All lumped nodal volumes must be positive.")

        # K_grad/V has units 1/L_mesh^2. Convert to 1/m^2.
        inv_vol_scale2 = 1.0 / vol / (self.coordinate_scale_m**2)
        self.scale11_1 = -2.0 * self.A11_1 / (self.mu0 * self.Ms1) * inv_vol_scale2
        self.scale11_2 = -2.0 * self.A11_2 / (self.mu0 * self.Ms2) * inv_vol_scale2
        self.scale12_1 = -self.A12 / (self.mu0 * self.Ms1) * inv_vol_scale2
        self.scale12_2 = -self.A12 / (self.mu0 * self.Ms2) * inv_vol_scale2

        self.c1 = self.J_af / (self.mu0 * self.Ms1)
        self.c2 = self.J_af / (self.mu0 * self.Ms2)

        self.tmp1 = fem.Function(V, name="exchange_grad_tmp1")
        self.tmp2 = fem.Function(V, name="exchange_grad_tmp2")
        self.H1_ex = fem.Function(V, name="H1_exchange_total")
        self.H2_ex = fem.Function(V, name="H2_exchange_total")

        imap = V.dofmap.index_map
        bs = V.dofmap.index_map_bs
        if bs != 3:
            raise ValueError("Expected a three-component blocked vector space.")
        self.owned_scalar_size = int(bs * imap.size_local)
        self.owned_nodes = int(imap.size_local)
        self.vol_node_m3 = (
            vol.reshape((-1, 3)).mean(axis=1) * self.coordinate_scale_m**3
        )


    def compute(self, m1, m2, out1=None, out2=None):
        """Compute the complete exchange fields for both sublattices."""
        if out1 is None:
            out1 = self.H1_ex
        if out2 is None:
            out2 = self.H2_ex

        m1.x.scatter_forward()
        m2.x.scatter_forward()
        self.K_grad.mult(m1.x.petsc_vec, self.tmp1.x.petsc_vec)
        self.K_grad.mult(m2.x.petsc_vec, self.tmp2.x.petsc_vec)
        self.tmp1.x.scatter_forward()
        self.tmp2.x.scatter_forward()

        t1 = self.tmp1.x.array
        t2 = self.tmp2.x.array
        out1.x.array[:] = self.scale11_1 * t1 + self.scale12_1 * t2 - self.c1 * m2.x.array
        out2.x.array[:] = self.scale11_2 * t2 + self.scale12_2 * t1 - self.c2 * m1.x.array
        out1.x.scatter_forward()
        out2.x.scatter_forward()
        return out1, out2

    def apply_variation(self, v1, v2, out1=None, out2=None):
        """Apply dH_ex/dm to a two-sublattice direction (v1, v2)."""
        return self.compute(v1, v2, out1=out1, out2=out2)

    def Energy_terms(self, m1, m2, include_local_offset=True, reduce=False):
        """Return the exchange-energy decomposition in joules."""
        e11_1 = fem.assemble_scalar(
            fem.form(ufl.inner(ufl.grad(m1), ufl.grad(m1)) * ufl.dx)
        )
        e11_2 = fem.assemble_scalar(
            fem.form(ufl.inner(ufl.grad(m2), ufl.grad(m2)) * ufl.dx)
        )
        e12 = fem.assemble_scalar(
            fem.form(ufl.inner(ufl.grad(m1), ufl.grad(m2)) * ufl.dx)
        )

        E_ex11_1 = float(self.A11_1 * self.coordinate_scale_m * e11_1)
        E_ex11_2 = float(self.A11_2 * self.coordinate_scale_m * e11_2)
        E_ex12 = float(self.A12 * self.coordinate_scale_m * e12)

        m1_owned = m1.x.array[: self.owned_scalar_size].reshape((-1, 3))
        m2_owned = m2.x.array[: self.owned_scalar_size].reshape((-1, 3))
        dot12 = np.einsum("ij,ij->i", m1_owned, m2_owned)
        if include_local_offset:
            density_factor = 1.0 + dot12
        else:
            density_factor = dot12
        E_ex0 = float(
            self.J_af * np.sum(self.vol_node_m3[: self.owned_nodes] * density_factor)
        )

        if reduce:
            E_ex11_1 = float(self.comm.allreduce(E_ex11_1, op=MPI.SUM))
            E_ex11_2 = float(self.comm.allreduce(E_ex11_2, op=MPI.SUM))
            E_ex12 = float(self.comm.allreduce(E_ex12, op=MPI.SUM))
            E_ex0 = float(self.comm.allreduce(E_ex0, op=MPI.SUM))

        return {
            "E_ex11_1": E_ex11_1,
            "E_ex11_2": E_ex11_2,
            "E_ex12": E_ex12,
            "E_ex0": E_ex0,
            "E_exchange_total": E_ex11_1 + E_ex11_2 + E_ex12 + E_ex0,
        }

    def Energy_terms_global(self, m1, m2, include_local_offset=True):
        return self.Energy_terms(
            m1,
            m2,
            include_local_offset=include_local_offset,
            reduce=True,
        )

    def Energy_global(self, m1, m2, include_local_offset=True):
        return self.Energy_terms_global(
            m1, m2, include_local_offset=include_local_offset
        )["E_exchange_total"]

    def diagonal_blocks_owned(self):
        """
        Return the nodal diagonal approximation used by the local 6x6 PC.

        The off-diagonal sublattice blocks include both the diagonal of the
        spatial A12 operator and the exact local homogeneous AFM coupling.
        """
        diagK = self.K_grad.getDiagonal().getArray(readonly=True).copy()
        s = slice(0, self.owned_scalar_size)
        diag11 = (self.scale11_1[s] * diagK).reshape((-1, 3))
        diag22 = (self.scale11_2[s] * diagK).reshape((-1, 3))
        diag12 = (self.scale12_1[s] * diagK - self.c1).reshape((-1, 3))
        diag21 = (self.scale12_2[s] * diagK - self.c2).reshape((-1, 3))
        return diag11, diag22, diag12, diag21


__all__ = ["ExchangeField"]
