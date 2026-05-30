"""
GPU nodal/lumped uniaxial anisotropy field.

    H_ani_i = 2 Ku/(mu0 Ms) * (m_i cdot n_i) n_i

and the coherent lumped nodal energy

    E_ani = - Ku * sum_i V_i (m_i cdot n_i)^2

or, with include_offset=True,

    E_ani = Ku * sum_i V_i [1 - (m_i cdot n_i)^2].

"""

from __future__ import annotations

import numpy as np
import cupy as cp
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem


_ANISOTROPY_NODAL_KERNEL = cp.ElementwiseKernel(
    "float64 mx, float64 my, float64 mz, "
    "float64 nx, float64 ny, float64 nz, "
    "float64 pref",
    "float64 hx, float64 hy, float64 hz",
    """
    double mdotn = mx*nx + my*ny + mz*nz;
    double c = pref * mdotn;
    hx = c * nx;
    hy = c * ny;
    hz = c * nz;
    """,
    name="anisotropy_nodal_field",
)


_ANISOTROPY_DIAG_KERNEL = cp.ElementwiseKernel(
    "float64 nx, float64 ny, float64 nz, float64 pref",
    "float64 dx, float64 dy, float64 dz",
    """
    dx = pref * nx * nx;
    dy = pref * ny * ny;
    dz = pref * nz * nz;
    """,
    name="anisotropy_nodal_diag",
)


def _set_vec_cuda(vec: PETSc.Vec, block_size: int | None = None) -> PETSc.Vec:
    try:
        vec.setType(PETSc.Vec.Type.CUDA)
    except Exception:
        pass

    if block_size is not None:
        try:
            vec.setBlockSize(block_size)
        except Exception:
            pass

    try:
        vec.bindToCPU(False)
    except Exception:
        pass

    return vec


def _vec_to_cupy(vec: PETSc.Vec, mode: str = "rw") -> cp.ndarray:
    return cp.from_dlpack(vec.toDLPack(mode))


class AnisotropyField:
    """
    Nodal/lumped GPU uniaxial anisotropy field.

    Public API intentionally keeps the old constructor and compute/Energy names:

        AnisotropyField(mesh, V, Ku, Ms, AniVec, VolN)
        compute(m_gpu) -> dolfinx.fem.Function
        compute_vec(m_gpu, out_gpu) -> PETSc.Vec
        Energy(m_fun) -> float
        Energy_lumped_gpu(m_gpu) -> float
        diagonal_vec(out_gpu) -> PETSc.Vec

    This class does NOT expose a `.K` PETSc matrix. The llg_module must treat it
    as a local/non-matrix term.
    """

    is_matrix_free = True

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

        ani = np.asarray(AniVec, dtype=np.float64).reshape(-1)
        if ani.size != self.n.x.array.size:
            raise ValueError(
                f"AniVec has size {ani.size}, but V.x.array has size {self.n.x.array.size}. "
                "Pass AniVec in the same ordering as the vector function space."
            )

        self.n.x.array[:] = ani
        n_view = self.n.x.array.reshape((-1, 3))

        if normalize_axis:
            nrm = np.linalg.norm(n_view, axis=1)
            nrm = np.maximum(nrm, 1e-300)
            n_view[:, :] = n_view / nrm[:, None]

        self.n.x.scatter_forward()

        self.n_cpu = np.asarray(n_view, dtype=np.float64).copy()
        self.n_gpu = cp.asarray(self.n_cpu)

        imap = self.V.dofmap.index_map
        bs = self.V.dofmap.index_map_bs
        self.owned_scalar_size = int(bs * imap.size_local)
        self.owned_nodes = self.owned_scalar_size // 3

        vol = np.asarray(VolN, dtype=np.float64).reshape(-1)
        if vol.size != self.n.x.array.size:
            raise ValueError(
                f"VolN has size {vol.size}, but V.x.array has size {self.n.x.array.size}. "
                "Pass the vector-valued lumped volume array in the same ordering as V.x.array."
            )

        # VolN is vector-valued and usually repeats the same nodal volume three times.
        self.vol_node_nm3 = vol.reshape((-1, 3)).mean(axis=1).copy()
        self.vol_node_m3 = self.vol_node_nm3 * 1e-27
        self.vol_gpu_m3 = cp.asarray(self.vol_node_m3)

        self.h_gpu = self.H_anis.x.petsc_vec.duplicate()
        _set_vec_cuda(self.h_gpu, block_size=3)
        self.h_gpu.zeroEntries()

    def compute_vec(self, m_gpu: PETSc.Vec, out_gpu: PETSc.Vec) -> PETSc.Vec:
        """Compute H_ani(m_gpu) into out_gpu, both PETSc CUDA vectors."""
        m_all = _vec_to_cupy(m_gpu, "r")
        h_all = _vec_to_cupy(out_gpu, "rw")

        n_nodes = min(m_all.size, h_all.size) // 3
        n_nodes = min(n_nodes, self.n_gpu.shape[0])
        n_scalar = 3 * n_nodes

        m = m_all[:n_scalar].reshape((-1, 3))
        h = h_all[:n_scalar].reshape((-1, 3))
        n = self.n_gpu[:n_nodes]

        _ANISOTROPY_NODAL_KERNEL(
            m[:, 0], m[:, 1], m[:, 2],
            n[:, 0], n[:, 1], n[:, 2],
            float(self.prefactor),
            h[:, 0], h[:, 1], h[:, 2],
        )

        if h_all.size > n_scalar:
            h_all[n_scalar:] = 0.0

        return out_gpu

    def compute(self, m_gpu: PETSc.Vec) -> fem.Function:
        """
        Compatibility wrapper matching the old Anisotropy_GPU.compute API.
        """
        self.compute_vec(m_gpu, self.h_gpu)
        self.h_gpu.copy(self.H_anis.x.petsc_vec)
        self.H_anis.x.scatter_forward()
        return self.H_anis

    def diagonal_vec(self, out_gpu: PETSc.Vec) -> PETSc.Vec:
        """
        Put the pointwise diagonal of dH_ani/dm into out_gpu.

        Since H_i = pref * (n_i n_i^T) m_i, the diagonal entries are
        pref * (nx^2, ny^2, nz^2). This is what a PETSc Mat.getDiagonal()
        would return for the equivalent block-diagonal nodal operator.
        """
        out_all = _vec_to_cupy(out_gpu, "rw")
        n_nodes = min(out_all.size // 3, self.n_gpu.shape[0])
        n_scalar = 3 * n_nodes

        out = out_all[:n_scalar].reshape((-1, 3))
        n = self.n_gpu[:n_nodes]

        _ANISOTROPY_DIAG_KERNEL(
            n[:, 0], n[:, 1], n[:, 2],
            float(self.prefactor),
            out[:, 0], out[:, 1], out[:, 2],
        )

        if out_all.size > n_scalar:
            out_all[n_scalar:] = 0.0

        return out_gpu

    def jac_times_vec(self, m_state_vec: PETSc.Vec, v_vec: PETSc.Vec, out_vec: PETSc.Vec) -> PETSc.Vec:
        """
        Jacobian-vector product for anisotropy.

        The uniaxial anisotropy field is linear in m for fixed n, so J v = H_ani(v).
        m_state_vec is unused and kept only for API compatibility.
        """
        return self.compute_vec(v_vec, out_vec)

    def Energy(self, m: fem.Function, include_offset: bool | None = None) -> float:
        """Host-side nodal/lumped uniaxial anisotropy energy in Joule."""
        if include_offset is None:
            include_offset = self.energy_with_offset

        m_owned = np.asarray(
            m.x.array[: self.owned_scalar_size],
            dtype=np.float64,
        ).reshape((-1, 3))
        n_owned = self.n_cpu[: self.owned_nodes]
        vol = self.vol_node_m3[: self.owned_nodes]

        mdotn = np.einsum("ij,ij->i", m_owned, n_owned)

        if include_offset:
            local_E = self.Ku * np.sum(vol * (1.0 - mdotn * mdotn))
        else:
            local_E = -self.Ku * np.sum(vol * mdotn * mdotn)

        return float(self.comm.allreduce(float(local_E), op=MPI.SUM))

    def Energy_lumped_gpu(self, m_gpu: PETSc.Vec, include_offset: bool | None = None) -> float:
        """GPU-side nodal/lumped uniaxial anisotropy energy in Joule."""
        if include_offset is None:
            include_offset = self.energy_with_offset

        m_all = _vec_to_cupy(m_gpu, "r")
        n_nodes = min(m_all.size // 3, self.n_gpu.shape[0], self.vol_gpu_m3.size)

        m = m_all[: 3 * n_nodes].reshape((-1, 3))
        n = self.n_gpu[:n_nodes]
        vol = self.vol_gpu_m3[:n_nodes]

        mdotn = cp.sum(m * n, axis=1)

        if include_offset:
            local_E = self.Ku * cp.sum(vol * (1.0 - mdotn * mdotn))
        else:
            local_E = -self.Ku * cp.sum(vol * mdotn * mdotn)

        return float(self.comm.allreduce(float(local_E.get()), op=MPI.SUM))

    # Backward-compatible alias used in some experiments.
    Energy_gpu = Energy_lumped_gpu


__all__ = ["AnisotropyField"]
