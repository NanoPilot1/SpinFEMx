import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

import numpy as np
import cupy as cp
import jax
import jax.numpy as jnp

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
import ufl

from jaxfmm.strayfield import init_strayfield, eval_strayfield

jax.config.update("jax_enable_x64", True)


def cupy_to_jax(a_cp):
    """
    CuPy array -> JAX array using DLPack, without copy host.
    """
    try:
        return jax.dlpack.from_dlpack(a_cp)
    except TypeError:
        return jax.dlpack.from_dlpack(a_cp.toDlpack())


def jax_to_cupy(a_jax):
    """
    JAX array -> CuPy array using DLPack, without copy host.
    """
    try:
        return cp.from_dlpack(a_jax)
    except TypeError:
        return cp.from_dlpack(jax.dlpack.to_dlpack(a_jax))



"""
GPU demagnetizing field using the external jaxfmm library.

This module evaluates the demagnetizing field with jaxfmm
(https://gitlab.com/jaxfmm/jaxfmm/-/tree/strayfield/jaxfmm) and uses
JAX/CuPy DLPack interoperability to exchange data with PETSc CUDA vectors.

The jaxfmm library is used as an external backend; this module provides the
DOLFINx/PETSc-CUDA interface required by the micromagnetic solver.

Conventions
-----------
- Mesh coordinates are assumed to be in nm.
- Execution is intended for serial GPU runs with one MPI rank.
- compute_vec(m_vec, out_vec) expects PETSc.Vec CUDA vectors.
- Energy_lumped_gpu(m_vec, H_vec) evaluates the demagnetizing energy directly
  on the GPU using lumped nodal volumes.
"""


class DemagFieldFMMJAXGPU:
    """
    Demagnetizing field computed with jaxfmm on the GPU.

    The class converts PETSc CUDA vectors to CuPy/JAX arrays using DLPack,
    evaluates the stray field with jaxfmm, and writes the result back into a
    PETSc CUDA vector.

    This implementation is intended for single-rank GPU execution and should
    not be added to the sparse linear_terms list used by the LLG solvers.
    """

    def __init__(self, domain_mesh, V, V1, Ms, VolN, mem_limit=2_000_000):
        self.mesh = domain_mesh
        self.V = V
        self.V1 = V1
        self.Ms = float(Ms)
        self.mu0 = 4.0 * np.pi * 1e-7
        self.comm = self.mesh.comm

        if self.comm.size != 1:
            raise RuntimeError(
                "DemagFieldFMMJAXGPU work in single core"
            )

        self.H_d = fem.Function(self.V)

        self.start, self.end = self.V.dofmap.index_map.local_range
        self.local_dofs = self.end - self.start
        self.local_size = 3 * self.local_dofs


        vol = np.asarray(VolN[: self.local_size], dtype=np.float64).reshape((-1, 3))
        self.vol_nodes_cp = cp.asarray(vol[:, 0])


        self.X = np.asarray(self.mesh.geometry.x, dtype=np.float64)

        tdim = self.mesh.topology.dim
        self.mesh.topology.create_connectivity(tdim, 0)
        conn = self.mesh.topology.connectivity(tdim, 0)

        ncells = self.mesh.topology.index_map(tdim).size_local
        nvpc = len(conn.links(0))
        

        self.cells = np.empty((ncells, nvpc), dtype=np.int64)
        for c in range(ncells):
            self.cells[c, :] = np.asarray(conn.links(c), dtype=np.int64)

        Msat = jnp.array([self.Ms], dtype=jnp.float64)

        self.tree_info = init_strayfield(
            self.X,
            self.cells,
            Msat,
            mem_limit=mem_limit,
        )

        self.H_gpu = None

    def compute_vec(self, m_vec: PETSc.Vec, out_vec: PETSc.Vec):

        m_cp_all = cp.from_dlpack(m_vec.toDLPack("r"))
        out_cp_all = cp.from_dlpack(out_vec.toDLPack("rw"))

        m_cp = m_cp_all[: self.local_size].reshape((-1, 3))
        out_cp = out_cp_all[: self.local_size].reshape((-1, 3))

        m_jax = cupy_to_jax(m_cp)

        H_jax = eval_strayfield(m_jax, **self.tree_info)

        H_cp = jax_to_cupy(H_jax)

        out_cp[:, :] = H_cp

        if out_cp_all.size > self.local_size:
            out_cp_all[self.local_size:] = 0.0

        self.H_gpu = out_vec
        return out_vec

    def copy_to_function(self, H_vec: PETSc.Vec):

        H_vec.copy(self.H_d.x.petsc_vec)
        self.H_d.x.scatter_forward()
        return self.H_d

    def Energy_lumped_gpu(self, m_vec: PETSc.Vec, H_vec: PETSc.Vec):

        m_cp_all = cp.from_dlpack(m_vec.toDLPack("r"))
        H_cp_all = cp.from_dlpack(H_vec.toDLPack("r"))

        m_cp = m_cp_all[: self.local_size].reshape((-1, 3))
        H_cp = H_cp_all[: self.local_size].reshape((-1, 3))

        mdH = cp.sum(m_cp * H_cp, axis=1)
        val = cp.sum(self.vol_nodes_cp * mdH)

        return float((-0.5 * self.mu0 * self.Ms * val * 1e-27).item())

    def Energy(self, m_fun):

        self.H_d.x.scatter_forward()
        dE = ufl.inner(m_fun, self.H_d) * ufl.dx(domain=self.mesh)
        energy = fem.assemble_scalar(fem.form(dE))
        return -0.5 * self.mu0 * self.Ms * energy * 1e-27