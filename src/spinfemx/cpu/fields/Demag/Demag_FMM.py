from mpi4py import MPI
import numpy as np

from pathlib import Path
from time import perf_counter

from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import apply_lifting, set_bc, create_vector
from petsc4py import PETSc

import ufl

from scipy.spatial import cKDTree
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from bempp_cl.api.external import fenicsx
import bempp_cl.api

from dolfinx import mesh as dmesh

from scipy.sparse.csgraph import connected_components

import os

# CPU:
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
from jaxfmm.strayfield import gen_unitcube_mesh, init_strayfield, eval_strayfield


'''
    We test the package jaxfmm that allow to calculate the demag field using the Fast Multipole Method.
    This only work in rank 0

'''

class DemagField_FMM:
    def __init__(self, domain_mesh, V, V1, Ms):
        self.mesh = domain_mesh
        self.mu0 = 4*np.pi*1e-7
        self.Ms = Ms
        self.V1 = V1
        self.V  = V
        self.comm = self.mesh.comm
        self.rank = self.comm.rank

        self.H_d = fem.Function(self.V)


        meshfile = "tmp_mesh.xdmf"
        if self.rank == 0:

            pass
        self.comm.Barrier()
        with io.XDMFFile(self.mesh.comm, meshfile, "w") as xdmf:
            xdmf.write_mesh(self.mesh)
        self.comm.Barrier()

        # --- Build serial mesh and tree ONLY in rank 0 ---
        self.X = None
        self.cells = None
        self.tree_info = None
        self.m_serial = None
        self.nloc = self.V1.dofmap.index_map.size_local

        if self.rank == 0:
            with io.XDMFFile(MPI.COMM_SELF, meshfile, "r") as xdmf:
                try:
                    serial_mesh = xdmf.read_mesh(name="Grid")
                except Exception:
                    serial_mesh = xdmf.read_mesh()

            self.X = serial_mesh.geometry.x.copy()

            tdim = serial_mesh.topology.dim
            serial_mesh.topology.create_connectivity(tdim, 0)
            conn = serial_mesh.topology.connectivity(tdim, 0)

            nc = serial_mesh.topology.index_map(tdim).size_local
            nvpc = len(conn.links(0))
            self.cells = np.empty((nc, nvpc), dtype=np.int32)
            for c in range(nc):
                self.cells[c, :] = conn.links(c)

            Msat = jnp.array([self.Ms])
            self.tree_info = init_strayfield(self.X, self.cells, Msat, mem_limit=2000000)

            self.m_serial = np.zeros((self.X.shape[0], 3), dtype=np.float64)

        # --- Broadcast X to all to map (coords only) ---
        self.X = self.comm.bcast(self.X, root=0)

        # --- Construct idx_serial_owned on each rank (P1 owned DOFs) ---
        imap = self.V1.dofmap.index_map
        nloc = imap.size_local  # owned dofs P1

        coords = self.V1.tabulate_dof_coordinates()[:nloc]  # (nloc,3)
        tree = cKDTree(self.X)
        dists, idx = tree.query(coords, k=1, workers=1)
        dmax = self.comm.allreduce(float(dists.max() if dists.size else 0.0), op=MPI.MAX)
        if self.rank == 0 and dmax > 1e-10:
            print(f"[WARN] dmax DOF->serial = {dmax:g}", flush=True)

        self.idx_serial_owned = idx.astype(np.int32)





        self.nloc = self.V1.dofmap.index_map.size_local  # owned P1 nodes per rank


        self.counts_nodes = self.comm.allgather(self.nloc)
        self.counts_f = np.array([3*c for c in self.counts_nodes], dtype=np.int32)

        self.displs_f = np.zeros_like(self.counts_f)
        self.displs_f[1:] = np.cumsum(self.counts_f[:-1])

        self.ntot_f = int(np.sum(self.counts_f))

        # buffers 
        self._send_m = np.empty(3*self.nloc, dtype=np.float64)
        self._recv_m = np.empty(self.ntot_f, dtype=np.float64) if self.rank == 0 else None

        self._send_h = np.empty(3*self.nloc, dtype=np.float64)
        self._recv_h = np.empty(self.ntot_f, dtype=np.float64) if self.rank == 0 else None

        # combine mapping ONLY ONCE (pickle here DOESN'T matter: it happens 1 time)
        idx_list = self.comm.gather(self.idx_serial_owned.astype(np.int32, copy=False), root=0)
        if self.rank == 0:
            self.idx_by_rank = idx_list  # list: idx_by_rank[r] = (nloc_r,)
        else:
            self.idx_by_rank = None




    def compute(self, m):

        m_loc = m.x.array[:3*self.nloc] 


        self._send_m[:] = m_loc


        self.comm.Gatherv(
            sendbuf=self._send_m,
            recvbuf=(self._recv_m, (self.counts_f, self.displs_f)),
            root=0
        )

        if self.rank == 0:

            self.m_serial.fill(0.0)

            offset = 0
            for r, nloc_r in enumerate(self.counts_nodes):
                block = self._recv_m[offset:offset + 3*nloc_r].reshape(nloc_r, 3)
                idx_r = self.idx_by_rank[r]
                self.m_serial[idx_r, :] = block
                offset += 3*nloc_r


            H_serial = np.array(eval_strayfield(self.m_serial, **self.tree_info), dtype=np.float64)

   
            offset = 0
            for r, nloc_r in enumerate(self.counts_nodes):
                idx_r = self.idx_by_rank[r]
                self._recv_h[offset:offset + 3*nloc_r] = H_serial[idx_r, :].reshape(-1)
                offset += 3*nloc_r


        self.comm.Scatterv(
            sendbuf=(self._recv_h, (self.counts_f, self.displs_f)),
            recvbuf=self._send_h,
            root=0
        )


        self.H_d.x.array[:3*self.nloc] = self._send_h
        self.H_d.x.scatter_forward()
        return self.H_d

    def Energy(self, m: fem.Function):
        self.H_d.x.scatter_forward()
        dE = ufl.inner(m, self.H_d) * ufl.dx(domain=self.mesh)
        energy = fem.assemble_scalar(fem.form(dE))
        return -0.5 * self.mu0 * self.Ms * energy / 1e27


