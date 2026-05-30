import math
import numpy as np
import cupy as cp
import ufl

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.mesh import exterior_facet_indices
from dolfinx.fem.petsc import assemble_matrix

try:
    from .Demag_Lindholm import (
        DoubleLayerDenseOwnedRowsMPI,
        build_nullspace_per_component,
    )
except ImportError:
    from Demag_Lindholm import (
        DoubleLayerDenseOwnedRowsMPI,
        build_nullspace_per_component,
    )


def _set_vec_cuda(vec: PETSc.Vec, block_size=None):
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


def _set_mat_cuda(mat: PETSc.Mat):
    try:
        mat = mat.convert(PETSc.Mat.Type.AIJCUSPARSE)
    except Exception:
        mat = mat.convert("aijcusparse")

    try:
        mat.bindToCPU(False)
    except Exception:
        pass

    return mat


def _vec_to_cupy(vec: PETSc.Vec, mode="rw"):
    return cp.from_dlpack(vec.toDLPack(mode))


class DemagFieldLindholmGPU:
    """
    GPU-oriented FEM/BEM demagnetizing field using the Lindholm double-layer
    operator.

    Notes
    -----
    - Matrices are assembled with DOLFINx on the host.
    - FEM matrices are converted to PETSc AIJCUSPARSE.
    - The dense boundary operator B is assembled once on CPU and copied to CuPy.
    - The per-step B @ u1_boundary multiplication runs on GPU.
    - The implementation is intended first for single-GPU / one MPI rank.
      MPI support is possible but needs careful treatment of boundary reductions.
    """

    def __init__(
        self,
        mesh,
        V,
        V1,
        Ms,
        VolN,
        serial_mesh_path=None,
        cache_dir=None,
        cache_prefix="Bcache_lindholm_ownedrows",
        cache_tag=None,
        map_tol=1e-10,
        ksp_rtol_u1=1e-6,
        ksp_rtol_u2=1e-6,
        incident_policy="skip",
        volume_scale=1e27,
        B_dtype=np.float64,
    ):
        self.mesh = mesh
        self.V = V
        self.V1 = V1
        self.Ms = float(Ms)
        self.mu0 = 4.0 * math.pi * 1e-7
        self.comm = mesh.comm
        self.rank = self.comm.rank
        self.volume_scale = float(volume_scale)
        self.last_its_u1 = 1
        self.last_its_u2 = 1
        if self.comm.size != 1:
            raise RuntimeError(
                "DemagFieldLindholmGPU is currently implemented for single-rank GPU execution only."
            )

        self.local_dofs = V.dofmap.index_map.size_local
        self.local_size = 3 * self.local_dofs
        self.B_dtype = B_dtype



        # Scalar nodal volume from vector VolN.
        VolN = np.asarray(VolN, dtype=np.float64)
        self.vol_scalar_gpu = cp.asarray(
            VolN[: self.local_size].reshape((-1, 3))[:, 0]
        )

        self.inv_vol_scalar_gpu = 1.0 / self.vol_scalar_gpu

        # ------------------------------------------------------------
        # u1 problem: A_u1 u1 = Div m
        # ------------------------------------------------------------
        v1 = ufl.TestFunction(V1)
        u1 = ufl.TrialFunction(V1)

        a_u1 = fem.form(ufl.inner(ufl.grad(v1), ufl.grad(u1)) * ufl.dx)
        A_u1_cpu = assemble_matrix(a_u1)
        A_u1_cpu.assemble()

        self.ns_u1 = build_nullspace_per_component(mesh, V1, A_u1_cpu)

        self.A_u1 = _set_mat_cuda(A_u1_cpu)

        v2 = ufl.TestFunction(V1)
        m_trial = ufl.TrialFunction(V)

        Div_form = fem.form(
            self.Ms * ufl.inner(ufl.grad(v2), m_trial) * ufl.dx
        )

        Div_cpu = assemble_matrix(Div_form)
        Div_cpu.assemble()
        self.Div = _set_mat_cuda(Div_cpu)

        self.b_u1 = self.Div.createVecLeft()
        self.u1 = self.A_u1.createVecRight()

        _set_vec_cuda(self.b_u1)
        _set_vec_cuda(self.u1)

        self.ksp_u1 = PETSc.KSP().create(self.comm)
        self.ksp_u1.setType("cg")
        self.ksp_u1.setOperators(self.A_u1)
        self.ksp_u1.setTolerances(rtol=ksp_rtol_u1, atol=1e-12, max_it=2000)
        #self.ksp_u1.setInitialGuessNonzero(True)

        pc1 = self.ksp_u1.getPC()
        pc1.setType("gamg")
        pc1.setReusePreconditioner(True)    

        self.ksp_u1.setFromOptions()

        # ------------------------------------------------------------
        # u2 problem with Dirichlet data on boundary
        # ------------------------------------------------------------
        tdim = mesh.topology.dim
        fdim = tdim - 1

        mesh.topology.create_connectivity(fdim, tdim)
        mesh.topology.create_connectivity(fdim, 0)

        boundary_facets = exterior_facet_indices(mesh.topology)
        self.boundary_dofs = fem.locate_dofs_topological(V1, fdim, boundary_facets)
        self.boundary_dofs = np.unique(self.boundary_dofs).astype(np.int32)

        zero = fem.Function(V1)
        zero.x.array[:] = 0.0
        zero.x.scatter_forward()

        bc_zero = fem.dirichletbc(zero, self.boundary_dofs)

        uu = ufl.TrialFunction(V1)
        vv = ufl.TestFunction(V1)

        a_u2 = fem.form(ufl.inner(ufl.grad(uu), ufl.grad(vv)) * ufl.dx)

        # Raw matrix: used to form b = -A_raw * g.
        A_u2_raw_cpu = assemble_matrix(a_u2)
        A_u2_raw_cpu.assemble()

        # BC matrix: identity rows on boundary.
        A_u2_bc_cpu = assemble_matrix(a_u2, bcs=[bc_zero])
        A_u2_bc_cpu.assemble()

        self.A_u2_raw = _set_mat_cuda(A_u2_raw_cpu)
        self.A_u2 = _set_mat_cuda(A_u2_bc_cpu)

        self.g_u2 = self.A_u2.createVecRight()
        self.b_u2 = self.A_u2.createVecLeft()
        self.u2 = self.A_u2.createVecRight()

        _set_vec_cuda(self.g_u2)
        _set_vec_cuda(self.b_u2)
        _set_vec_cuda(self.u2)

        self.ksp_u2 = PETSc.KSP().create(self.comm)
        self.ksp_u2.setType("cg")
        self.ksp_u2.setOperators(self.A_u2)
        self.ksp_u2.setTolerances(rtol=ksp_rtol_u2, atol=1e-12, max_it=12000)
        

        pc2 = self.ksp_u2.getPC()
        pc2.setType("gamg")
        pc2.setReusePreconditioner(True)     

        self.ksp_u2.setFromOptions()

        # ------------------------------------------------------------
        # Lindholm B operator
        # ------------------------------------------------------------
        self.opB = DoubleLayerDenseOwnedRowsMPI(
            mesh_par=mesh,
            V1_par=V1,
            serial_mesh_path=serial_mesh_path,
            cache_dir=cache_dir,
            cache_prefix=cache_prefix,
            cache_tag=cache_tag,
            map_tol=map_tol,
            incident_policy=incident_policy,
        )

        self.opB.assemble(force=False)

        # Dense B block copied once to GPU.
        self.B_gpu = cp.asarray(self.opB.Bloc,  dtype=self.B_dtype)

        self.xb_gpu = cp.zeros(self.opB.Nb, dtype=self.B_dtype)

        self.bdofs_owned_gpu = cp.asarray(self.opB.bdofs_owned, dtype=cp.int32)

        self.bidx_owned = self.opB.bidx_owned.astype(np.int32)
        self.bidx_owned_gpu = cp.asarray(self.bidx_owned, dtype=cp.int32)

        self.rowpos_for_bdof_gpu = cp.asarray(self.opB.rowpos_for_bdof, dtype=cp.int32)
        self.boundary_dofs_gpu = cp.asarray(self.boundary_dofs, dtype=cp.int32)
        self.y_rows_gpu = cp.empty(self.opB.nrows, dtype=self.B_dtype)
        # ------------------------------------------------------------
        # H = -grad(u1 + u2)
        # ------------------------------------------------------------
        vvec = ufl.TestFunction(V)
        phi = ufl.TrialFunction(V1)

        G_form = fem.form(ufl.inner(vvec, -ufl.grad(phi)) * ufl.dx)
        G_cpu = assemble_matrix(G_form)
        G_cpu.assemble()

        self.G = _set_mat_cuda(G_cpu)

        self.phi = self.G.createVecRight()
        self.H_tmp = self.G.createVecLeft()

        _set_vec_cuda(self.phi)
        _set_vec_cuda(self.H_tmp, block_size=3)

    def _solve_u1_gpu(self, m_vec):
        self.Div.mult(m_vec, self.b_u1)

        try:
            self.ns_u1.remove(self.b_u1)
        except Exception:
            pass

        self.ksp_u1.solve(self.b_u1, self.u1)

        try:
            self.ns_u1.remove(self.u1)
        except Exception:
            pass

 
    def _apply_B_gpu(self):
        """
        Compute boundary Dirichlet data:

            g = B u1_boundary

        Single-rank GPU version.
        """

        u1_cp = _vec_to_cupy(self.u1, "r")

        self.xb_gpu.fill(0.0)

        self.xb_gpu[self.bidx_owned_gpu] = u1_cp[self.bdofs_owned_gpu]

        cp.matmul(self.B_gpu, self.xb_gpu, out=self.y_rows_gpu)

        g_cp = _vec_to_cupy(self.g_u2, "rw")

        g_cp.fill(0.0)

        g_cp[self.bdofs_owned_gpu] = self.y_rows_gpu[self.rowpos_for_bdof_gpu]

    def _solve_u2_gpu(self):
        """
        Solve Laplace equation for u2 with Dirichlet data g_u2.

        Equivalent to:
            A_raw u = 0 in the interior,
            u = g on the boundary.

        Algebraically:
            b = -A_raw g
            b[boundary] = g[boundary]
            A_bc u = b
        """

        self.A_u2_raw.mult(self.g_u2, self.b_u2)
        self.b_u2.scale(-1.0)

        b_cp = _vec_to_cupy(self.b_u2, "rw")
        g_cp = _vec_to_cupy(self.g_u2, "r")

        b_cp[self.boundary_dofs_gpu] = g_cp[self.boundary_dofs_gpu]

        self.ksp_u2.solve(self.b_u2, self.u2)


    def compute_vec(self, m_vec: PETSc.Vec, out_vec: PETSc.Vec):
        """
        Compute demagnetizing field on GPU.

        Parameters
        ----------
        m_vec:
            PETSc CUDA vector with magnetization.
        out_vec:
            PETSc CUDA vector where H_demag is written.
        """

        self._solve_u1_gpu(m_vec)
        self._apply_B_gpu()
        self._solve_u2_gpu()

        self.u1.copy(self.phi)
        self.phi.axpy(1.0, self.u2)

        self.G.mult(self.phi, out_vec)

        h_cp = _vec_to_cupy(out_vec, "rw")
        h_owned = h_cp[: self.local_size].reshape((-1, 3))

        h_owned *= self.inv_vol_scalar_gpu[:, None]

        if h_cp.size > self.local_size:
            h_cp[self.local_size :] = 0.0

    def Energy_lumped_gpu(self, m_vec: PETSc.Vec, h_vec: PETSc.Vec):
        """
        Lumped demagnetizing energy:

            E = -0.5 mu0 Ms int m · H_demag dV
        """

        m_cp = _vec_to_cupy(m_vec, "r")
        h_cp = _vec_to_cupy(h_vec, "r")

        m = m_cp[: self.local_size].reshape((-1, 3))
        h = h_cp[: self.local_size].reshape((-1, 3))

        density = cp.sum(m * h, axis=1)
        integral = cp.sum(density * self.vol_scalar_gpu)

        local_energy = float(integral.get())

        return -0.5 * self.mu0 * self.Ms * local_energy / self.volume_scale
    





def synchronize_gpu():
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def set_vec_cuda(vec: PETSc.Vec, block_size=None):
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
