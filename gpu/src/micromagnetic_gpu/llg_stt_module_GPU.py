import sys
from pathlib import Path
from time import perf_counter

import adios4dolfinx as ad
import cupy as cp
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io
from dolfinx.fem import Constant, form
from dolfinx.fem.petsc import assemble_matrix, assemble_vector

from .Exchange_GPU import ExchangeField
from .Anisotropy_GPU import AnisotropyField
from .DMI_Interfacial_GPU import DMIInterfacial
from .DMI_Bulk_GPU import DMIBULK




# -----------------------------------------------------------------------------
# PETSc / CuPy helpers
# -----------------------------------------------------------------------------
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


def _set_mat_cuda(mat: PETSc.Mat) -> PETSc.Mat:
    try:
        mtype = str(mat.getType()).lower()
    except Exception:
        mtype = ""
    if "cuda" not in mtype:
        try:
            mat = mat.convert(PETSc.Mat.Type.AIJCUSPARSE)
        except Exception:
            mat = mat.convert("aijcusparse")
    try:
        mat.bindToCPU(False)
    except Exception:
        pass
    return mat


def _dup_cuda_vec(template: PETSc.Vec, block_size: int | None = None) -> PETSc.Vec:
    out = template.duplicate()
    _set_vec_cuda(out, block_size=block_size)
    out.zeroEntries()
    return out


def _vec_to_cupy(vec: PETSc.Vec, mode: str = "rw") -> cp.ndarray:
    return cp.from_dlpack(vec.toDLPack(mode))


def _sync_vec_to_function(vec: PETSc.Vec, fun: fem.Function) -> fem.Function:
    vec.copy(fun.x.petsc_vec)
    fun.x.scatter_forward()
    return fun


# -----------------------------------------------------------------------------
# Zhang-Li advection operator on GPU
# -----------------------------------------------------------------------------
class ZhangLiGPU:
    """
    GPU version of Zhang-Li directional derivative:

        Z(m) = (Jdir cdot nabla) m

    discretized as a linear FEM operator K_J and scaled by 1/VolN.
    """

    def __init__(self, mesh, V, Jdir_vec, VolN):
        self.mesh = mesh
        self.V = V

        self.Jdir = fem.Function(V)
        self.Jdir.x.array[:] = np.asarray(Jdir_vec, dtype=np.float64)
        self.Jdir.x.scatter_forward()

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        directional_derivative = ufl.dot(ufl.grad(u), self.Jdir)
        advection_form = ufl.inner(directional_derivative, v) * ufl.dx

        K_cpu = assemble_matrix(fem.form(advection_form))
        K_cpu.assemble()

        prefactor = fem.Function(V)
        prefactor.x.array[:] = 1.0 / VolN[:]
        prefactor.x.scatter_forward()
        K_cpu.diagonalScale(prefactor.x.petsc_vec, None)

        self.K_J = _set_mat_cuda(K_cpu)

        self.z_gpu = self.K_J.createVecLeft()
        _set_vec_cuda(self.z_gpu, block_size=3)

        self.ZhangLi_host = fem.Function(V)

    def compute_vec(self, m_gpu: PETSc.Vec, out_gpu: PETSc.Vec | None = None) -> PETSc.Vec:
        if out_gpu is None:
            out_gpu = self.z_gpu
        self.K_J.mult(m_gpu, out_gpu)
        return out_gpu

    def to_function(self, vec: PETSc.Vec | None = None):
        if vec is None:
            vec = self.z_gpu
        vec.copy(self.ZhangLi_host.x.petsc_vec)
        self.ZhangLi_host.x.scatter_forward()
        return self.ZhangLi_host


# -----------------------------------------------------------------------------
# Stopping criterion
# -----------------------------------------------------------------------------
class StopByMaxDmdtFDGPU:
    def __init__(self, comm, stopping_dm_dt_deg_ns, vec_template, n_owned_scalar, check_every=10, print_every_hit=True):
        self.comm = comm
        self.thresh_deg_ns = float(stopping_dm_dt_deg_ns)
        self.check_every = int(check_every)
        self.print_every_hit = bool(print_every_hit)
        self.n_owned_scalar = int(n_owned_scalar)
        self.n_owned_scalar3 = (self.n_owned_scalar // 3) * 3

        self.u_prev = _dup_cuda_vec(vec_template, block_size=3)
        self.du = _dup_cuda_vec(vec_template, block_size=3)
        vec_template.copy(self.u_prev)

        self.t_prev = None
        self.last_max_dmdt_deg_ns = float("nan")

    def __call__(self, ts):
        if self.thresh_deg_ns <= 0.0:
            return 0

        step = ts.getStepNumber()
        if self.check_every > 1 and (step % self.check_every) != 0:
            return 0

        u = ts.getSolution()
        t = ts.getTime()

        if self.t_prev is None:
            u.copy(self.u_prev)
            self.t_prev = t
            return 0

        dt = t - self.t_prev
        if dt <= 0.0:
            u.copy(self.u_prev)
            self.t_prev = t
            return 0

        # du = u - u_prev
        self.u_prev.copy(self.du)
        self.du.scale(-1.0)
        self.du.axpy(1.0, u)

        du_all = _vec_to_cupy(self.du, "r")
        du = du_all[: self.n_owned_scalar3].reshape((-1, 3))

        if du.size:
            max_local = float(cp.sqrt((du * du).sum(axis=1)).max().item()) / dt
        else:
            max_local = 0.0

        max_global = self.comm.allreduce(max_local, op=MPI.MAX)
        max_deg_ns = max_global * (180.0 / np.pi) * 1e-9
        self.last_max_dmdt_deg_ns = max_deg_ns

        if self.comm.rank == 0 and self.print_every_hit:
            print(f"[dmdt] step={step} t={t*1e9:.6f} ns  max|dm/dt|={max_deg_ns:.6e} deg/ns", flush=True)

        if max_deg_ns < self.thresh_deg_ns:
            ts.setConvergedReason(PETSc.TS.ConvergedReason.CONVERGED_USER)

        u.copy(self.u_prev)
        self.t_prev = t
        return 0

    def reset(self, vec_current):
        vec_current.copy(self.u_prev)
        self.t_prev = None
        self.last_max_dmdt_deg_ns = float("nan")


# -----------------------------------------------------------------------------
# Effective field + Zhang-Li STT on GPU
# -----------------------------------------------------------------------------
class EffectiveFieldSTTGPU:
    def __init__(
        self,
        mesh,
        Ms,
        Aex,
        Ku,
        n_ani_vec,
        D_bulk,
        D_int,
        n0_int_vec,
        H0_static=None,
        gamma=2.211e5,
        alpha=0.5,
        do_precess=1,
        Jmagnitude=0.0,
        Jdir_vec=None,
        P=0.0,
        beta=0.0,
        use_demag=False,
        demag_method="fmm",
        demag_kwargs=None,
        H_time_func=None,
    ):
        self.mesh = mesh
        self.comm = mesh.comm
        self.V = fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
        self.V1 = fem.functionspace(mesh, ("Lagrange", 1))

        self.Ms = float(Ms)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.A = float(Aex)
        self.Ku = float(Ku)
        self.D_bulk = float(D_bulk)
        self.D_int = float(D_int)
        self.do_precess = float(do_precess)
        self.use_demag = bool(use_demag)
        self.H_time_func = H_time_func
        self.current_time = 0.0

        self.P = float(P)
        self.Jmagnitude = float(Jmagnitude)
        self.beta = float(beta)

        e = 1.6021766e-19
        muB = 9.27400915e-24
        self.prefZhang = (
            self.Jmagnitude
            * self.P
            * muB
            / (e * self.Ms * (1.0 + self.beta**2))
            / (1.0 + self.alpha**2)
            / 1e-9
        )

        self.prefactor = -self.gamma / (1.0 + self.alpha**2)
        self.Stab = self.Ms * self.gamma / (1.0 + self.alpha**2) * 0.5

        # Host-side functions for I/O only
        self.m = fem.Function(self.V)
        self.dmdt = fem.Function(self.V)
        self.H_eff = fem.Function(self.V)
        self.H0_host = fem.Function(self.V)
        self.Zhang_host = fem.Function(self.V)

        self.start, self.end = self.V.dofmap.index_map.local_range
        self.local_dofs = self.end - self.start
        self.local_size = 3 * self.local_dofs

        # -------------------------------------------------------------
        # Lumped nodal volumes
        # -------------------------------------------------------------
        v = ufl.TestFunction(self.V)
        tmp = ufl.dot(v, Constant(self.mesh, PETSc.ScalarType((1.0, 1.0, 1.0)))) * ufl.dx
        volN_f = fem.Function(self.V)
        volN_f.x.petsc_vec.set(0.0)
        assemble_vector(volN_f.x.petsc_vec, form(tmp))
        volN_f.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )
        volN_f.x.scatter_forward()
        volN = volN_f.x.array.copy()
        self.volN = volN

        # -------------------------------------------------------------
        # Linear magnetic operators
        # -------------------------------------------------------------
        self.exchange_field = None
        self.anisotropy_field = None
        self.DMIBULK = None
        self.DMI_int = None
        self.demag_field = None
        self.H_demag_gpu = None
        self.linear_terms: list[tuple[str, PETSc.Mat, PETSc.Vec]] = []

        template_in = None
        template_out = None

        if abs(self.A) > 0.0:
            self.exchange_field = ExchangeField(self.mesh, self.V, self.A, self.Ms, volN)
            self.exchange_field.K = _set_mat_cuda(self.exchange_field.K)
            buf = self.exchange_field.K.createVecLeft()
            _set_vec_cuda(buf, block_size=3)
            self.linear_terms.append(("exchange", self.exchange_field.K, buf))
            template_in = self.exchange_field.K.createVecRight()
            template_out = self.exchange_field.K.createVecLeft()

        if abs(self.Ku) > 0.0:
            self.anisotropy_field = AnisotropyField(self.mesh, self.V, self.Ku, self.Ms, n_ani_vec, volN)
            self.anisotropy_field.K = _set_mat_cuda(self.anisotropy_field.K)
            buf = self.anisotropy_field.K.createVecLeft()
            _set_vec_cuda(buf, block_size=3)
            self.linear_terms.append(("anisotropy", self.anisotropy_field.K, buf))
            if template_in is None:
                template_in = self.anisotropy_field.K.createVecRight()
                template_out = self.anisotropy_field.K.createVecLeft()

        if abs(self.D_bulk) > 0.0:
            if DMIBULK is None:
                raise NotImplementedError("DMI_Bulk_GPU is not available. Provide it or set D_bulk=0.")
            self.DMIBULK = DMIBULK(self.mesh, self.V, self.V1, self.D_bulk, self.Ms, volN)
            self.DMIBULK.K = _set_mat_cuda(self.DMIBULK.K)
            buf = self.DMIBULK.K.createVecLeft()
            _set_vec_cuda(buf, block_size=3)
            self.linear_terms.append(("dmi_bulk", self.DMIBULK.K, buf))
            if template_in is None:
                template_in = self.DMIBULK.K.createVecRight()
                template_out = self.DMIBULK.K.createVecLeft()

        if abs(self.D_int) > 0.0:
            self.DMI_int = DMIInterfacial(self.mesh, self.V, self.V1, self.D_int, n0_int_vec, self.Ms, volN)
            self.DMI_int.K = _set_mat_cuda(self.DMI_int.K)
            buf = self.DMI_int.K.createVecLeft()
            _set_vec_cuda(buf, block_size=3)
            self.linear_terms.append(("dmi_interfacial", self.DMI_int.K, buf))
            if template_in is None:
                template_in = self.DMI_int.K.createVecRight()
                template_out = self.DMI_int.K.createVecLeft()

        # -------------------------------------------------------------
        # Zhang-Li GPU operator.
        # It also supplies a template if no magnetic linear operator is active.
        # -------------------------------------------------------------
        if Jdir_vec is None:
            Jdir_vec = np.zeros(3 * len(self.mesh.geometry.x), dtype=np.float64)
        self.ZhangLi = ZhangLiGPU(self.mesh, self.V, Jdir_vec, volN)

        if template_in is None:
            template_in = self.ZhangLi.K_J.createVecRight()
            template_out = self.ZhangLi.K_J.createVecLeft()

        _set_vec_cuda(template_in, block_size=3)
        _set_vec_cuda(template_out, block_size=3)

        # Persistent GPU vectors
        self.m_gpu = _dup_cuda_vec(template_in, block_size=3)
        self.dmdt_gpu = _dup_cuda_vec(template_in, block_size=3)

        self.H_eff_gpu = _dup_cuda_vec(template_out, block_size=3)
        self.Hm_gpu = _dup_cuda_vec(template_out, block_size=3)
        self.Hv_gpu = _dup_cuda_vec(template_out, block_size=3)
        self.H0_gpu = _dup_cuda_vec(template_out, block_size=3)
        self.Ht_gpu = _dup_cuda_vec(template_out, block_size=3)

        self.Zm_gpu = _dup_cuda_vec(template_out, block_size=3)
        self.Zv_gpu = _dup_cuda_vec(template_out, block_size=3)

        self.Jv_buffer = _dup_cuda_vec(template_in, block_size=3)

        # State cached for Jv
        self.M_state_gpu = _dup_cuda_vec(template_in, block_size=3)
        self.JdotGrad_m_gpu = _dup_cuda_vec(template_out, block_size=3)

        # Linear diagonal for preconditioner diagnostics/approximation.
        # This only contains the sparse/local linear magnetic operators
        # (exchange, anisotropy, DMI). Demag is intentionally excluded.
        self.diagK = _dup_cuda_vec(template_out, block_size=3)
        self.diagK.zeroEntries()
        for _, K, _ in self.linear_terms:
            d = K.getDiagonal()
            _set_vec_cuda(d, block_size=3)
            self.diagK.axpy(1.0, d)

        self.diagK_abs = _dup_cuda_vec(template_out, block_size=3)
        self.diagK.copy(self.diagK_abs)
        self.diagK_abs.abs()

        # Static external field
        if H0_static is not None:
            self.H0_host.x.array[:] = np.asarray(H0_static, dtype=np.float64)
            self.H0_host.x.scatter_forward()
            self.H0_host.x.petsc_vec.copy(self.H0_gpu)
        else:
            self.H0_gpu.zeroEntries()

        # Optional demag
        if self.use_demag:
            method = str(demag_method).lower()
            kwargs = {} if demag_kwargs is None else dict(demag_kwargs)

            if method in ["fmm", "jaxfmm"]:
                from .Demag_FMM_GPU import DemagFieldFMMJAXGPU

                mem_limit = kwargs.pop("mem_limit", 4_000_000)

                self.demag_field = DemagFieldFMMJAXGPU(
                    self.mesh,
                    self.V,
                    self.V1,
                    self.Ms,
                    volN,
                    mem_limit=mem_limit,
                    **kwargs,
                )

            elif method in ["lindholm", "lindholm_gpu"]:
                from .Demag_Lindholm_GPU import DemagFieldLindholmGPU

                self.demag_field = DemagFieldLindholmGPU(
                    mesh=self.mesh,
                    V=self.V,
                    V1=self.V1,
                    Ms=self.Ms,
                    VolN=volN,
                    **kwargs,
                )

            else:
                raise NotImplementedError(
                    "Unsupported demag_method. Use 'jaxfmm', 'fmm', 'lindholm', or 'lindholm_gpu'."
                )

            self.H_demag_gpu = _dup_cuda_vec(template_out, block_size=3)

        self.LLGSteps = 0
        self.JacSteps = 0
    # -----------------------------------------------------------------
    # Host/device sync
    # -----------------------------------------------------------------
    def set_m_from_cpu(self, m0_array):
        self.m.x.array[:] = np.asarray(m0_array, dtype=np.float64)
        self.m.x.scatter_forward()
        self.m.x.petsc_vec.copy(self.m_gpu)

    def sync_m_to_function(self):
        return _sync_vec_to_function(self.m_gpu, self.m)

    def sync_H_to_function(self):
        return _sync_vec_to_function(self.H_eff_gpu, self.H_eff)

    def sync_Zhang_to_function(self):
        return _sync_vec_to_function(self.Zm_gpu, self.Zhang_host)

    def set_uniform_field(self, Hx, Hy, Hz):
        H = _vec_to_cupy(self.H0_gpu, "rw")
        owned = H[: self.local_size].reshape((-1, 3))
        owned[:, 0] = Hx
        owned[:, 1] = Hy
        owned[:, 2] = Hz
        if H.size > self.local_size:
            H[self.local_size:] = 0.0

    # -----------------------------------------------------------------
    # Fields
    # -----------------------------------------------------------------
    def _eval_H_time_vec(self, t):
        self.Ht_gpu.zeroEntries()
        if self.H_time_func is None:
            return None

        try:
            out = self.H_time_func(float(t), self.mesh.geometry.x)
        except TypeError:
            out = self.H_time_func(float(t))

        H = _vec_to_cupy(self.Ht_gpu, "rw")
        owned = H[: self.local_size].reshape((-1, 3))

        if isinstance(out, PETSc.Vec):
            out.copy(self.Ht_gpu)
            return self.Ht_gpu

        if isinstance(out, cp.ndarray):
            arr = out
        else:
            arr = cp.asarray(np.asarray(out, dtype=np.float64))

        if arr.ndim == 1 and arr.size == 3:
            owned[:, 0] = arr[0]
            owned[:, 1] = arr[1]
            owned[:, 2] = arr[2]
            return self.Ht_gpu

        if arr.ndim == 1 and arr.size == self.local_size:
            H[: self.local_size] = arr
            return self.Ht_gpu

        if arr.shape == (self.local_dofs, 3):
            owned[:, :] = arr
            return self.Ht_gpu

        raise ValueError(f"Unsupported H_time_func output shape: {arr.shape}")

    def apply_linear_field_vec(self, x_vec: PETSc.Vec, out_vec: PETSc.Vec):
        out_vec.zeroEntries()
        for _, K, buf in self.linear_terms:
            K.mult(x_vec, buf)
            out_vec.axpy(1.0, buf)

    def compute_H_eff_vec(self, m_vec: PETSc.Vec, out_vec: PETSc.Vec):

        self.apply_linear_field_vec(m_vec, out_vec)

        if self.demag_field is not None:
            self.demag_field.compute_vec(m_vec, self.H_demag_gpu)
            out_vec.axpy(1.0, self.H_demag_gpu)

        out_vec.axpy(1.0, self.H0_gpu)

        if self.H_time_func is not None:
            self._eval_H_time_vec(self.current_time)
            out_vec.axpy(1.0, self.Ht_gpu)

    def compute_H_jac_vec(self, m_vec: PETSc.Vec, out_vec: PETSc.Vec):
        """
        Field used only for JV and preconditioner.

        Includes:

        exchange + anisotropy + DMI + external field

        Excludes:

        demag

        This allows maintaining demag in the RHS, but completely removing it
        from jac_vec_times_STT_vec and the local preconditioner.
        """
        self.apply_linear_field_vec(m_vec, out_vec)

        out_vec.axpy(1.0, self.H0_gpu)

        if self.H_time_func is not None:
            self._eval_H_time_vec(self.current_time)
            out_vec.axpy(1.0, self.Ht_gpu)

    def zhang_li_vec(self, m_vec: PETSc.Vec, out_vec: PETSc.Vec):
        self.ZhangLi.compute_vec(m_vec, out_vec)

    # -----------------------------------------------------------------
    # RHS: LLG + Zhang-Li STT
    # -----------------------------------------------------------------
    def llg_rhs_STT_vec(self, m_vec: PETSc.Vec, out_vec: PETSc.Vec):
        self.LLGSteps += 1

        self.compute_H_eff_vec(m_vec, self.H_eff_gpu)
        self.zhang_li_vec(m_vec, self.Zm_gpu)

        m_all = _vec_to_cupy(m_vec, "r")
        h_all = _vec_to_cupy(self.H_eff_gpu, "r")
        z_all = _vec_to_cupy(self.Zm_gpu, "r")
        rhs_all = _vec_to_cupy(out_vec, "rw")

        M = m_all[: self.local_size].reshape((-1, 3))
        H = h_all[: self.local_size].reshape((-1, 3))
        Z = z_all[: self.local_size].reshape((-1, 3))
        RHS = rhs_all[: self.local_size].reshape((-1, 3))

        mx = M[:, 0]; my = M[:, 1]; mz = M[:, 2]
        hx = H[:, 0]; hy = H[:, 1]; hz = H[:, 2]
        zx = Z[:, 0]; zy = Z[:, 1]; zz = Z[:, 2]

        # m x H
        mcx = my * hz - mz * hy
        mcy = mz * hx - mx * hz
        mcz = mx * hy - my * hx

        # m x (m x H)
        mcmx = my * mcz - mz * mcy
        mcmy = mz * mcx - mx * mcz
        mcmz = mx * mcy - my * mcx

        # m x Z
        zcx = my * zz - mz * zy
        zcy = mz * zx - mx * zz
        zcz = mx * zy - my * zx

        # m x (m x Z)
        zzx = my * zcz - mz * zcy
        zzy = mz * zcx - mx * zcz
        zzz = mx * zcy - my * zcx

        norm2 = mx * mx + my * my + mz * mz
        stab = self.Stab * (1.0 - norm2)

        stt_a = self.beta - self.alpha
        stt_b = 1.0 + self.alpha * self.beta

        RHS[:, 0] = (
            self.prefactor * (self.do_precess * mcx + self.alpha * mcmx)
            - self.prefZhang * (stt_a * zcx + stt_b * zzx)
            + stab * mx
        )
        RHS[:, 1] = (
            self.prefactor * (self.do_precess * mcy + self.alpha * mcmy)
            - self.prefZhang * (stt_a * zcy + stt_b * zzy)
            + stab * my
        )
        RHS[:, 2] = (
            self.prefactor * (self.do_precess * mcz + self.alpha * mcmz)
            - self.prefZhang * (stt_a * zcz + stt_b * zzz)
            + stab * mz
        )

        if rhs_all.size > self.local_size:
            rhs_all[self.local_size:] = 0.0

    # Alias expected by explicit TS wrapper
    def rhs_function(self, ts, t, y, f):
        self.current_time = float(t)
        self.llg_rhs_STT_vec(y, f)
        return 0

    def ifunction_STT(self, ts, t, y, ydot, f):
        """
        Implicit residual for PETSc TS/BDF:

            F(t, y, ydot) = ydot - RHS_STT(t, y)

        Important petsc4py convention:
            source.copy(target) copies source -> target.
        Therefore this must be ydot.copy(f), not f.copy(ydot).
        """
        self.current_time = float(t)

        # dmdt_gpu = RHS_STT(t, y)
        self.llg_rhs_STT_vec(y, self.dmdt_gpu)

        # f = ydot - RHS_STT(t, y)
        ydot.copy(f)
        f.axpy(-1.0, self.dmdt_gpu)

        return 0

    # -----------------------------------------------------------------
    # Optional Jv_STT for validation/future implicit method.
    # -----------------------------------------------------------------
    def update_jac_state_STT(self, m_vec: PETSc.Vec):
        m_vec.copy(self.M_state_gpu)

        self.compute_H_jac_vec(m_vec, self.Hm_gpu)

        self.zhang_li_vec(m_vec, self.JdotGrad_m_gpu)

    def jac_vec_times_STT_vec(self, v_vec: PETSc.Vec, out_vec: PETSc.Vec):
        self.JacSteps += 1

        self.apply_linear_field_vec(v_vec, self.Hv_gpu)

        self.zhang_li_vec(v_vec, self.Zv_gpu)

        m_all = _vec_to_cupy(self.M_state_gpu, "r")
        hm_all = _vec_to_cupy(self.Hm_gpu, "r")
        jgm_all = _vec_to_cupy(self.JdotGrad_m_gpu, "r")
        v_all = _vec_to_cupy(v_vec, "r")
        hv_all = _vec_to_cupy(self.Hv_gpu, "r")
        jgv_all = _vec_to_cupy(self.Zv_gpu, "r")
        out_all = _vec_to_cupy(out_vec, "rw")

        M = m_all[: self.local_size].reshape((-1, 3))
        Hm = hm_all[: self.local_size].reshape((-1, 3))
        Jgm = jgm_all[: self.local_size].reshape((-1, 3))
        V = v_all[: self.local_size].reshape((-1, 3))
        Hv = hv_all[: self.local_size].reshape((-1, 3))
        Jgv = jgv_all[: self.local_size].reshape((-1, 3))
        OUT = out_all[: self.local_size].reshape((-1, 3))

        mx = M[:, 0]; my = M[:, 1]; mz = M[:, 2]
        hx = Hm[:, 0]; hy = Hm[:, 1]; hz = Hm[:, 2]
        vx = V[:, 0]; vy = V[:, 1]; vz = V[:, 2]
        hvx = Hv[:, 0]; hvy = Hv[:, 1]; hvz = Hv[:, 2]

        jmx = Jgm[:, 0]; jmy = Jgm[:, 1]; jmz = Jgm[:, 2]
        jvx = Jgv[:, 0]; jvy = Jgv[:, 1]; jvz = Jgv[:, 2]

        # v x H(m)
        cvx = vy * hz - vz * hy
        cvy = vz * hx - vx * hz
        cvz = vx * hy - vy * hx

        # m x H(v)
        cmx = my * hvz - mz * hvy
        cmy = mz * hvx - mx * hvz
        cmz = mx * hvy - my * hvx

        prec_x = self.do_precess * (cvx + cmx)
        prec_y = self.do_precess * (cvy + cmy)
        prec_z = self.do_precess * (cvz + cmz)

        mdHm = mx * hx + my * hy + mz * hz
        mdHv = mx * hvx + my * hvy + mz * hvz
        vdHm = vx * hx + vy * hy + vz * hz
        mdv = mx * vx + my * vy + mz * vz
        mdmm = mx * mx + my * my + mz * mz

        common = vdHm + mdHv

        damp_x = vx * mdHm - 2.0 * hx * mdv + mx * common - hvx * mdmm
        damp_y = vy * mdHm - 2.0 * hy * mdv + my * common - hvy * mdmm
        damp_z = vz * mdHm - 2.0 * hz * mdv + mz * common - hvz * mdmm

        # STT derivative
        # M x JgradV
        mxjvx = my * jvz - mz * jvy
        mxjvy = mz * jvx - mx * jvz
        mxjvz = mx * jvy - my * jvx

        # V x JgradM
        vxjmx = vy * jmz - vz * jmy
        vxjmy = vz * jmx - vx * jmz
        vxjmz = vx * jmy - vy * jmx

        MjgradV = mx * jvx + my * jvy + mz * jvz
        VjgradM = vx * jmx + vy * jmy + vz * jmz
        MjgradM = mx * jmx + my * jmy + mz * jmz

        prec_stt_x = -(self.beta - self.alpha) * self.prefZhang * (mxjvx + vxjmx)
        prec_stt_y = -(self.beta - self.alpha) * self.prefZhang * (mxjvy + vxjmy)
        prec_stt_z = -(self.beta - self.alpha) * self.prefZhang * (mxjvz + vxjmz)

        stt_factor = -(1.0 + self.beta * self.alpha) * self.prefZhang

        damp_stt_x = stt_factor * (
            vx * MjgradM
            + mx * VjgradM
            + mx * MjgradV
            - jvx * mdmm
            - 2.0 * jmx * mdv
        )
        damp_stt_y = stt_factor * (
            vy * MjgradM
            + my * VjgradM
            + my * MjgradV
            - jvy * mdmm
            - 2.0 * jmy * mdv
        )
        damp_stt_z = stt_factor * (
            vz * MjgradM
            + mz * VjgradM
            + mz * MjgradV
            - jvz * mdmm
            - 2.0 * jmz * mdv
        )

        stab_common = self.Stab * (1.0 - mdmm)

        OUT[:, 0] = self.prefactor * (prec_x + self.alpha * damp_x) + prec_stt_x + damp_stt_x
        OUT[:, 1] = self.prefactor * (prec_y + self.alpha * damp_y) + prec_stt_y + damp_stt_y
        OUT[:, 2] = self.prefactor * (prec_z + self.alpha * damp_z) + prec_stt_z + damp_stt_z

        OUT[:, 0] += stab_common * vx - 2.0 * self.Stab * mx * mdv
        OUT[:, 1] += stab_common * vy - 2.0 * self.Stab * my * mdv
        OUT[:, 2] += stab_common * vz - 2.0 * self.Stab * mz * mdv

        if out_all.size > self.local_size:
            out_all[self.local_size:] = 0.0


    def compute_Energy_terms(self):
        m_fun = self.sync_m_to_function()

        E_exch = 0.0
        E_ani = 0.0
        E_dmi_bulk = 0.0
        E_dmi_int = 0.0
        E_demag = 0.0

        if self.exchange_field is not None:
            E_exch = float(self.exchange_field.Energy(m_fun))

        if self.anisotropy_field is not None:
    
            E_ani = float(self.anisotropy_field.Energy(m_fun))


        if self.demag_field is not None:
            self.demag_field.compute_vec(self.m_gpu, self.H_demag_gpu)
            E_demag = float(self.demag_field.Energy_lumped_gpu(self.m_gpu, self.H_demag_gpu))

        if self.DMIBULK is not None:
            self.DMIBULK.compute(self.m_gpu)
            E_dmi_bulk = float(self.DMIBULK.Energy(m_fun))

        if self.DMI_int is not None:
            self.DMI_int.compute(self.m_gpu)
            E_dmi_int = float(self.DMI_int.Energy(m_fun))

        E_total = E_exch + E_demag + E_ani + E_dmi_bulk + E_dmi_int

        return {
            "E_demag": E_demag,
            "E_exch": E_exch,
            "E_ani": E_ani,
            "E_dmi_bulk": E_dmi_bulk,
            "E_dmi_int": E_dmi_int,
            "E_total": E_total,
        }


# -----------------------------------------------------------------------------
# Matrix-free Jacobian context for BDF/STT: A*x = shift*x - J_STT*x
# plus local 3x3 Python/CuPy preconditioner
# -----------------------------------------------------------------------------
class JvContextSTT:
    def __init__(self, hef: EffectiveFieldSTTGPU, eps_reg=1e-14, det_eps=1e-30):
        self.hef = hef
        self.shift = 0.0
        self.calls = 0
        self.callsPre = 0

        self.eps_reg = float(eps_reg)
        self.det_eps = float(det_eps)
        self.enable_pc = True
        self._pc_ready = False

        n = self.hef.local_dofs

        self.i00 = cp.empty(n); self.i01 = cp.empty(n); self.i02 = cp.empty(n)
        self.i10 = cp.empty(n); self.i11 = cp.empty(n); self.i12 = cp.empty(n)
        self.i20 = cp.empty(n); self.i21 = cp.empty(n); self.i22 = cp.empty(n)

        g = float(self.hef.gamma)
        a = float(self.hef.alpha)
        b = float(self.hef.beta)
        dp = float(self.hef.do_precess)

        self.gamma = g
        self.alpha = a
        self.beta = b
        self.do_precess = dp

        # LLG local coefficients. 
        self.c1 = (g / (1.0 + a * a)) * dp
        self.c2 = (g * a / (1.0 + a * a))
        self.Stab = float(self.hef.Stab)

        # STT local coefficients. 
        self.prefZ = float(self.hef.prefZhang)
        self.c3 = (b - a) * self.prefZ
        self.c4 = -(1.0 + a * b) * self.prefZ

    def update_pc_full_fast_gpu(
        self,
        shift,
        include_stab=True,
        use_abs_kappa=True,
    ):
        """
        Local 3x3 CuPy preconditioner for STT-BDF:

            P_i sim shift*I_3 - J_i^approx

        It includes:
            - local LLG precession/damping terms
            - kappa approximation for dH_lin/dm using diagK
            - norm stabilization term
            - local STT terms using G = (Jdir cdot grad)m frozen at the state

        It excludes demag completely, provided update_jac_state_STT uses
        compute_H_jac_vec(...). In this file that is intentional.
        """
        self.shift = float(shift)
        local_size = self.hef.local_size

        M_all = cp.from_dlpack(self.hef.M_state_gpu.toDLPack("r"))
        H_all = cp.from_dlpack(self.hef.Hm_gpu.toDLPack("r"))
        G_all = cp.from_dlpack(self.hef.JdotGrad_m_gpu.toDLPack("r"))

        if use_abs_kappa:
            D_all = cp.from_dlpack(self.hef.diagK_abs.toDLPack("r"))
        else:
            D_all = cp.from_dlpack(self.hef.diagK.toDLPack("r"))

        M = M_all[:local_size].reshape((-1, 3))
        H = H_all[:local_size].reshape((-1, 3))
        G = G_all[:local_size].reshape((-1, 3))
        D = D_all[:local_size].reshape((-1, 3))

        mx = M[:, 0]; my = M[:, 1]; mz = M[:, 2]
        hx = H[:, 0]; hy = H[:, 1]; hz = H[:, 2]
        gx = G[:, 0]; gy = G[:, 1]; gz = G[:, 2]

        kappa = cp.mean(cp.abs(D), axis=1) if use_abs_kappa else cp.mean(D, axis=1)

        c1 = self.c1
        c2 = self.c2
        c3 = self.c3
        c4 = self.c4
        Stab = self.Stab

        mdH = mx * hx + my * hy + mz * hz
        mdm = mx * mx + my * my + mz * mz

        # ------------------------------------------------------------
        # LLG precession approximate block:
        #     -c1 * (S_H - kappa*S_m)
        # ------------------------------------------------------------
        Jp00 = 0.0
        Jp01 = -c1 * (-hz - kappa * (-mz))
        Jp02 = -c1 * ( hy - kappa * ( my))

        Jp10 = -c1 * ( hz - kappa * ( mz))
        Jp11 = 0.0
        Jp12 = -c1 * (-hx - kappa * (-mx))

        Jp20 = -c1 * (-hy - kappa * (-my))
        Jp21 = -c1 * ( hx - kappa * ( mx))
        Jp22 = 0.0

        # ------------------------------------------------------------
        # LLG damping approximate block:
        #     c2 * (B + kappa*C)
        # ------------------------------------------------------------
        B00 = mdH - hx * mx
        B11 = mdH - hy * my
        B22 = mdH - hz * mz

        B01 = mx * hy - 2.0 * hx * my
        B02 = mx * hz - 2.0 * hx * mz

        B10 = my * hx - 2.0 * hy * mx
        B12 = my * hz - 2.0 * hy * mz

        B20 = mz * hx - 2.0 * hz * mx
        B21 = mz * hy - 2.0 * hz * my

        C00 = mx * mx - mdm
        C11 = my * my - mdm
        C22 = mz * mz - mdm

        C01 = mx * my
        C02 = mx * mz

        C10 = my * mx
        C12 = my * mz

        C20 = mz * mx
        C21 = mz * my

        Jd00 = c2 * (B00 + kappa * C00)
        Jd11 = c2 * (B11 + kappa * C11)
        Jd22 = c2 * (B22 + kappa * C22)

        Jd01 = c2 * (B01 + kappa * C01)
        Jd02 = c2 * (B02 + kappa * C02)

        Jd10 = c2 * (B10 + kappa * C10)
        Jd12 = c2 * (B12 + kappa * C12)

        Jd20 = c2 * (B20 + kappa * C20)
        Jd21 = c2 * (B21 + kappa * C21)

        # ------------------------------------------------------------
        # Norm stabilization block
        # ------------------------------------------------------------
        if include_stab:
            s0 = 1.0 - mdm

            Js00 = Stab * (s0 - 2.0 * mx * mx)
            Js11 = Stab * (s0 - 2.0 * my * my)
            Js22 = Stab * (s0 - 2.0 * mz * mz)

            Js01 = Stab * (-2.0 * mx * my)
            Js02 = Stab * (-2.0 * mx * mz)

            Js10 = Stab * (-2.0 * my * mx)
            Js12 = Stab * (-2.0 * my * mz)

            Js20 = Stab * (-2.0 * mz * mx)
            Js21 = Stab * (-2.0 * mz * my)
        else:
            Js00 = Js11 = Js22 = 0.0
            Js01 = Js02 = Js10 = Js12 = Js20 = Js21 = 0.0

        # Base LLG block.
        J00 = Jd00 + Js00
        J11 = Jd11 + Js11
        J22 = Jd22 + Js22

        J01 = Jp01 + Jd01 + Js01
        J02 = Jp02 + Jd02 + Js02

        J10 = Jp10 + Jd10 + Js10
        J12 = Jp12 + Jd12 + Js12

        J20 = Jp20 + Jd20 + Js20
        J21 = Jp21 + Jd21 + Js21

        # ------------------------------------------------------------
        # STT local block using G = (Jdir \cdot grad)m frozen.
        # This mirrors the CPU preconditioner approximation.
        # ------------------------------------------------------------
        Jstt01 = c3 * (-gz)
        Jstt02 = c3 * ( gy)

        Jstt10 = c3 * ( gz)
        Jstt12 = c3 * (-gx)

        Jstt20 = c3 * (-gy)
        Jstt21 = c3 * ( gx)

        mdG = mx * gx + my * gy + mz * gz

        Bg00 = mdG - gx * mx
        Bg11 = mdG - gy * my
        Bg22 = mdG - gz * mz

        Bg01 = mx * gy - 2.0 * gx * my
        Bg02 = mx * gz - 2.0 * gx * mz

        Bg10 = my * gx - 2.0 * gy * mx
        Bg12 = my * gz - 2.0 * gy * mz

        Bg20 = mz * gx - 2.0 * gz * mx
        Bg21 = mz * gy - 2.0 * gz * my

        J00 = J00 + c4 * Bg00
        J11 = J11 + c4 * Bg11
        J22 = J22 + c4 * Bg22

        J01 = J01 + Jstt01 + c4 * Bg01
        J02 = J02 + Jstt02 + c4 * Bg02

        J10 = J10 + Jstt10 + c4 * Bg10
        J12 = J12 + Jstt12 + c4 * Bg12

        J20 = J20 + Jstt20 + c4 * Bg20
        J21 = J21 + Jstt21 + c4 * Bg21

        # A = shift*I - J + eps*I
        s = self.shift + self.eps_reg

        A00 = s - J00
        A01 = -J01
        A02 = -J02

        A10 = -J10
        A11 = s - J11
        A12 = -J12

        A20 = -J20
        A21 = -J21
        A22 = s - J22

        det = (
            A00 * (A11 * A22 - A12 * A21)
            - A01 * (A10 * A22 - A12 * A20)
            + A02 * (A10 * A21 - A11 * A20)
        )

        det = cp.where(
            cp.abs(det) < self.det_eps,
            det + cp.sign(det + self.det_eps) * self.det_eps,
            det,
        )

        invdet = 1.0 / det

        self.i00[:] =  (A11 * A22 - A12 * A21) * invdet
        self.i01[:] = -(A01 * A22 - A02 * A21) * invdet
        self.i02[:] =  (A01 * A12 - A02 * A11) * invdet

        self.i10[:] = -(A10 * A22 - A12 * A20) * invdet
        self.i11[:] =  (A00 * A22 - A02 * A20) * invdet
        self.i12[:] = -(A00 * A12 - A02 * A10) * invdet

        self.i20[:] =  (A10 * A21 - A11 * A20) * invdet
        self.i21[:] = -(A00 * A21 - A01 * A20) * invdet
        self.i22[:] =  (A00 * A11 - A01 * A10) * invdet

        self._pc_ready = True

    def apply(self, pc, x, y):
        self.callsPre += 1

        if (not self.enable_pc) or (not self._pc_ready):
            x.copy(y)
            return

        local_size = self.hef.local_size

        x_all = cp.from_dlpack(x.toDLPack("r"))
        y_all = cp.from_dlpack(y.toDLPack("rw"))

        X = x_all[:local_size].reshape((-1, 3))
        Y = y_all[:local_size].reshape((-1, 3))

        x0 = X[:, 0]
        x1 = X[:, 1]
        x2 = X[:, 2]

        Y[:, 0] = self.i00 * x0 + self.i01 * x1 + self.i02 * x2
        Y[:, 1] = self.i10 * x0 + self.i11 * x1 + self.i12 * x2
        Y[:, 2] = self.i20 * x0 + self.i21 * x1 + self.i22 * x2

        if y_all.size > local_size:
            y_all[local_size:] = 0.0

    def mult(self, A, x, y):
        self.calls += 1

        # J_STT*x
        self.hef.jac_vec_times_STT_vec(x, self.hef.Jv_buffer)

        # y = shift*x - J_STT*x
        x.copy(y)
        y.scale(self.shift)
        y.axpy(-1.0, self.hef.Jv_buffer)


# -----------------------------------------------------------------------------
# Main STT driver
# -----------------------------------------------------------------------------

class LLG_STT_GPU:
    def __init__(self, mesh, Ms, gamma=2.211e5, alpha=0.5, do_precess=1):
        self.mesh = mesh
        self.Ms = float(Ms)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.do_precess = float(do_precess)

        self._Aex = 0.0
        self._Ku = 0.0
        self._n_ani = None
        self._D_bulk = 0.0
        self._D_int = 0.0
        self._n0_int = None
        self._H0_vec = None
        self._H_time_func = None

        self._Jmag = 0.0
        self._Jdir_vec = None
        self._P = 0.0
        self._beta = 0.0

        self._has_exchange = False
        self._has_demag = False
        self._has_anisotropy = False
        self._has_dmi_bulk = False
        self._has_dmi_int = False
        self._has_H0 = False
        self._has_current = False

        self._demag_method = "fmm"
        self._demag_kwargs = {}

        self.hef: EffectiveFieldSTTGPU | None = None
        self.ts = None
        self.y = None
        self.stopper = None
        self._solver_ready = False

    def add_exchange(self, Aex):
        self._Aex = float(Aex)
        self._has_exchange = True

    def add_demag(self, method="fmm", **kwargs):
        self._has_demag = True
        self._demag_method = method
        self._demag_kwargs = dict(kwargs)

    def add_anisotropy(self, Ku, n_vec):
        self._Ku = float(Ku)
        self._n_ani = n_vec
        self._has_anisotropy = True

    def add_dmi_bulk(self, D_bulk):
        self._D_bulk = float(D_bulk)
        self._has_dmi_bulk = True

    def add_dmi_interfacial(self, D_int, n0_vec):
        self._D_int = float(D_int)
        self._n0_int = n0_vec
        self._has_dmi_int = True

    def add_external_field(self, H0_vec=None, H_time_func=None):
        self._H0_vec = H0_vec
        self._H_time_func = H_time_func
        self._has_H0 = H0_vec is not None

    def add_current(self, Jmagnitude, Jdir_vec, P, beta):
        self._Jmag = float(Jmagnitude)
        self._Jdir_vec = Jdir_vec
        self._P = float(P)
        self._beta = float(beta)
        self._has_current = True

    def _build_effective_field(self):
        Aex = self._Aex if self._has_exchange else 0.0

        if self._has_anisotropy and self._n_ani is not None:
            Ku = self._Ku
            n_ani_vec = self._n_ani
        else:
            Ku = 0.0
            n_ani_vec = np.zeros(3 * len(self.mesh.geometry.x), dtype=np.float64)

        D_bulk = self._D_bulk if self._has_dmi_bulk else 0.0

        if self._has_dmi_int and self._n0_int is not None:
            D_int = self._D_int
            n0_int_vec = self._n0_int
        else:
            D_int = 0.0
            n0_int_vec = np.zeros(3 * len(self.mesh.geometry.x), dtype=np.float64)

        if self._has_H0 and self._H0_vec is not None:
            H0_vec = self._H0_vec
        else:
            H0_vec = np.zeros(3 * len(self.mesh.geometry.x), dtype=np.float64)

        if self._has_current and self._Jdir_vec is not None:
            Jmag = self._Jmag
            Jdir_vec = self._Jdir_vec
            P = self._P
            beta = self._beta
        else:
            Jmag = 0.0
            Jdir_vec = np.zeros(3 * len(self.mesh.geometry.x), dtype=np.float64)
            P = 0.0
            beta = 0.0

        self.hef = EffectiveFieldSTTGPU(
            self.mesh,
            self.Ms,
            Aex,
            Ku,
            n_ani_vec,
            D_bulk,
            D_int,
            n0_int_vec,
            H0_static=H0_vec,
            gamma=self.gamma,
            alpha=self.alpha,
            do_precess=self.do_precess,
            Jmagnitude=Jmag,
            Jdir_vec=Jdir_vec,
            P=P,
            beta=beta,
            use_demag=self._has_demag,
            demag_method=self._demag_method,
            demag_kwargs=self._demag_kwargs,
            H_time_func=self._H_time_func,
        )

    def _cancel_ts_monitors(self):
        if self.ts is None:
            return
        try:
            self.ts.monitorCancel()
        except Exception:
            try:
                self.ts.setMonitor(None)
            except Exception:
                pass

    def _ensure_solver_bdf(
        self,
        m0_array,
        dt_init,
        ts_rtol=1e-6,
        ts_atol=1e-6,
        snes_rtol=1e-2,
        snes_atol=1e-4,
        ksp_rtol=1e-4,
        stopping_dmdt=0.0,
        check_every_stop=10,
        stop_print=False,
    ):

        if self.hef is None:
            self._build_effective_field()

        hef = self.hef

        if m0_array is not None:
            hef.set_m_from_cpu(m0_array)

        ts = PETSc.TS().create(self.mesh.comm)

        opts = PETSc.Options()
        opts["ts_type"] = "bdf"
        opts["ts_adapt_type"] = "basic"
        opts["ts_adapt_clip"] = "0.1, 3.0"
        opts["ts_adapt_safety"] = 0.9
        opts["ts_adapt_reject_safety"] = 0.1
        opts["ts_adapt_scale_solve_failed"] = 0.25
        opts["ts_adapt_dt_min"] = 1e-17
        opts["ts_adapt_dt_max"] = 1e-10

        opts["snes_type"] = "newtonls"
        opts["snes_linesearch_type"] = "bt"
        opts["snes_linesearch_order"] = 2

        opts["ts_rtol"] = ts_rtol
        opts["ts_atol"] = ts_atol
        opts["snes_rtol"] = snes_rtol
        opts["snes_atol"] = snes_atol
        opts["snes_max_it"] = 8
        opts["ksp_type"] = "gmres"
        opts["ksp_rtol"] = ksp_rtol
        opts["ts_max_snes_failures"] = -1
        opts["ts_max_steps"] = 5000000

        ts.setTime(0.0)
        ts.setTimeStep(float(dt_init))
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.STEPOVER)

        snes = ts.getSNES()
        n_loc = hef.m_gpu.getLocalSize()
        n_glob = hef.m_gpu.getSize()

        J = PETSc.Mat().create(comm=self.mesh.comm)
        ctx = JvContextSTT(hef)
        J.setSizes([[n_loc, n_glob], [n_loc, n_glob]])
        J.setType("python")
        J.setPythonContext(ctx)
        J.setUp()

        ksp = snes.getKSP()
        ksp.setType(PETSc.KSP.Type.GMRES)
        ksp.setTolerances(rtol=ksp_rtol, max_it=200)

        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.PYTHON)
        pc.setPythonContext(ctx)


        def IJac(ts_, t, y, ydot, shift, A, B):
            hef.current_time = float(t)

            hef.update_jac_state_STT(y)

            ctx.update_pc_full_fast_gpu(float(shift),include_stab=True,use_abs_kappa=True, )

            return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

        F_gpu = _dup_cuda_vec(hef.m_gpu, block_size=3)

        ts.setIFunction(hef.ifunction_STT, F_gpu)
        ts.setIJacobian(IJac, J, J)
        ts.setFromOptions()

        y = _dup_cuda_vec(hef.m_gpu, block_size=3)
        hef.m_gpu.copy(y)
        ts.setSolution(y)

        stopper = StopByMaxDmdtFDGPU(
            self.mesh.comm,
            stopping_dm_dt_deg_ns=stopping_dmdt,
            vec_template=y,
            n_owned_scalar=hef.local_size,
            check_every=check_every_stop,
            print_every_hit=stop_print,
        )
        ts.setPostStep(stopper)

        self.ts = ts
        self.y = y
        self.stopper = stopper
        self.ctx = ctx
        self.J = J
        self._solver_ready = True

    def solve_bdf(
        self,
        m0_array,
        t0,
        t_final,
        dt_init,
        dt_save=None,
        dt_snap=None,
        output_dir="output_stt_gpu_bdf",
        ts_rtol=1e-6,
        ts_atol=1e-6,
        snes_rtol=1e-2,
        snes_atol=1e-5,
        ksp_rtol=1e-4,
        stopping_dmdt=0.0,
        monitor_fn=None,
        save_final_state=True,
        check_every_stop=5,
        stop_print=False,
        return_stats=False,
    ):
        """
        BDF implicit STT solve. This is the default solve path.
        """
        self._ensure_solver_bdf(
            m0_array=m0_array,
            dt_init=dt_init,
            ts_rtol=ts_rtol,
            ts_atol=ts_atol,
            snes_rtol=snes_rtol,
            snes_atol=snes_atol,
            ksp_rtol=ksp_rtol,
            stopping_dmdt=stopping_dmdt,
            check_every_stop=check_every_stop,
            stop_print=stop_print,
        )

        ts = self.ts
        hef = self.hef
        comm = self.mesh.comm

        ts.setTime(float(t0))
        ts.setMaxTime(float(t_final))
        ts.setTimeStep(float(dt_init))
        ts.restartStep()

        if m0_array is not None:
            hef.set_m_from_cpu(m0_array)
            hef.m_gpu.copy(self.y)

        if comm.rank == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        comm.barrier()

        self._cancel_ts_monitors()

        if dt_save is not None:
            if dt_snap is None:
                dt_snap = dt_save

            log_path = Path(output_dir) / "log_stt_gpu_bdf.txt"
            last_save_n = {"n": -1}
            last_snap_n = {"n": -1}
            snap_counter = {"k": 0}
            first_print = {"done": False}

            def default_monitor(ts_, step, t, u, hef_, mesh_):
                dt_ts = ts_.getTimeStep()

                _sync_vec_to_function(u, hef_.m)
                u.copy(hef_.m_gpu)

                mag_local = hef_.m.x.array[: hef_.local_size].reshape((-1, 3))
                mag = mesh_.comm.gather(mag_local, root=0)

                energy = hef_.compute_Energy_terms()

                n_snap = int(np.trunc(t / dt_snap))
                if n_snap != last_snap_n["n"]:
                    last_snap_n["n"] = n_snap
                    filename = Path(output_dir) / f"m_stt_bdf_{snap_counter['k']:03d}.xdmf"
                    snap_counter["k"] += 1
                    with io.XDMFFile(mesh_.comm, str(filename), "w") as xdmf:
                        xdmf.write_mesh(mesh_)
                        xdmf.write_function(hef_.m)

                if mesh_.comm.rank == 0:
                    mag = np.reshape(np.concatenate(mag), (-1, 3))
                    maxdmdt_deg_ns = (
                        float(self.stopper.last_max_dmdt_deg_ns)
                        if self.stopper is not None
                        else 0.0
                    )
                    if not np.isfinite(maxdmdt_deg_ns):
                        maxdmdt_deg_ns = 0.0

                    if not first_print["done"]:
                        header = (
                            f"{'time':>10} {'dt':>10} "
                            f"{'<mx>':>15} {'<my>':>15} {'<mz>':>15} "
                            f"{'maxdmdt(deg/ns)':>18} "
                            f"{'E_demag':>15} {'E_exch':>15} {'E_ani':>15} "
                            f"{'E_dmi_bulk':>15} {'E_dmi_int':>15} {'E_total':>15}"
                        )
                        print(header)
                        with open(log_path, "w") as f:
                            f.write(header + "\n")
                        first_print["done"] = True

                    line = (
                        f"{t*1e9:10.4f} {dt_ts*1e9:10.4f}"
                        f"{mag[:,0].mean():15.6f} "
                        f"{mag[:,1].mean():15.6f} "
                        f"{mag[:,2].mean():15.6f} "
                        f"{maxdmdt_deg_ns:18.6e} "
                        f"{energy['E_demag']:15.4e} "
                        f"{energy['E_exch']:15.4e} "
                        f"{energy['E_ani']:15.4e} "
                        f"{energy['E_dmi_bulk']:15.4e} "
                        f"{energy['E_dmi_int']:15.4e} "
                        f"{energy['E_total']:15.4e}"
                    )
                    print(line)
                    with open(log_path, "a") as f:
                        f.write(line + "\n")
                    sys.stdout.flush()

            def monitor(ts_, step, t, u):
                n = int(np.trunc(t / dt_save))
                if n != last_save_n["n"]:
                    last_save_n["n"] = n
                    if monitor_fn is not None:
                        monitor_fn(ts_, step, t, u, hef, self.mesh)
                    else:
                        default_monitor(ts_, step, t, u, hef, self.mesh)

            ts.setMonitor(monitor)

        tstart = perf_counter()
        ts.solve(self.y)
        elapsed = perf_counter() - tstart

        self.y.copy(hef.m_gpu)
        hef.sync_m_to_function()

        stats = {
            "t_end": float(ts.getTime()),
            "dt_last": float(ts.getTimeStep()),
            "nsteps": int(ts.getStepNumber()),
            "reason": int(ts.getConvergedReason()),
            "maxdmdt_deg_ns": (
                float(self.stopper.last_max_dmdt_deg_ns)
                if self.stopper is not None
                else float("nan")
            ),
            "jv_calls": getattr(self.ctx, "calls", None),
            "pc_calls": getattr(self.ctx, "callsPre", None),
        }

        filename = Path(output_dir) / "Relax_STT_GPU_BDF.xdmf"
        with io.XDMFFile(self.mesh.comm, str(filename), "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            xdmf.write_function(hef.m)

        if save_final_state:
            fname = Path(output_dir) / "Relax_STT_GPU_BDF.bp"
            ad.write_mesh(fname, self.mesh)
            ad.write_function(fname, hef.m, time=0.0, name="m")

        if self.mesh.comm.rank == 0:
            print("STT BDF finished")
            print("nsteps:", stats["nsteps"])
            print("t_end:", stats["t_end"])
            print("dt_last:", stats["dt_last"])
            print("reason:", stats["reason"])
            print("Jv calls:", stats["jv_calls"])
            print("PC calls:", stats["pc_calls"])
            print("wall-clock:", elapsed)

        if return_stats:
            return self.y, self.ctx, elapsed, stats

        return self.y, self.ctx, elapsed

    def _ensure_solver_explicit(
        self,
        m0_array,
        dt_init,
        ts_rtol=1e-6,
        ts_atol=1e-6,
        rk_type="5dp",
        stopping_dmdt=0.0,
        check_every_stop=10,
        stop_print=False,
    ):
        if self.hef is None:
            self._build_effective_field()

        hef = self.hef

        if m0_array is not None:
            hef.set_m_from_cpu(m0_array)

        ts = PETSc.TS().create(self.mesh.comm)
        ts.setType("rk")
        ts.setRKType(rk_type)
        ts.setTime(0.0)
        ts.setTimeStep(float(dt_init))
        ts.setTolerances(rtol=ts_rtol, atol=ts_atol)
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.STEPOVER)

        ts.setRHSFunction(hef.rhs_function, hef.dmdt_gpu)

        opts = PETSc.Options()
        opts["ts_adapt_type"] = "basic"
        opts["ts_adapt_clip"] = "0.1, 3.0"
        opts["ts_adapt_safety"] = 0.9
        opts["ts_adapt_reject_safety"] = 0.1
        opts["ts_adapt_dt_min"] = 1e-18
        opts["ts_adapt_dt_max"] = 1e-11
        opts["ts_max_steps"] = 50000000
        ts.setFromOptions()

        y = _dup_cuda_vec(hef.m_gpu, block_size=3)
        hef.m_gpu.copy(y)
        ts.setSolution(y)

        stopper = StopByMaxDmdtFDGPU(
            self.mesh.comm,
            stopping_dm_dt_deg_ns=stopping_dmdt,
            vec_template=y,
            n_owned_scalar=hef.local_size,
            check_every=check_every_stop,
            print_every_hit=stop_print,
        )
        ts.setPostStep(stopper)

        self.ts = ts
        self.y = y
        self.stopper = stopper
        self._solver_ready = True

    def solve_explicit(
        self,
        m0_array,
        t0,
        t_final,
        dt_init,
        rk_type="5dp",
        dt_save=None,
        dt_snap=None,
        output_dir="output_stt_gpu",
        ts_rtol=1e-6,
        ts_atol=1e-6,
        stopping_dmdt=0.0,
        monitor_fn=None,
        save_final_state=True,
        check_every_stop=5,
        stop_print=False,
        return_stats=False,
    ):
        self._ensure_solver_explicit(
            m0_array=m0_array,
            dt_init=dt_init,
            ts_rtol=ts_rtol,
            ts_atol=ts_atol,
            rk_type=rk_type,
            stopping_dmdt=stopping_dmdt,
            check_every_stop=check_every_stop,
            stop_print=stop_print,
        )

        ts = self.ts
        hef = self.hef
        comm = self.mesh.comm

        ts.setTime(float(t0))
        ts.setMaxTime(float(t_final))
        ts.setTimeStep(float(dt_init))
        ts.restartStep()

        if comm.rank == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        comm.barrier()

        self._cancel_ts_monitors()

        if dt_save is not None:
            if dt_snap is None:
                dt_snap = dt_save

            log_path = Path(output_dir) / "log_stt_gpu.txt"
            last_save_n = {"n": -1}
            last_snap_n = {"n": -1}
            snap_counter = {"k": 0}
            first_print = {"done": False}

            def default_monitor(ts_, step, t, u, hef_, mesh_):
                dt_ts = ts_.getTimeStep()

                _sync_vec_to_function(u, hef_.m)
                u.copy(hef_.m_gpu)

                mag_local = hef_.m.x.array[: hef_.local_size].reshape((-1, 3))
                mag = mesh_.comm.gather(mag_local, root=0)

                energy = hef_.compute_Energy_terms()


                n_snap = int(np.trunc(t / dt_snap))
                if n_snap != last_snap_n["n"]:
                    last_snap_n["n"] = n_snap
                    filename = Path(output_dir) / f"m_stt_{snap_counter['k']:03d}.xdmf"
                    snap_counter["k"] += 1
                    with io.XDMFFile(mesh_.comm, str(filename), "w") as xdmf:
                        xdmf.write_mesh(mesh_)
                        xdmf.write_function(hef_.m)

                if mesh_.comm.rank == 0:
                    mag = np.reshape(np.concatenate(mag), (-1, 3))
                    maxdmdt_deg_ns = (
                        float(self.stopper.last_max_dmdt_deg_ns)
                        if self.stopper is not None
                        else 0.0
                    )
                    if not np.isfinite(maxdmdt_deg_ns):
                        maxdmdt_deg_ns = 0.0

                    if not first_print["done"]:
                        header = (
                            f"{'time':>10} {'dt':>10} "
                            f"{'<mx>':>15} {'<my>':>15} {'<mz>':>15} "
                            f"{'maxdmdt(deg/ns)':>18} "
                            f"{'E_demag':>15} {'E_exch':>15} {'E_ani':>15} "
                            f"{'E_dmi_bulk':>15} {'E_dmi_int':>15} {'E_total':>15}"
                        )
                        print(header)
                        with open(log_path, "w") as f:
                            f.write(header + "\n")
                        first_print["done"] = True

                    line = (
                        f"{t*1e9:10.4f} {dt_ts*1e9:10.4f}"
                        f"{mag[:,0].mean():15.6f} "
                        f"{mag[:,1].mean():15.6f} "
                        f"{mag[:,2].mean():15.6f} "
                        f"{maxdmdt_deg_ns:18.6e} "
                        f"{energy['E_demag']:15.4e} "
                        f"{energy['E_exch']:15.4e} "
                        f"{energy['E_ani']:15.4e} "
                        f"{energy['E_dmi_bulk']:15.4e} "
                        f"{energy['E_dmi_int']:15.4e} "
                        f"{energy['E_total']:15.4e}"
                    )
                    print(line)
                    with open(log_path, "a") as f:
                        f.write(line + "\n")
                    sys.stdout.flush()

            def monitor(ts_, step, t, u):
                n = int(np.trunc(t / dt_save))
                if n != last_save_n["n"]:
                    last_save_n["n"] = n
                    if monitor_fn is not None:
                        monitor_fn(ts_, step, t, u, hef, self.mesh)
                    else:
                        default_monitor(ts_, step, t, u, hef, self.mesh)

            ts.setMonitor(monitor)

        tstart = perf_counter()
        ts.solve(self.y)
        elapsed = perf_counter() - tstart

        self.y.copy(hef.m_gpu)
        hef.sync_m_to_function()

        stats = {
            "t_end": float(ts.getTime()),
            "dt_last": float(ts.getTimeStep()),
            "nsteps": int(ts.getStepNumber()),
            "reason": int(ts.getConvergedReason()),
            "maxdmdt_deg_ns": (
                float(self.stopper.last_max_dmdt_deg_ns)
                if self.stopper is not None
                else float("nan")
            ),
        }

        filename = Path(output_dir) / "Relax_STT_GPU.xdmf"
        with io.XDMFFile(self.mesh.comm, str(filename), "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            xdmf.write_function(hef.m)

        if save_final_state:
            fname = Path(output_dir) / "Relax_STT_GPU.bp"
            ad.write_mesh(fname, self.mesh)
            ad.write_function(fname, hef.m, time=0.0, name="m")

        if self.mesh.comm.rank == 0:
            print("STT explicit RK finished")
            print("RK type:", rk_type)
            print("nsteps:", stats["nsteps"])
            print("t_end:", stats["t_end"])
            print("dt_last:", stats["dt_last"])
            print("reason:", stats["reason"])
            print("wall-clock:", elapsed)

        if return_stats:
            return self.y, elapsed, stats

        return self.y, elapsed

    def solve(self, *args, **kwargs):
        return self.solve_bdf(*args, **kwargs)
