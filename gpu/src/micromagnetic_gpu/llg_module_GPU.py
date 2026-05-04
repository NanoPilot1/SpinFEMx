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
from dolfinx.fem.petsc import assemble_vector

# GPU-enabled contributions
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
# Stopping criterion on GPU
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
# Effective field + RHS/Jv on GPU
# -----------------------------------------------------------------------------
class EffectiveField:
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
        Kc1=0.0,
        u1_cub=None,
        u2_cub=None,
        gamma=2.211e5,
        alpha=0.5,
        do_precess=1,
        use_demag=False,
        demag_method="lindholm",
        demag_kwargs=None,
        H0_static=None,
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
        self.Kc1 = float(Kc1)
        self.do_precess = float(do_precess)
        self.use_demag = bool(use_demag)
        self.H_time_func = H_time_func
        self.current_time = 0.0

        if abs(self.Kc1) > 0.0:
            raise NotImplementedError(
                "Cubic anisotropy full-GPU path is not included here. Set Kc1=0."
            )

        self.prefactorEQ = -self.gamma / (1.0 + self.alpha**2)
        self.Stab = self.Ms * self.gamma / (1.0 + self.alpha**2) * 0.5

        # Host-side Functions only for I/O / diagnostics.
        self.m = fem.Function(self.V)
        self.dmdt = fem.Function(self.V)
        self.H_eff = fem.Function(self.V)
        self.H0_host = fem.Function(self.V)

        self.start, self.end = self.V.dofmap.index_map.local_range
        self.local_dofs = self.end - self.start
        self.local_size = 3 * self.local_dofs

        # -------------------------------------------------------------
        # Lumped nodal volumes (assembled once on DOLFINx layout)
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

        # -------------------------------------------------------------
        # Linear GPU operators
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
            if template_in is None:
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
        # If there are no linear terms, allow demag-only.
        # This also protects against template_in=None.
        # -------------------------------------------------------------
        if template_in is None:
            if self.use_demag:
                template_in = self.m.x.petsc_vec.duplicate()
                template_out = self.m.x.petsc_vec.duplicate()
            else:
                raise ValueError(
                    "There are no active GPU linear terms and no demag."
                    "You must add exchange, anisotropy, DMI, or demag."
                )

        _set_vec_cuda(template_in, block_size=3)
        _set_vec_cuda(template_out, block_size=3)

        # Persistent GPU state
        self.m_gpu = _dup_cuda_vec(template_in, block_size=3)
        self.dmdt_gpu = _dup_cuda_vec(template_in, block_size=3)
        self.H_eff_gpu = _dup_cuda_vec(template_out, block_size=3)
        self.Hm_gpu = _dup_cuda_vec(template_out, block_size=3)
        self.Hv_gpu = _dup_cuda_vec(template_out, block_size=3)
        self.H0_gpu = _dup_cuda_vec(template_out, block_size=3)
        self.Ht_gpu = _dup_cuda_vec(template_out, block_size=3)
        self.Jv_buffer = _dup_cuda_vec(template_in, block_size=3)






        # State used by Jv
        self.M_state_gpu = _dup_cuda_vec(template_in, block_size=3)

        if H0_static is not None:
            self.H0_host.x.array[:] = np.asarray(H0_static, dtype=np.float64)
            self.H0_host.x.scatter_forward()
            self.H0_host.x.petsc_vec.copy(self.H0_gpu)
        else:
            self.H0_gpu.zeroEntries()

        self.LLGSteps = 0
        self.JacSteps = 0

        self.diagK = _dup_cuda_vec(template_out, block_size=3)
        self.diagK_abs = _dup_cuda_vec(template_out, block_size=3)
        self.diagK.zeroEntries()
        for _, K, _ in self.linear_terms:
            d = K.getDiagonal()
            _set_vec_cuda(d, block_size=3)
            self.diagK.axpy(1.0, d)

        self.diagK.copy(self.diagK_abs)
        self.diagK_abs.abs()






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

    # -----------------------------------------------------------------
    # Host/device sync helpers
    # -----------------------------------------------------------------

    def external_field_mean(self, t=None):
        """
        Mean total external magnetic field for single-rank GPU execution.

        Computes:

            <H_ext> = <H0_static + H_time(t)>

        Returns
        -------
        np.ndarray, shape (3,)
            Mean total external magnetic field.
        """

        if t is not None:
            self.current_time = float(t)

        if self.H_time_func is not None:
            self._eval_H_time_vec(self.current_time)
        else:
            self.Ht_gpu.zeroEntries()

        H0_all = _vec_to_cupy(self.H0_gpu, "r")
        Ht_all = _vec_to_cupy(self.Ht_gpu, "r")

        H0 = H0_all[: self.local_size].reshape((-1, 3))
        Ht = Ht_all[: self.local_size].reshape((-1, 3))

        if H0.shape[0] == 0:
            return np.zeros(3, dtype=np.float64)

        Hext_mean = cp.mean(H0 + Ht, axis=0).get()

        return np.asarray(Hext_mean, dtype=np.float64)


    def set_m_from_cpu(self, m0_array):
        self.m.x.array[:] = np.asarray(m0_array, dtype=np.float64)
        self.m.x.scatter_forward()
        self.m.x.petsc_vec.copy(self.m_gpu)

    def sync_m_to_function(self):
        return _sync_vec_to_function(self.m_gpu, self.m)

    def sync_H_to_function(self):
        return _sync_vec_to_function(self.H_eff_gpu, self.H_eff)

    def set_uniform_field(self, Hx, Hy, Hz):
        H = _vec_to_cupy(self.H0_gpu, "rw")
        owned = H[: self.local_size].reshape((-1, 3))
        owned[:, 0] = Hx
        owned[:, 1] = Hy
        owned[:, 2] = Hz
        if H.size > self.local_size:
            H[self.local_size :] = 0.0

    # -----------------------------------------------------------------
    # External time-dependent field
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
            raise ValueError(f"Unsupported CuPy H_time_func shape {arr.shape}")

        arr = np.asarray(out, dtype=np.float64)
        if arr.ndim == 1 and arr.size == 3:
            owned[:, 0] = arr[0]
            owned[:, 1] = arr[1]
            owned[:, 2] = arr[2]
            return self.Ht_gpu
        if arr.ndim == 1 and arr.size == self.local_size:
            H[: self.local_size] = cp.asarray(arr)
            return self.Ht_gpu
        if arr.shape == (self.local_dofs, 3):
            owned[:, :] = cp.asarray(arr)
            return self.Ht_gpu

        raise ValueError(
            "H_time_func must return PETSc.Vec, CuPy/NumPy (3,), CuPy/NumPy (local_size,), or (local_dofs, 3)."
        )

    # -----------------------------------------------------------------
    # Effective field
    # -----------------------------------------------------------------

    def apply_linear_field_vec(self, x_vec: PETSc.Vec, out_vec: PETSc.Vec):
        out_vec.zeroEntries()

        for _, K, buf in self.linear_terms:
            K.mult(x_vec, buf)
            out_vec.axpy(1.0, buf)

    def compute_H_eff_vec(self, m_vec: PETSc.Vec, out_vec: PETSc.Vec):
        # Exchange + anisotropy + DMI
        self.apply_linear_field_vec(m_vec, out_vec)

        # Demag FMM/JAX on GPU
        if self.demag_field is not None:
            self.demag_field.compute_vec(m_vec, self.H_demag_gpu)
            out_vec.axpy(1.0, self.H_demag_gpu)

        # static external magnetic field
        out_vec.axpy(1.0, self.H0_gpu)

        # Time dependent external magnetic field
        if self.H_time_func is not None:
            self._eval_H_time_vec(self.current_time)
            out_vec.axpy(1.0, self.Ht_gpu)

    def compute_H_eff(self, m_fun):
        self.compute_H_eff_vec(m_fun.x.petsc_vec, self.H_eff.x.petsc_vec)
        return self.H_eff


    def compute_H_jac_vec(self, m_vec: PETSc.Vec, out_vec: PETSc.Vec):
        """
        Field used ONLY for Jacobian and preconditioner.

        Includes:

        exchange + anisotropy + DMI + external field

        Excludes:

        demagnetism

        This maintains demagnetism in the RHS, but not in the Jacobian or PC.
        """

        # Linear local part
        self.apply_linear_field_vec(m_vec, out_vec)

        # static external magnetic field
        out_vec.axpy(1.0, self.H0_gpu)

        # Time dependent external magnetic field
        if self.H_time_func is not None:
            self._eval_H_time_vec(self.current_time)
            out_vec.axpy(1.0, self.Ht_gpu)
        # -----------------------------------------------------------------
    # RHS and Jacobian-times-vector in CuPy
    # -----------------------------------------------------------------
    def llg_rhs_vec(self, m_vec: PETSc.Vec, out_vec: PETSc.Vec):
        self.LLGSteps += 1

        self.compute_H_eff_vec(m_vec, self.H_eff_gpu)

        m_all = _vec_to_cupy(m_vec, "r")
        h_all = _vec_to_cupy(self.H_eff_gpu, "r")
        rhs_all = _vec_to_cupy(out_vec, "rw")

        m = m_all[: self.local_size].reshape((-1, 3))
        h = h_all[: self.local_size].reshape((-1, 3))
        rhs = rhs_all[: self.local_size].reshape((-1, 3))

        mx = m[:, 0]; my = m[:, 1]; mz = m[:, 2]
        hx = h[:, 0]; hy = h[:, 1]; hz = h[:, 2]

        # m x H
        mcx = my * hz - mz * hy
        mcy = mz * hx - mx * hz
        mcz = mx * hy - my * hx

        # m x (m x H)
        mcmx = my * mcz - mz * mcy
        mcmy = mz * mcx - mx * mcz
        mcmz = mx * mcy - my * mcx

        norm2 = mx * mx + my * my + mz * mz
        stab = self.Stab * (1.0 - norm2)

        rhs[:, 0] = self.prefactorEQ * (self.do_precess * mcx + self.alpha * mcmx) + stab * mx
        rhs[:, 1] = self.prefactorEQ * (self.do_precess * mcy + self.alpha * mcmy) + stab * my
        rhs[:, 2] = self.prefactorEQ * (self.do_precess * mcz + self.alpha * mcmz) + stab * mz

        if rhs_all.size > self.local_size:
            rhs_all[self.local_size :] = 0.0

    def update_jac_state(self, m_vec: PETSc.Vec):
        m_vec.copy(self.M_state_gpu)

        self.compute_H_jac_vec(m_vec, self.Hm_gpu)


    def jac_vec_times_vec(self, v_vec: PETSc.Vec, out_vec: PETSc.Vec):
        self.JacSteps += 1

        self.apply_linear_field_vec(v_vec, self.Hv_gpu)

        m_all = _vec_to_cupy(self.M_state_gpu, "r")
        hm_all = _vec_to_cupy(self.Hm_gpu, "r")
        v_all = _vec_to_cupy(v_vec, "r")
        hv_all = _vec_to_cupy(self.Hv_gpu, "r")
        out_all = _vec_to_cupy(out_vec, "rw")

        m = m_all[: self.local_size].reshape((-1, 3))
        hm = hm_all[: self.local_size].reshape((-1, 3))
        v = v_all[: self.local_size].reshape((-1, 3))
        hv = hv_all[: self.local_size].reshape((-1, 3))
        out = out_all[: self.local_size].reshape((-1, 3))

        mx = m[:, 0]; my = m[:, 1]; mz = m[:, 2]
        hx = hm[:, 0]; hy = hm[:, 1]; hz = hm[:, 2]
        vx = v[:, 0]; vy = v[:, 1]; vz = v[:, 2]
        hvx = hv[:, 0]; hvy = hv[:, 1]; hvz = hv[:, 2]

        # v x H(m)
        cvx = vy * hz - vz * hy
        cvy = vz * hx - vx * hz
        cvz = vx * hy - vy * hx

        # m x H(v)
        cmx = my * hvz - mz * hvy
        cmy = mz * hvx - mx * hvz
        cmz = mx * hvy - my * hvx

        mdHm = mx * hx + my * hy + mz * hz
        mdHv = mx * hvx + my * hvy + mz * hvz
        vdHm = vx * hx + vy * hy + vz * hz
        mdv = mx * vx + my * vy + mz * vz
        mdmm = mx * mx + my * my + mz * mz

        common = vdHm + mdHv

        dampx = vx * mdHm - 2.0 * hx * mdv + mx * common - hvx * mdmm
        dampy = vy * mdHm - 2.0 * hy * mdv + my * common - hvy * mdmm
        dampz = vz * mdHm - 2.0 * hz * mdv + mz * common - hvz * mdmm

        coef = -self.gamma / (1.0 + self.alpha**2)
        stab_common = self.Stab * (1.0 - mdmm)

        out[:, 0] = coef * (self.do_precess * (cvx + cmx) + self.alpha * dampx)
        out[:, 1] = coef * (self.do_precess * (cvy + cmy) + self.alpha * dampy)
        out[:, 2] = coef * (self.do_precess * (cvz + cmz) + self.alpha * dampz)

        out[:, 0] += stab_common * vx - 2.0 * self.Stab * mx * mdv
        out[:, 1] += stab_common * vy - 2.0 * self.Stab * my * mdv
        out[:, 2] += stab_common * vz - 2.0 * self.Stab * mz * mdv

        if out_all.size > self.local_size:
            out_all[self.local_size :] = 0.0

    # -----------------------------------------------------------------
    # PETSc TS callbacks
    # -----------------------------------------------------------------
    def ifunction(self, ts, t, y, ydot, f):

        self.current_time = float(t)

        # dmdt_gpu = RHS(t, y)
        self.llg_rhs_vec(y, self.dmdt_gpu)

        # f = ydot - RHS(t, y)
        ydot.copy(f)
        f.axpy(-1.0, self.dmdt_gpu)

        return 0

    # -----------------------------------------------------------------
    # Diagnostics / output
    # -----------------------------------------------------------------
    def compute_Energy_terms(self):

        m_fun = self.sync_m_to_function()

        E_exch = 0.0
        E_ani = 0.0
        E_dmi_bulk = 0.0
        E_dmi_int = 0.0
        E_demag = 0.0
        E_cubic = 0.0

        if self.exchange_field is not None:
            E_exch = float(self.exchange_field.Energy(m_fun))

        if self.anisotropy_field is not None:
            E_ani = float(self.anisotropy_field.Energy(m_fun))

        if self.demag_field is not None:
            self.demag_field.compute_vec(self.m_gpu, self.H_demag_gpu)
            E_demag = self.demag_field.Energy_lumped_gpu(self.m_gpu, self.H_demag_gpu)

        if self.DMIBULK is not None:
            self.DMIBULK.compute(self.m_gpu)
            E_dmi_bulk = float(self.DMIBULK.Energy(m_fun))

        if self.DMI_int is not None:
            self.DMI_int.compute(self.m_gpu)
            E_dmi_int = float(self.DMI_int.Energy(m_fun))

        E_total = (
            E_exch
            + E_demag
            + E_ani
            + E_dmi_bulk
            + E_dmi_int
            + E_cubic
        )

        return {
            "E_demag": E_demag,
            "E_exch": E_exch,
            "E_ani": E_ani,
            "E_dmi_bulk": E_dmi_bulk,
            "E_dmi_int": E_dmi_int,
            "E_cubic": E_cubic,
            "E_total": E_total,
        }

    def compute_Energy(self):

        return self.compute_Energy_terms()["E_total"]

    def rhs_function(self, ts, t, y, f):


        self.current_time = float(t)
        self.llg_rhs_vec(y, f)
        return 0
# -----------------------------------------------------------------------------
# Jacobian matrix-free context: y <- shift*x - Jx
# -----------------------------------------------------------------------------
class JvContext:
    def __init__(self, hef, eps_reg=1e-14, det_eps=1e-30):
        self.hef = hef
        self.shift = 0.0
        self.calls = 0
        self.callsPre = 0

        self.diagA = _dup_cuda_vec(self.hef.diagK_abs, block_size=3)
        self.inv_diagA = _dup_cuda_vec(self.hef.diagK_abs, block_size=3)

        self.hef = hef
        self.shift = 0.0
        self.calls = 0
        self.callsPre = 0

        self.eps_reg = float(eps_reg)
        self.det_eps = float(det_eps)

        n = self.hef.local_dofs

        self.i00 = cp.empty(n); self.i01 = cp.empty(n); self.i02 = cp.empty(n)
        self.i10 = cp.empty(n); self.i11 = cp.empty(n); self.i12 = cp.empty(n)
        self.i20 = cp.empty(n); self.i21 = cp.empty(n); self.i22 = cp.empty(n)

        self._pc_ready = False

        self.gamma = float(self.hef.gamma)
        self.alpha = float(self.hef.alpha)
        self.do_precess = float(self.hef.do_precess)
        self.Stab = float(self.hef.Stab)

        self.c1 = self.gamma * self.do_precess / (1.0 + self.alpha*self.alpha)
        self.c2 = self.gamma * self.alpha / (1.0 + self.alpha*self.alpha)

    def update_pc_full_fast_gpu(self, shift, include_stab=True, use_abs_kappa=True):
        self.shift = float(shift)

        local_size = self.hef.local_size

        M_all = cp.from_dlpack(self.hef.M_state_gpu.toDLPack("r"))
        H_all = cp.from_dlpack(self.hef.Hm_gpu.toDLPack("r"))
        D_all = cp.from_dlpack(self.hef.diagK.toDLPack("r"))

        M = M_all[:local_size].reshape((-1, 3))
        H = H_all[:local_size].reshape((-1, 3))
        D = D_all[:local_size].reshape((-1, 3))

        mx = M[:, 0]; my = M[:, 1]; mz = M[:, 2]
        hx = H[:, 0]; hy = H[:, 1]; hz = H[:, 2]

        if use_abs_kappa:
            kappa = cp.mean(cp.abs(D), axis=1)
        else:
            kappa = cp.mean(D, axis=1)

        c1 = self.c1
        c2 = self.c2
        Stab = self.Stab

        mdH = mx*hx + my*hy + mz*hz
        mdm = mx*mx + my*my + mz*mz

        # Precession approximate block
        Jp00 = 0.0
        Jp01 = -c1*(-hz - kappa*(-mz))
        Jp02 = -c1*( hy - kappa*( my))

        Jp10 = -c1*( hz - kappa*( mz))
        Jp11 = 0.0
        Jp12 = -c1*(-hx - kappa*(-mx))

        Jp20 = -c1*(-hy - kappa*(-my))
        Jp21 = -c1*( hx - kappa*( mx))
        Jp22 = 0.0

        # Damping approximate block
        B00 = mdH - hx*mx
        B11 = mdH - hy*my
        B22 = mdH - hz*mz

        B01 = mx*hy - 2.0*hx*my
        B02 = mx*hz - 2.0*hx*mz
        B10 = my*hx - 2.0*hy*mx
        B12 = my*hz - 2.0*hy*mz
        B20 = mz*hx - 2.0*hz*mx
        B21 = mz*hy - 2.0*hz*my

        C00 = mx*mx - mdm
        C11 = my*my - mdm
        C22 = mz*mz - mdm

        C01 = mx*my
        C02 = mx*mz
        C10 = my*mx
        C12 = my*mz
        C20 = mz*mx
        C21 = mz*my

        Jd00 = c2*(B00 + kappa*C00)
        Jd11 = c2*(B11 + kappa*C11)
        Jd22 = c2*(B22 + kappa*C22)

        Jd01 = c2*(B01 + kappa*C01)
        Jd02 = c2*(B02 + kappa*C02)
        Jd10 = c2*(B10 + kappa*C10)
        Jd12 = c2*(B12 + kappa*C12)
        Jd20 = c2*(B20 + kappa*C20)
        Jd21 = c2*(B21 + kappa*C21)

        if include_stab:
            s0 = 1.0 - mdm

            Js00 = Stab*(s0 - 2.0*mx*mx)
            Js11 = Stab*(s0 - 2.0*my*my)
            Js22 = Stab*(s0 - 2.0*mz*mz)

            Js01 = Stab*(-2.0*mx*my)
            Js02 = Stab*(-2.0*mx*mz)
            Js10 = Stab*(-2.0*my*mx)
            Js12 = Stab*(-2.0*my*mz)
            Js20 = Stab*(-2.0*mz*mx)
            Js21 = Stab*(-2.0*mz*my)
        else:
            Js00 = Js11 = Js22 = 0.0
            Js01 = Js02 = Js10 = Js12 = Js20 = Js21 = 0.0

        J00 = Jd00 + Js00
        J11 = Jd11 + Js11
        J22 = Jd22 + Js22

        J01 = Jp01 + Jd01 + Js01
        J02 = Jp02 + Jd02 + Js02
        J10 = Jp10 + Jd10 + Js10
        J12 = Jp12 + Jd12 + Js12
        J20 = Jp20 + Jd20 + Js20
        J21 = Jp21 + Jd21 + Js21

        # A = shift I - J + eps I
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
            A00*(A11*A22 - A12*A21)
            - A01*(A10*A22 - A12*A20)
            + A02*(A10*A21 - A11*A20)
        )

        det = cp.where(cp.abs(det) < self.det_eps,
                    det + cp.sign(det + self.det_eps)*self.det_eps,
                    det)

        invdet = 1.0 / det

        self.i00[:] =  (A11*A22 - A12*A21) * invdet
        self.i01[:] = -(A01*A22 - A02*A21) * invdet
        self.i02[:] =  (A01*A12 - A02*A11) * invdet

        self.i10[:] = -(A10*A22 - A12*A20) * invdet
        self.i11[:] =  (A00*A22 - A02*A20) * invdet
        self.i12[:] = -(A00*A12 - A02*A10) * invdet

        self.i20[:] =  (A10*A21 - A11*A20) * invdet
        self.i21[:] = -(A00*A21 - A01*A20) * invdet
        self.i22[:] =  (A00*A11 - A01*A10) * invdet

        self._pc_ready = True


    def update_pc(self, shift):
        self.shift = float(shift)

        self.hef.diagK_abs.copy(self.diagA)
        self.diagA.scale(self.jacobi_scale)
        self.diagA.shift(abs(self.shift) + abs(self.hef.Stab) + self.eps)

        self.diagA.copy(self.inv_diagA)
        self.inv_diagA.reciprocal()

    def mult(self, A, x, y):
        self.calls += 1
        self.hef.jac_vec_times_vec(x, self.hef.Jv_buffer)

        # y = shift*x - Jv
        x.copy(y)
        y.scale(self.shift)
        y.axpy(-1.0, self.hef.Jv_buffer)

    def apply(self, pc, x, y):
        self.callsPre += 1

        if not self._pc_ready:
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

        Y[:, 0] = self.i00*x0 + self.i01*x1 + self.i02*x2
        Y[:, 1] = self.i10*x0 + self.i11*x1 + self.i12*x2
        Y[:, 2] = self.i20*x0 + self.i21*x1 + self.i22*x2

        if y_all.size > local_size:
            y_all[local_size:] = 0.0


# -----------------------------------------------------------------------------
# Main LLG driver
# -----------------------------------------------------------------------------
class LLG_GPU:
    def __init__(self, mesh, Ms, gamma=2.211e5, alpha=0.5, do_precess=1):
        self.mesh = mesh
        self.Ms = Ms
        self.gamma = gamma
        self.alpha = alpha
        self.do_precess = do_precess

        self._Aex = 0.0
        self._Ku = 0.0
        self._n_ani = None
        self._D_bulk = 0.0
        self._D_int = 0.0
        self._n0_int = None
        self._H0_vec = None
        self._H_time_func = None
        self._Kc1 = 0.0
        self._u1_cub = None
        self._u2_cub = None

        self._has_exchange = False
        self._has_demag = False
        self._has_anisotropy = False
        self._has_dmi_bulk = False
        self._has_dmi_int = False
        self._has_cubic = False

        self._demag_method = "lindholm"
        self._demag_kwargs = {}

        self.hef: EffectiveField | None = None
        self.ts = None
        self.ctx = None
        self.J = None
        self.y = None
        self.stopper = None
        self._solver_ready = False

    def add_exchange(self, Aex):
        self._Aex = Aex
        self._has_exchange = True

    def add_demag(self, method="lindholm", **kwargs):
        self._has_demag = True
        self._demag_method = method
        self._demag_kwargs = dict(kwargs)

    def add_anisotropy(self, Ku, n_vec):
        self._Ku = Ku
        self._n_ani = n_vec
        self._has_anisotropy = True

    def add_dmi_bulk(self, D_bulk):
        self._D_bulk = D_bulk
        self._has_dmi_bulk = True

    def add_dmi_interfacial(self, D_int, n0_vec):
        self._D_int = D_int
        self._n0_int = n0_vec
        self._has_dmi_int = True

    def add_external_field(self, H0_vec=None, H_time_func=None):
        self._H0_vec = H0_vec
        self._H_time_func = H_time_func

    def add_cubic_anisotropy(self, Kc1, u1_vec, u2_vec):
        self._Kc1 = float(Kc1)
        self._u1_cub = u1_vec
        self._u2_cub = u2_vec
        self._has_cubic = True

    def _build_effective_field(self):
        Aex = self._Aex if self._has_exchange else 0.0

        if self._has_anisotropy and self._n_ani is not None:
            Ku = self._Ku
            n_ani_vec = self._n_ani
        else:
            Ku = 0.0
            n_ani_vec = np.zeros(3 * len(self.mesh.geometry.x), dtype=np.float64)

        if self._has_dmi_bulk:
            D_bulk = self._D_bulk
        else:
            D_bulk = 0.0

        if self._has_dmi_int and self._n0_int is not None:
            D_int = self._D_int
            n0_int_vec = self._n0_int
        else:
            D_int = 0.0
            n0_int_vec = np.zeros(3 * len(self.mesh.geometry.x), dtype=np.float64)

        if self._has_cubic and self._u1_cub is not None and self._u2_cub is not None:
            Kc1 = self._Kc1
            u1_cub = self._u1_cub
            u2_cub = self._u2_cub
        else:
            Kc1 = 0.0
            u1_cub = None
            u2_cub = None

        self.hef = EffectiveField(
            self.mesh,
            self.Ms,
            Aex,
            Ku,
            n_ani_vec,
            D_bulk,
            D_int,
            n0_int_vec,
            Kc1=Kc1,
            u1_cub=u1_cub,
            u2_cub=u2_cub,
            gamma=self.gamma,
            alpha=self.alpha,
            do_precess=self.do_precess,
            use_demag=self._has_demag,
            demag_method=self._demag_method,
            demag_kwargs=self._demag_kwargs,
            H0_static=self._H0_vec,
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

    def _reset_ts_run(self, t0, t_final, dt_init):
        ts = self.ts
        ts.setTime(float(t0))
        ts.setMaxTime(float(t_final))
        ts.setTimeStep(float(dt_init))
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.STEPOVER)
        ts.restartStep()
        try:
            ts.setStepNumber(0)
        except Exception:
            pass
        if self.stopper is not None:
            self.stopper.reset(self.y)

    def _run_ts(self):
        ts = self.ts
        hef = self.hef
        y = self.y

        tstart = perf_counter()
        ts.solve(y)
        elapsed = perf_counter() - tstart

        y.copy(hef.m_gpu)
        hef.sync_m_to_function()

        if self.mesh.comm.rank == 0:
            print("LLGCalls", hef.LLGSteps)

        stats = {
            "t_end": float(ts.getTime()),
            "dt_last": float(ts.getTimeStep()),
            "nsteps": int(ts.getStepNumber()),
            "reason": int(ts.getConvergedReason()),
            "maxdmdt_deg_ns": float(self.stopper.last_max_dmdt_deg_ns) if self.stopper is not None else float("nan"),
        }
        return elapsed, stats

    def _ensure_solver_explicit(
        self,
        m0_array,
        dt_init,
        ts_rtol=1e-5,
        ts_atol=1e-5,
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

        # Explicit Runge-Kutta
        ts.setType("rk")
        ts.setRKType(rk_type)

        ts.setTime(0.0)
        ts.setTimeStep(float(dt_init))
        ts.setTolerances(rtol=ts_rtol, atol=ts_atol)
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.STEPOVER)

        # RHS: dm/dt = F(t, m)
        ts.setRHSFunction(hef.rhs_function, hef.dmdt_gpu)

        # Estado inicial en GPU
        y = _dup_cuda_vec(hef.m_gpu, block_size=3)
        hef.m_gpu.copy(y)
        ts.setSolution(y)

        # RK adaptive setting
        opts = PETSc.Options()
        opts["ts_adapt_type"] = "basic"
        opts["ts_adapt_clip"] = "0.1, 3.0"
        opts["ts_adapt_safety"] = 0.9
        opts["ts_adapt_reject_safety"] = 0.1
        opts["ts_adapt_dt_min"] = 1e-18
        opts["ts_adapt_dt_max"] = 1e-10
        opts["ts_max_steps"] = 50000000

        ts.setFromOptions()

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


    def relax_explicit(
        self,
        m0_array,
        t0,
        t_final,
        dt_init,
        rk_type="5dp",
        dt_save=None,
        dt_snap=None,
        output_dir="output_explicit",
        ts_rtol=1e-6,
        ts_atol=1e-6,
        stopping_dmdt=0.0,
        monitor_fn=None,
        save_final_state=True,
        check_every_stop=5,
        stop_print=False,
        return_stats=False,
    ):
        """
        Explicit RK relaxation.
        Useful to test the GPU RHS without Newton/KSP/preconditioner.
        """

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

            log_path = Path(output_dir) / "log_explicit.txt"
            last_save_n = {"n": -1}
            last_snap_n = {"n": -1}
            snap_counter = {"k": 0}
            first_print = {"done": False}

            def default_monitor(ts_, step, t, u, hef_, mesh_):
                dt_ts = ts_.getTimeStep()

                _sync_vec_to_function(u, hef_.m)
                mag_local = hef_.m.x.array[: hef_.local_size].reshape((-1, 3))
                mag = mesh_.comm.gather(mag_local, root=0)

                n_snap = int(np.trunc(t / dt_snap))
                if n_snap != last_snap_n["n"]:
                    last_snap_n["n"] = n_snap
                    filename = Path(output_dir) / f"m_explicit_{snap_counter['k']:03d}.xdmf"
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
                            f"{'maxdmdt(deg/ns)':>18}"
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
                        f"{maxdmdt_deg_ns:18.6e}"
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

        filename = Path(output_dir) / "Relax_explicit.xdmf"
        with io.XDMFFile(self.mesh.comm, str(filename), "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            xdmf.write_function(hef.m)

        if save_final_state:
            fname = Path(output_dir) / "Relax_explicit.bp"
            ad.write_mesh(fname, self.mesh)
            ad.write_function(fname, hef.m, time=0.0, name="m")

        if self.mesh.comm.rank == 0:
            print("Explicit RK finished")
            print("RK type:", rk_type)
            print("nsteps:", stats["nsteps"])
            print("t_end:", stats["t_end"])
            print("dt_last:", stats["dt_last"])
            print("reason:", stats["reason"])
            print("wall-clock:", elapsed)

        if return_stats:
            return self.y, elapsed, stats

        return self.y, elapsed




    def _ensure_solver(
        self,
        m0_array,
        dt_init,
        ts_rtol=1e-6,
        ts_atol=1e-6,
        snes_rtol=1e-3,
        snes_atol=1e-5,
        ksp_rtol=1e-5,
        stopping_dmdt=0.0,
        check_every_stop=10,
        stop_print=False,
    ):
        if self.hef is None:
            self._build_effective_field()
        hef = self.hef

        if not self._solver_ready:
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
            opts["ksp_reuse_preconditioner"] = "true"
            opts["snes_lag_preconditioner"] = 1


            ts.setTime(0.0)
            ts.setTimeStep(dt_init)
            ts.setExactFinalTime(PETSc.TS.ExactFinalTime.STEPOVER)

            snes = ts.getSNES()
            n_loc = hef.m_gpu.getLocalSize()
            n_glob = hef.m_gpu.getSize()

            J = PETSc.Mat().create(comm=self.mesh.comm)
            ctx = JvContext(hef)
            J.setSizes([[n_loc, n_glob], [n_loc, n_glob]])
            J.setType("python")
            J.setPythonContext(ctx)
            J.setUp()

            ksp = snes.getKSP()
            ksp.setType(PETSc.KSP.Type.GMRES)
            ksp.setTolerances(rtol=ksp_rtol, max_it=200)

            pc = ksp.getPC()
            #pc.setType(PETSc.PC.Type.NONE)
            pc.setType(PETSc.PC.Type.PYTHON)
            pc.setPythonContext(ctx)

            def IJac(ts_, t, y, ydot, shift, A, B):
                hef.current_time = float(t)
                hef.update_jac_state(y)

                #ctx.shift = float(shift)

                ctx.update_pc_full_fast_gpu(float(shift), include_stab=True, use_abs_kappa=True)

                return PETSc.Mat.Structure.SAME_NONZERO_PATTERN
            
            # Explicitly CUDA residual vector. Prevents PETSc from creating a CPU vector.
            # For implicit residual vectors while RHS/Jv reside in CUDA/CuPy.

            F_gpu = _dup_cuda_vec(hef.m_gpu, block_size=3)

            ts.setIFunction(hef.ifunction, F_gpu)
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

            self.ts, self.ctx, self.J, self.y, self.stopper = ts, ctx, J, y, stopper
            self._solver_ready = True
        else:
            if m0_array is not None:
                hef.set_m_from_cpu(m0_array)
                hef.m_gpu.copy(self.y)

    def relax(
        self,
        m0_array,
        t0,
        t_final,
        dt_init,
        dt_save=None,
        dt_snap=None,
        output_dir="output",
        ts_rtol=1e-6,
        ts_atol=1e-6,
        snes_rtol=1e-2,
        snes_atol=1e-4,
        ksp_rtol=1e-4,
        stopping_dmdt=0.0,
        monitor_fn=None,
        save_final_state=True,
        check_every_stop=5,
        stop_print=False,
        return_stats=False,
    ):
        self._ensure_solver(
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

        if dt_save is not None:
            if dt_snap is None:
                dt_snap = dt_save
            if comm.rank == 0:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            comm.barrier()

        self._cancel_ts_monitors()

        if dt_save is not None:
            log_path = Path(output_dir) / "log.txt"
            last_save_n = {"n": -1}
            last_snap_n = {"n": -1}
            snap_counter = {"k": 0}
            first_print = {"done": False}

            def default_monitor(ts_, step, t, u, hef_, mesh_):
                dt_ts = ts_.getTimeStep()

                # 1. synchronize solution
                _sync_vec_to_function(u, hef_.m)
                u.copy(hef_.m_gpu)

                # 2. Average magnetization.
                mag_local = hef_.m.x.array[: hef_.local_size].reshape((-1, 3))
                mag = mesh_.comm.gather(mag_local, root=0)

                # 3. Compute energies
                energy = hef_.compute_Energy_terms()
                Hext_mean = hef_.external_field_mean(t)


                # 4. Snapshot XDMF.
                n_snap = int(np.trunc(t / dt_snap))
                if n_snap != last_snap_n["n"]:
                    last_snap_n["n"] = n_snap
                    filename = Path(output_dir) / f"m{snap_counter['k']:03d}.xdmf"
                    snap_counter["k"] += 1
                    with io.XDMFFile(mesh_.comm, str(filename), "w") as xdmf:
                        xdmf.write_mesh(mesh_)
                        xdmf.write_function(hef_.m)

                # 5. Print/log in rank 0.
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
                            f"{'Hx_ext':>15} {'Hy_ext':>15} {'Hz_ext':>15} "
                            f"{'maxdmdt(deg/ns)':>18} "
                            f"{'E_demag':>15} {'E_exch':>15} {'E_ani':>15} "
                            f"{'E_dmi_bulk':>15} {'E_dmi_int':>15} "
                            f"{'E_cubic':>15} {'E_total':>15}"
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
                        f"{Hext_mean[0]:15.6e} "
                        f"{Hext_mean[1]:15.6e} "
                        f"{Hext_mean[2]:15.6e} "
                        f"{maxdmdt_deg_ns:18.6e} "
                        f"{energy['E_demag']:15.4e} "
                        f"{energy['E_exch']:15.4e} "
                        f"{energy['E_ani']:15.4e} "
                        f"{energy['E_dmi_bulk']:15.4e} "
                        f"{energy['E_dmi_int']:15.4e} "
                        f"{energy['E_cubic']:15.4e} "
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

        self._reset_ts_run(t0=t0, t_final=t_final, dt_init=dt_init)
        elapsed, stats = self._run_ts()

        if comm.rank == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        comm.barrier()

        hef.sync_m_to_function()
        filename = Path(output_dir) / "Relax.xdmf"
        with io.XDMFFile(self.mesh.comm, str(filename), "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            xdmf.write_function(hef.m)

        if save_final_state:
            fname = Path(output_dir) / "Relax.bp"
            ad.write_mesh(fname, self.mesh)
            ad.write_function(fname, hef.m, time=0.0, name="m")

        if return_stats:
            return self.y, self.ctx, elapsed, stats
        return self.y, self.ctx, elapsed

    def hysteresis(
        self,
        m0_array,
        H_steps,
        t_final_per_step,
        dt_init,
        output_dir="hyst_out",
        ts_rtol=1e-6,
        ts_atol=1e-6,
        snes_rtol=1e-2,
        snes_atol=1e-4,
        ksp_rtol=1e-4,
        stopping_dmdt=0.0,
        check_every_stop=10,
        stop_print=False,
        xdmf_name="Hysteresis.xdmf",
        log_name="hysteresis_log.txt",
        write_xdmf_series=False,
        write_xdmf_per_step=True,
        write_bp_series=True,
        bp_name="Hysteresis.bp",
    ):
        H_steps = np.asarray(list(H_steps), dtype=float).reshape((-1, 3))
        comm = self.mesh.comm

        self._ensure_solver(
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

        self._cancel_ts_monitors()

        if comm.rank == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        comm.barrier()

        log_path = Path(output_dir) / log_name
        if comm.rank == 0:
            with open(log_path, "w") as f:
                f.write("# step Hx Hy Hz  <mx> <my> <mz>  maxdmdt(deg/ns)  reason nsteps dt_last\n")

        xdmf = None
        if write_xdmf_series:
            xdmf_path = Path(output_dir) / xdmf_name
            xdmf = io.XDMFFile(comm, str(xdmf_path), "w")
            xdmf.write_mesh(self.mesh)

        bp_path = Path(output_dir) / bp_name
        if write_bp_series:
            ad.write_mesh(bp_path, self.mesh)

        results = []

        for i, (Hx, Hy, Hz) in enumerate(H_steps):
            hef.set_uniform_field(Hx, Hy, Hz)
            self._reset_ts_run(t0=0.0, t_final=t_final_per_step, dt_init=dt_init)
            elapsed, stats = self._run_ts()

            hef.sync_m_to_function()

            if xdmf is not None:
                xdmf.write_function(hef.m, float(i))

            if write_xdmf_per_step:
                fname = Path(output_dir) / f"m_{i:05d}.xdmf"
                with io.XDMFFile(comm, str(fname), "w") as xf:
                    xf.write_mesh(self.mesh)
                    xf.write_function(hef.m)

            if write_bp_series:
                ad.write_function(bp_path, hef.m, time=float(i), name="m")

            mloc = hef.m.x.array[: hef.local_size].reshape((-1, 3))
            s_loc = mloc.sum(axis=0) if mloc.size else np.zeros(3)
            n_loc = mloc.shape[0]
            s_glob = np.array(comm.allreduce(s_loc, op=MPI.SUM), dtype=float)
            n_glob = comm.allreduce(n_loc, op=MPI.SUM)
            mmean = s_glob / float(n_glob)

            maxdmdt = float(self.stopper.last_max_dmdt_deg_ns) if self.stopper is not None else 0.0
            if not np.isfinite(maxdmdt):
                maxdmdt = 0.0

            entry = {
                "step": int(i),
                "H": (float(Hx), float(Hy), float(Hz)),
                "m_mean": (float(mmean[0]), float(mmean[1]), float(mmean[2])),
                "elapsed": float(elapsed),
                **stats,
            }
            results.append(entry)

            if comm.rank == 0:
                with open(log_path, "a") as f:
                    f.write(
                        f"{i:d} {Hx:.6e} {Hy:.6e} {Hz:.6e} "
                        f"{mmean[0]:.6e} {mmean[1]:.6e} {mmean[2]:.6e} "
                        f"{maxdmdt:.6e} {stats['reason']:d} {stats['nsteps']:d} {stats['dt_last']:.6e}\n"
                    )
                print(
                    f"[HYST] i={i:05d}  H=({Hx:+.6e},{Hy:+.6e},{Hz:+.6e})  "
                    f"<m>=({mmean[0]:+.6e},{mmean[1]:+.6e},{mmean[2]:+.6e})  "
                    f"max|dm/dt|={maxdmdt:.6e} deg/ns  nsteps={stats['nsteps']}  "
                    f"t_end={stats['t_end']*1e9:.6f} ns",
                    flush=True,
                )

        if xdmf is not None:
            xdmf.close()

        return results
