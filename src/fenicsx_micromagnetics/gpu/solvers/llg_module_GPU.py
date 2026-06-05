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
from ..fields.Exchange_GPU import ExchangeField
from ..fields.Anisotropy_GPU import AnisotropyField
from ..fields.DMI_Bulk_GPU import DMIBULK
from ..fields.DMI_Interfacial_GPU import DMIInterfacial
from ..fields.Cubic_Anisotropy_GPU import CubicAnisotropyFieldGPU


# PETSc / CuPy helpers

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



# Fused CuPy kernels for hot local algebra

_LLG_RHS_KERNEL = cp.ElementwiseKernel(
    "float64 mx, float64 my, float64 mz, "
    "float64 hx, float64 hy, float64 hz, "
    "float64 prefactor, float64 alpha, float64 do_precess, float64 Stab",
    "float64 rx, float64 ry, float64 rz",
    """
    double mcx = my*hz - mz*hy;
    double mcy = mz*hx - mx*hz;
    double mcz = mx*hy - my*hx;

    double mcmx = my*mcz - mz*mcy;
    double mcmy = mz*mcx - mx*mcz;
    double mcmz = mx*mcy - my*mcx;

    double norm2 = mx*mx + my*my + mz*mz;
    double stab = Stab * (1.0 - norm2);

    rx = prefactor * (do_precess*mcx + alpha*mcmx) + stab*mx;
    ry = prefactor * (do_precess*mcy + alpha*mcmy) + stab*my;
    rz = prefactor * (do_precess*mcz + alpha*mcmz) + stab*mz;
    """,
    name="llg_rhs_fused_kernel",
)


_JAC_VEC_KERNEL = cp.ElementwiseKernel(
    "float64 mx, float64 my, float64 mz, "
    "float64 hx, float64 hy, float64 hz, "
    "float64 vx, float64 vy, float64 vz, "
    "float64 hvx, float64 hvy, float64 hvz, "
    "float64 gamma, float64 alpha, float64 do_precess, float64 Stab",
    "float64 ox, float64 oy, float64 oz",
    """
    double cvx = vy*hz - vz*hy;
    double cvy = vz*hx - vx*hz;
    double cvz = vx*hy - vy*hx;

    double cmx = my*hvz - mz*hvy;
    double cmy = mz*hvx - mx*hvz;
    double cmz = mx*hvy - my*hvx;

    double mdHm = mx*hx + my*hy + mz*hz;
    double mdHv = mx*hvx + my*hvy + mz*hvz;
    double vdHm = vx*hx + vy*hy + vz*hz;
    double mdv = mx*vx + my*vy + mz*vz;
    double mdmm = mx*mx + my*my + mz*mz;

    double common = vdHm + mdHv;

    double dampx = vx*mdHm - 2.0*hx*mdv + mx*common - hvx*mdmm;
    double dampy = vy*mdHm - 2.0*hy*mdv + my*common - hvy*mdmm;
    double dampz = vz*mdHm - 2.0*hz*mdv + mz*common - hvz*mdmm;

    double coef = -gamma / (1.0 + alpha*alpha);
    double stab_common = Stab * (1.0 - mdmm);

    ox = coef * (do_precess*(cvx + cmx) + alpha*dampx)
         + stab_common*vx - 2.0*Stab*mx*mdv;
    oy = coef * (do_precess*(cvy + cmy) + alpha*dampy)
         + stab_common*vy - 2.0*Stab*my*mdv;
    oz = coef * (do_precess*(cvz + cmz) + alpha*dampz)
         + stab_common*vz - 2.0*Stab*mz*mdv;
    """,
    name="jac_vec_fused_kernel",
)


_PC_BUILD_INV3_KERNEL = cp.ElementwiseKernel(
    "float64 mx, float64 my, float64 mz, "
    "float64 hx, float64 hy, float64 hz, "
    "float64 dx, float64 dy, float64 dz, "
    "float64 dcx, float64 dcy, float64 dcz, "
    "int32 use_cubic, int32 include_stab, int32 use_abs_kappa, "
    "float64 shift, float64 eps_reg, float64 det_eps, "
    "float64 c1, float64 c2, float64 Stab",
    "float64 i00, float64 i01, float64 i02, "
    "float64 i10, float64 i11, float64 i12, "
    "float64 i20, float64 i21, float64 i22",
    """
    double dex = dx + (use_cubic ? dcx : 0.0);
    double dey = dy + (use_cubic ? dcy : 0.0);
    double dez = dz + (use_cubic ? dcz : 0.0);

    double kappa;
    if (use_abs_kappa) {
        kappa = (fabs(dex) + fabs(dey) + fabs(dez)) / 3.0;
    } else {
        kappa = (dex + dey + dez) / 3.0;
    }

    double mdH = mx*hx + my*hy + mz*hz;
    double mdm = mx*mx + my*my + mz*mz;

    double Jp01 = -c1*(-hz - kappa*(-mz));
    double Jp02 = -c1*( hy - kappa*( my));
    double Jp10 = -c1*( hz - kappa*( mz));
    double Jp12 = -c1*(-hx - kappa*(-mx));
    double Jp20 = -c1*(-hy - kappa*(-my));
    double Jp21 = -c1*( hx - kappa*( mx));

    double B00 = mdH - hx*mx;
    double B11 = mdH - hy*my;
    double B22 = mdH - hz*mz;

    double B01 = mx*hy - 2.0*hx*my;
    double B02 = mx*hz - 2.0*hx*mz;
    double B10 = my*hx - 2.0*hy*mx;
    double B12 = my*hz - 2.0*hy*mz;
    double B20 = mz*hx - 2.0*hz*mx;
    double B21 = mz*hy - 2.0*hz*my;

    double C00 = mx*mx - mdm;
    double C11 = my*my - mdm;
    double C22 = mz*mz - mdm;

    double C01 = mx*my;
    double C02 = mx*mz;
    double C10 = my*mx;
    double C12 = my*mz;
    double C20 = mz*mx;
    double C21 = mz*my;

    double Jd00 = c2*(B00 + kappa*C00);
    double Jd11 = c2*(B11 + kappa*C11);
    double Jd22 = c2*(B22 + kappa*C22);
    double Jd01 = c2*(B01 + kappa*C01);
    double Jd02 = c2*(B02 + kappa*C02);
    double Jd10 = c2*(B10 + kappa*C10);
    double Jd12 = c2*(B12 + kappa*C12);
    double Jd20 = c2*(B20 + kappa*C20);
    double Jd21 = c2*(B21 + kappa*C21);

    double Js00 = 0.0, Js11 = 0.0, Js22 = 0.0;
    double Js01 = 0.0, Js02 = 0.0, Js10 = 0.0;
    double Js12 = 0.0, Js20 = 0.0, Js21 = 0.0;

    if (include_stab) {
        double s0 = 1.0 - mdm;
        Js00 = Stab*(s0 - 2.0*mx*mx);
        Js11 = Stab*(s0 - 2.0*my*my);
        Js22 = Stab*(s0 - 2.0*mz*mz);
        Js01 = Stab*(-2.0*mx*my);
        Js02 = Stab*(-2.0*mx*mz);
        Js10 = Stab*(-2.0*my*mx);
        Js12 = Stab*(-2.0*my*mz);
        Js20 = Stab*(-2.0*mz*mx);
        Js21 = Stab*(-2.0*mz*my);
    }

    double J00 = Jd00 + Js00;
    double J11 = Jd11 + Js11;
    double J22 = Jd22 + Js22;
    double J01 = Jp01 + Jd01 + Js01;
    double J02 = Jp02 + Jd02 + Js02;
    double J10 = Jp10 + Jd10 + Js10;
    double J12 = Jp12 + Jd12 + Js12;
    double J20 = Jp20 + Jd20 + Js20;
    double J21 = Jp21 + Jd21 + Js21;

    double s = shift + eps_reg;

    double A00 = s - J00;
    double A01 = -J01;
    double A02 = -J02;
    double A10 = -J10;
    double A11 = s - J11;
    double A12 = -J12;
    double A20 = -J20;
    double A21 = -J21;
    double A22 = s - J22;

    double det = A00*(A11*A22 - A12*A21)
               - A01*(A10*A22 - A12*A20)
               + A02*(A10*A21 - A11*A20);

    if (fabs(det) < det_eps) {
        double sgn = ((det + det_eps) > 0.0) ? 1.0 : -1.0;
        det = det + sgn*det_eps;
    }

    double invdet = 1.0 / det;

    i00 =  (A11*A22 - A12*A21) * invdet;
    i01 = -(A01*A22 - A02*A21) * invdet;
    i02 =  (A01*A12 - A02*A11) * invdet;
    i10 = -(A10*A22 - A12*A20) * invdet;
    i11 =  (A00*A22 - A02*A20) * invdet;
    i12 = -(A00*A12 - A02*A10) * invdet;
    i20 =  (A10*A21 - A11*A20) * invdet;
    i21 = -(A00*A21 - A01*A20) * invdet;
    i22 =  (A00*A11 - A01*A10) * invdet;
    """,
    name="pc_build_inv3_fused_kernel",
)


_PC_APPLY_KERNEL = cp.ElementwiseKernel(
    "float64 i00, float64 i01, float64 i02, "
    "float64 i10, float64 i11, float64 i12, "
    "float64 i20, float64 i21, float64 i22, "
    "float64 x0, float64 x1, float64 x2",
    "float64 y0, float64 y1, float64 y2",
    """
    y0 = i00*x0 + i01*x1 + i02*x2;
    y1 = i10*x0 + i11*x1 + i12*x2;
    y2 = i20*x0 + i21*x1 + i22*x2;
    """,
    name="pc_apply_fused_kernel",
)



# Stopping criterion on GPU

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



# Effective field + RHS/Jv on GPU

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

        # Spatial-resolution diagnostic. Build the mesh-edge list only once.
        # In the common single-rank GPU path, the edge endpoints are also kept
        # on the device so max_neighbor_angle_deg_gpu() avoids CPU transfers.
        self.neighbor_pairs_host = self._build_neighbor_pairs()
        if self.comm.size == 1 and self.neighbor_pairs_host.size:
            self.neighbor_i_gpu = cp.asarray(
                self.neighbor_pairs_host[:, 0],
                dtype=cp.int32,
            )
            self.neighbor_j_gpu = cp.asarray(
                self.neighbor_pairs_host[:, 1],
                dtype=cp.int32,
            )
        else:
            self.neighbor_i_gpu = None
            self.neighbor_j_gpu = None


        # Lumped nodal volumes (assembled once on DOLFINx layout)

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

        self.mu0 = 4.0 * np.pi * 1e-7

        vol_scalar = np.asarray(volN[: self.local_size], dtype=np.float64).reshape((-1, 3))[:, 0]
        self.vol_nodes_gpu = cp.asarray(vol_scalar)

        # Mesh coordinates are in nm, so nodal volume is nm^3.
        self.volume_scale_energy = 1e-27


        # Linear GPU operators

        self.exchange_field = None
        self.anisotropy_field = None
        self.DMIBULK = None
        self.DMI_int = None
        self.demag_field = None
        self.H_demag_gpu = None
        self.linear_terms: list[tuple[str, PETSc.Mat, PETSc.Vec]] = []
        # Matrix-free/local field terms. Used for nodal anisotropy and similar
        # contributions that do not expose a PETSc Mat .K. Each entry is
        # (name, field_object, work_vector). field_object must provide
        # compute_vec(x_vec, out_vec).
        self.local_terms: list[tuple[str, object, PETSc.Vec]] = []
        self.cubic_field = None
        self.H_cubic_gpu = None

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

            # Backward compatibility:
            # - old Anisotropy_GPU exposes a PETSc matrix .K and is handled as a
            #   normal linear sparse term.
            # - nodal anisotropy is matrix-free and provides
            #   compute_vec(...), Energy_lumped_gpu(...), diagonal_vec(...).
            if hasattr(self.anisotropy_field, "K"):
                self.anisotropy_field.K = _set_mat_cuda(self.anisotropy_field.K)
                buf = self.anisotropy_field.K.createVecLeft()
                _set_vec_cuda(buf, block_size=3)
                self.linear_terms.append(("anisotropy", self.anisotropy_field.K, buf))
                if template_in is None:
                    template_in = self.anisotropy_field.K.createVecRight()
                    template_out = self.anisotropy_field.K.createVecLeft()
            elif hasattr(self.anisotropy_field, "compute_vec"):
                if template_in is None:
                    template_in = self.m.x.petsc_vec.duplicate()
                    template_out = self.m.x.petsc_vec.duplicate()
                buf = getattr(self.anisotropy_field, "h_gpu", None)
                if buf is None:
                    buf = template_out.duplicate()
                _set_vec_cuda(buf, block_size=3)
                self.local_terms.append(("anisotropy", self.anisotropy_field, buf))
            else:
                raise TypeError(
                    "AnisotropyField must expose either a PETSc matrix .K or "
                    "a matrix-free compute_vec(m_vec, out_vec) method."
                )

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

        if abs(self.Kc1) > 0.0:
            if u1_cub is None or u2_cub is None:
                raise ValueError(
                    "Cubic anisotropy requires u1_cub and u2_cub."
                )

            self.cubic_field = CubicAnisotropyFieldGPU(
                mesh=self.mesh,
                V=self.V,
                K1=self.Kc1,
                Ms=self.Ms,
                u1=u1_cub,
                u2=u2_cub,
                VolN=volN,
            )

        # -------------------------------------------------------------
        # If there are no linear terms, allow demag-only.
        # This also protects against template_in=None.
        # -------------------------------------------------------------
        if template_in is None:
            if self.use_demag or self.cubic_field is not None:
                template_in = self.m.x.petsc_vec.duplicate()
                template_out = self.m.x.petsc_vec.duplicate()
            else:
                raise ValueError(
                    "There are no active GPU terms. You must add exchange, anisotropy, "
                    "DMI, cubic anisotropy, or demag."
                )

        _set_vec_cuda(template_in, block_size=3)
        _set_vec_cuda(template_out, block_size=3)

        # Persistent GPU state
        self.m_gpu = _dup_cuda_vec(template_in, block_size=3)
        self.dmdt_gpu = _dup_cuda_vec(template_in, block_size=3)
        self.H_eff_gpu = _dup_cuda_vec(template_out, block_size=3)

        # Minimizer cache flags.  ``compute_Energy_terms_minimize_gpu`` can
        # fill H_eff_gpu while evaluating E(m_trial).  After the trial is
        # accepted and copied into m_gpu, the LaBonte-BB minimizer can reuse
        # this buffer instead of recomputing the full effective field.
        self._h_eff_valid_for_m_gpu = False
        self._h_eff_gpu_filled_by_energy = False

        self.Hm_gpu = _dup_cuda_vec(template_out, block_size=3)
        self.Hv_gpu = _dup_cuda_vec(template_out, block_size=3)
        self.H0_gpu = _dup_cuda_vec(template_out, block_size=3)
        self.Ht_gpu = _dup_cuda_vec(template_out, block_size=3)
        self.Jv_buffer = _dup_cuda_vec(template_in, block_size=3)


        if self.cubic_field is not None:
            self.H_cubic_gpu = _dup_cuda_vec(template_out, block_size=3)
            self.diagK_cubic_gpu = _dup_cuda_vec(template_out, block_size=3)
        else:
            self.H_cubic_gpu = None
            self.diagK_cubic_gpu = None

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

        # Matrix-free/local terms can still contribute to the diagonal used by
        # the approximate block preconditioner. For nodal anisotropy this is
        # diag(2 Ku/(mu0 Ms) * n n^T) = pref * (nx^2, ny^2, nz^2).
        for _, field, _ in self.local_terms:
            if hasattr(field, "diagonal_vec"):
                field.diagonal_vec(self.Hv_gpu)
                self.diagK.axpy(1.0, self.Hv_gpu)

        self.diagK.copy(self.diagK_abs)
        self.diagK_abs.abs()



        if self.use_demag:
            method = str(demag_method).lower()
            kwargs = {} if demag_kwargs is None else dict(demag_kwargs)

            if method in ["fmm", "jaxfmm"]:
                from ..fields.Demag_FMM_GPU import DemagFieldFMMJAXGPU

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
                from ..fields.Demag_Lindholm_GPU import DemagFieldLindholmGPU

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


    def _build_neighbor_pairs(self):
        """
        Build unique pairs of neighboring P1 nodes connected by mesh edges.

        The scalar P1 space V1 and the blocked vector P1 space V share the same
        nodal ordering. For tetrahedra this produces the six edges per cell;
        duplicate edges are removed once during initialization.
        """
        tdim = self.mesh.topology.dim
        cell_map = self.mesh.topology.index_map(tdim)
        n_cells = int(cell_map.size_local + cell_map.num_ghosts)

        chunks = []

        for cell in range(n_cells):
            dofs = np.asarray(self.V1.dofmap.cell_dofs(cell), dtype=np.int32)

            if dofs.size < 2:
                continue

            ia, ib = np.triu_indices(dofs.size, k=1)
            pairs = np.column_stack((dofs[ia], dofs[ib])).astype(
                np.int32,
                copy=False,
            )
            pairs.sort(axis=1)
            chunks.append(pairs)

        if not chunks:
            return np.empty((0, 2), dtype=np.int32)

        return np.unique(np.vstack(chunks), axis=0).astype(
            np.int32,
            copy=False,
        )

    def max_neighbor_angle_deg_gpu(self, m_vec: PETSc.Vec | None = None):
        """
        Return the global maximum angle, in degrees, between neighboring nodal
        moments.

        Single-rank execution stays entirely on GPU except for transferring the
        final scalar maximum. The MPI fallback uses the host-side Function after
        synchronizing ghost values.
        """
        if self.neighbor_pairs_host.size == 0:
            return 0.0

        if m_vec is None:
            m_vec = self.m_gpu

        if self.comm.size == 1:
            m_all = _vec_to_cupy(m_vec, "r")
            moments = m_all[: self.local_size].reshape((-1, 3))

            mi = moments[self.neighbor_i_gpu]
            mj = moments[self.neighbor_j_gpu]

            dot = cp.sum(mi * mj, axis=1)
            norm_product = cp.sqrt(
                cp.sum(mi * mi, axis=1) * cp.sum(mj * mj, axis=1)
            )

            # Ignore defensively any zero-length vectors by assigning angle 0.
            cosine = cp.where(
                norm_product > 1e-30,
                dot / norm_product,
                1.0,
            )
            cosine = cp.clip(cosine, -1.0, 1.0)

            local_max = float(
                (cp.arccos(cosine) * (180.0 / np.pi)).max().item()
            )
            return local_max

        # Conservative MPI fallback. This path is not used by the current
        # single-GPU backend but keeps the diagnostic well-defined.
        if m_vec is not self.m_gpu:
            m_vec.copy(self.m_gpu)

        self.sync_m_to_function()
        self.m.x.scatter_forward()

        moments = self.m.x.array.reshape((-1, 3))
        pairs = self.neighbor_pairs_host

        mi = moments[pairs[:, 0]]
        mj = moments[pairs[:, 1]]

        dot = np.einsum("ij,ij->i", mi, mj)
        norm_product = np.linalg.norm(mi, axis=1) * np.linalg.norm(mj, axis=1)

        cosine = np.ones_like(dot)
        valid = norm_product > 1e-30
        cosine[valid] = dot[valid] / norm_product[valid]
        cosine = np.clip(cosine, -1.0, 1.0)

        local_max = float(np.degrees(np.arccos(cosine)).max())
        return float(self.comm.allreduce(local_max, op=MPI.MAX))


    # Host/device sync helpers


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

    def apply_jacobian_field_vec(self, m_state_vec: PETSc.Vec, v_vec: PETSc.Vec, out_vec: PETSc.Vec):
        """
        Compute out = dH_eff/dm[m_state] v.

        Includes:
            - linear terms: exchange, uniaxial anisotropy, DMI
            - nonlinear cubic anisotropy derivative

        Excludes:
            - demag derivative
            - external field derivative
        """

        # Linear part
        self.apply_linear_field_vec(v_vec, out_vec)

        # Cubic anisotropy derivative
        if self.cubic_field is not None:
            self.cubic_field.jac_times_vec(
                m_state_vec,
                v_vec,
                self.H_cubic_gpu,
            )
            out_vec.axpy(1.0, self.H_cubic_gpu)


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

        for _, field, buf in self.local_terms:
            field.compute_vec(x_vec, buf)
            out_vec.axpy(1.0, buf)

    def compute_H_eff_vec(self, m_vec: PETSc.Vec, out_vec: PETSc.Vec):
        """
        Compute the full effective field H_eff(m_vec) into out_vec.

        When the output buffer is self.H_eff_gpu and the input vector is
        self.m_gpu, the result is marked as reusable by the minimizer.
        For every other output/input combination, the reusable H_eff_gpu cache
        is considered invalid.
        """
        # Exchange + uniaxial anisotropy + DMI
        self.apply_linear_field_vec(m_vec, out_vec)

        # Cubic anisotropy is nonlinear, so it is added separately.
        if self.cubic_field is not None:
            self.cubic_field.compute_vec(m_vec, self.H_cubic_gpu)
            out_vec.axpy(1.0, self.H_cubic_gpu)

        # Demag FMM/JAX or Lindholm on GPU.
        if self.demag_field is not None:
            self.demag_field.compute_vec(m_vec, self.H_demag_gpu)
            out_vec.axpy(1.0, self.H_demag_gpu)

        # Static external magnetic field.
        out_vec.axpy(1.0, self.H0_gpu)

        # Time-dependent external magnetic field.
        if self.H_time_func is not None:
            self._eval_H_time_vec(self.current_time)
            out_vec.axpy(1.0, self.Ht_gpu)

        if out_vec is self.H_eff_gpu and m_vec is self.m_gpu:
            self._h_eff_valid_for_m_gpu = True
            self._h_eff_gpu_filled_by_energy = False
        else:
            self._h_eff_valid_for_m_gpu = False

    def compute_H_eff(self, m_fun):
        self.compute_H_eff_vec(m_fun.x.petsc_vec, self.H_eff.x.petsc_vec)
        return self.H_eff



    def _energy_from_field_lumped_gpu(
        self,
        m_vec: PETSc.Vec,
        H_vec: PETSc.Vec,
        factor: float,
    ):
        """
        Lumped GPU energy from m · H.

        Parameters
        ----------
        m_vec:
            PETSc CUDA vector with magnetization.
        H_vec:
            PETSc CUDA vector with field contribution.
        factor:
            Energy prefactor:
                -0.5 for self-consistent linear energy terms
                -1.0 for external Zeeman field

        Returns
        -------
        float
            Energy in Joules.
        """

        m_all = _vec_to_cupy(m_vec, "r")
        H_all = _vec_to_cupy(H_vec, "r")

        m = m_all[: self.local_size].reshape((-1, 3))
        H = H_all[: self.local_size].reshape((-1, 3))

        mdH = cp.sum(m * H, axis=1)
        val = cp.sum(self.vol_nodes_gpu * mdH)

        E = factor * self.mu0 * self.Ms * val * self.volume_scale_energy

        return float(E.item())


    def _external_energy_lumped_gpu(self, m_vec: PETSc.Vec):
        """
        GPU Zeeman energy:

            E_Z = -mu0 Ms int m cdot H_ext dV

        H_ext = H0_static + H_time(current_time)
        """

        self.Ht_gpu.zeroEntries()

        if self.H_time_func is not None:
            self._eval_H_time_vec(self.current_time)

        # H_ext = H0 + Ht
        self.H0_gpu.copy(self.Hv_gpu)
        self.Hv_gpu.axpy(1.0, self.Ht_gpu)

        return self._energy_from_field_lumped_gpu(
            m_vec=m_vec,
            H_vec=self.Hv_gpu,
            factor=-1.0,
        )


    def compute_Energy_terms_minimize_gpu(self, m_vec: PETSc.Vec | None = None):
        """
        Fast GPU energy evaluation for the minimizer.
        """

        if m_vec is None:
            m_vec = self.m_gpu

        # Rebuild the full effective-field cache from scratch.
        self.H_eff_gpu.zeroEntries()
        self._h_eff_valid_for_m_gpu = False
        self._h_eff_gpu_filled_by_energy = False

        E_exch = 0.0
        E_ani = 0.0
        E_dmi_bulk = 0.0
        E_dmi_int = 0.0
        E_demag = 0.0
        E_ext = 0.0
        E_cubic = 0.0

        # ---------------------------------------------------------
        # Linear self-energy terms:
        # exchange, anisotropy, DMI bulk, DMI interfacial.
        # ---------------------------------------------------------
        for name, K, buf in self.linear_terms:
            K.mult(m_vec, buf)

            # Store the full H_eff contribution for later reuse.
            self.H_eff_gpu.axpy(1.0, buf)

            E_term = self._energy_from_field_lumped_gpu(
                m_vec=m_vec,
                H_vec=buf,
                factor=-0.5,
            )

            if name == "exchange":
                E_exch += E_term
            elif name == "anisotropy":
                E_ani += E_term
            elif name == "dmi_bulk":
                E_dmi_bulk += E_term
            elif name == "dmi_interfacial":
                E_dmi_int += E_term

        # ---------------------------------------------------------
        # Matrix-free/local self-energy terms, e.g. nodal anisotropy.
        # ---------------------------------------------------------
        for name, field, buf in self.local_terms:
            field.compute_vec(m_vec, buf)

            # Store the full H_eff contribution for later reuse.
            self.H_eff_gpu.axpy(1.0, buf)

            if hasattr(field, "Energy_lumped_gpu"):
                E_term = float(field.Energy_lumped_gpu(m_vec))
            else:
                E_term = self._energy_from_field_lumped_gpu(
                    m_vec=m_vec,
                    H_vec=buf,
                    factor=-0.5,
                )

            if name == "anisotropy":
                E_ani += E_term

        # ---------------------------------------------------------
        # Demagnetizing field.
        # ---------------------------------------------------------
        if self.demag_field is not None:
            self.demag_field.compute_vec(m_vec, self.H_demag_gpu)
            self.H_eff_gpu.axpy(1.0, self.H_demag_gpu)

            if hasattr(self.demag_field, "Energy_lumped_gpu"):
                E_demag = float(
                    self.demag_field.Energy_lumped_gpu(
                        m_vec,
                        self.H_demag_gpu,
                    )
                )
            else:
                E_demag = self._energy_from_field_lumped_gpu(
                    m_vec=m_vec,
                    H_vec=self.H_demag_gpu,
                    factor=-0.5,
                )

        # ---------------------------------------------------------
        # Cubic anisotropy.
        # ---------------------------------------------------------
        if self.cubic_field is not None:
            self.cubic_field.compute_vec(m_vec, self.H_cubic_gpu)
            self.H_eff_gpu.axpy(1.0, self.H_cubic_gpu)

            if hasattr(self.cubic_field, "Energy_lumped_gpu"):
                E_cubic = float(self.cubic_field.Energy_lumped_gpu(m_vec))
            else:
                E_cubic = self._energy_from_field_lumped_gpu(
                    m_vec=m_vec,
                    H_vec=self.H_cubic_gpu,
                    factor=-0.5,
                )

        # ---------------------------------------------------------
        # Static/time-dependent external field.
        # For minimization, H_time_func should normally be fixed.
        # ---------------------------------------------------------
        has_static_field = True
        try:
            h0_norm = self.H0_gpu.norm()
            has_static_field = h0_norm > 0.0
        except Exception:
            has_static_field = True

        if has_static_field or self.H_time_func is not None:
            # _external_energy_lumped_gpu builds self.Hv_gpu = H0 + Ht.
            E_ext = self._external_energy_lumped_gpu(m_vec)
            self.H_eff_gpu.axpy(1.0, self.Hv_gpu)

        E_total = (
            E_exch
            + E_ani
            + E_dmi_bulk
            + E_dmi_int
            + E_demag
            + E_ext
            + E_cubic
        )

        self._h_eff_gpu_filled_by_energy = True
        self._h_eff_valid_for_m_gpu = m_vec is self.m_gpu

        return {
            "E_demag": float(E_demag),
            "E_exch": float(E_exch),
            "E_ani": float(E_ani),
            "E_dmi_bulk": float(E_dmi_bulk),
            "E_dmi_int": float(E_dmi_int),
            "E_ext": float(E_ext),
            "E_cubic": float(E_cubic),
            "E_total": float(E_total),
        }


    def compute_Energy_minimize_gpu(self, m_vec: PETSc.Vec | None = None):
        """
        Fast total energy used only by the minimizer.
        """

        return self.compute_Energy_terms_minimize_gpu(m_vec)["E_total"]


    def compute_H_jac_vec(self, m_vec: PETSc.Vec, out_vec: PETSc.Vec):
        """
        Field used ONLY for Jacobian and preconditioner.

        Includes:
            exchange + uniaxial anisotropy + DMI + cubic anisotropy + external field

        Excludes:
            demagnetism
        """

        # Linear local part: exchange + uniaxial anisotropy + DMI
        self.apply_linear_field_vec(m_vec, out_vec)

        # Nonlinear local cubic anisotropy
        if self.cubic_field is not None:
            self.cubic_field.compute_vec(m_vec, self.H_cubic_gpu)
            out_vec.axpy(1.0, self.H_cubic_gpu)

        # Static external magnetic field
        out_vec.axpy(1.0, self.H0_gpu)

        # Time-dependent external magnetic field
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

        _LLG_RHS_KERNEL(
            m[:, 0], m[:, 1], m[:, 2],
            h[:, 0], h[:, 1], h[:, 2],
            float(self.prefactorEQ),
            float(self.alpha),
            float(self.do_precess),
            float(self.Stab),
            rhs[:, 0], rhs[:, 1], rhs[:, 2],
        )

        if rhs_all.size > self.local_size:
            rhs_all[self.local_size :] = 0.0

    def update_jac_state(self, m_vec: PETSc.Vec):
        m_vec.copy(self.M_state_gpu)

        self.compute_H_jac_vec(m_vec, self.Hm_gpu)


    def jac_vec_times_vec(self, v_vec: PETSc.Vec, out_vec: PETSc.Vec):
        self.JacSteps += 1

        self.apply_jacobian_field_vec(self.M_state_gpu, v_vec, self.Hv_gpu)

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

        _JAC_VEC_KERNEL(
            m[:, 0], m[:, 1], m[:, 2],
            hm[:, 0], hm[:, 1], hm[:, 2],
            v[:, 0], v[:, 1], v[:, 2],
            hv[:, 0], hv[:, 1], hv[:, 2],
            float(self.gamma),
            float(self.alpha),
            float(self.do_precess),
            float(self.Stab),
            out[:, 0], out[:, 1], out[:, 2],
        )

        if out_all.size > self.local_size:
            out_all[self.local_size :] = 0.0


    # PETSc TS callbacks

    def ifunction(self, ts, t, y, ydot, f):

        self.current_time = float(t)

        # dmdt_gpu = RHS(t, y)
        self.llg_rhs_vec(y, self.dmdt_gpu)

        # f = ydot - RHS(t, y)
        ydot.copy(f)
        f.axpy(-1.0, self.dmdt_gpu)

        return 0


    # Diagnostics 
    def compute_Energy_terms(self, sync: bool = True):

        if sync:
            m_fun = self.sync_m_to_function()
        else:
            m_fun = self.m

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

        if self.cubic_field is not None:
            E_cubic = float(
                self.cubic_field.Energy(m_fun)
            )

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

# Jacobian matrix-free context: y <- shift*x - Jx

class JvContext:
    def __init__(self, hef, eps_reg=1e-14, det_eps=1e-30):
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

        # Persistent CuPy views for PETSc CUDA vectors owned by EffectiveField.
        # Do not cache callback vectors x/y: PETSc may pass different work vectors.
        self._M_state_view = _vec_to_cupy(self.hef.M_state_gpu, "r")
        self._Hm_view = _vec_to_cupy(self.hef.Hm_gpu, "r")
        self._diagK_view = _vec_to_cupy(self.hef.diagK, "r")
        self._diagK_cubic_view = (
            _vec_to_cupy(self.hef.diagK_cubic_gpu, "r")
            if self.hef.diagK_cubic_gpu is not None
            else None
        )

    def update_pc_full_fast_gpu(self, shift, include_stab=True, use_abs_kappa=True):
        """
        Build the block-diagonal 3x3 inverse used by the Python PC.

        This is the fused version of the old CuPy-expression implementation.
        It computes the approximate local Jacobian block, forms

            A_i = (shift + eps_reg) I - J_i,

        and stores A_i^{-1} directly in the nine persistent CuPy arrays i**.
        """
        self.shift = float(shift)

        local_size = self.hef.local_size

        M_all = self._M_state_view
        H_all = self._Hm_view
        D_all = self._diagK_view

        M = M_all[:local_size].reshape((-1, 3))
        H = H_all[:local_size].reshape((-1, 3))
        D = D_all[:local_size].reshape((-1, 3))

        use_cubic = 0
        if self.hef.cubic_field is not None and self.hef.diagK_cubic_gpu is not None:
            self.hef.cubic_field.jac_diag_vec(
                self.hef.M_state_gpu,
                self.hef.diagK_cubic_gpu,
            )
            Dc_all = self._diagK_cubic_view
            if Dc_all is None:
                Dc_all = _vec_to_cupy(self.hef.diagK_cubic_gpu, "r")
                self._diagK_cubic_view = Dc_all
            Dc = Dc_all[:local_size].reshape((-1, 3))
            use_cubic = 1
        else:
            # Dummy view. The fused kernel ignores it when use_cubic == 0.
            Dc = D

        _PC_BUILD_INV3_KERNEL(
            M[:, 0], M[:, 1], M[:, 2],
            H[:, 0], H[:, 1], H[:, 2],
            D[:, 0], D[:, 1], D[:, 2],
            Dc[:, 0], Dc[:, 1], Dc[:, 2],
            int(use_cubic),
            int(bool(include_stab)),
            int(bool(use_abs_kappa)),
            float(self.shift),
            float(self.eps_reg),
            float(self.det_eps),
            float(self.c1),
            float(self.c2),
            float(self.Stab),
            self.i00, self.i01, self.i02,
            self.i10, self.i11, self.i12,
            self.i20, self.i21, self.i22,
        )

        self._pc_ready = True


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

        x_all = _vec_to_cupy(x, "r")
        y_all = _vec_to_cupy(y, "rw")

        X = x_all[:local_size].reshape((-1, 3))
        Y = y_all[:local_size].reshape((-1, 3))

        _PC_APPLY_KERNEL(
            self.i00, self.i01, self.i02,
            self.i10, self.i11, self.i12,
            self.i20, self.i21, self.i22,
            X[:, 0], X[:, 1], X[:, 2],
            Y[:, 0], Y[:, 1], Y[:, 2],
        )

        if y_all.size > local_size:
            y_all[local_size:] = 0.0



# Main LLG driver

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

        # initial state
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

                # GPU spatial-resolution diagnostic, evaluated only at log time.
                max_neighbor_angle_deg = hef_.max_neighbor_angle_deg_gpu(u)

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
                            f"{'maxdmdt(deg/ns)':>18} "
                            f"{'max_nn_angle(deg)':>18}"
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
                        f"{max_neighbor_angle_deg:18.6f}"
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



    def _save_final_magnetization_state(
        self,
        hef,
        output_dir,
        xdmf_name="Minimize_LaBonte.xdmf",
        bp_name="Minimize_LaBonte.bp",
        write_xdmf=True,
        write_bp=True,
        time=0.0,
        function_name="m",
    ):
        """
        Save final magnetization state.

        Writes:
        - XDMF file for visualization.
        - ADIOS2 .bp file for restart/read-back with adios4dolfinx.

        Parameters
        ----------
        hef:
            EffectiveField instance.
        output_dir:
            Output directory.
        xdmf_name:
            Name of the XDMF output file.
        bp_name:
            Name of the ADIOS2 BP output.
        write_xdmf:
            Whether to write XDMF.
        write_bp:
            Whether to write ADIOS2 BP.
        time:
            Time/value associated with the saved function.
        function_name:
            Name used for the saved magnetization function.
        """

        comm = self.mesh.comm
        output_dir = Path(output_dir)

        if comm.rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)

        comm.barrier()

        # Ensure GPU state is copied to host-side dolfinx Function.
        hef.sync_m_to_function()

        try:
            hef.m.name = function_name
        except Exception:
            pass

        if write_xdmf:
            xdmf_path = output_dir / xdmf_name

            with io.XDMFFile(comm, str(xdmf_path), "w") as xdmf:
                xdmf.write_mesh(self.mesh)
                try:
                    xdmf.write_function(hef.m, float(time))
                except TypeError:
                    xdmf.write_function(hef.m)

        if write_bp:
            bp_path = output_dir / bp_name

            ad.write_mesh(bp_path, self.mesh)
            ad.write_function(bp_path, hef.m, time=float(time), name=function_name)

        comm.barrier()

        return {
            "xdmf_path": str(output_dir / xdmf_name) if write_xdmf else None,
            "bp_path": str(output_dir / bp_name) if write_bp else None,
        }
    

    def _ensure_solver(
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
        use_python_pc = True,
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
            #opts["ksp_converged_reason"] = ""
            #opts['ksp_converged_reason'] = ""
            #opts['snes_converged_reason'] = ""
            #opts["log_view"] = "" 

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

            if use_python_pc:
                pc.setType(PETSc.PC.Type.PYTHON)
                pc.setPythonContext(ctx)

                def IJac(ts_, t, y, ydot, shift, A, B):
                    hef.current_time = float(t)
                    hef.update_jac_state(y)

                    #ctx.shift = float(shift)

                    ctx.update_pc_full_fast_gpu(float(shift), include_stab=True, use_abs_kappa=True)

                    return PETSc.Mat.Structure.SAME_NONZERO_PATTERN


            else:

                pc.setType(PETSc.PC.Type.NONE)

                def IJac(ts_, t, y, ydot, shift, A, B):
                    hef.current_time = float(t)
                    hef.update_jac_state(y)

                    ctx.shift = float(shift)

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
        use_pc_python = True

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
            use_python_pc = use_pc_python
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

                # 3. Compute energies. The solution was already synchronized above.
                m_fun = hef_.m

                # GPU spatial-resolution diagnostic, evaluated only at log time.
                max_neighbor_angle_deg = hef_.max_neighbor_angle_deg_gpu(
                    hef_.m_gpu
                )

                energy = hef_.compute_Energy_terms(sync=False)

                if hef_.demag_field is not None:
                    hef_.demag_field.compute_vec(hef_.m_gpu, hef_.H_demag_gpu)

                    if hasattr(hef_.demag_field, "copy_to_function") and hasattr(hef_.demag_field, "Energy"):
                        hef_.demag_field.copy_to_function(hef_.H_demag_gpu)
                        E_demag_exact = float(hef_.demag_field.Energy(m_fun))

                        energy["E_total"] += E_demag_exact - energy["E_demag"]
                        energy["E_demag"] = E_demag_exact



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
                            f"{'max_nn_angle(deg)':>18} "
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
                        f"{max_neighbor_angle_deg:18.6f} "
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
        t_final_per_step=1e-9,
        dt_init=1e-15,
        method="minimize",
        output_dir="hyst_out",

        ts_rtol=1e-6,
        ts_atol=1e-6,
        snes_rtol=1e-2,
        snes_atol=1e-4,
        ksp_rtol=1e-4,
        stopping_dmdt=0.0,
        check_every_stop=10,
        stop_print=False,

        min_max_iter=1000,
        min_tol=10.0,
        min_alpha0=1e-13,
        min_alpha_min=1e-13,
        min_alpha_max=1e-5,
        min_bb_variant="alternate",
        min_nonmonotone_window=10,
        min_max_backtracking=8,
        min_shrink=0.5,
        min_print_every=10,
        min_energy_accept_rtol=1e-12,
        min_energy_accept_atol=1e-30,
        min_energy_stagnation_rtol=1e-11,
        min_energy_stagnation_atol=1e-30,
        min_stagnation_window=30,
        min_max_rejected_at_alpha_min=20,
        min_alpha_restart=1e-9,
        min_max_alpha_restarts=3,
        min_restart_residual_factor=10.0,

        xdmf_name="Hysteresis.xdmf",
        bp_name="Hysteresis.bp",
        log_name="hysteresis_log.txt",
        per_step_dir="states",
        write_xdmf_series=False,
        write_xdmf_per_step=True,
        write_bp_series=False,
        write_bp_per_step=False,
        return_stats=True,
        use_pc_python = True,
    ):
        """
        Compute a hysteresis curve using either:

            method="minimize"
                Quasi-static energy minimization with LaBonte-BB.

            method="llg"
                Dynamic relaxation using PETSc.TS through _ensure_solver().

        At each external-field step:
            1. Set the external field.
            2. Relax/minimize the magnetization.
            3. Save the final magnetization state.
            4. Use that final state as the initial state for the next field step.

        This version assumes single-rank GPU execution.
        """

        comm = self.mesh.comm

        if comm.size != 1:
            raise RuntimeError(
                "hysteresis() is implemented here for single-rank GPU execution only."
            )

        method = str(method).lower()

        if method in ["minimize", "minimizer", "labonte", "bb", "labonte-bb"]:
            method = "minimize"
        elif method in ["llg", "ts", "solver", "ensure_solver", "dynamic"]:
            method = "llg"
        else:
            raise ValueError(
                "Unsupported hysteresis method. Use method='llg' or method='minimize'."
            )

        H_steps = np.asarray(list(H_steps), dtype=np.float64).reshape((-1, 3))

        if H_steps.shape[0] == 0:
            raise ValueError("H_steps must contain at least one external-field step.")
        
        if self.hef is None:
            self._build_effective_field()

        hef = self.hef

        if m0_array is not None:
            hef.set_m_from_cpu(m0_array)


        # Output directories.

        output_path = Path(output_dir)
        states_path = output_path / per_step_dir

        output_path.mkdir(parents=True, exist_ok=True)
        states_path.mkdir(parents=True, exist_ok=True)


        if method == "llg":
            self._ensure_solver(
                m0_array=None,
                dt_init=dt_init,
                ts_rtol=ts_rtol,
                ts_atol=ts_atol,
                snes_rtol=snes_rtol,
                snes_atol=snes_atol,
                ksp_rtol=ksp_rtol,
                stopping_dmdt=stopping_dmdt,
                check_every_stop=check_every_stop,
                stop_print=stop_print,
                use_python_pc = use_pc_python,
            )

            self._cancel_ts_monitors()


        # Optional XDMF time series.

        xdmf_series = None
        xdmf_series_path = None

        if write_xdmf_series:
            xdmf_series_path = output_path / xdmf_name
            xdmf_series = io.XDMFFile(comm, str(xdmf_series_path), "w")
            xdmf_series.write_mesh(self.mesh)


        # Optional ADIOS2 BP time series.

        bp_series_path = output_path / bp_name

        if write_bp_series:
            ad.write_mesh(bp_series_path, self.mesh)


        # Log file.

        log_path = output_path / log_name

        with open(log_path, "w") as f:
            f.write(
                "# step method Hx Hy Hz "
                "<mx> <my> <mz> "
                "E_demag E_exch E_ani E_dmi_bulk E_dmi_int E_ext E_cubic E_total "
                "max_projected_field maxdmdt_deg_ns max_nn_angle_deg "
                "iterations nsteps dt_last stop_reason elapsed "
                "xdmf_step bp_step\n"
            )

        results = []


        # Hysteresis loop.

        # Import minimizer locally to keep standalone minimization decoupled
        # from LLG_GPU at module-import time. This avoids a circular import
        # when Minimizer_GPU imports EffectiveField from this module.
        from .Minimizer_GPU import LaBonteBBMinimizerGPU

        for i, H in enumerate(H_steps):
            Hx, Hy, Hz = map(float, H)

            mu0 = 4.0 * np.pi * 1e-7

            Hx_mT = mu0 * Hx * 1e3
            Hy_mT = mu0 * Hy * 1e3
            Hz_mT = mu0 * Hz * 1e3

            print(
                f"\n[hysteresis] step={i:04d}/{len(H_steps)-1:04d} "
                f"method={method} "
                f"H=({Hx_mT:+.6f}, {Hy_mT:+.6f}, {Hz_mT:+.6f}) mT",
                flush=True,
            )


            # Set external field for this step.

            hef.set_uniform_field(Hx, Hy, Hz)
            hef.current_time = 0.0

            t_start = perf_counter()


            # Branch 1: LaBonte-BB minimization.

            if method == "minimize":
                minimizer = LaBonteBBMinimizerGPU(hef)

                step_stats = minimizer.minimize(
                    max_iter=min_max_iter,
                    tol=min_tol,
                    alpha0=min_alpha0,
                    alpha_min=min_alpha_min,
                    alpha_max=min_alpha_max,
                    bb_variant=min_bb_variant,
                    nonmonotone_window=min_nonmonotone_window,
                    max_backtracking=min_max_backtracking,
                    shrink=min_shrink,
                    print_every=min_print_every,
                    energy_accept_rtol=min_energy_accept_rtol,
                    energy_accept_atol=min_energy_accept_atol,
                    energy_stagnation_rtol=min_energy_stagnation_rtol,
                    energy_stagnation_atol=min_energy_stagnation_atol,
                    stagnation_window=min_stagnation_window,
                    max_rejected_at_alpha_min=min_max_rejected_at_alpha_min,
                    alpha_restart=min_alpha_restart,
                    max_alpha_restarts=min_max_alpha_restarts,
                    restart_residual_factor=min_restart_residual_factor,
                )

                elapsed = perf_counter() - t_start

                # The final minimizer state lives in hef.m_gpu.
                hef.sync_m_to_function()

                max_projected_field = float(
                    step_stats.get("max_projected_field", np.nan)
                )

                maxdmdt_deg_ns = float("nan")
                nsteps = 0
                dt_last = float("nan")
                stop_reason = str(step_stats.get("stop_reason", "unknown"))
                iterations = int(step_stats.get("iterations", -1))




            # Branch 2: LLG / PETSc.TS relaxation.

            else:
                # Start TS from the current final state of the previous field step.
                hef.m_gpu.copy(self.y)

                self._reset_ts_run(
                    t0=0.0,
                    t_final=float(t_final_per_step),
                    dt_init=float(dt_init),
                )

                elapsed, ts_stats = self._run_ts()

                # _run_ts() copies self.y -> hef.m_gpu and syncs hef.m.

                # ----------------------------------------------------
                # Compute projected effective-field residual inline.
                #
                # projH = H_eff - (m cdot H_eff) m
                #
                # This is the same residual used by the minimizer.
                # ----------------------------------------------------
                hef.compute_H_eff_vec(hef.m_gpu, hef.H_eff_gpu)

                m_all = _vec_to_cupy(hef.m_gpu, "r")
                h_all = _vec_to_cupy(hef.H_eff_gpu, "r")

                m = m_all[: hef.local_size].reshape((-1, 3))
                h = h_all[: hef.local_size].reshape((-1, 3))

                if m.shape[0] > 0:
                    mdh = cp.sum(m * h, axis=1)

                    proj_x = h[:, 0] - mdh * m[:, 0]
                    proj_y = h[:, 1] - mdh * m[:, 1]
                    proj_z = h[:, 2] - mdh * m[:, 2]

                    proj_norm = cp.sqrt(
                        proj_x * proj_x
                        + proj_y * proj_y
                        + proj_z * proj_z
                    )

                    max_projected_field = float(cp.max(proj_norm).item())
                else:
                    max_projected_field = 0.0

                maxdmdt_deg_ns = float(ts_stats.get("maxdmdt_deg_ns", np.nan))
                nsteps = int(ts_stats.get("nsteps", -1))
                dt_last = float(ts_stats.get("dt_last", np.nan))
                stop_reason = str(ts_stats.get("reason", "unknown"))
                iterations = nsteps

                step_stats = dict(ts_stats)
                step_stats["iterations"] = iterations
                step_stats["stop_reason"] = stop_reason
                step_stats["max_projected_field"] = max_projected_field


            # Sync final state and compute diagnostics.

            hef.sync_m_to_function()

            max_neighbor_angle_deg = hef.max_neighbor_angle_deg_gpu(
                hef.m_gpu
            )

            m_local = hef.m.x.array[: hef.local_size].reshape((-1, 3))

            if m_local.size:
                m_mean = m_local.mean(axis=0)
            else:
                m_mean = np.zeros(3, dtype=np.float64)

            # Prefer the GPU minimization energy because it includes E_ext.
            # Fallback to the exact diagnostic energy if the fast GPU path is absent.
            try:
                energy = hef.compute_Energy_terms_minimize_gpu(hef.m_gpu)
            except Exception:
                energy = hef.compute_Energy_terms()

                if "E_ext" not in energy:
                    energy["E_ext"] = 0.0

                if "E_total" not in energy:
                    energy["E_total"] = sum(
                        float(energy.get(k, 0.0))
                        for k in [
                            "E_demag",
                            "E_exch",
                            "E_ani",
                            "E_dmi_bulk",
                            "E_dmi_int",
                            "E_ext",
                            "E_cubic",
                        ]
                    )

            for key in [
                "E_demag",
                "E_exch",
                "E_ani",
                "E_dmi_bulk",
                "E_dmi_int",
                "E_ext",
                "E_cubic",
                "E_total",
            ]:
                energy.setdefault(key, 0.0)


            # Save per-step XDMF.

            xdmf_step_path = None

            if write_xdmf_per_step:
                xdmf_step_path = states_path / f"m_hyst_{i:04d}.xdmf"

                with io.XDMFFile(comm, str(xdmf_step_path), "w") as xdmf:
                    xdmf.write_mesh(self.mesh)
                    try:
                        xdmf.write_function(hef.m, float(i))
                    except TypeError:
                        xdmf.write_function(hef.m)


            # Save XDMF series.

            if xdmf_series is not None:
                try:
                    xdmf_series.write_function(hef.m, float(i))
                except TypeError:
                    xdmf_series.write_function(hef.m)


            # Save per-step ADIOS2 BP.

            bp_step_path = None

            if write_bp_per_step:
                bp_step_path = states_path / f"m_hyst_{i:04d}.bp"

                ad.write_mesh(bp_step_path, self.mesh)
                ad.write_function(
                    bp_step_path,
                    hef.m,
                    time=float(i),
                    name="m",
                )


            # Save BP series.

            if write_bp_series:
                ad.write_function(
                    bp_series_path,
                    hef.m,
                    time=float(i),
                    name="m",
                )


            # Keep TS vector synchronized if solver exists.

            if self.y is not None:
                hef.m_gpu.copy(self.y)


            # Store result.

            row = {
                "step": int(i),
                "method": method,
                "H": (Hx, Hy, Hz),
                "m_mean": (
                    float(m_mean[0]),
                    float(m_mean[1]),
                    float(m_mean[2]),
                ),
                "energy": {
                    k: float(v) for k, v in energy.items()
                },
                "max_projected_field": float(max_projected_field),
                "maxdmdt_deg_ns": float(maxdmdt_deg_ns),
                "max_nn_angle_deg": float(max_neighbor_angle_deg),
                "iterations": int(iterations),
                "nsteps": int(nsteps),
                "dt_last": float(dt_last),
                "stop_reason": stop_reason,
                "elapsed": float(elapsed),
                "xdmf_step": str(xdmf_step_path) if xdmf_step_path is not None else None,
                "bp_step": str(bp_step_path) if bp_step_path is not None else None,
            }

            results.append(row)

            # --------------------------------------------------------
            # Log line.
            # --------------------------------------------------------
            line = (
                f"{i:06d} {method:>10s} "
                f"{Hx:+.8e} {Hy:+.8e} {Hz:+.8e} "
                f"{m_mean[0]:+.8e} {m_mean[1]:+.8e} {m_mean[2]:+.8e} "
                f"{energy['E_demag']:+.8e} "
                f"{energy['E_exch']:+.8e} "
                f"{energy['E_ani']:+.8e} "
                f"{energy['E_dmi_bulk']:+.8e} "
                f"{energy['E_dmi_int']:+.8e} "
                f"{energy['E_ext']:+.8e} "
                f"{energy['E_cubic']:+.8e} "
                f"{energy['E_total']:+.8e} "
                f"{max_projected_field:+.8e} "
                f"{maxdmdt_deg_ns:+.8e} "
                f"{max_neighbor_angle_deg:+.8e} "
                f"{iterations:d} {nsteps:d} "
                f"{dt_last:+.8e} "
                f"{stop_reason} "
                f"{elapsed:+.8e} "
                f"{str(xdmf_step_path)} {str(bp_step_path)}"
            )

            print(line, flush=True)

            with open(log_path, "a") as f:
                f.write(line + "\n")


        # Close XDMF series.

        if xdmf_series is not None:
            xdmf_series.close()


        # Final state remains available in hef.m_gpu, hef.m, and self.y.

        hef.sync_m_to_function()

        if self.y is not None:
            hef.m_gpu.copy(self.y)

        print("\n[hysteresis] finished")
        print("method:", method)
        print("steps:", len(H_steps))
        print("output_dir:", str(output_path))
        print("log:", str(log_path))

        if write_bp_series:
            print("BP series:", str(bp_series_path))

        if write_xdmf_series:
            print("XDMF series:", str(xdmf_series_path))

        if return_stats:
            return results

        return None