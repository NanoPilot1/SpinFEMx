# llg_stt_module.py

import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import njit
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import fem, io
from dolfinx.fem import Function, Constant, form
from dolfinx.fem.petsc import assemble_vector

import adios4dolfinx as ad

from ..fields.Exchange import ExchangeField
from ..fields.Anisotropy import AnisotropyField
from ..fields.DMI_Bulk import DMIBULK
from ..fields.DMI_Interfacial import DMIInterfacial
from .Zhang_Li import ZhangLi



@njit(fastmath=True)
def jac_vec_stt_local_kernel(M, Hm, V, Hv, Jgrad_m, Jgrad_v, out,
                             gamma, alpha, beta, do_precess, Stab, prefZhang):
    coef = -gamma / (1.0 + alpha * alpha)
    c_stt_prec = -(beta - alpha) * prefZhang
    c_stt_damp = -(1.0 + beta * alpha) * prefZhang
    n = M.shape[0]

    for i in range(n):
        mx = M[i, 0]
        my = M[i, 1]
        mz = M[i, 2]

        vx = V[i, 0]
        vy = V[i, 1]
        vz = V[i, 2]

        hx = Hm[i, 0]
        hy = Hm[i, 1]
        hz = Hm[i, 2]

        hvx = Hv[i, 0]
        hvy = Hv[i, 1]
        hvz = Hv[i, 2]

        gmx = Jgrad_m[i, 0]
        gmy = Jgrad_m[i, 1]
        gmz = Jgrad_m[i, 2]

        gvx = Jgrad_v[i, 0]
        gvy = Jgrad_v[i, 1]
        gvz = Jgrad_v[i, 2]

        # v x Hm
        cvhx = vy * hz - vz * hy
        cvhy = vz * hx - vx * hz
        cvhz = vx * hy - vy * hx

        # m x Hv
        cmhx = my * hvz - mz * hvy
        cmhy = mz * hvx - mx * hvz
        cmhz = mx * hvy - my * hvx

        prec_x = do_precess * (cvhx + cmhx)
        prec_y = do_precess * (cvhy + cmhy)
        prec_z = do_precess * (cvhz + cmhz)

        mdHm = mx * hx + my * hy + mz * hz
        mdHv = mx * hvx + my * hvy + mz * hvz
        vdHm = vx * hx + vy * hy + vz * hz
        mdv = mx * vx + my * vy + mz * vz
        mdmm = mx * mx + my * my + mz * mz

        s = vdHm + mdHv

        damp_x = vx * mdHm - 2.0 * hx * mdv + mx * s - hvx * mdmm
        damp_y = vy * mdHm - 2.0 * hy * mdv + my * s - hvy * mdmm
        damp_z = vz * mdHm - 2.0 * hz * mdv + mz * s - hvz * mdmm

        # STT precessional-like part:
        # M x Jgrad_v + V x Jgrad_m
        cmgv_x = my * gvz - mz * gvy
        cmgv_y = mz * gvx - mx * gvz
        cmgv_z = mx * gvy - my * gvx

        cvgm_x = vy * gmz - vz * gmy
        cvgm_y = vz * gmx - vx * gmz
        cvgm_z = vx * gmy - vy * gmx

        prec_stt_x = c_stt_prec * (cmgv_x + cvgm_x)
        prec_stt_y = c_stt_prec * (cmgv_y + cvgm_y)
        prec_stt_z = c_stt_prec * (cmgv_z + cvgm_z)

        MjgradV = mx * gvx + my * gvy + mz * gvz
        VjgradM = vx * gmx + vy * gmy + vz * gmz
        MjgradM = mx * gmx + my * gmy + mz * gmz

        damp_stt_x = c_stt_damp * (
            vx * MjgradM + mx * VjgradM + mx * MjgradV - gvx * mdmm - 2.0 * gmx * mdv
        )
        damp_stt_y = c_stt_damp * (
            vy * MjgradM + my * VjgradM + my * MjgradV - gvy * mdmm - 2.0 * gmy * mdv
        )
        damp_stt_z = c_stt_damp * (
            vz * MjgradM + mz * VjgradM + mz * MjgradV - gvz * mdmm - 2.0 * gmz * mdv
        )

        one_minus = 1.0 - mdmm

        out[3 * i + 0] = (
            coef * (prec_x + alpha * damp_x)
            + prec_stt_x
            + damp_stt_x
            + Stab * (vx * one_minus - 2.0 * mx * mdv)
        )
        out[3 * i + 1] = (
            coef * (prec_y + alpha * damp_y)
            + prec_stt_y
            + damp_stt_y
            + Stab * (vy * one_minus - 2.0 * my * mdv)
        )
        out[3 * i + 2] = (
            coef * (prec_z + alpha * damp_z)
            + prec_stt_z
            + damp_stt_z
            + Stab * (vz * one_minus - 2.0 * mz * mdv)
        )


@njit(fastmath=True)
def llg_rhs_stt_local_kernel(M, H, Z, out, gamma, alpha, beta, do_precess, Stab, prefZhang):
    """
    CPU/Numba RHS for LLG + Zhang-Li STT.

    Computes, per owned node,

        dm/dt = -gamma/(1+alpha^2) [do_precess m x H + alpha m x (m x H)]
                - prefZhang [(beta-alpha) m x Z + (1+alpha beta) m x (m x Z)]
                + Stab (1-|m|^2) m

    where Z = (Jdir dot grad) m.
    """
    coef = -gamma / (1.0 + alpha * alpha)
    stt_a = beta - alpha
    stt_b = 1.0 + alpha * beta
    n = M.shape[0]

    for i in range(n):
        mx = M[i, 0]
        my = M[i, 1]
        mz = M[i, 2]

        hx = H[i, 0]
        hy = H[i, 1]
        hz = H[i, 2]

        zx = Z[i, 0]
        zy = Z[i, 1]
        zz = Z[i, 2]

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
        stab = Stab * (1.0 - norm2)

        out[3 * i + 0] = (
            coef * (do_precess * mcx + alpha * mcmx)
            - prefZhang * (stt_a * zcx + stt_b * zzx)
            + stab * mx
        )
        out[3 * i + 1] = (
            coef * (do_precess * mcy + alpha * mcmy)
            - prefZhang * (stt_a * zcy + stt_b * zzy)
            + stab * my
        )
        out[3 * i + 2] = (
            coef * (do_precess * mcz + alpha * mcmz)
            - prefZhang * (stt_a * zcz + stt_b * zzz)
            + stab * mz
        )


@njit(fastmath=True)
def pc_build_inv3_stt_kernel(
    M, H, G, kappa,
    i00, i01, i02,
    i10, i11, i12,
    i20, i21, i22,
    shift, c1, c2, c3, c4, Stab,
    include_stab, eps_reg, det_eps,
):
    """
    Fused local 3x3 STT preconditioner builder.

    Builds inv(A_i) with

        A_i = (shift + eps_reg) I - J_i^approx,

    including the LLG local block, Zhang-Li local frozen-gradient block,
    and optional norm-stabilization block.  The determinant is guarded to
    avoid NaN/Inf when the local 3x3 block is nearly singular.
    """
    n = M.shape[0]
    s = shift + eps_reg

    for idx in range(n):
        mx = M[idx, 0]
        my = M[idx, 1]
        mz = M[idx, 2]

        hx = H[idx, 0]
        hy = H[idx, 1]
        hz = H[idx, 2]

        gx = G[idx, 0]
        gy = G[idx, 1]
        gz = G[idx, 2]

        kap = kappa[idx]

        mdH = mx * hx + my * hy + mz * hz
        mdm = mx * mx + my * my + mz * mz

        # LLG precession approximate block: -c1 * (S_H - kappa*S_m)
        Jp01 = -c1 * (-hz + kap * mz)
        Jp02 = -c1 * ( hy - kap * my)
        Jp10 = -c1 * ( hz - kap * mz)
        Jp12 = -c1 * (-hx + kap * mx)
        Jp20 = -c1 * (-hy + kap * my)
        Jp21 = -c1 * ( hx - kap * mx)

        # LLG damping approximate block: c2 * (B + kappa*C)
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

        Jd00 = c2 * (B00 + kap * C00)
        Jd11 = c2 * (B11 + kap * C11)
        Jd22 = c2 * (B22 + kap * C22)
        Jd01 = c2 * (B01 + kap * C01)
        Jd02 = c2 * (B02 + kap * C02)
        Jd10 = c2 * (B10 + kap * C10)
        Jd12 = c2 * (B12 + kap * C12)
        Jd20 = c2 * (B20 + kap * C20)
        Jd21 = c2 * (B21 + kap * C21)

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
            Js00 = 0.0
            Js11 = 0.0
            Js22 = 0.0
            Js01 = 0.0
            Js02 = 0.0
            Js10 = 0.0
            Js12 = 0.0
            Js20 = 0.0
            Js21 = 0.0

        J00 = Jd00 + Js00
        J11 = Jd11 + Js11
        J22 = Jd22 + Js22
        J01 = Jp01 + Jd01 + Js01
        J02 = Jp02 + Jd02 + Js02
        J10 = Jp10 + Jd10 + Js10
        J12 = Jp12 + Jd12 + Js12
        J20 = Jp20 + Jd20 + Js20
        J21 = Jp21 + Jd21 + Js21

        # STT frozen-gradient block.
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

        J00 += c4 * Bg00
        J11 += c4 * Bg11
        J22 += c4 * Bg22
        J01 += Jstt01 + c4 * Bg01
        J02 += Jstt02 + c4 * Bg02
        J10 += Jstt10 + c4 * Bg10
        J12 += Jstt12 + c4 * Bg12
        J20 += Jstt20 + c4 * Bg20
        J21 += Jstt21 + c4 * Bg21

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

        if abs(det) < det_eps:
            if det >= 0.0:
                det = det + det_eps
            else:
                det = det - det_eps

        invdet = 1.0 / det

        i00[idx] =  (A11 * A22 - A12 * A21) * invdet
        i01[idx] = -(A01 * A22 - A02 * A21) * invdet
        i02[idx] =  (A01 * A12 - A02 * A11) * invdet
        i10[idx] = -(A10 * A22 - A12 * A20) * invdet
        i11[idx] =  (A00 * A22 - A02 * A20) * invdet
        i12[idx] = -(A00 * A12 - A02 * A10) * invdet
        i20[idx] =  (A10 * A21 - A11 * A20) * invdet
        i21[idx] = -(A00 * A21 - A01 * A20) * invdet
        i22[idx] =  (A00 * A11 - A01 * A10) * invdet


@njit(fastmath=True)
def pc_apply_inv3_kernel(
    x,
    y,
    i00, i01, i02,
    i10, i11, i12,
    i20, i21, i22,
):
    n = x.size // 3
    for i in range(n):
        j = 3 * i
        x0 = x[j]
        x1 = x[j + 1]
        x2 = x[j + 2]

        y[j] = i00[i] * x0 + i01[i] * x1 + i02[i] * x2
        y[j + 1] = i10[i] * x0 + i11[i] * x1 + i12[i] * x2
        y[j + 2] = i20[i] * x0 + i21[i] * x1 + i22[i] * x2


class JvContextSTTCPU:
    """
    Matrix-free Jacobian context for implicit CPU STT BDF.

    The matrix action is

        A x = shift*x - J_STT(x),

    and the Python PC applies an approximate local inverse based on one 3x3
    block per owned node.  The PC includes local LLG terms, Zhang-Li frozen
    gradient terms, and the norm-stabilization block; it excludes demag.
    """

    def __init__(self, hef):
        self.hef = hef
        self.shift = 0.0
        self.calls = 0
        self.callsPre = 0
        self.enable_pc = True
        self._pc_ready = False

        diag = self.hef.K_total.getDiagonal()
        diagK = diag.getArray(readonly=True).copy()

        ani = getattr(self.hef, "anisotropy_field", None)
        if (
            ani is not None
            and not hasattr(ani, "K")
            and hasattr(ani, "diagonal_array")
        ):
            ani_diag = ani.diagonal_array(owned_only=True)
            diagK[: ani_diag.size] += ani_diag

        self.diagK = diagK
        d = diagK.reshape(-1, 3)
        self.kappa_abs = np.mean(np.abs(d), axis=1).astype(np.float64)
        self.kappa_sgn = np.mean(d, axis=1).astype(np.float64)

        g = float(self.hef.gamma)
        a = float(self.hef.alpha)
        b = float(self.hef.beta)
        dp = float(self.hef.do_precess)

        self.gamma = g
        self.alpha = a
        self.beta = b
        self.do_precess = dp
        self.Stab = float(self.hef.Stab)
        self.prefZ = float(self.hef.prefZhang)

        self.c1 = (g / (1.0 + a * a)) * dp
        self.c2 = (g * a / (1.0 + a * a))
        self.c3 = (b - a) * self.prefZ
        self.c4 = -(1.0 + a * b) * self.prefZ

        n = self.hef.M_cached.shape[0]
        self.i00 = np.empty(n, dtype=np.float64)
        self.i01 = np.empty(n, dtype=np.float64)
        self.i02 = np.empty(n, dtype=np.float64)
        self.i10 = np.empty(n, dtype=np.float64)
        self.i11 = np.empty(n, dtype=np.float64)
        self.i12 = np.empty(n, dtype=np.float64)
        self.i20 = np.empty(n, dtype=np.float64)
        self.i21 = np.empty(n, dtype=np.float64)
        self.i22 = np.empty(n, dtype=np.float64)

    def update_pc_full_fast(
        self,
        shift,
        include_stab=True,
        use_abs_kappa=True,
        eps_reg=1e-14,
        det_eps=1e-30,
    ):
        self.shift = float(shift)
        kappa = self.kappa_abs if use_abs_kappa else self.kappa_sgn

        pc_build_inv3_stt_kernel(
            self.hef.M_cached,
            self.hef.Hm_cached,
            self.hef.JdotGrad_m_cache,
            kappa,
            self.i00, self.i01, self.i02,
            self.i10, self.i11, self.i12,
            self.i20, self.i21, self.i22,
            float(self.shift),
            float(self.c1),
            float(self.c2),
            float(self.c3),
            float(self.c4),
            float(self.Stab),
            bool(include_stab),
            float(eps_reg),
            float(det_eps),
        )
        self._pc_ready = True

    def apply(self, pc, x, y):
        self.callsPre += 1
        if (not self.enable_pc) or (not self._pc_ready):
            x.copy(y)
            return

        xv = x.getArray(readonly=True)
        yv = y.getArray()
        pc_apply_inv3_kernel(
            xv,
            yv,
            self.i00, self.i01, self.i02,
            self.i10, self.i11, self.i12,
            self.i20, self.i21, self.i22,
        )

    def mult(self, A, x, y):
        self.calls += 1
        xv = x.getArray(readonly=True)
        yv = y.getArray()

        self.hef.jac_vec_times_STT(xv, out=self.hef.Jv_buffer)
        yv[:] = self.shift * xv - self.hef.Jv_buffer


# ---------------------------------------------------------
#  EffectiveFieldSTT (LLG + STT Zhang-Li)
# ---------------------------------------------------------
class EffectiveFieldSTT:
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
        H0_vec,
        gamma=2.211e5,
        alpha=0.5,
        do_precess=1,
        Jmagnitude=0.0,
        Jdir_vec=None,
        P=0.0,
        beta=0.0,
        use_demag=True, 
        demag_method="lindholm",
        demag_kwargs=None,
    ):
        self.mesh = mesh
        self.V = fem.functionspace(
            self.mesh, ("Lagrange", 1, (self.mesh.geometry.dim,))
        )
        self.V1 = fem.functionspace(self.mesh, ("Lagrange", 1))

        self.Ms = Ms
        self.gamma = gamma
        self.alpha = alpha
        self.A = Aex
        self.Ku = Ku
        self.D_bulk = D_bulk
        self.D_int = D_int
        self.do_precess = do_precess
        self.use_demag = use_demag  

        self.n_ani = fem.Function(self.V)
        self.n_ani.x.array[:] = n_ani_vec

        self.n0_int = fem.Function(self.V)
        self.n0_int.x.array[:] = n0_int_vec

        self.H0_ext = fem.Function(self.V)
        self.H0_ext.x.array[:] = H0_vec.copy()
        self.H0_ext.x.scatter_forward()


        self.comm = self.mesh.comm
        self.m = fem.Function(self.V)
        self.H_eff = Function(self.V)

        self.prefactor = -self.gamma / (1.0 + self.alpha**2)
        self.Stab = self.Ms * self.gamma / (1 + self.alpha**2)*0.5

        self.P = P
        self.Jmagnitude = Jmagnitude
        self.beta = beta
        self.Jdir_vec = Jdir_vec

        e = 1.6021766e-19
        muB = 9.27400915e-24

        self.prefZhang = (
            self.Jmagnitude
            * self.P
            * muB
            / (e * self.Ms * (1.0 + self.beta**2))
            * 1.0
            / (1.0 + self.alpha**2)
            / 1e-9
        )

        self.n_nodes_local =  self.V.dofmap.index_map.size_local

        self.start, self.end = self.V.dofmap.index_map.local_range
        owned_dofs = self.end - self.start
        self.local_dofs = self.end - self.start
        self.local_size = 3 * self.local_dofs


        self.mx = np.zeros(self.n_nodes_local)
        self.my = np.zeros(self.n_nodes_local)
        self.mz = np.zeros(self.n_nodes_local)

        self.Hx = np.zeros(self.n_nodes_local)
        self.Hy = np.zeros(self.n_nodes_local)
        self.Hz = np.zeros(self.n_nodes_local)

        self.mcx = np.zeros(self.n_nodes_local)
        self.mcy = np.zeros(self.n_nodes_local)
        self.mcz = np.zeros(self.n_nodes_local)

        self.mcmx = np.zeros(self.n_nodes_local)
        self.mcmy = np.zeros(self.n_nodes_local)
        self.mcmz = np.zeros(self.n_nodes_local)

        self.Hfield = np.zeros(3 * self.n_nodes_local)

        # STT arrays
        self.Zcx = np.zeros(self.n_nodes_local)
        self.Zcy = np.zeros(self.n_nodes_local)
        self.Zcz = np.zeros(self.n_nodes_local)

        self.ZZcx = np.zeros(self.n_nodes_local)
        self.ZZcy = np.zeros(self.n_nodes_local)
        self.ZZcz = np.zeros(self.n_nodes_local)

        self.Zhang_x = np.zeros(self.n_nodes_local)
        self.Zhang_y = np.zeros(self.n_nodes_local)
        self.Zhang_z = np.zeros(self.n_nodes_local)

        self.zhang_le = np.zeros(3 * self.n_nodes_local)

        self.He = np.zeros(3 * self.n_nodes_local)
        
        v = ufl.TestFunction(self.V)
        tmp_0 = ufl.dot( v, Constant(self.mesh, PETSc.ScalarType((1.0, 1.0, 1.0)))) * ufl.dx

        volN_f = fem.Function(self.V)
        volN_f.x.petsc_vec.set(0.0)
        assemble_vector(volN_f.x.petsc_vec, form(tmp_0))
        volN_f.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,mode=PETSc.ScatterMode.REVERSE,)
        volN_f.x.scatter_forward()
        volN = volN_f.x.array

        self.Hvec = np.zeros(3 * self.n_nodes_local)
        self.dmdt = Function(self.V)


        self.demag_field = None
        if self.use_demag:
            from ..fields.Demag import make_demag_field

            if demag_kwargs is None:
                demag_kwargs = {}

            if self.mesh.comm.rank == 0:
                print(f"[Demag-STT] Precomputing demag method='{demag_method}' ...", flush=True)

            t0 = perf_counter()
            self.demag_field = make_demag_field(
                demag_method, self.mesh, self.V, self.V1, self.Ms, **demag_kwargs
            )
            t1 = perf_counter()

            if self.mesh.comm.rank == 0:
                print(f"[Demag-STT] Precomputation finished in {t1 - t0:.2f} s", flush=True)


            

        self.exchange_field = ExchangeField(
            self.mesh, self.V, self.A, self.Ms, volN
        )

        self.anisotropy_field = AnisotropyField(
            self.mesh, self.V, self.Ku, self.Ms, n_ani_vec, volN
        )

        self.DMIBULK = DMIBULK(
            self.mesh, self.V, self.V1, self.D_bulk, self.Ms, volN
        )


        self.DMI_int = None
        if abs(self.D_int) > 0.0:
            self.DMI_int = DMIInterfacial(
                self.mesh,
                self.V,
                self.V1,
                self.D_int,
                n0_int_vec,
                self.Ms,
                volN,
            )


        self.ZhangLi = ZhangLi(self.mesh, self.V, self.Jdir_vec, volN)

        self.JacStep = 0
        self.LLGStep = 0


        # Sparse linear operator used by the Jacobian/preconditioner.
        #
        # Exchange and DMI are differential FEM operators and remain in K_total.
        # Uniaxial anisotropy may now be nodal/matrix-free.  If the anisotropy
        # object still exposes a PETSc matrix K, keep backward compatibility and
        # add it here; otherwise it is applied explicitly in update_jac_state()
        # and jac_vec_times_STT().
        self.K_total = self.exchange_field.K + self.DMIBULK.K

        if self.anisotropy_field is not None and hasattr(self.anisotropy_field, "K"):
            self.K_total = self.K_total + self.anisotropy_field.K

        if self.DMI_int is not None:
            self.K_total = self.K_total + self.DMI_int.K

        self.v_jac = Function(self.V)
        self.v_jac.x.array[:] = 0


        self.H_m = Function(self.V)
        self.H_v = Function(self.V)


        self.Jv_buffer = np.zeros(3 * owned_dofs, dtype=np.float64)

        self.M_cached = np.zeros((owned_dofs, 3))
        self.Hm_cached = np.zeros((owned_dofs, 3))
        self.JdotGrad_m_cache = np.zeros((owned_dofs, 3))

        # Work buffers for matrix-free/nodal anisotropy in the Jacobian path.
        # They are only used when the anisotropy object does not expose a K matrix.
        self.Hani_m_cache = np.zeros((owned_dofs, 3), dtype=np.float64)
        self.Hani_v_cache = np.zeros((owned_dofs, 3), dtype=np.float64)
        self.Hv_total = np.zeros((owned_dofs, 3), dtype=np.float64)

        self.local_dofs = owned_dofs
        self.local_size = 3 * self.local_dofs

        jac_vec_stt_local_kernel(
            self.M_cached,
            self.Hm_cached,
            self.M_cached,
            self.Hm_cached,
            self.JdotGrad_m_cache,
            self.JdotGrad_m_cache,
            self.Jv_buffer,
            self.gamma,
            self.alpha,
            self.beta,
            self.do_precess,
            self.Stab,
            self.prefZhang,
        )


    def compute_H_eff(self, m):
  
        self.He[:] = self.exchange_field.compute(m).x.petsc_vec.array

        if self.demag_field is not None:  
            self.He += self.demag_field.compute(m).x.petsc_vec.array

        if abs(self.Ku) > 0.0:
            self.He += self.anisotropy_field.compute(m).x.petsc_vec.array
        if abs(self.D_bulk) > 0.0:
            self.He += self.DMIBULK.compute(m).x.petsc_vec.array

        if self.DMI_int is not None and abs(self.D_int) > 0.0:
            self.He += self.DMI_int.compute(m).x.petsc_vec.array

        self.He += self.H0_ext.x.petsc_vec.array

        return self.He

 
    def _field_energy_global(self, value):
        """
        Convert a local field-energy contribution into a global scalar.

        The field Energy(...) methods used by the legacy CPU stack are treated
        as local contributions.  The monitor in this file already gathered and
        summed them manually; this helper gives compute_Energy(...) the same MPI
        semantics for direct calls.
        """
        return float(self.comm.allreduce(float(value), op=MPI.SUM))

    def compute_Energy(self, m):
        E_exch = self._field_energy_global(self.exchange_field.Energy(m))

        E_demag = 0.0
        if self.demag_field is not None:
            E_demag = self._field_energy_global(self.demag_field.Energy(m))

        E_ani = 0.0
        if abs(self.Ku) > 0.0:
            E_ani = self._field_energy_global(self.anisotropy_field.Energy(m))

        E_dmi_bulk = 0.0
        if abs(self.D_bulk) > 0.0:
            E_dmi_bulk = self._field_energy_global(self.DMIBULK.Energy(m))

        E_dmi_int = 0.0
        if self.DMI_int is not None and abs(self.D_int) > 0.0:
            E_dmi_int = self._field_energy_global(self.DMI_int.Energy(m))

        return E_exch + E_demag + E_ani + E_dmi_bulk + E_dmi_int


    def update_jac_state(self):


        self.K_total.mult(self.m.x.petsc_vec, self.H_m.x.petsc_vec)

        M_loc =  self.m.x.petsc_vec.getArray(readonly=True).reshape(-1, 3)
        Hm_loc = self.H_m.x.petsc_vec.getArray(readonly=True).reshape(-1, 3)

        self.M_cached[:, :] = M_loc

        # If anisotropy is nodal/matrix-free, it is not part of K_total and
        # must be added explicitly to H(m) for the Jacobian linearization.
        if (
            abs(self.Ku) > 0.0
            and self.anisotropy_field is not None
            and not hasattr(self.anisotropy_field, "K")
            and hasattr(self.anisotropy_field, "apply_array")
        ):
            self.anisotropy_field.apply_array(
                self.m.x.petsc_vec.getArray(readonly=True),
                out_flat=self.Hani_m_cache.reshape(-1),
                owned_only=True,
            )
            self.Hm_cached[:, :] = Hm_loc + self.Hani_m_cache
        else:
            self.Hm_cached[:, :] = Hm_loc

        self.Hm_cached[:, :] += self.H0_ext.x.petsc_vec.getArray(readonly=True).reshape(-1, 3)


        Jgm = self.ZhangLi.compute(self.m).x.petsc_vec.getArray(readonly=True).reshape(-1, 3)
        self.JdotGrad_m_cache[:, :] = Jgm

    def jac_vec_times_STT(self, v, out):
        self.JacStep += 1

        self.v_jac.x.array[:self.local_size] = v
        self.K_total.mult(self.v_jac.x.petsc_vec, self.H_v.x.petsc_vec)

        JdotGrad_v = self.ZhangLi.compute(self.v_jac).x.petsc_vec.getArray(readonly=True).reshape(-1, 3)

        M = self.M_cached
        Hm = self.Hm_cached
        JdotGrad_m = self.JdotGrad_m_cache
        V = v.reshape(-1, 3)
        Hv_lin = self.H_v.x.petsc_vec.getArray(readonly=True).reshape(-1, 3)

        # Add J_ani*v explicitly when anisotropy is nodal/matrix-free.
        if (
            abs(self.Ku) > 0.0
            and self.anisotropy_field is not None
            and not hasattr(self.anisotropy_field, "K")
            and hasattr(self.anisotropy_field, "apply_array")
        ):
            self.anisotropy_field.apply_array(
                v,
                out_flat=self.Hani_v_cache.reshape(-1),
                owned_only=True,
            )
            np.add(Hv_lin, self.Hani_v_cache, out=self.Hv_total)
            Hv = self.Hv_total
        else:
            Hv = Hv_lin

        jac_vec_stt_local_kernel(
            M,
            Hm,
            V,
            Hv,
            JdotGrad_m,
            JdotGrad_v,
            out,
            self.gamma,
            self.alpha,
            self.beta,
            self.do_precess,
            self.Stab,
            self.prefZhang,
        )


    def llg_rhs_STT(self, m):
        self.LLGStep += 1

        m_numeric = m.x.petsc_vec.getArray(readonly=True)
        self.Hfield[:] = self.compute_H_eff(m)
        self.zhang_le[:] = self.ZhangLi.compute(m).x.petsc_vec.array

        M = m_numeric[: self.local_size].reshape((-1, 3))
        H = self.Hfield[: self.local_size].reshape((-1, 3))
        Z = self.zhang_le[: self.local_size].reshape((-1, 3))
        out = self.dmdt.x.petsc_vec.array[: self.local_size]

        llg_rhs_stt_local_kernel(
            M,
            H,
            Z,
            out,
            float(self.gamma),
            float(self.alpha),
            float(self.beta),
            float(self.do_precess),
            float(self.Stab),
            float(self.prefZhang),
        )

        return self.dmdt

    def ifunction_STT(self, ts, t, y, ydot, f):
        self.LLGStep += 1

        y.copy(self.m.x.petsc_vec)
        self.m.x.scatter_forward()

        dmdt = self.llg_rhs_STT(self.m)
        #dmdt.x.scatter_forward()

        f.waxpy(-1.0, dmdt.x.petsc_vec, ydot)
        return 0



class LLG_STT:
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


        self._Jmag = 0.0
        self._Jdir_vec = None
        self._P = 0.0
        self._beta = 0.0

        self._has_exchange = False
        self._has_demag = False  
        self._demag_method = "lindholm"
        self._demag_kwargs = {}

        self._has_anisotropy = False
        self._has_dmi_bulk = False
        self._has_dmi_int = False
        self._has_H0 = False
        self._has_current = False

        self.hef: EffectiveFieldSTT | None = None


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

    def add_external_field(self, H0_vec):
        """
        Add a static external field.

        This legacy CPU STT module intentionally supports only a static H0_vec.
        Time-dependent fields are available in the GPU STT path, but are not
        implemented here.
        """
        self._H0_vec = H0_vec
        self._has_H0 = True

    def add_current(self, Jmagnitude, Jdir_vec, P, beta):

        self._Jmag = Jmagnitude
        self._Jdir_vec = Jdir_vec
        self._P = P
        self._beta = beta
        self._has_current = True


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

        self.hef = EffectiveFieldSTT(
            self.mesh,
            self.Ms,
            Aex,
            Ku,
            n_ani_vec,
            D_bulk,
            D_int,
            n0_int_vec,
            H0_vec,
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
        )

    def solve(
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
        monitor_fn=None,
    ):
        

        if self.hef is None:
            self._build_effective_field()
        hef = self.hef

        hef.m.x.array[:] = m0_array
        hef.m.x.scatter_forward()

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
        opts["ts_max_steps"] = 5000000
        opts["snes_rtol"] = snes_rtol
        opts["snes_atol"] = snes_atol
        opts["snes_max_it"] = 8
        opts["ksp_type"] = "gmres"
        opts["ksp_rtol"] = ksp_rtol
        opts["ts_max_snes_failures"] = -1

        ts.setTime(t0)
        ts.setTimeStep(dt_init)
        ts.setMaxTime(t_final)
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.STEPOVER)

        dm = PETSc.DMShell().create(comm=self.mesh.comm)
        dm.setGlobalVector(hef.m.x.petsc_vec.copy())
        ts.setDM(dm)

        snes = ts.getSNES()

        n_loc = hef.m.x.petsc_vec.getLocalSize()
        n_glob = hef.m.x.petsc_vec.getSize()


        # Matrix-free Jacobian/preconditioner context.
        # Kept at module level so it is reusable and testable outside solve().
        J = PETSc.Mat().create(comm=self.mesh.comm)
        ctx = JvContextSTTCPU(hef)
        J.setSizes([[n_loc, n_glob], [n_loc, n_glob]])
        J.setType("python")
        J.setPythonContext(ctx)
        J.setUp()

        ksp = snes.getKSP()
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.PYTHON)
        pc.setPythonContext(ctx)



        def IJac(ts_, t, y, ydot, shift, A, B):
            y.copy(hef.m.x.petsc_vec)
            hef.m.x.scatter_forward()

            hef.update_jac_state()

            ctx.update_pc_full_fast(shift, include_stab=True, use_abs_kappa=True)

            return PETSc.Mat.Structure.SAME_NONZERO_PATTERN


        ts.setIFunction(hef.ifunction_STT)
        ts.setIJacobian(IJac, J)

        ts.setFromOptions()

        y = hef.m.x.petsc_vec.copy()
        y.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,mode=PETSc.ScatterMode.FORWARD,)
  
        hef.m.x.scatter_forward()
        hef.compute_H_eff(hef.m)

        if dt_save is not None:
            if dt_snap is None:
                dt_snap = dt_save

            if self.mesh.comm.rank == 0:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            log_path = Path(output_dir) / "log.txt"

            last_save_n = {"n": -1}
            last_snap_n = {"n": -1}
            snap_counter = {"k": 0}
            first_print = {"done": False}





            def default_monitor(ts_, step, t, u, hef_, mesh_):

                u.copy(hef_.m.x.petsc_vec)
                hef_.m.x.scatter_forward()

                Exch = hef_.exchange_field.Energy(hef_.m)
                Demag = 0.0                      
                if hef_.demag_field is not None:  
                    Demag = hef_.demag_field.Energy(hef_.m)

                Ani = 0.0
                if getattr(hef_, "Ku", 0.0) != 0.0:
                    Ani = hef_.anisotropy_field.Energy(hef_.m)

                DMI_bulk = 0.0
                if getattr(hef_, "D_bulk", 0.0) != 0.0:
                    DMI_bulk = hef_.DMIBULK.Energy(hef_.m)

                DMI_int = 0.0
                if getattr(hef_, "D_int", 0.0) != 0.0 and hef_.DMI_int is not None:
                    DMI_int = hef_.DMI_int.Energy(hef_.m)

                Exch_total = mesh_.comm.gather(Exch, root=0)
                Demag_total = mesh_.comm.gather(Demag, root=0)
                Ani_total = mesh_.comm.gather(Ani, root=0)
                DMI_bulk_total = mesh_.comm.gather(DMI_bulk, root=0)
                DMI_int_total = mesh_.comm.gather(DMI_int, root=0)

                mag = mesh_.comm.gather(hef_.m.x.petsc_vec.getArray(readonly=True), root=0)

                # snapshots
                n_snap = int(np.trunc(t / dt_snap))
                if n_snap != last_snap_n["n"]:
                    last_snap_n["n"] = n_snap
                    filename = Path(output_dir) / f"m{snap_counter['k']:03d}.xdmf"
                    snap_counter["k"] += 1
                    with io.XDMFFile(mesh_.comm, str(filename), "w") as xdmf:
                        xdmf.write_mesh(mesh_)
                        xdmf.write_function(hef_.m)

                if mesh_.comm.rank == 0:
                    mag = np.reshape(np.concatenate(mag), (-1, 3))

                    E_exch = np.sum(Exch_total)
                    E_demag = np.sum(Demag_total)
                    E_ani = np.sum(Ani_total)
                    E_db = np.sum(DMI_bulk_total)
                    E_di = np.sum(DMI_int_total)
                    E_tot = E_exch + E_demag + E_ani + E_db + E_di


                    if not first_print["done"]:
                        header = (
                            f"{'time':>10} {'<mx>':>15} {'<my>':>15} {'<mz>':>15} "
                            f"{'E_demag':>15} {'E_exch':>15} {'E_ani':>15} "
                            f"{'E_dmi_bulk':>15} {'E_dmi_int':>15} {'E_total':>15}"
                        )
                        print(header)
                        with open(log_path, "w") as f:
                            f.write(header + "\n")
                        first_print["done"] = True

                    line = (
                        f"{t*1e9:10.4f} "
                        f"{mag[:,0].mean():15.6f} {mag[:,1].mean():15.6f} {mag[:,2].mean():15.6f} "
                        f"{E_demag:15.4e} {E_exch:15.4e} {E_ani:15.4e} "
                        f"{E_db:15.4e} {E_di:15.4e} {E_tot:15.4e}"
                    )
                    print(line)
                    with open(log_path, "a") as f:
                        f.write(line + "\n")
                    sys.stdout.flush()

            def monitor(ts_, step, t, u):
                n = int(np.trunc(t / dt_save))

                #if step % 2 == 0:

                if n != last_save_n["n"]:
                    last_save_n["n"] = n
                    if monitor_fn is not None:
                        monitor_fn(ts_, step, t, u, hef, self.mesh)
                    else:
                        default_monitor(ts_, step, t, u, hef, self.mesh)

            ts.setMonitor(monitor)

        ts.setSolution(y)


        tstart = perf_counter()
        ts.solve(y)
        elapsed = perf_counter() - tstart


        comm = self.mesh.comm
        #mag_all = comm.gather(y.array, root=0)
        #Pre_calls = comm.gather(ctx.callsPre, root=0)
        #Jac_calls = comm.gather(ctx.calls, root=0)
        #LLG_calls = comm.gather(hef.LLGStep, root=0)
        #sizecores = comm.Get_size()

        if comm.rank == 0:
            print(f"\n  ts.solve : {elapsed:.3f} s")
            #print("jac calls", Jac_calls)
            #print("prec calls", Pre_calls)
            #print("LLG calls", LLG_calls)




        return y, ctx, elapsed