import dolfinx
from dolfinx import fem, io
from dolfinx.fem import  Constant, functionspace, form
from dolfinx.fem.petsc import (assemble_vector,)
from mpi4py import MPI

from ..fields.Exchange import ExchangeField
from ..fields.Anisotropy import AnisotropyField
from ..fields.DMI_Bulk import DMIBULK
from ..fields.DMI_Interfacial import DMIInterfacial
from ..fields.Cubic_Anisotropy import CubicAnisotropyField
import adios4dolfinx as ad
import ufl
import numpy as np
from petsc4py import PETSc
from time import perf_counter
from pathlib import Path
import sys
import hashlib
from numba import njit



@njit(fastmath=True)
def jac_vec_local_kernel(M, Hm, V, Hv, out, gamma, alpha, do_precess, Stab):
    coef = -gamma / (1.0 + alpha * alpha)
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

        cvhx = vy * hz - vz * hy
        cvhy = vz * hx - vx * hz
        cvhz = vx * hy - vy * hx

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

        one_minus = 1.0 - mdmm

        out[3 * i + 0] = coef * (prec_x + alpha * damp_x) + Stab * (vx * one_minus - 2.0 * mx * mdv)
        out[3 * i + 1] = coef * (prec_y + alpha * damp_y) + Stab * (vy * one_minus - 2.0 * my * mdv)
        out[3 * i + 2] = coef * (prec_z + alpha * damp_z) + Stab * (vz * one_minus - 2.0 * mz * mdv)


# ---------------------------------------------------------
#  Effective Field Class
# ---------------------------------------------------------
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
        use_demag=True,
        demag_method="lindholm", 
        demag_kwargs=None,
        H0_static=None,   
        H_time_func=None   
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

        # Anisotropy axis
        self.n_ani = fem.Function(self.V)
        self.n_ani.x.array[:] = n_ani_vec

        self.Kc1 = float(Kc1)
        self.cubic_field = None

        self.n0_int = fem.Function(self.V)
        self.n0_int.x.array[:] = n0_int_vec

        self.comm = self.mesh.comm
        self.m = fem.Function(self.V)


        self.H_time_func = H_time_func

        self.H_eff = fem.Function(self.V)
        self.prefactorEQ= -self.gamma / ((1 + self.alpha**2))


        # A weak penalty term Stab*(1-|m|^2)m is used to mitigate norm drift (|m|<1)
        # observed during time integration due to discretization/solver tolerances
        # (see Sci. Rep. 15, 15775 (2025)). In skyrmion-on-curved-geometry tests,
        # the full prefactor slightly perturbed the trajectory; using 0.5*Stab
        # eliminate this effect while keeping |m| close to 1.
                

        self.Stab = self.Ms * self.gamma / (1 + self.alpha**2)*0.5

        self.n_nodes_local =  self.V.dofmap.index_map.size_local

        self.start, self.end = self.V.dofmap.index_map.local_range
        owned_dofs = self.end - self.start
        self.local_dofs = self.end - self.start
        self.local_size = 3 * self.local_dofs


        self.coords = self.mesh.geometry.x

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
        self.norm = np.zeros(self.n_nodes_local)
        self.Hv_total = np.zeros((owned_dofs, 3), dtype=np.float64)


        self.Hfield = np.zeros(3 * self.n_nodes_local)

        self.He = np.zeros(3 * self.n_nodes_local)

    
        if self.H_time_func is not None:
            self.Ht = np.zeros(self.local_size, dtype=np.float64)     # flat (3*N_owned)
            self.Ht_view = self.Ht.reshape((-1, 3))                   
        else:
            self.Ht = None
            self.Ht_view = None

        v = ufl.TestFunction(self.V)
        tmp_0 = ufl.dot( v, Constant(self.mesh, PETSc.ScalarType((1.0, 1.0, 1.0)))) * ufl.dx

        volN_f = fem.Function(self.V)
        volN_f.x.petsc_vec.set(0.0)
        assemble_vector(volN_f.x.petsc_vec, form(tmp_0))
        volN_f.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,mode=PETSc.ScatterMode.REVERSE,)
        volN_f.x.scatter_forward()
        volN = volN_f.x.array


        self.vol_nodes = np.asarray(volN[: self.local_size], dtype=np.float64).reshape((-1, 3))[:, 0]
        self.volume_scale_energy = 1e-27
        self.mu0 = 4.0 * np.pi * 1e-7

        self.dmdt = fem.Function(self.V)

        # ---------------- Effective fields contributions ----------------


        self.demag_field = None
        if self.use_demag:
            from ..fields.Demag import make_demag_field
            if demag_kwargs is None:
                demag_kwargs = {}

            if self.mesh.comm.rank == 0:
                print(f"[Demag] Precomputing demag method='{demag_method}' ...", flush=True)

            t0 = perf_counter()
            self.demag_field = make_demag_field(
                demag_method, self.mesh, self.V, self.V1, self.Ms, **demag_kwargs
            )
            t1 = perf_counter()

            if self.mesh.comm.rank == 0:
                print(f"[Demag] Precomputation finished in {t1 - t0:.2f} s", flush=True)



        self.exchange_field = ExchangeField(
            self.mesh, self.V, self.A, self.Ms, volN 
        )

        self.anisotropy_field = AnisotropyField(
            self.mesh, self.V, self.Ku, self.Ms, n_ani_vec, volN 
        )

        # The anisotropy implementation can be either:
        #   (i) matrix-based, exposing .K, or
        #   (ii) nodal/matrix-free, exposing compute(), Energy(), and
        #        optionally diagonal_array().
        # The nodal version is:
        # H_ani_i = 2 Ku/(mu0 Ms) * (m_i dot n_i) n_i.
        self.anisotropy_has_matrix = hasattr(self.anisotropy_field, "K")

        self.DMIBULK = DMIBULK(
            self.mesh, self.V, self.V1, self.D_bulk, self.Ms, volN 
        )

        # Interfacial DMI: can be None if D_int = 0
        self.DMI_int = None
        if abs(self.D_int) > 0.0:
            self.DMI_int = DMIInterfacial(
                self.mesh, self.V, self.V1, self.D_int, n0_int_vec, self.Ms, volN 
            )


        if abs(self.Kc1) > 0.0:
            if (u1_cub is None) or (u2_cub is None):
                raise ValueError("Kc1 != 0 but u1_cub/u2_cub were not provided")

            self.cubic_field = CubicAnisotropyField(self.mesh, self.V, self.Kc1, self.Ms,u1=u1_cub, u2=u2_cub)

            # buffer for Hv_cubic in jac_times_vec (owned)
            self.Hv_cubic = np.zeros((self.local_dofs, 3), dtype=np.float64)


        self.JacSteps = 0
        self.LLGSteps = 0

        # ---------------- K_total: sum of matrix-based linear terms ----------------
        # Exchange and DMI are derivative operators and remain matrix-based.
        # Uniaxial anisotropy may be matrix-based or nodal/matrix-free.
        # If it is nodal, it is NOT included in K_total and is added separately
        # in compute_H_eff_fast(), energy, Jacobian-vector products, and the PC.

        self.K_total = self.exchange_field.K + self.DMIBULK.K
        if self.anisotropy_has_matrix:
            self.K_total = self.K_total + self.anisotropy_field.K
        if self.DMI_int is not None:
            self.K_total = self.K_total + self.DMI_int.K

        # ---------------- Fast-path linear field buffer ----------------
        # H_lin_buf stores K_total @ m (i.e. exchange + uniaxial anis + DMI bulk
        # + DMI int, all in one sparse mat-vec).  Used by compute_H_eff_fast()
        # and the *_fast energy methods so the minimizer does not pay for one
        # K.mult per linear interaction.
        self.H_lin_buf = fem.Function(self.V)
        # Tracks whether H_lin_buf and the demag/cubic buffers are coherent
        # with the current self.m.  Set to True by compute_H_eff_fast(), and
        # to False by any path that does not refresh H_lin_buf.
        self._h_lin_valid = False

        self.m_jac = fem.Function(self.V)
        self.v_jac = fem.Function(self.V)



        self.H_m = fem.Function(self.V)
        self.H_v = fem.Function(self.V)



        self.Jv_buffer = np.zeros(3 * owned_dofs, dtype=np.float64)

        self.M_cached = np.zeros((owned_dofs, 3))
        self.Hm_cached = np.zeros((owned_dofs, 3))

        jac_vec_local_kernel(
            self.M_cached,
            self.Hm_cached,
            self.M_cached,
            self.Hm_cached,
            self.Jv_buffer,
            self.gamma,
            self.alpha,
            self.do_precess,
            self.Stab,
        )

        self.H0_ext = fem.Function(self.V)

        # H0_static must be 3*N (flattened as m)
        if H0_static is not None:
            self.H0_ext.x.array[:] = np.array(H0_static, copy=True)
            self.H0_ext.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,     mode=PETSc.ScatterMode.FORWARD)    

        else:
            self.H0_ext.x.array[:] = 0

        self.H0_static = self.H0_ext.x.array
        
        #self.H0_owned = self.H0_static[:self.local_size].reshape((-1, 3))

        self.current_time = 0.0

        # Work buffer for H_ext = H0 + H_time(t) in flat owned layout.
        # Kept separate from self.He to avoid corrupting the cached effective field.
        self.H_ext_work = np.zeros(self.local_size, dtype=np.float64)

        # Cache-validity metadata for self.He.  The CPU minimizer may reuse
        # self.He after a line-search energy evaluation; the signature prevents
        # silently reusing a field computed for a different magnetization.
        self._he_valid = False
        self._he_valid_signature = None

    def _field_energy_global(self, field, m, *, prefer_lumped=False):

        if hasattr(field, "Energy_global"):
            return float(field.Energy_global(m))

        if prefer_lumped and hasattr(field, "Energy_lumped_global"):
            return float(field.Energy_lumped_global(m))

        if prefer_lumped and hasattr(field, "Energy_lumped"):
            local = float(field.Energy_lumped(m))
        else:
            local = float(field.Energy(m))

        return float(self.comm.allreduce(local, op=MPI.SUM))

    def _owned_signature_from_flat(self, arr_flat):
        """Stable local signature for the owned magnetization entries.

        This is intentionally local: each rank only needs to know whether its
        local slice of self.He matches the local slice of the PETSc vector it is
        about to use.
        """
        arr = np.ascontiguousarray(np.asarray(arr_flat[: self.local_size], dtype=np.float64))
        h = hashlib.blake2b(digest_size=16)
        h.update(memoryview(arr).cast("B"))
        h.update(str(arr.shape).encode("ascii"))
        return h.digest()

    def _owned_signature_from_function(self, m):
        return self._owned_signature_from_flat(m.x.array)

    def he_matches_vec(self, vec):
        """True iff self.He is valid for the owned entries of PETSc Vec vec."""
        if not getattr(self, "_he_valid", False):
            return False
        arr = vec.getArray(readonly=True)
        return self._owned_signature_from_flat(arr) == getattr(self, "_he_valid_signature", None)

    def _mark_he_valid_for(self, m):
        self._he_valid = True
        self._he_valid_signature = self._owned_signature_from_function(m)

    def _invalidate_he_cache(self):
        self._he_valid = False
        self._he_valid_signature = None

    def _eval_H_time(self, t):
        """Evaluate H_time_func and return self.Ht in flat owned layout.

        Accepted return shapes:
          - (3,), uniform field
          - (local_size,), flat vector field
          - (local_dofs, 3), owned nodal vector field
          - (N, 3) with N >= local_dofs, where the owned prefix is used

        The callable may accept either H_time_func(t, coords) or H_time_func(t).
        """
        if self.H_time_func is None:
            return None

        try:
            out = self.H_time_func(float(t), self.coords)
        except TypeError:
            out = self.H_time_func(float(t))

        if out is None:
            raise RuntimeError("H_time_func returned None")

        arr = np.asarray(out, dtype=np.float64)

        if arr.ndim == 1 and arr.size == 3:
            self.Ht_view[:, 0] = arr[0]
            self.Ht_view[:, 1] = arr[1]
            self.Ht_view[:, 2] = arr[2]
            return self.Ht

        if arr.ndim == 1 and arr.size == self.local_size:
            self.Ht[:] = arr
            return self.Ht

        if arr.ndim == 2 and arr.shape[1] == 3:
            if arr.shape[0] < self.local_dofs:
                raise ValueError(
                    f"H_time_func returned shape {arr.shape}; expected at least "
                    f"({self.local_dofs}, 3) for the owned vector field."
                )
            self.Ht[:] = arr[: self.local_dofs, :].reshape(-1)
            return self.Ht

        if arr.size == self.local_size:
            self.Ht[:] = arr.reshape(-1)
            return self.Ht

        raise ValueError(
            f"H_time_func returned shape {arr.shape}. Expected (3,), "
            f"({self.local_size},), or ({self.local_dofs}, 3)."
        )

    def _add_time_field_inplace(self, H_flat, t):
        """Add H_time(t) to H_flat using the same evaluator as the Jacobian."""
        if self.H_time_func is None:
            return
        H_flat[: self.local_size] += self._eval_H_time(t)[: self.local_size]

    def _external_field_flat(self):
        """Return H0 + H_time(current_time) in self.H_ext_work."""
        self.H_ext_work[:] = self.H0_ext.x.array[: self.local_size]
        if self.H_time_func is not None:
            self.H_ext_work[:] += self._eval_H_time(self.current_time)[: self.local_size]
        return self.H_ext_work

    def _has_external_field(self):
        return self.H_time_func is not None or np.any(self.H0_ext.x.array[: self.local_size] != 0.0)

    def _energy_from_field_lumped_cpu(self, m_flat, H_flat, factor):
        m = m_flat[: self.local_size].reshape((-1, 3))
        H = H_flat[: self.local_size].reshape((-1, 3))

        local = np.sum(self.vol_nodes * np.einsum("ij,ij->i", m, H))

        global_val = self.comm.allreduce(float(local), op=MPI.SUM)

        return factor * self.mu0 * self.Ms * global_val * self.volume_scale_energy

    def compute_Energy_terms_minimize_cpu(self, m):
        """
        Fast lumped energy for minimization.

        Assumes compute_H_eff(m) is called here, so all field buffers correspond
        to the current m.
        """

        self.compute_H_eff(m)

        m_flat = m.x.array

        E_exch = 0.0
        E_ani = 0.0
        E_dmi_bulk = 0.0
        E_dmi_int = 0.0
        E_demag = 0.0
        E_cubic = 0.0
        E_ext = 0.0

        if self.exchange_field is not None:
            H = self.exchange_field.H_exch.x.array
            E_exch = self._energy_from_field_lumped_cpu(m_flat, H, factor=-0.5)

        if abs(self.Ku) > 0.0:
            H = self.anisotropy_field.H_anis.x.array
            E_ani = self._energy_from_field_lumped_cpu(m_flat, H, factor=-0.5)

        if abs(self.D_bulk) > 0.0:
            H = self.DMIBULK.H_DMI.x.array
            E_dmi_bulk = self._energy_from_field_lumped_cpu(m_flat, H, factor=-0.5)

        if self.DMI_int is not None and abs(self.D_int) > 0.0:
            H = self.DMI_int.H_DMI.x.array
            E_dmi_int = self._energy_from_field_lumped_cpu(m_flat, H, factor=-0.5)

        if self.demag_field is not None:
            H = self.demag_field.H_d.x.array
            E_demag = self._energy_from_field_lumped_cpu(m_flat, H, factor=-0.5)

        if self.cubic_field is not None:
            # Cubic Energy(...) is treated as a local contribution; reduce it
            # here so all minimization energies are global in MPI.
            E_cubic = self._field_energy_global(self.cubic_field, m, prefer_lumped=True)

        if self._has_external_field():
            H = self._external_field_flat()
            E_ext = self._energy_from_field_lumped_cpu(m_flat, H, factor=-1.0)

        E_total = (
            E_exch
            + E_ani
            + E_dmi_bulk
            + E_dmi_int
            + E_demag
            + E_cubic
            + E_ext
        )

        return {
            "E_demag": float(E_demag),
            "E_exch": float(E_exch),
            "E_ani": float(E_ani),
            "E_dmi_bulk": float(E_dmi_bulk),
            "E_dmi_int": float(E_dmi_int),
            "E_cubic": float(E_cubic),
            "E_ext": float(E_ext),
            "E_total": float(E_total),
        }

    def compute_Energy_minimize_cpu(self, m):
        return self.compute_Energy_terms_minimize_cpu(m)["E_total"]


    def compute_H_eff(self, m):
        """
        Compute the total effective field H_eff(m).

        Contributions:
        - exchange (always)
        - demagnetizing field (optional)
        - uniaxial anisotropy (optional)
        - bulk DMI (optional)
        - interfacial DMI (optional)
        - external field: static + time-dependent (optional)
        """


        self.He[:] = self.exchange_field.compute(m).x.petsc_vec.array
        
        if self.demag_field is not None:
            self.He += self.demag_field.compute(m).x.petsc_vec.array

        if abs(self.Ku) > 0.0:
            self.He += self.anisotropy_field.compute(m).x.petsc_vec.array

        if abs(self.D_bulk) > 0.0:
            self.He += self.DMIBULK.compute(m).x.petsc_vec.array

        if self.DMI_int is not None and abs(self.D_int) > 0.0:
            self.He += self.DMI_int.compute(m).x.petsc_vec.array

        #if self.H_time_func is not None:
        #    self.He += self._eval_H_time(self.current_time)

        if self.cubic_field is not None:
            self.He += self.cubic_field.compute(m).x.petsc_vec.array

        self.He += self.H0_ext.x.petsc_vec.array

        if self.H_time_func is not None:
            self._add_time_field_inplace(self.He, self.current_time)


            

        # self.H_lin_buf was NOT refreshed by this slow path.
        self._h_lin_valid = False
        self._mark_he_valid_for(m)
        #self.H_eff.x.scatter_forward()

        return self.He

    # ---------- Fast H_eff using K_total ----------
    def compute_H_eff_fast(self, m):
        """
        Fast variant of compute_H_eff(m) for the energy minimizer.

        Replaces matrix-based K.mult calls (exchange + bulk DMI + optional
        matrix anisotropy + interfacial DMI) with a single K_total.mult into
        self.H_lin_buf, then adds demag / cubic / external / time-dependent
        contributions on top.

        """
        # 1. Linear contributions in one shot: H_lin = K_total m
        self.K_total.mult(m.x.petsc_vec, self.H_lin_buf.x.petsc_vec)

        # 2. Start He from the matrix-based linear part.
        self.He[:] = self.H_lin_buf.x.petsc_vec.array

        # 2b. Add matrix-free/nodal uniaxial anisotropy, if used.
        if abs(self.Ku) > 0.0 and not self.anisotropy_has_matrix:
            self.He += self.anisotropy_field.compute(m).x.petsc_vec.array

        # 3. Non-linear / non-K contributions (demag is a solve; cubic is
        #    polynomial in m; static + time-dependent external fields).
        if self.demag_field is not None:
            self.He += self.demag_field.compute(m).x.petsc_vec.array

        if self.cubic_field is not None:
            self.He += self.cubic_field.compute(m).x.petsc_vec.array

        self.He += self.H0_ext.x.petsc_vec.array

        if self.H_time_func is not None:
            self._add_time_field_inplace(self.He, self.current_time)

        self._h_lin_valid = True
        self._mark_he_valid_for(m)
        return self.He

    # ---------- Fast energy methods (lumped, use *_fast buffers) ----------
    def compute_Energy_terms_minimize_fast_cpu(self, m):
        """
        Fast lumped energy for the minimizer.

        """
        self.compute_H_eff_fast(m)

        m_flat = m.x.array

        # E_lin = -1/2 mu0 Ms <m, K_total m>_V
        H_lin_arr = self.H_lin_buf.x.petsc_vec.array
        E_lin = self._energy_from_field_lumped_cpu(m_flat, H_lin_arr, factor=-0.5)

        E_demag = 0.0
        if self.demag_field is not None:
            H = self.demag_field.H_d.x.array
            E_demag = self._energy_from_field_lumped_cpu(m_flat, H, factor=-0.5)

        E_ani = 0.0
        if abs(self.Ku) > 0.0 and not self.anisotropy_has_matrix:
            # The nodal anisotropy is not part of K_total, so add its
            # coherent lumped energy separately.  This fast minimization path
            # returns global energies, unlike diagnostic Energy(...) methods
            # that return local contributions and are summed by the caller.
            if hasattr(self.anisotropy_field, "Energy_global"):
                E_ani = float(self.anisotropy_field.Energy_global(m))
            else:
                E_ani_local = float(self.anisotropy_field.Energy(m))
                E_ani = float(self.comm.allreduce(E_ani_local, op=MPI.SUM))

        E_cubic = 0.0
        if self.cubic_field is not None:
            E_cubic = self._field_energy_global(self.cubic_field, m, prefer_lumped=True)

        E_ext = 0.0
        if self._has_external_field():
            H = self._external_field_flat()
            E_ext = self._energy_from_field_lumped_cpu(m_flat, H, factor=-1.0)

        E_total = E_lin + E_ani + E_demag + E_cubic + E_ext

        return {
            "E_lin": float(E_lin),
            "E_ani": float(E_ani),
            "E_demag": float(E_demag),
            "E_cubic": float(E_cubic),
            "E_ext": float(E_ext),
            "E_total": float(E_total),
        }

    def compute_Energy_minimize_fast_cpu(self, m):
        return self.compute_Energy_terms_minimize_fast_cpu(m)["E_total"]

    def compute_Energy_from_current_fields_fast_cpu(self, m):
        """
        Reuse H_lin_buf / demag.H_d / cubic buffers already populated by the
        most recent compute_H_eff_fast() call.  Avoids one K_total.mult and
        one demag solve when the minimizer just evaluated H_eff at this m.
        """
        if not getattr(self, "_h_lin_valid", False):
            return self.compute_Energy_minimize_fast_cpu(m)

        m_flat = m.x.array

        H_lin_arr = self.H_lin_buf.x.petsc_vec.array
        E_lin = self._energy_from_field_lumped_cpu(m_flat, H_lin_arr, factor=-0.5)

        E_demag = 0.0
        if self.demag_field is not None:
            H = self.demag_field.H_d.x.array
            E_demag = self._energy_from_field_lumped_cpu(m_flat, H, factor=-0.5)

        E_ani = 0.0
        if abs(self.Ku) > 0.0 and not self.anisotropy_has_matrix:
            if hasattr(self.anisotropy_field, "Energy_global"):
                E_ani = float(self.anisotropy_field.Energy_global(m))
            else:
                E_ani_local = float(self.anisotropy_field.Energy(m))
                E_ani = float(self.comm.allreduce(E_ani_local, op=MPI.SUM))

        E_cubic = 0.0
        if self.cubic_field is not None:
            E_cubic = self._field_energy_global(self.cubic_field, m, prefer_lumped=True)

        E_ext = 0.0
        if self._has_external_field():
            H = self._external_field_flat()
            E_ext = self._energy_from_field_lumped_cpu(m_flat, H, factor=-1.0)

        return float(E_lin + E_ani + E_demag + E_cubic + E_ext)

    # ---------- Compute total energy ----------
    def compute_Energy(self, m):
        """Return global diagnostic energy in MPI.

        The legacy field Energy(...) methods are treated as local
        contributions unless the field exposes Energy_global(...).
        """
        E_exch = self._field_energy_global(self.exchange_field, m)

        E_demag = 0.0
        if self.demag_field is not None:
            E_demag = self._field_energy_global(self.demag_field, m)

        E_ani = 0.0
        if abs(self.Ku) > 0.0:
            E_ani = self._field_energy_global(self.anisotropy_field, m)

        E_dmi_bulk = 0.0
        if abs(self.D_bulk) > 0.0:
            E_dmi_bulk = self._field_energy_global(self.DMIBULK, m)

        E_dmi_int = 0.0
        if self.DMI_int is not None and abs(self.D_int) > 0.0:
            E_dmi_int = self._field_energy_global(self.DMI_int, m)

        E_cub = 0.0
        if self.cubic_field is not None:
            E_cub = self._field_energy_global(self.cubic_field, m)

        return float(E_exch + E_demag + E_ani + E_dmi_bulk + E_dmi_int + E_cub)

    # ---------- Jacobian times vector ----------
    def update_jac_state(self):
        """
        m_vec: local NumPy view of m.x.array (length = 3*local_dofs)
        """
        self.K_total.mult(self.m.x.petsc_vec, self.H_m.x.petsc_vec)

        M_loc = self.m.x.petsc_vec.getArray(readonly=True).reshape(-1, 3)
        Hm_loc = self.H_m.x.petsc_vec.getArray(readonly=True).reshape(-1, 3)

        if abs(self.Ku) > 0.0 and not self.anisotropy_has_matrix:
            Hani = self.anisotropy_field.compute(self.m).x.petsc_vec.getArray(readonly=True).reshape(-1, 3)
            Hm_loc = Hm_loc + Hani

        if self.cubic_field is not None:
            self.m.x.scatter_forward()
            Hc = self.cubic_field.compute(self.m).x.petsc_vec.getArray(readonly=True).reshape(-1, 3)
            Hm_loc = Hm_loc + Hc

        self.M_cached[:, :] = M_loc
        Hext = self.H0_ext.x.petsc_vec.getArray(readonly=True).reshape(-1, 3)


        if self.H_time_func is not None:
            Ht = self._eval_H_time(self.current_time).reshape(-1, 3)
            self.Hm_cached[:, :] = Hm_loc + Hext + Ht
        else:
            self.Hm_cached[:, :] = Hm_loc + Hext

    def jac_vec_times(self, m_unused, v, out):
        self.JacSteps += 1

        self.v_jac.x.array[:self.local_size] = v
        self.K_total.mult(self.v_jac.x.petsc_vec, self.H_v.x.petsc_vec)

        M = self.M_cached
        Hm = self.Hm_cached
        V = v.reshape(-1, 3)

        Hv_lin = self.H_v.x.petsc_vec.getArray(readonly=True).reshape(-1, 3)

        if (abs(self.Ku) > 0.0 and not self.anisotropy_has_matrix) or (self.cubic_field is not None):
            self.Hv_total[:, :] = Hv_lin

            if abs(self.Ku) > 0.0 and not self.anisotropy_has_matrix:
                # The Jacobian of nodal anisotropy is the same linear map
                # H_ani(v) = prefactor*(v dot n)*n.
                Hani_v = self.anisotropy_field.compute(self.v_jac).x.petsc_vec.getArray(readonly=True).reshape(-1, 3)
                self.Hv_total[:, :] += Hani_v

            if self.cubic_field is not None:
                self.cubic_field.jac_times_vec_owned(M, V, self.Hv_cubic)
                self.Hv_total[:, :] += self.Hv_cubic

            Hv = self.Hv_total
        else:
            Hv = Hv_lin

        jac_vec_local_kernel(
            M,
            Hm,
            V,
            Hv,
            out,
            self.gamma,
            self.alpha,
            self.do_precess,
            self.Stab,
        )


    def llg_rhs(self, m):
        self.LLGSteps += 1

        m_numeric = m.x.petsc_vec.getArray(readonly=True)           
        self.Hfield[:] = self.compute_H_eff(m)                     

        self.mx[:] = m_numeric[0::3]
        self.my[:] = m_numeric[1::3]
        self.mz[:] = m_numeric[2::3]

        self.norm[:] = self.mx[:] * self.mx[:]+ self.my[:] * self.my[:]+ self.mz[:] * self.mz[:]

        self.Hx[:] = self.Hfield[0::3]
        self.Hy[:] = self.Hfield[1::3]
        self.Hz[:] = self.Hfield[2::3]

        self.mcx[:] = self.my * self.Hz - self.mz * self.Hy
        self.mcy[:] = self.mz * self.Hx - self.mx * self.Hz
        self.mcz[:] = self.mx * self.Hy - self.my * self.Hx

        self.mcmx[:] = self.my * self.mcz - self.mz * self.mcy
        self.mcmy[:] = self.mz * self.mcx - self.mx * self.mcz
        self.mcmz[:] = self.mx * self.mcy - self.my * self.mcx

        self.dmdt.x.petsc_vec.array[0::3] = self.prefactorEQ * (self.do_precess * self.mcx + self.alpha * self.mcmx) + self.Stab*(1.0 - self.norm) * self.mx[:]
        self.dmdt.x.petsc_vec.array[1::3] = self.prefactorEQ* (self.do_precess * self.mcy + self.alpha * self.mcmy) + self.Stab*(1.0 - self.norm) * self.my[:]
        self.dmdt.x.petsc_vec.array[2::3] = self.prefactorEQ * (self.do_precess * self.mcz + self.alpha * self.mcmz) + self.Stab*(1.0 - self.norm) * self.mz[:] 

        return self.dmdt



    # ---------- IFunction  ----------
    def ifunction(self, ts, t, y, ydot, f):


        #self.LLGSteps += 1

        self.current_time = t

        y.copy(self.m.x.petsc_vec)   #copy y in m local
        self.m.x.scatter_forward()   

        dmdt = self.llg_rhs(self.m)
        #dmdt.x.scatter_forward()

        f.waxpy(-1.0, dmdt.x.petsc_vec, ydot)
        return 0



    def set_uniform_field(self, Hx, Hy, Hz):

        self.H0_ext.x.array[0::3] = Hx
        self.H0_ext.x.array[1::3] = Hy
        self.H0_ext.x.array[2::3] = Hz

        self.H0_ext.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,     mode=PETSc.ScatterMode.FORWARD)
        self._invalidate_he_cache()







class StopByMaxDmdtFD:
    """
    Estimates max |dm/dt| (deg/ns) using a finite difference between accepted time steps.
    Does not evaluate H_eff
    Does not depend on ydot
    """
    def __init__(self, comm, stopping_dm_dt_deg_ns, vec_template,
                check_every=10, print_every_hit=True):
        self.comm = comm
        self.thresh_deg_ns = float(stopping_dm_dt_deg_ns)
        self.check_every = int(check_every)
        self.print_every_hit = bool(print_every_hit)


        self.u_prev = vec_template.duplicate()
        self.du = vec_template.duplicate()


        r0, r1 = vec_template.getOwnershipRange()
        self.n_owned = int(r1 - r0)

        self.n_owned3 = (self.n_owned // 3) * 3

        self.t_prev = None
        self.last_max_dmdt_deg_ns = float("nan")


        vec_template.copy(self.u_prev)

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
        u.copy(self.du)
        self.du.axpy(-1.0, self.u_prev)

        arr = self.du.getArray(readonly=True)
        a = np.asarray(arr[:self.n_owned3]).reshape((-1, 3))
        if a.size:
            d = np.sqrt((a * a).sum(axis=1))
            max_local = float(d.max()) / dt          # 1/s
        else:
            max_local = 0.0

        max_global = self.comm.allreduce(max_local, op=MPI.MAX)
        max_deg_ns = max_global * (180.0 / np.pi) * 1e-9
        self.last_max_dmdt_deg_ns = max_deg_ns

        if self.comm.rank == 0 and self.print_every_hit:
            print(f"[dmdt] step={step} t={t*1e9:.6f} ns  max|dm/dt|={max_deg_ns:.6e} deg/ns", flush=True)

        if max_deg_ns < self.thresh_deg_ns:
            #if self.comm.rank == 0:
                #print(f"[STOP] step={step} t={t*1e9:.6f} ns  max|dm/dt|={max_deg_ns:.6e} < {self.thresh_deg_ns:.6e} deg/ns",flush=True)
            ts.setConvergedReason(PETSc.TS.ConvergedReason.CONVERGED_USER)


        u.copy(self.u_prev)
        self.t_prev = t
        return 0
        
    def reset(self, vec_current):
        vec_current.copy(self.u_prev)
        self.t_prev = None
        self.last_max_dmdt_deg_ns = float("nan")


def skew_from_vec(a):

    ax = a[:, 0]; ay = a[:, 1]; az = a[:, 2]
    S = np.zeros((a.shape[0], 3, 3), dtype=a.dtype)
    S[:, 0, 1] = -az
    S[:, 0, 2] =  ay
    S[:, 1, 0] =  az
    S[:, 1, 2] = -ax
    S[:, 2, 0] = -ay
    S[:, 2, 1] =  ax
    return S


class JvContext:
    def __init__(self, hef_):
        self.hef = hef_
        self.shift = 0.0
        self.calls = 0
        self.callsPre = 0

        diag = self.hef.K_total.getDiagonal()
        self.diagK = diag.getArray(readonly=True).copy()

        # Add diagonal contribution from matrix-free/nodal anisotropy to the
        # preconditioner.  This mirrors what Mat.getDiagonal() would return
        # for the equivalent block-diagonal nodal operator.
        if (
            hasattr(self.hef, "anisotropy_field")
            and abs(getattr(self.hef, "Ku", 0.0)) > 0.0
            and not getattr(self.hef, "anisotropy_has_matrix", True)
            and hasattr(self.hef.anisotropy_field, "diagonal_array")
        ):
            self.diagK += self.hef.anisotropy_field.diagonal_array(owned_only=True)

        self.gamma = float(self.hef.gamma)
        self.do_precess = float(self.hef.do_precess)

        self.enable_pc = True


        self.w = None
        self.denom = None
        self.s_eff = None

        diagK = self.diagK.copy()
        d = diagK.reshape(-1, 3)
        self.kappa_abs = np.mean(np.abs(d), axis=1).astype(np.float64)  
        self.kappa_sgn = np.mean(d, axis=1).astype(np.float64)         


        g  = float(self.hef.gamma)
        a  = float(self.hef.alpha)
        dp = float(self.hef.do_precess)

        self.c1 = (g / (1.0 + a*a)) * dp
        self.c2 = (g * a / (1.0 + a*a))
        self.Stab = float(self.hef.Stab)

        self.enable_pc = True

        n = self.hef.M_cached.shape[0]
        self.A00 = np.empty(n); self.A01 = np.empty(n); self.A02 = np.empty(n)
        self.A10 = np.empty(n); self.A11 = np.empty(n); self.A12 = np.empty(n)
        self.A20 = np.empty(n); self.A21 = np.empty(n); self.A22 = np.empty(n)

        self.i00 = np.empty(n); self.i01 = np.empty(n); self.i02 = np.empty(n)
        self.i10 = np.empty(n); self.i11 = np.empty(n); self.i12 = np.empty(n)
        self.i20 = np.empty(n); self.i21 = np.empty(n); self.i22 = np.empty(n)

        self._pc_ready = False

        self.base_kappa_abs = self.kappa_abs.copy()
        self.base_kappa_sgn = self.kappa_sgn.copy()
        self.kappa_work = np.empty_like(self.base_kappa_abs)

        self.include_cubic_pc = True
        self._kappa_cub = np.empty_like(self.base_kappa_abs)

    def _compute_kappa_cubic(self, M, out_kappa):

        cub = getattr(self.hef, "cubic_field", None)
        if cub is None:
            out_kappa[:] = 0.0
            return

        u1 = cub.u1A[:M.shape[0]]
        u2 = cub.u2A[:M.shape[0]]
        u3 = cub.u3A[:M.shape[0]]
        pref = float(cub.pref)

        a1 = np.einsum("ij,ij->i", M, u1)
        a2 = np.einsum("ij,ij->i", M, u2)
        a3 = np.einsum("ij,ij->i", M, u3)

        a1_2 = a1*a1; a2_2 = a2*a2; a3_2 = a3*a3

        # gradient of s1,s2,s3:
        g1 = u1*(a2_2 + a3_2)[:,None] + (2.0*a1*a2)[:,None]*u2 + (2.0*a1*a3)[:,None]*u3
        g2 = u2*(a3_2 + a1_2)[:,None] + (2.0*a2*a3)[:,None]*u3 + (2.0*a2*a1)[:,None]*u1
        g3 = u3*(a1_2 + a2_2)[:,None] + (2.0*a3*a1)[:,None]*u1 + (2.0*a3*a2)[:,None]*u2

        # diagonal of JH = pref*(u1 o g1 + u2 o g2 + u3 o g3)
        # diag(j) = pref*(u1*g1 + u2*g2 + u3*g3) 
        diag = pref*(u1*g1 + u2*g2 + u3*g3)     # (n,3)

        out_kappa[:] = (np.abs(diag[:,0]) + np.abs(diag[:,1]) + np.abs(diag[:,2])) / 3.0


    def mult(self, A, x, y):
        self.calls += 1
        xv = x.getArray(readonly=True)
        yv = y.getArray()

        #m_vec = self.hef.m.x.petsc_vec.getArray(readonly=True)
        self.hef.jac_vec_times(None, xv, out=self.hef.Jv_buffer)

        yv[:] = self.shift * xv - self.hef.Jv_buffer



    def update_pc_full_fast(self, shift, include_stab=True, use_abs_kappa=True,
                            eps_reg=1e-14, det_eps=1e-30):
        self.shift = float(shift)

        M = self.hef.M_cached   
        H = self.hef.Hm_cached  
        mx, my, mz = M[:,0], M[:,1], M[:,2]
        hx, hy, hz = H[:,0], H[:,1], H[:,2]

        kappa_base = self.base_kappa_abs if use_abs_kappa else self.base_kappa_sgn
        kappa = self.kappa_work
        kappa[:] = kappa_base

        cub = getattr(self.hef, "cubic_field", None)
        if self.include_cubic_pc and (cub is not None):
            self._compute_kappa_cubic(M, self._kappa_cub)
            kappa += self._kappa_cub


        c1 = self.c1
        c2 = self.c2
        Stab = self.Stab

        mdH = mx*hx + my*hy + mz*hz
        mdm = mx*mx + my*my + mz*mz

        Jp00 = 0.0
        Jp01 = -c1*(-hz - kappa*(-mz))   # -c1*(S_H01 - kappa*S_m01)
        Jp02 = -c1*( hy - kappa*( my))
        Jp10 = -c1*( hz - kappa*( mz))
        Jp11 = 0.0
        Jp12 = -c1*(-hx - kappa*(-mx))
        Jp20 = -c1*(-hy - kappa*(-my))
        Jp21 = -c1*( hx - kappa*( mx))
        Jp22 = 0.0


        # Diagonal:
        B00 = mdH + mx*hx - 2*hx*mx  # = mdH - hx*mx
        B11 = mdH - hy*my
        B22 = mdH - hz*mz
        C00 = mx*mx - mdm
        C11 = my*my - mdm
        C22 = mz*mz - mdm
        #
        # Off-diagonal:
        B01 = mx*hy - 2*hx*my
        B02 = mx*hz - 2*hx*mz
        B10 = my*hx - 2*hy*mx
        B12 = my*hz - 2*hy*mz
        B20 = mz*hx - 2*hz*mx
        B21 = mz*hy - 2*hz*my

        C01 = mx*my; C02 = mx*mz
        C10 = my*mx; C12 = my*mz
        C20 = mz*mx; C21 = mz*my

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
            s0 = (1.0 - mdm)
            # diag: s0 - 2 m_i^2 ; offdiag: -2 m_i m_j
            Js00 = Stab*(s0 - 2*mx*mx)
            Js11 = Stab*(s0 - 2*my*my)
            Js22 = Stab*(s0 - 2*mz*mz)
            Js01 = Stab*(-2*mx*my); Js02 = Stab*(-2*mx*mz)
            Js10 = Stab*(-2*my*mx); Js12 = Stab*(-2*my*mz)
            Js20 = Stab*(-2*mz*mx); Js21 = Stab*(-2*mz*my)
        else:
            Js00=Js11=Js22=0.0
            Js01=Js02=Js10=Js12=Js20=Js21=0.0

        # J = Jp + Jd + Js
        J00 = (0.0)     + Jd00 + Js00
        J11 = (0.0)     + Jd11 + Js11
        J22 = (0.0)     + Jd22 + Js22

        J01 = Jp01 + Jd01 + Js01
        J02 = Jp02 + Jd02 + Js02
        J10 = Jp10 + Jd10 + Js10
        J12 = Jp12 + Jd12 + Js12
        J20 = Jp20 + Jd20 + Js20
        J21 = Jp21 + Jd21 + Js21

        # A = shift*I - J + eps_reg*I
        s = self.shift + eps_reg
        A00 = self.A00; A01 = self.A01; A02 = self.A02
        A10 = self.A10; A11 = self.A11; A12 = self.A12
        A20 = self.A20; A21 = self.A21; A22 = self.A22

        A00[:] = s - J00;  A01[:] =   - J01;  A02[:] =   - J02
        A10[:] =   - J10;  A11[:] = s - J11;  A12[:] =   - J12
        A20[:] =   - J20;  A21[:] =   - J21;  A22[:] = s - J22


        # det = a00*(a11*a22-a12*a21) - a01*(a10*a22-a12*a20) + a02*(a10*a21-a11*a20)
        m00 = A11*A22 - A12*A21
        m01 = A10*A22 - A12*A20
        m02 = A10*A21 - A11*A20
        det = A00*m00 - A01*m01 + A02*m02


        det_abs = np.abs(det)
        det = np.where(det_abs < det_eps, det + np.sign(det + det_eps)*det_eps, det)
        invdet = 1.0/det

        i00=i00_ = self.i00; i01=self.i01; i02=self.i02
        i10=self.i10; i11=self.i11; i12=self.i12
        i20=self.i20; i21=self.i21; i22=self.i22

        i00[:] =  (A11*A22 - A12*A21) * invdet
        i01[:] = -(A01*A22 - A02*A21) * invdet
        i02[:] =  (A01*A12 - A02*A11) * invdet

        i10[:] = -(A10*A22 - A12*A20) * invdet
        i11[:] =  (A00*A22 - A02*A20) * invdet
        i12[:] = -(A00*A12 - A02*A10) * invdet

        i20[:] =  (A10*A21 - A11*A20) * invdet
        i21[:] = -(A00*A21 - A01*A20) * invdet
        i22[:] =  (A00*A11 - A01*A10) * invdet

        self._pc_ready = True

    def apply(self, pc, x, y):
        self.callsPre += 1
        if (not self.enable_pc) or (not self._pc_ready):
            x.copy(y)
            return

        xv = x.getArray(readonly=True)
        yv = y.getArray()

        x0 = xv[0::3]; x1 = xv[1::3]; x2 = xv[2::3]

        y0 = self.i00*x0 + self.i01*x1 + self.i02*x2
        y1 = self.i10*x0 + self.i11*x1 + self.i12*x2
        y2 = self.i20*x0 + self.i21*x1 + self.i22*x2

        yv[0::3] = y0
        yv[1::3] = y1
        yv[2::3] = y2


# ---------------------------------------------------------
#  LLG main class
# ---------------------------------------------------------
class LLG:
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
        """
        Zeeman:
          - H0_vec: static field (3*N, array ).
          - H_time_func: H(t, coords) -> (N, 3) or (3N,).
        """
        self._H0_vec = H0_vec
        self._H_time_func = H_time_func


    def add_cubic_anisotropy(self, Kc1, u1_vec, u2_vec):
        """
        u1_vec, u2_vec: arrays of length 3*N, ordered according to the DOFs of V (same convention as n_ani_vec).
        u3 is constructed internally as the cross product u1 x u2 in the cubic anisotropy class.
        """
        self._Kc1 = float(Kc1)
        self._u1_cub = u1_vec
        self._u2_cub = u2_vec
        self._has_cubic = True

    # build effective field

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

        H0_static = self._H0_vec
        H_time_func = self._H_time_func


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
            H0_static=H0_static,
            H_time_func=H_time_func,
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

        y.copy(hef.m.x.petsc_vec)
        hef.m.x.scatter_forward()


        stats = {
            "t_end": float(ts.getTime()),
            "dt_last": float(ts.getTimeStep()),
            "nsteps": int(ts.getStepNumber()),
            "reason": int(ts.getConvergedReason()),
            "maxdmdt_deg_ns": float(self.stopper.last_max_dmdt_deg_ns) if self.stopper is not None else float("nan"),
        }
        return elapsed, stats




    def _reset_run(self, t_final, dt_init=None):
        ts = self.ts
        if dt_init is not None:
            ts.setTimeStep(dt_init)
        ts.setTime(0.0)
        ts.setMaxTime(t_final)
        try:
            ts.setStepNumber(0)
        except Exception:
            pass
        self.stopper.reset(self.y)

    def _ensure_solver(self, m0_array, dt_init,
                    ts_rtol=1e-6, ts_atol=1e-6,
                    snes_rtol=1e-3, snes_atol=1e-5,
                    ksp_rtol=1e-5,
                    stopping_dmdt=0.0,
                    check_every_stop=10, stop_print=False, 
                    set_pc = True,
                    ):

        if self.hef is None:
            self._build_effective_field()
        hef = self.hef

        if not self._solver_ready:
            if m0_array is not None:
                hef.m.x.array[:] = m0_array
                hef.m.x.scatter_forward()

            ts = PETSc.TS().create(self.mesh.comm)


            opts = PETSc.Options()
            opts["ts_type"] = "bdf"
            opts["ts_bdf_order"] = 2                      
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
            opts["ksp_reuse_preconditioner"] = "true"
            opts["snes_lag_preconditioner"] = 1
            opts["ts_max_snes_failures"] = -1
            opts["ts_max_steps"] = 5000000
            #opts["ksp_converged_reason"] = ""
            #opts['ksp_converged_reason'] = ""
            #opts['snes_converged_reason'] = ""
            #opts["log_view"] = "" 

   

            ts.setTime(0.0)
            ts.setTimeStep(dt_init)
            ts.setExactFinalTime(PETSc.TS.ExactFinalTime.STEPOVER)

            snes = ts.getSNES()
            n_loc = hef.m.x.petsc_vec.getLocalSize()
            n_glob = hef.m.x.petsc_vec.getSize()

            J = PETSc.Mat().create(comm=self.mesh.comm)
            ctx = JvContext(hef)
            J.setSizes([[n_loc, n_glob], [n_loc, n_glob]])
            J.setType("python")
            J.setPythonContext(ctx)
            J.setUp()



            ksp = snes.getKSP()
            pc = ksp.getPC()

            if set_pc:
                pc.setType("python")
                pc.setPythonContext(ctx)

                def IJac(ts_, t, y, ydot, shift, A, B):
                    
                    hef.current_time = t
                    y.copy(hef.m.x.petsc_vec)      #copy y in m local
                    hef.update_jac_state()

                    ctx.update_pc_full_fast(shift, include_stab=True, use_abs_kappa=True)
                    return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

            else:
                pc.setType(PETSc.PC.Type.NONE)

                def IJac(ts_, t, y, ydot, shift, A, B):
                    hef.current_time = float(t)

                    y.copy(hef.m.x.petsc_vec)
                    hef.m.x.scatter_forward()

                    hef.update_jac_state()

                    ctx.shift = float(shift)

                    return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

            ts.setIFunction(hef.ifunction)
            ts.setIJacobian(IJac, J)
            ts.setFromOptions()

            y = hef.m.x.petsc_vec.copy()
            y.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,mode=PETSc.ScatterMode.FORWARD)
            ts.setSolution(y)

            stopper = StopByMaxDmdtFD(
                self.mesh.comm,
                stopping_dm_dt_deg_ns=stopping_dmdt,
                vec_template=y,
                check_every=check_every_stop,
                print_every_hit=stop_print,  
            )
            ts.setPostStep(stopper)
            self.stopper = stopper

            self.ts, self.ctx, self.J, self.y, self.stopper = ts, ctx, J, y, stopper
            self._solver_ready = True

        else:

            if m0_array is not None:
                hef.m.x.array[:] = m0_array
                hef.m.x.scatter_forward()
                self.y.copy(hef.m.x.petsc_vec)





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
        pc_python = True,
    ):
        """
        Single temporary integrator. Reuses the persistent TS (_ensure_solver).

        Returns: (y, ctx, elapsed) or (y, ctx, elapsed, stats) if return_stats=True.
        """

        
        self._ensure_solver(
            m0_array=m0_array,
            dt_init=dt_init,
            ts_rtol=ts_rtol, ts_atol=ts_atol,
            snes_rtol=snes_rtol, snes_atol=snes_atol,
            ksp_rtol=ksp_rtol,
            stopping_dmdt=stopping_dmdt,
            check_every_stop=check_every_stop,
            stop_print=stop_print,
            set_pc = pc_python,
        )

        ts = self.ts
        hef = self.hef
        y = self.y
        comm = self.mesh.comm

        if m0_array is not None:
            hef.m.x.array[:] = m0_array
            y.copy(hef.m.x.petsc_vec)
            hef.m.x.scatter_forward()

        hef.compute_H_eff(hef.m)
            

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

                Exch = hef_.exchange_field.Energy(hef_.m)
                Demag = hef_.demag_field.Energy(hef_.m) if hef_.demag_field is not None else 0.0
                Ani = hef_.anisotropy_field.Energy(hef_.m) if getattr(hef_, "Ku", 0.0) != 0.0 else 0.0
                DMI_bulk = hef_.DMIBULK.Energy(hef_.m) if getattr(hef_, "D_bulk", 0.0) != 0.0 else 0.0
                DMI_int = hef_.DMI_int.Energy(hef_.m) if (getattr(hef_, "D_int", 0.0) != 0.0 and hef_.DMI_int is not None) else 0.0
                E_cub = hef_.cubic_field.Energy(hef_.m) if hef_.cubic_field is not None else 0.0


                Exch_total = mesh_.comm.gather(Exch, root=0)
                Demag_total = mesh_.comm.gather(Demag, root=0)
                Ani_total = mesh_.comm.gather(Ani, root=0)
                DMI_bulk_total = mesh_.comm.gather(DMI_bulk, root=0)
                DMI_int_total = mesh_.comm.gather(DMI_int, root=0)
                E_cub_total = mesh_.comm.gather(E_cub, root=0)


                mag = mesh_.comm.gather(hef_.m.x.petsc_vec.getArray(readonly=True), root=0)




                torque_norm = np.sqrt(hef_.mcx**2 + hef_.mcy**2 + hef_.mcz**2)
                max_torque_local = float(np.max(torque_norm)) if torque_norm.size else 0.0
                max_torque_all = mesh_.comm.gather(max_torque_local, root=0)

                Hext_local = np.zeros_like(hef_.H_eff.x.array)
                Hext_local += hef_.H0_static  
                Hext_all = mesh_.comm.gather(Hext_local, root=0)

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




                    E_exch = float(np.sum(Exch_total))
                    E_demag = float(np.sum(Demag_total))
                    E_ani = float(np.sum(Ani_total))
                    E_db = float(np.sum(DMI_bulk_total))
                    E_di = float(np.sum(DMI_int_total))
                    E_cub = float(np.sum(E_cub_total))

                    E_tot = E_exch + E_demag + E_ani + E_db + E_di+ E_cub

                    Hext = np.reshape(np.concatenate(Hext_all), (-1, 3))
                    Hx_ext_mean = Hext[:, 0].mean()
                    Hy_ext_mean = Hext[:, 1].mean()
                    Hz_ext_mean = Hext[:, 2].mean()

                    if hef_.H_time_func is not None:
                        Ht = np.asarray(hef_.H_time_func(float(t)), dtype=float)
                        if Ht.shape != (3,):
                            raise ValueError("For uniform field, H_time_func(t) must be return (3,) vector.")
                        Hx_ext_mean += Ht[0]
                        Hy_ext_mean += Ht[1]
                        Hz_ext_mean += Ht[2]


                    maxtorque = 4 * np.pi * 1e-7 * float(max(max_torque_all)) if max_torque_all else 0.0
                    maxdmdt_deg_ns = float(self.stopper.last_max_dmdt_deg_ns) if self.stopper is not None else 0.0
                    if not np.isfinite(maxdmdt_deg_ns):
                        maxdmdt_deg_ns = 0.0

                    if not first_print["done"]:
                        header = (
                            f"{'time':>10} {'dt':>10} {'<mx>':>15} {'<my>':>15} {'<mz>':>15} "
                            f"{'Hx_ext':>15} {'Hy_ext':>15} {'Hz_ext':>15} "
                            f"{'maxdmdt(deg/ns)':>18} {'max(mxh)':>15} "
                            f"{'E_demag':>15} {'E_exch':>15} {'E_ani':>15} "
                            f"{'E_dmi_bulk':>15} {'E_dmi_int':>15} {'E_cubic':>15}  {'E_total':>15}"
                        )
                        print(header)
                        with open(log_path, "w") as f:
                            f.write(header + "\n")
                        first_print["done"] = True

                    line = (
                        f"{t*1e9:10.4f} {dt_ts*1e9:10.4f}"
                        f"{mag[:,0].mean():15.6f} {mag[:,1].mean():15.6f} {mag[:,2].mean():15.6f} "
                        f"{Hx_ext_mean:15.6e} {Hy_ext_mean:15.6e} {Hz_ext_mean:15.6e} "
                        f"{maxdmdt_deg_ns:18.6e} {maxtorque:15.4e} "
                        f"{E_demag:15.4e} {E_exch:15.4e} {E_ani:15.4e} "
                        f"{E_db:15.4e} {E_di:15.4e} {E_cub:15.4e} {E_tot:15.4e}"
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

        filename = Path(output_dir) / "Relax.xdmf"
        with io.XDMFFile(self.mesh.comm, str(filename), "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            xdmf.write_function(self.hef.m)

        if save_final_state:
            fname = Path(output_dir) / "Relax.bp"
            ad.write_mesh(fname, self.mesh)
            ad.write_function(fname, self.hef.m, time=0.0, name="m")

        if return_stats:
            return y, self.ctx, elapsed, stats
        return y, self.ctx, elapsed








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
        pc_python = True,
    ):
        H_steps = np.asarray(list(H_steps), dtype=float).reshape((-1, 3))
        comm = self.mesh.comm


        self._ensure_solver(
            m0_array=m0_array,
            dt_init=dt_init,
            ts_rtol=ts_rtol, ts_atol=ts_atol,
            snes_rtol=snes_rtol, snes_atol=snes_atol,
            ksp_rtol=ksp_rtol,
            stopping_dmdt=stopping_dmdt,
            check_every_stop=check_every_stop,
            stop_print=stop_print,
            set_pc = pc_python,
        )

        ts = self.ts
        hef = self.hef
        y = self.y

    
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

        def global_mean_m():

            mloc = hef.m.x.petsc_vec.getArray(readonly=True).reshape((-1, 3))
            s_loc = mloc.sum(axis=0) if mloc.size else np.zeros(3)
            n_loc = mloc.shape[0]
            s_glob = np.array(comm.allreduce(s_loc, op=MPI.SUM), dtype=float)
            n_glob = comm.allreduce(n_loc, op=MPI.SUM)
            return s_glob / float(n_glob)

        results = []

        for i, (Hx, Hy, Hz) in enumerate(H_steps):

            hef.set_uniform_field(Hx, Hy, Hz)
            self._reset_ts_run(t0=0.0, t_final=t_final_per_step, dt_init=dt_init)
            elapsed, stats = self._run_ts()


            if xdmf is not None:
                xdmf.write_function(hef.m, float(i))


            if write_xdmf_per_step:
                fname = Path(output_dir) / f"m_{i:05d}.xdmf"
                with io.XDMFFile(comm, str(fname), "w") as xf:
                    xf.write_mesh(self.mesh)
                    xf.write_function(hef.m)


            if write_bp_series:
                ad.write_function(bp_path, hef.m, time=float(i), name="m")



            mmean = global_mean_m()
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

            if comm.rank == 0:

                print(
                    f"[HYST] i={i:05d}  H(mT)=({Hx*4*np.pi*1e-4:+.6e},{Hy*4*np.pi*1e-4:+.6e},{Hz*4*np.pi*1e-4:+.6e})  "
                    f"<m>=({mmean[0]:+.6e},{mmean[1]:+.6e},{mmean[2]:+.6e})  "
                    f"max|dm/dt|={maxdmdt:.6e} deg/ns  "
                    f"nsteps={stats['nsteps']}  t_end={stats['t_end']*1e9:.6f} ns",
                    flush=True
                )

        if xdmf is not None:
            xdmf.close()

        return results
