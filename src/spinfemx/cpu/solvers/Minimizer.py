from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numba import njit
from mpi4py import MPI
from petsc4py import PETSc

from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Optional
import sys
from contextlib import contextmanager

try:
    from .llg_module import EffectiveField
except ImportError:  
    from llg_module import EffectiveField



_EPS = 1e-300


@njit(fastmath=True)
def _normalize_flat_owned_kernel(a_flat):
    n = a_flat.size // 3
    for i in range(n):
        j = 3 * i
        mx = a_flat[j]
        my = a_flat[j + 1]
        mz = a_flat[j + 2]
        nrm = (mx * mx + my * my + mz * mz) ** 0.5
        if nrm < 1e-30:
            nrm = 1e-30
        inv = 1.0 / nrm
        a_flat[j] = mx * inv
        a_flat[j + 1] = my * inv
        a_flat[j + 2] = mz * inv


@njit(fastmath=True)
def _max_norm3_flat_kernel(a_flat):
    n = a_flat.size // 3
    maxv = 0.0
    for i in range(n):
        j = 3 * i
        x = a_flat[j]
        y = a_flat[j + 1]
        z = a_flat[j + 2]
        val = (x * x + y * y + z * z) ** 0.5
        if val > maxv:
            maxv = val
    return maxv


@njit(fastmath=True)
def _projected_direction_with_g_kernel(m_flat, h_flat, d_flat, g_flat):
    n = m_flat.size // 3
    for i in range(n):
        j = 3 * i
        mx = m_flat[j]
        my = m_flat[j + 1]
        mz = m_flat[j + 2]

        hx = h_flat[j]
        hy = h_flat[j + 1]
        hz = h_flat[j + 2]

        mdh = mx * hx + my * hy + mz * hz

        dx = hx - mdh * mx
        dy = hy - mdh * my
        dz = hz - mdh * mz

        d_flat[j] = dx
        d_flat[j + 1] = dy
        d_flat[j + 2] = dz

        g_flat[j] = -dx
        g_flat[j + 1] = -dy
        g_flat[j + 2] = -dz


@njit(fastmath=True)
def _curvilinear_step_kernel(m_flat, d_flat, alpha, y_flat):
    n = m_flat.size // 3
    half_alpha = 0.5 * alpha

    for i in range(n):
        j = 3 * i

        mx = m_flat[j]
        my = m_flat[j + 1]
        mz = m_flat[j + 2]

        dx = d_flat[j]
        dy = d_flat[j + 1]
        dz = d_flat[j + 2]

        # p = m x d
        px = my * dz - mz * dy
        py = mz * dx - mx * dz
        pz = mx * dy - my * dx

        cx = half_alpha * px
        cy = half_alpha * py
        cz = half_alpha * pz

        # b = m + c x m
        bx = mx + (cy * mz - cz * my)
        by = my + (cz * mx - cx * mz)
        bz = mz + (cx * my - cy * mx)

        cdotb = cx * bx + cy * by + cz * bz
        c2 = cx * cx + cy * cy + cz * cz

        # c x b
        cbx = cy * bz - cz * by
        cby = cz * bx - cx * bz
        cbz = cx * by - cy * bx

        inv = 1.0 / (1.0 + c2)

        y_flat[j] = (bx + cbx + cx * cdotb) * inv
        y_flat[j + 1] = (by + cby + cy * cdotb) * inv
        y_flat[j + 2] = (bz + cbz + cz * cdotb) * inv


def _dup_vec(template: PETSc.Vec, block_size: int | None = 3) -> PETSc.Vec:
    out = template.duplicate()
    if block_size is not None:
        try:
            out.setBlockSize(block_size)
        except Exception:
            pass
    out.zeroEntries()
    return out


def _arr(vec: PETSc.Vec, readonly: bool = False) -> np.ndarray:
    return vec.getArray(readonly=readonly)


class _TeeStdout:
    """
    Rank-0 helper: duplicate stdout into a log file while preserving console
    output. This captures the existing print-based monitor without changing the
    minimizer algorithms into file-aware classes.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def _rank0_stdout_tee(comm, log_path, enabled=True):
    """
    Redirect stdout to console+file on rank 0 only.
    """
    if (not enabled) or comm.rank != 0 or log_path is None:
        yield
        return

    old_stdout = sys.stdout
    with open(log_path, "a", buffering=1) as f:
        sys.stdout = _TeeStdout(old_stdout, f)
        try:
            yield
        finally:
            sys.stdout = old_stdout


@dataclass
class _EnergyPolicy:
    mode: str = "auto"
    rtol: float = 1e-10
    atol: float = 1e-30

    def reduce(self, comm, value: float) -> float:
        value = float(value)
        mode = str(self.mode).lower()

        if comm.size == 1:
            return value

        if mode == "sum":
            return float(comm.allreduce(value, op=MPI.SUM))

        if mode == "as_returned":
            return value

        if mode == "rank0":
            out = value if comm.rank == 0 else None
            return float(comm.bcast(out, root=0))

        if mode == "auto":
            vals = comm.allgather(value)
            vals = np.asarray(vals, dtype=np.float64)
            vmax = float(np.nanmax(vals))
            vmin = float(np.nanmin(vals))
            scale = max(abs(vmax), abs(vmin), 1.0)

            # If all ranks returned the same scalar, treat it as already global.
            if abs(vmax - vmin) <= self.atol + self.rtol * scale:
                return float(vals[0])

            # Otherwise interpret values as local energy contributions.
            return float(np.nansum(vals))

        raise ValueError(
            "energy_reduction must be one of: 'auto', 'sum', 'as_returned', 'rank0'."
        )


class _CPUFieldAdapterMPI:
    """Vector-style adapter over the CPU EffectiveField used by llg_module.py.

    The important part is energy_at():

    1. Prefer a fast/vectorial minimization energy if the EffectiveField
       provides it.  This is the recommended route for line-search/minimization.

    2. Fall back to the exact/legacy compute_Energy(m) only if no fast method is
       available.

    """

    def __init__(
        self,
        effective_field,
        energy_reduction: str = "auto",
        energy_backend: str = "fast",
        verbose_energy_backend: bool = False,
    ):
        self.hef = effective_field
        self.comm = effective_field.comm
        self.local_size = int(effective_field.local_size)
        self.energy_policy = _EnergyPolicy(energy_reduction)

        self.energy_backend = str(energy_backend).lower()
        self.verbose_energy_backend = bool(verbose_energy_backend)
        self._reported_energy_backend = False
        self.last_energy_backend = None

        if self.energy_backend not in ("fast", "exact", "auto"):
            raise ValueError("energy_backend must be 'fast', 'exact', or 'auto'.")

    def copy_vec_to_m_function(self, m_vec: PETSc.Vec):
        """Copy distributed PETSc vector into hef.m and update ghosts."""
        m_vec.copy(self.hef.m.x.petsc_vec)
        self.hef.m.x.scatter_forward()

    def compute_H_eff_vec(self, m_vec: PETSc.Vec, out_vec: PETSc.Vec):
        """Compute H_eff(m_vec) into out_vec on the CPU layout."""
        self.copy_vec_to_m_function(m_vec)
        # Prefer the fast K_total-based path when EffectiveField provides it.
        # It does one sparse matvec for the linear terms instead of one per
        # exchange / uniaxial anisotropy / DMI contribution.
        fast_fn = getattr(self.hef, "compute_H_eff_fast", None)
        if callable(fast_fn):
            H = np.asarray(fast_fn(self.hef.m), dtype=np.float64)
        else:
            H = np.asarray(self.hef.compute_H_eff(self.hef.m), dtype=np.float64)

        out = _arr(out_vec)
        if H.size < self.local_size:
            raise ValueError(
                f"compute_H_eff returned size {H.size}, expected at least {self.local_size}."
            )
        out[: self.local_size] = H[: self.local_size]
        if out.size > self.local_size:
            out[self.local_size :] = 0.0
        return out_vec

    @staticmethod
    def _extract_total_energy(value):
        """
        Accept either a scalar or a dictionary-like return value.
        """
        if isinstance(value, dict):
            for key in ("E_total", "energy", "total", "E"):
                if key in value:
                    return float(value[key])
            raise KeyError(
                "Energy dictionary does not contain one of: "
                "'E_total', 'energy', 'total', 'E'."
            )
        return float(value)

    def _energy_fast_raw(self):
        """
        Try the fast/vectorial energy routes exposed by EffectiveField.

        """
        h = self.hef
        m = self.hef.m

        # Preferred simple scalar methods.
        # *_fast_cpu variants use K_total in a single matvec; prefer them.
        scalar_methods = (
            "compute_Energy_minimize_fast_cpu",
            "compute_Energy_minimize_cpu",
        )

        for name in scalar_methods:
            fn = getattr(h, name, None)
            if callable(fn):
                return self._extract_total_energy(fn(m)), name

        # Dict-returning term methods.
        term_methods = (
            "compute_Energy_terms_minimize_fast_cpu",
            "compute_Energy_terms_minimize_cpu",
        )

        for name in term_methods:
            fn = getattr(h, name, None)
            if callable(fn):
                return self._extract_total_energy(fn(m)), name

        return None, None

    def _energy_exact_raw(self):
        """
        Legacy/exact energy route.
        """
        return float(self.hef.compute_Energy(self.hef.m)), "compute_Energy"

    def energy_at(self, m_vec: PETSc.Vec) -> float:

        self.copy_vec_to_m_function(m_vec)

        backend = None
        local_or_global = None

        if self.energy_backend in ("fast", "auto"):
            local_or_global, backend = self._energy_fast_raw()

        if local_or_global is None:
            if self.energy_backend == "fast":
                # Fall back, but make the backend explicit.
                local_or_global, backend = self._energy_exact_raw()
            elif self.energy_backend in ("exact", "auto"):
                local_or_global, backend = self._energy_exact_raw()

        self.last_energy_backend = backend

        if (
            self.verbose_energy_backend
            and not self._reported_energy_backend
            and self.comm.rank == 0
        ):
            print(f"[EnergyMinimizer] energy backend: {backend}", flush=True)
            self._reported_energy_backend = True

        return self.energy_policy.reduce(self.comm, float(local_or_global))

    def sync_solution_to_hef(self, m_vec: PETSc.Vec):
        self.copy_vec_to_m_function(m_vec)
        return self.hef.m


class _DistributedVectorOps:

    def _owned3(self, vec: PETSc.Vec, readonly: bool = False) -> np.ndarray:
        return _arr(vec, readonly=readonly)[: self.local_size].reshape((-1, 3))

    def _zero_ghost_part(self, vec: PETSc.Vec):
        a = _arr(vec)
        if a.size > self.local_size:
            a[self.local_size :] = 0.0

    def _normalize(self, vec: PETSc.Vec):
        a = _arr(vec)[: self.local_size]
        _normalize_flat_owned_kernel(a)
        self._zero_ghost_part(vec)

    def _dot(self, a_vec: PETSc.Vec, b_vec: PETSc.Vec) -> float:
        a = _arr(a_vec, readonly=True)[: self.local_size]
        b = _arr(b_vec, readonly=True)[: self.local_size]
        local = float(np.dot(a, b))
        return float(self.comm.allreduce(local, op=MPI.SUM))

    def _max_norm3(self, vec: PETSc.Vec) -> float:
        a = _arr(vec, readonly=True)[: self.local_size]
        local = float(_max_norm3_flat_kernel(a)) if a.size else 0.0
        return float(self.comm.allreduce(local, op=MPI.MAX))

    def _mean_m(self, m_vec: PETSc.Vec) -> np.ndarray:
        m = self._owned3(m_vec, readonly=True)
        local_sum = m.sum(axis=0) if m.shape[0] else np.zeros(3, dtype=np.float64)
        local_n = np.array(float(m.shape[0]), dtype=np.float64)

        global_sum = np.zeros(3, dtype=np.float64)
        global_n = np.array(0.0, dtype=np.float64)
        self.comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
        self.comm.Allreduce(local_n, global_n, op=MPI.SUM)

        if float(global_n) <= 0.0:
            return np.zeros(3, dtype=np.float64)
        return global_sum / float(global_n)



class LaBonteBBMinimizerCPUMPI(_DistributedVectorOps):
    """
    MPI-safe CPU version of the LaBonte/Exl steepest descent minimizer.

    Algorithm:
      d = H_eff - (m cdot H_eff)m
      g = -d
      BB step from s = m_n - m_{n-1}, y = g_n - g_{n-1}
      Cayley-like norm-preserving update on |m| = 1
      nonmonotone energy acceptance with backtracking
    """

    def __init__(
        self,
        effective_field,
        energy_reduction: str = "auto",
        energy_backend: str = "fast",
        verbose_energy_backend: bool = False,
    ):
        self.hef = effective_field
        self.ad = _CPUFieldAdapterMPI(
            effective_field,
            energy_reduction=energy_reduction,
            energy_backend=energy_backend,
            verbose_energy_backend=verbose_energy_backend,
        )
        self.comm = effective_field.comm
        self.local_size = self.ad.local_size
        self.m_vec = _dup_vec(self.hef.m.x.petsc_vec, block_size=3)
        self.hef.m.x.petsc_vec.copy(self.m_vec)

        self.m_old = _dup_vec(self.m_vec, block_size=3)
        self.m_trial = _dup_vec(self.m_vec, block_size=3)
        self.H = _dup_vec(self.m_vec, block_size=3)
        self.d = _dup_vec(self.m_vec, block_size=3)
        self.g = _dup_vec(self.m_vec, block_size=3)
        self.g_old = _dup_vec(self.m_vec, block_size=3)
        self.s_vec = _dup_vec(self.m_vec, block_size=3)
        self.y_vec = _dup_vec(self.m_vec, block_size=3)

        self.last_energy = None
        self.last_max_grad = None
        self.last_iterations = 0

    def _compute_projected_direction(self, m_vec: PETSc.Vec):
        self.ad.compute_H_eff_vec(m_vec, self.H)

        m = _arr(m_vec, readonly=True)[: self.local_size]
        h = _arr(self.H, readonly=True)[: self.local_size]
        d = _arr(self.d)[: self.local_size]
        g = _arr(self.g)[: self.local_size]

        _projected_direction_with_g_kernel(m, h, d, g)

        self._zero_ghost_part(self.d)
        self._zero_ghost_part(self.g)

    def _compute_projected_direction_reuse(self, m_vec: PETSc.Vec):
        """
        Reuse self.hef.He only if the EffectiveField explicitly marks it as
        valid for the current PETSc vector.  Otherwise fall back to a fresh
        H_eff evaluation.
        """
        He_arr = getattr(self.hef, "He", None)
        matches = False
        he_matches_vec = getattr(self.hef, "he_matches_vec", None)
        if callable(he_matches_vec):
            matches = bool(he_matches_vec(m_vec))

        if (
            He_arr is None
            or not hasattr(He_arr, "size")
            or He_arr.size < self.local_size
            or not matches
        ):
            self._compute_projected_direction(m_vec)
            return

        H_out = _arr(self.H)
        H_out[: self.local_size] = He_arr[: self.local_size]
        if H_out.size > self.local_size:
            H_out[self.local_size :] = 0.0

        m = _arr(m_vec, readonly=True)[: self.local_size]
        h = _arr(self.H, readonly=True)[: self.local_size]
        d = _arr(self.d)[: self.local_size]
        g = _arr(self.g)[: self.local_size]

        _projected_direction_with_g_kernel(m, h, d, g)

        self._zero_ghost_part(self.d)
        self._zero_ghost_part(self.g)

    def _curvilinear_step(
        self,
        m_vec: PETSc.Vec,
        d_vec: PETSc.Vec,
        alpha: float,
        out_vec: PETSc.Vec,
    ):
        m = _arr(m_vec, readonly=True)[: self.local_size]
        d = _arr(d_vec, readonly=True)[: self.local_size]
        y = _arr(out_vec)[: self.local_size]

        _curvilinear_step_kernel(m, d, float(alpha), y)
        self._zero_ghost_part(out_vec)

    def _bb_step(self, alpha_old, alpha_min, alpha_max, variant="alternate", iteration=0):
        sty = self._dot(self.s_vec, self.y_vec)
        sts = self._dot(self.s_vec, self.s_vec)
        yty = self._dot(self.y_vec, self.y_vec)

        if abs(sty) < _EPS or yty < _EPS or sts < _EPS:
            return alpha_old

        alpha1 = sts / sty
        alpha2 = sty / yty

        if variant == "bb1":
            alpha = alpha1
        elif variant == "bb2":
            alpha = alpha2
        else:
            alpha = alpha1 if iteration % 2 == 0 else alpha2

        if not np.isfinite(alpha):
            return alpha_old
        return min(max(abs(alpha), alpha_min), alpha_max)

    def minimize(
        self,
        max_iter=6000,
        tol=1.0,
        alpha0=1e-11,
        alpha_min=1e-13,
        alpha_max=1e-5,
        bb_variant="alternate",
        nonmonotone_window=10,
        max_backtracking=8,
        shrink=0.5,
        print_every=10,
        energy_accept_rtol=1e-12,
        energy_accept_atol=1e-30,
        energy_stagnation_rtol=1e-11,
        energy_stagnation_atol=1e-30,
        stagnation_window=30,
        max_rejected_at_alpha_min=20,
        alpha_restart=1e-9,
        max_alpha_restarts=3,
        restart_residual_factor=10.0,
    ):
        self._normalize(self.m_vec)
        E = self.ad.energy_at(self.m_vec)
        energy_history = [E]
        E_print_prev = None

        alpha = min(max(float(alpha0), alpha_min), alpha_max)
        rejected_at_alpha_min = 0
        alpha_restarts = 0
        stop_reason = "max_iter"

        self._compute_projected_direction(self.m_vec)

        for it in range(int(max_iter)):
            self.last_iterations = it
            max_g = self._max_norm3(self.d)
            self.last_max_grad = max_g

            do_print = (it == 0 or it % int(print_every) == 0)
            if do_print:
                # _mean_m uses MPI.Allreduce, therefore all ranks must call it.
                m_mean = self._mean_m(self.m_vec)

                if E_print_prev is not None and np.isfinite(E) and np.isfinite(E_print_prev):
                    dE_print = E - E_print_prev
                else:
                    dE_print = float("nan")

                if np.isfinite(E):
                    E_print_prev = float(E)

                if self.comm.rank == 0:
                    print(
                        f"[LaBonte-BB-CPU-MPI] it={it:06d} "
                        f"E={E:.12e} dE={dE_print:+.6e} "
                        f"max|projH|={max_g:.6e} alpha={alpha:.6e} "
                        f"<m>=({m_mean[0]:+.6e}, {m_mean[1]:+.6e}, {m_mean[2]:+.6e})",
                        flush=True,
                    )

            if max_g < tol:
                stop_reason = "projected_field_tol"
                break

            if len(energy_history) > int(stagnation_window):
                E_old = energy_history[-int(stagnation_window)]
                E_new = energy_history[-1]
                dE = abs(E_new - E_old)
                scale = max(abs(E_old), abs(E_new), 1e-300)
                if dE <= energy_stagnation_atol + energy_stagnation_rtol * scale:
                    if max_g <= restart_residual_factor * tol:
                        stop_reason = "energy_stagnation_low_residual"
                        break

                    if alpha_restarts < max_alpha_restarts:
                        alpha = min(max(alpha_restart, alpha_min), alpha_max)
                        alpha_restarts += 1
                        rejected_at_alpha_min = 0
                        energy_history = [E]
                        self._compute_projected_direction(self.m_vec)
                        continue

                    if max_g < 400.0:
                        stop_reason = "energy_stagnation_high_residual"
                        break



            self.m_vec.copy(self.m_old)
            self.g.copy(self.g_old)

            E_ref = max(energy_history[-int(nonmonotone_window) :])
            accepted = False
            alpha_try = alpha
            E_trial = np.nan

            for _ in range(int(max_backtracking) + 1):
                self._curvilinear_step(self.m_old, self.d, alpha_try, self.m_trial)
                self._normalize(self.m_trial)
                E_trial = self.ad.energy_at(self.m_trial)

                scale = max(abs(E_ref), abs(E_trial), 1e-300)
                accept_limit = E_ref + energy_accept_atol + energy_accept_rtol * scale
                if E_trial <= accept_limit:
                    accepted = True
                    break

                alpha_try *= shrink
                if alpha_try <= alpha_min:
                    alpha_try = alpha_min

            accepted_int = 1 if accepted and np.isfinite(E_trial) else 0
            accepted_int = int(self.comm.allreduce(accepted_int, op=MPI.MIN))
            accepted = bool(accepted_int)

            if not accepted:
                rejected_at_alpha_min += 1
                alpha = alpha_min
                if rejected_at_alpha_min >= max_rejected_at_alpha_min:
                    stop_reason = "repeated_rejection_at_alpha_min"
                    break
                continue

            rejected_at_alpha_min = 0
            self.m_trial.copy(self.m_vec)
            E = float(E_trial)
            energy_history.append(E)

            self._compute_projected_direction_reuse(self.m_vec)

            self.m_vec.copy(self.s_vec)
            self.s_vec.axpy(-1.0, self.m_old)

            self.g.copy(self.y_vec)
            self.y_vec.axpy(-1.0, self.g_old)

            alpha = self._bb_step(alpha_try, alpha_min, alpha_max, bb_variant, it)

        self._normalize(self.m_vec)
        self.ad.sync_solution_to_hef(self.m_vec)
        self._compute_projected_direction(self.m_vec)
        self.last_energy = self.ad.energy_at(self.m_vec)
        self.last_max_grad = self._max_norm3(self.d)

        # _mean_m uses MPI.Allreduce, therefore all ranks must call it.
        m_mean = self._mean_m(self.m_vec)
        if self.comm.rank == 0:
            print(
                f"[LaBonte-BB-CPU-MPI] finished: it={self.last_iterations}, "
                f"E={self.last_energy:.12e}, max|projH|={self.last_max_grad:.6e}, "
                f"reason={stop_reason}, "
                f"<m>=({m_mean[0]:+.6e}, {m_mean[1]:+.6e}, {m_mean[2]:+.6e})",
                flush=True,
            )

        return {
            "iterations": int(self.last_iterations),
            "energy": float(self.last_energy),
            "max_projected_field": float(self.last_max_grad),
            "alpha": float(alpha),
            "alpha_restarts": int(alpha_restarts),
            "rejected_at_alpha_min": int(rejected_at_alpha_min),
            "energy_history_len": int(len(energy_history)),
            "stop_reason": stop_reason,
            "converged": stop_reason in [
                "projected_field_tol",
                "energy_stagnation_low_residual",
            ],
        }



# -----------------------------------------------------------------------------
# Public high-level driver
# -----------------------------------------------------------------------------

@dataclass
class MinimizerContext:

    calls: int = 0
    callsPre: int = 0


class EnergyMinimizer:

    def __init__(
        self,
        mesh,
        Ms: float,
        gamma: float = 2.211e5,
        alpha: float = 0.5,
        do_precess: float = 1.0,
    ):
        self.mesh = mesh
        self.comm = mesh.comm

        self.Ms = float(Ms)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.do_precess = float(do_precess)

        # Physical coefficients / fields
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

        # Enabled flags
        self._has_exchange = False
        self._has_demag = False
        self._has_anisotropy = False
        self._has_dmi_bulk = False
        self._has_dmi_int = False
        self._has_cubic = False

        self._demag_method = "lindholm"
        self._demag_kwargs: dict[str, Any] = {}

        self.hef: Optional[EffectiveField] = None
        self.ctx = MinimizerContext()
        self.y: Optional[PETSc.Vec] = None

    # Emagnetic interactions
    def add_exchange(self, Aex: float):
        self._Aex = float(Aex)
        self._has_exchange = True

    def add_demag(self, method: str = "lindholm", **kwargs):
        self._has_demag = True
        self._demag_method = str(method)
        self._demag_kwargs = dict(kwargs)

    def add_anisotropy(self, Ku: float, n_vec):
        self._Ku = float(Ku)
        self._n_ani = n_vec
        self._has_anisotropy = True

    # Alias sometimes used in examples.
    def add_uniaxial_anisotropy(self, Ku: float, n_vec):
        self.add_anisotropy(Ku, n_vec)

    def add_dmi_bulk(self, D_bulk: float):
        self._D_bulk = float(D_bulk)
        self._has_dmi_bulk = True

    def add_dmi_interfacial(self, D_int: float, n0_vec):
        self._D_int = float(D_int)
        self._n0_int = n0_vec
        self._has_dmi_int = True

    def add_external_field(
        self,
        H0_vec=None,
        H_time_func: Optional[Callable[..., Any]] = None,
    ):
        """
        Add a static and/or time-dependent external magnetic field.

        H0_vec can be:
          - None
          - shape (3,), interpreted as uniform field
          - flat array with the same local layout as the vector Function
          - shape (N_local, 3), flattened internally

        H_time_func is passed directly to EffectiveField.
        """
        self._H0_vec = H0_vec
        self._H_time_func = H_time_func

        if self.hef is not None and H0_vec is not None:
            H0_flat = self._as_flat_field_vector(H0_vec, name="H0_vec")
            self.hef.H0_ext.x.array[: H0_flat.size] = H0_flat
            self.hef.H0_ext.x.scatter_forward()
            invalidate = getattr(self.hef, "_invalidate_he_cache", None)
            if callable(invalidate):
                invalidate()

    def set_uniform_field(self, Hx: float, Hy: float, Hz: float):
        """
        Convenience method for a uniform static external field.
        """
        self._H0_vec = np.asarray([Hx, Hy, Hz], dtype=np.float64)

        if self.hef is not None:
            self.hef.set_uniform_field(Hx, Hy, Hz)

    def add_cubic_anisotropy(self, Kc1: float, u1_vec, u2_vec):
        self._Kc1 = float(Kc1)
        self._u1_cub = u1_vec
        self._u2_cub = u2_vec
        self._has_cubic = True


    # Internal helpers

    def _local_vector_size(self) -> int:
        """
        Size of the local DOLFINx Function array for a vector field.

        This follows the convention already used by the CPU llg_module, namely
        arrays compatible with fem.Function(V).x.array.
        """
        return 3 * len(self.mesh.geometry.x)

    def _zeros_field(self) -> np.ndarray:
        return np.zeros(self._local_vector_size(), dtype=np.float64)

    def _as_flat_field_vector(self, value, name: str) -> np.ndarray:
        """
        Normalize user-provided vector fields to a flat local array.

        """
        if value is None:
            return self._zeros_field()

        arr = np.asarray(value, dtype=np.float64)

        n_local_geom = len(self.mesh.geometry.x)
        expected = 3 * n_local_geom

        if arr.ndim == 1 and arr.size == 3:
            out = np.zeros(expected, dtype=np.float64)
            out[0::3] = arr[0]
            out[1::3] = arr[1]
            out[2::3] = arr[2]
            return out

        if arr.ndim == 2 and arr.shape == (n_local_geom, 3):
            return np.ascontiguousarray(arr.reshape(-1), dtype=np.float64)

        if arr.ndim == 1 and arr.size == expected:
            return np.ascontiguousarray(arr, dtype=np.float64)

        raise ValueError(
            f"{name} has incompatible shape {arr.shape}. Expected (3,), "
            f"({n_local_geom}, 3), or ({expected},)."
        )

    def _build_effective_field(self):
        """
        Build the CPU EffectiveField used by the minimizers.
        """
        Aex = self._Aex if self._has_exchange else 0.0

        if self._has_anisotropy and self._n_ani is not None:
            Ku = self._Ku
            n_ani_vec = self._as_flat_field_vector(self._n_ani, name="n_ani_vec")
        else:
            Ku = 0.0
            n_ani_vec = self._zeros_field()

        if self._has_dmi_bulk:
            D_bulk = self._D_bulk
        else:
            D_bulk = 0.0

        if self._has_dmi_int and self._n0_int is not None:
            D_int = self._D_int
            n0_int_vec = self._as_flat_field_vector(self._n0_int, name="n0_int_vec")
        else:
            D_int = 0.0
            n0_int_vec = self._zeros_field()

        if self._has_cubic:
            if self._u1_cub is None or self._u2_cub is None:
                raise ValueError(
                    "Cubic anisotropy was enabled, but u1_vec/u2_vec were not provided."
                )

            Kc1 = self._Kc1
            u1_cub = self._as_flat_field_vector(self._u1_cub, name="u1_cub")
            u2_cub = self._as_flat_field_vector(self._u2_cub, name="u2_cub")
        else:
            Kc1 = 0.0
            u1_cub = None
            u2_cub = None

        H0_static = None
        if self._H0_vec is not None:
            H0_static = self._as_flat_field_vector(self._H0_vec, name="H0_vec")

        self.hef = EffectiveField(
            mesh=self.mesh,
            Ms=self.Ms,
            Aex=Aex,
            Ku=Ku,
            n_ani_vec=n_ani_vec,
            D_bulk=D_bulk,
            D_int=D_int,
            n0_int_vec=n0_int_vec,
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
            H_time_func=self._H_time_func,
        )

    def _load_initial_state(self, m0_array):
        """
        Load initial magnetization into self.hef.m.

        Accepted local layouts:
          - exactly len(hef.m.x.array)
          - exactly hef.local_size, assigned to owned entries only

        """
        if self.hef is None:
            raise RuntimeError("EffectiveField has not been built.")

        arr = np.asarray(m0_array, dtype=np.float64).reshape(-1)
        target = self.hef.m.x.array

        if arr.size == target.size:
            target[:] = arr
        elif arr.size == self.hef.local_size:
            target[: self.hef.local_size] = arr
        else:
            raise ValueError(
                "m0_array has incompatible size for the current MPI layout. "
                f"Got {arr.size}; expected either {target.size} "
                f"(local Function array including ghosts, if present) or "
                f"{self.hef.local_size} (owned vector entries)."
            )

        self._normalize_function_m(self.hef.m)
        self.hef.m.x.scatter_forward()

    @staticmethod
    def _normalize_flat_owned(arr: np.ndarray):
        """
        Normalize a flat vector field arranged as x0,y0,z0,x1,y1,z1,...
        """
        if arr.size == 0:
            return
        _normalize_flat_owned_kernel(arr)

    def _normalize_function_m(self, m_fun):
        """
        Normalize owned entries of a DOLFINx vector Function.
        """
        owned = m_fun.x.array[: self.hef.local_size]
        self._normalize_flat_owned(owned)

    def _write_final_outputs(
        self,
        output_dir: str | Path,
        *,
        write_xdmf: bool = True,
        save_final_state: bool = True,
        xdmf_name: str = "Minimize.xdmf",
        bp_name: str = "Minimize.bp",
        function_name: str = "m",
        time: float = 0.0,
    ):
        """
        Write final minimized magnetization.

        - XDMF is useful for visualization.
        - BP is useful as a restart/checkpoint file.
        - All ranks must call the writing routines.
        """
        if self.hef is None:
            raise RuntimeError("Cannot save before EffectiveField is built.")

        if output_dir is None:
            raise ValueError("output_dir must be provided when writing outputs.")

        out = Path(output_dir)
        if self.comm.rank == 0:
            out.mkdir(parents=True, exist_ok=True)
        self.comm.barrier()

        self.hef.m.x.scatter_forward()

        try:
            self.hef.m.name = function_name
        except Exception:
            pass

        xdmf_path = None
        bp_path = None

        if write_xdmf:
            from dolfinx import io

            xdmf_path = out / xdmf_name
            with io.XDMFFile(self.mesh.comm, str(xdmf_path), "w") as xdmf:
                xdmf.write_mesh(self.mesh)
                try:
                    xdmf.write_function(self.hef.m, float(time))
                except TypeError:
                    xdmf.write_function(self.hef.m)

        if save_final_state:
            import adios4dolfinx as ad

            bp_path = out / bp_name

            bp_written = False
            bp_error = None

            try:
                ad.write_mesh(self.mesh, bp_path)
                bp_written = True
            except Exception as e1:
                bp_error = e1
                try:
                    ad.write_mesh(self.mesh, str(bp_path))
                    bp_written = True
                    bp_error = None
                except Exception as e2:
                    bp_error = e2
                    try:
                        ad.write_mesh(str(bp_path), self.mesh)
                        bp_written = True
                        bp_error = None
                    except Exception as e3:
                        bp_error = e3

            if not bp_written:
                if self.comm.rank == 0:
                    print(
                        "[WARN] Could not write ADIOS/BP mesh checkpoint. "
                        f"XDMF output may still be available. Error: {repr(bp_error)}",
                        flush=True,
                    )
                bp_path = None
            else:
                function_written = False
                function_error = None

                function_write_attempts = (
                    lambda: ad.write_function(self.hef.m, bp_path, time=float(time), name=function_name),
                    lambda: ad.write_function(self.hef.m, str(bp_path), time=float(time), name=function_name),
                    lambda: ad.write_function(bp_path, self.hef.m, time=float(time), name=function_name),
                    lambda: ad.write_function(str(bp_path), self.hef.m, time=float(time), name=function_name),
                    lambda: ad.write_function(self.hef.m, bp_path),
                    lambda: ad.write_function(self.hef.m, str(bp_path)),
                    lambda: ad.write_function(str(bp_path), self.hef.m),
                )

                for attempt in function_write_attempts:
                    try:
                        attempt()
                        function_written = True
                        function_error = None
                        break
                    except Exception as e:
                        function_error = e

                if not function_written:
                    if self.comm.rank == 0:
                        print(
                            "[WARN] Could not write ADIOS/BP function checkpoint. "
                            f"XDMF output may still be available. Error: {repr(function_error)}",
                            flush=True,
                        )
                    bp_path = None

        self.comm.barrier()

        return {
            "xdmf_path": str(xdmf_path) if xdmf_path is not None else None,
            "bp_path": str(bp_path) if bp_path is not None else None,
        }

    def _save_final_state(self, output_dir: str | Path, filename: str = "m_final.bp"):
        """
        Backward-compatible BP-only writer.
        """
        return self._write_final_outputs(
            output_dir,
            write_xdmf=False,
            save_final_state=True,
            bp_name=filename,
        )

    # -------------------------------------------------------------------------
    # Main minimization API
    # -------------------------------------------------------------------------
    def minimize(
        self,
        m0_array,
        method: str = "labonte",
        max_iter: int = 3000,
        tol: float = 1.0,
        output_dir: str | Path | None = "output_minimize",
        save_final_state: bool = True,
        write_xdmf: bool = True,
        xdmf_name: str | None = None,
        bp_name: str | None = None,
        log_name: str | None = None,
        write_log: bool = True,
        energy_reduction: str = "auto",
        energy_backend: str = "fast",
        verbose_energy_backend: bool = False,
        print_every: int = 25,
        return_context: bool = False,
        **kwargs,
    ):
        """
        Minimize the micromagnetic energy with the LaBonte-BB minimizer.

        Parameters
        ----------
        m0_array:
            Initial magnetization in the local DOLFINx layout.  If None, the
            minimizer continues from the current ``self.hef.m`` state.
        method:
            Must be ``"labonte"`` (aliases: ``"labonte_bb"``, ``"bb"``,
            ``"minimize"``).  Other values raise ``ValueError``.
        max_iter:
            Maximum number of minimizer iterations.
        tol:
            Tolerance for max projected effective field.
        output_dir:
            Directory used if save_final_state=True.
        save_final_state:
            If True, writes output_dir/m_final.bp.
        energy_reduction:
            Reduction policy applied to the scalar returned by the energy backend.
            Use:
              - "auto"       default
              - "sum"        if the backend returns local contribution per rank
              - "as_returned" if the backend already returns a global energy
              - "rank0"      only for special legacy cases
        energy_backend:
            "fast" by default.  It first tries the vectorial/lumped minimization
            energy methods from EffectiveField, such as
            compute_Energy_minimize_cpu(m).  If unavailable, it falls back to
            compute_Energy(m).
            Use "exact" to force compute_Energy(m).
        verbose_energy_backend:
            If True, rank 0 prints which EffectiveField method is used for energy.
        print_every:
            Print interval.  Printing is rank-0 only inside the minimizer.
        return_context:
            If True, returns (y, ctx, elapsed, stats).  Otherwise returns
            (y, stats, elapsed).
        kwargs:
            Extra algorithm-specific parameters passed to LaBonteBBMinimizerCPUMPI.

        Returns
        -------
        By default:
            y, stats, elapsed

        If return_context=True:
            y, ctx, elapsed, stats
        """
        if self.hef is None:
            self._build_effective_field()

        if m0_array is not None:
            self._load_initial_state(m0_array)

        method_key = str(method).lower().replace("-", "_")

        # Backward-compatibility: SD has been removed.  Map ambiguous aliases
        # to LaBonte so existing scripts that just asked for "the default
        # minimizer" keep working, but reject explicit SD requests so the user
        # is aware of the change.
        if method_key in ("sd", "projected_sd", "projected_steepest_descent", "steepest_descent"):
            raise ValueError(
                "The projected steepest descent (SD) minimizer has been removed "
                "from this module.  Only LaBonte-BB is supported now.  Pass "
                "method='labonte' instead."
            )

        if method_key not in ("labonte", "labonte_bb", "bb", "minimize"):
            raise ValueError(
                f"Unsupported minimization method {method!r}.  Use method='labonte'."
            )

        method_tag = "LaBonte"
        default_xdmf_name = "Minimize_LaBonte.xdmf"
        default_bp_name = "Minimize_LaBonte.bp"
        default_log_name = "minimize_labonte_log.txt"

        if xdmf_name is None:
            xdmf_name = default_xdmf_name
        if bp_name is None:
            bp_name = default_bp_name
        if log_name is None:
            log_name = default_log_name

        output_path = Path(output_dir) if output_dir is not None else None
        log_path = None

        if output_path is not None:
            if self.comm.rank == 0:
                output_path.mkdir(parents=True, exist_ok=True)
            self.comm.barrier()

            if write_log:
                log_path = output_path / log_name
                if self.comm.rank == 0:
                    with open(log_path, "w") as f:
                        f.write(f"# Energy minimization log: {method_tag}\n")
                        f.write(f"method                  {method_key}\n")
                        f.write(f"max_iter                {int(max_iter)}\n")
                        f.write(f"tol                     {float(tol):.16e}\n")
                        f.write(f"energy_backend          {energy_backend}\n")
                        f.write(f"energy_reduction        {energy_reduction}\n")
                        f.write(f"mpi_size                {int(self.comm.size)}\n")
                        f.write("# monitor output\n")
            self.comm.barrier()

        t0 = perf_counter()

        alg = LaBonteBBMinimizerCPUMPI(
            self.hef,
            energy_reduction=energy_reduction,
            energy_backend=energy_backend,
            verbose_energy_backend=verbose_energy_backend,
        )

        with _rank0_stdout_tee(self.comm, log_path, enabled=write_log):
            stats = alg.minimize(
                max_iter=max_iter,
                tol=tol,
                print_every=print_every,
                **kwargs,
            )

        elapsed = perf_counter() - t0

        # Ensure final function and returned vector are synchronized.
        self.hef.m.x.scatter_forward()
        self.y = self.hef.m.x.petsc_vec.copy()
        self.ctx = MinimizerContext(calls=0, callsPre=0)

        output_info = {"xdmf_path": None, "bp_path": None}
        if output_path is not None and (write_xdmf or save_final_state):
            output_info = self._write_final_outputs(
                output_path,
                write_xdmf=write_xdmf,
                save_final_state=save_final_state,
                xdmf_name=xdmf_name,
                bp_name=bp_name,
                function_name="m",
                time=0.0,
            )

        # Add driver-level metadata without assuming exact keys from the
        # algorithm implementation.
        if isinstance(stats, dict):
            stats.setdefault("method", method_key)
            stats.setdefault("elapsed", float(elapsed))
            stats.setdefault("mpi_size", int(self.comm.size))
            stats.setdefault("energy_backend", str(energy_backend))
            stats.setdefault("energy_reduction", str(energy_reduction))
            stats.setdefault("output_dir", str(output_path) if output_path is not None else None)
            stats.setdefault("log_path", str(log_path) if log_path is not None else None)
            stats.setdefault("xdmf_path", output_info.get("xdmf_path"))
            stats.setdefault("bp_path", output_info.get("bp_path"))

        if write_log and log_path is not None and self.comm.rank == 0:
            with open(log_path, "a") as f:
                f.write("\n# final summary\n")
                f.write(f"elapsed                 {float(elapsed):.16e}\n")
                if isinstance(stats, dict):
                    for key in sorted(stats.keys()):
                        value = stats[key]
                        if isinstance(value, float):
                            f.write(f"{key:<24} {value:.16e}\n")
                        else:
                            f.write(f"{key:<24} {value}\n")

        if return_context:
            return self.y, self.ctx, elapsed, stats

        return self.y, stats, elapsed

    # -------------------------------------------------------------------------
    # Hysteresis loop via energy minimization
    # -------------------------------------------------------------------------
    def hysteresis(
        self,
        m0_array,
        H_steps,
        method: str = "labonte",
        max_iter: int = 3000,
        tol: float = 1.0,
        output_dir: str | Path = "hyst_minimize",
        write_xdmf_per_step: bool = True,
        write_bp_series: bool = True,
        xdmf_name: str = "Hysteresis.xdmf",
        bp_name: str = "Hysteresis.bp",
        log_name: str = "hysteresis_log.txt",
        write_xdmf_series: bool = False,
        energy_reduction: str = "auto",
        energy_backend: str = "fast",
        print_every: int = 25,
        verbose_energy_backend: bool = False,
        **kwargs,
    ):
        """
        Compute a quasistatic hysteresis loop via energy minimization.

        At each external-field step the minimizer (LaBonte-BB or projected SD)
        drives the magnetization to a local energy minimum.  The continuation
        of minima over decreasing/increasing field defines the hysteresis loop,
        following the approach of Exl et al. (J. Appl. Phys. 115, 17D118, 2014).

        Parameters
        ----------
        m0_array:
            Initial magnetization (same layout as ``minimize``).

        H_steps:
            Iterable of (Hx, Hy, Hz) external-field values in A/m.
            Can be any shape broadcastable to ``(N, 3)``.

        method:
            ``"labonte"`` or ``"sd"``.

        max_iter, tol:
            Passed to the minimizer at each field step.

        output_dir:
            Root directory for per-step XDMF/BP snapshots and the log.

        write_xdmf_per_step:
            If True, writes one XDMF file per field step.

        write_bp_series:
            If True, appends each step to a single ADIOS/BP checkpoint
            (useful for restarts / post-processing).

        write_xdmf_series:
            If True, writes all steps into a single time-series XDMF.

        kwargs:
            Extra algorithm-specific parameters forwarded to the minimizer
            (e.g. ``alpha0``, ``alpha_min``, ``alpha_max``, ``theta_max``).

        Returns
        -------
        list[dict]
            One entry per field step with keys ``step``, ``H``, ``m_mean``,
            ``energy``, ``elapsed``, ``stats``, etc.
        """
        H_steps = np.asarray(list(H_steps), dtype=float).reshape((-1, 3))
        comm = self.comm

        # Build EffectiveField once, load initial state.
        if self.hef is None:
            self._build_effective_field()
        self._load_initial_state(m0_array)

        hef = self.hef

        output_path = Path(output_dir)
        if comm.rank == 0:
            output_path.mkdir(parents=True, exist_ok=True)
        comm.barrier()

        # Log file.
        log_path = output_path / log_name
        if comm.rank == 0:
            with open(log_path, "w") as f:
                f.write(
                    "# step Hx Hy Hz  <mx> <my> <mz>  "
                    "E_total  max_projH  iterations  reason  elapsed\n"
                )

        # Optional XDMF time series.
        xdmf = None
        if write_xdmf_series:
            from dolfinx import io as _io

            xdmf_path = output_path / xdmf_name
            xdmf = _io.XDMFFile(comm, str(xdmf_path), "w")
            xdmf.write_mesh(self.mesh)

        # Optional ADIOS/BP series.
        bp_path = output_path / bp_name
        if write_bp_series:
            try:
                import adios4dolfinx as _ad

                _ad.write_mesh(bp_path, self.mesh)
            except Exception:
                write_bp_series = False

        def _global_mean_m():
            mloc = hef.m.x.petsc_vec.getArray(readonly=True).reshape((-1, 3))
            s_loc = mloc.sum(axis=0) if mloc.size else np.zeros(3)
            n_loc = mloc.shape[0]
            s_glob = np.array(comm.allreduce(s_loc, op=MPI.SUM), dtype=float)
            n_glob = comm.allreduce(n_loc, op=MPI.SUM)
            return s_glob / float(max(n_glob, 1))

        results = []

        for i, (Hx, Hy, Hz) in enumerate(H_steps):
            # Set the external field for this step.
            self.set_uniform_field(float(Hx), float(Hy), float(Hz))

            # Use the current hef.m (= output of the previous step, or m0 for
            # i==0) as the starting point.  minimize() reads from hef.m.
            _, stats_i, elapsed_i = self.minimize(
                m0_array=None,               # keep current state
                method=method,
                max_iter=max_iter,
                tol=tol,
                output_dir=None,             # per-step I/O handled here
                save_final_state=False,
                write_xdmf=False,
                write_log=False,
                energy_reduction=energy_reduction,
                energy_backend=energy_backend,
                verbose_energy_backend=verbose_energy_backend,
                print_every=print_every,
                **kwargs,
            )

            hef.m.x.scatter_forward()

            # Per-step outputs.
            if write_xdmf_series and xdmf is not None:
                xdmf.write_function(hef.m, float(i))

            if write_xdmf_per_step:
                from dolfinx import io as _io

                try:
                    self.mesh.name = "Grid"
                except Exception:
                    pass

                try:
                    hef.m.name = "m"
                except Exception:
                    pass

                fname = output_path / f"m_{i:05d}.xdmf"
                with _io.XDMFFile(comm, str(fname), "w") as xf:
                    xf.write_mesh(self.mesh)
                    xf.write_function(hef.m)
                    
            if write_bp_series:
                try:
                    import adios4dolfinx as _ad

                    _ad.write_function(bp_path, hef.m, time=float(i), name="m")
                except Exception:
                    pass

            mmean = _global_mean_m()

            E_total = float(stats_i.get("energy", float("nan")))
            max_projH = float(stats_i.get("max_projected_field",
                              stats_i.get("max_projH", float("nan"))))
            n_iter = int(stats_i.get("iterations", 0))
            reason = str(stats_i.get("stop_reason", ""))

            entry = {
                "step": int(i),
                "H": (float(Hx), float(Hy), float(Hz)),
                "m_mean": (float(mmean[0]), float(mmean[1]), float(mmean[2])),
                "energy": E_total,
                "max_projected_field": max_projH,
                "iterations": n_iter,
                "stop_reason": reason,
                "elapsed": float(elapsed_i),
                "stats": stats_i,
            }
            results.append(entry)

            if comm.rank == 0:
                with open(log_path, "a") as f:
                    f.write(
                        f"{i:d} {Hx:.6e} {Hy:.6e} {Hz:.6e} "
                        f"{mmean[0]:.6e} {mmean[1]:.6e} {mmean[2]:.6e} "
                        f"{E_total:.6e} {max_projH:.6e} {n_iter:d} "
                        f"{reason} {elapsed_i:.3f}\n"
                    )

                mu0 = 4.0 * np.pi * 1e-7
                print(
                    f"[HYST-MIN] i={i:05d}  "
                    f"H(mT)=({Hx*mu0*1e3:+.4e},{Hy*mu0*1e3:+.4e},{Hz*mu0*1e3:+.4e})  "
                    f"<m>=({mmean[0]:+.6e},{mmean[1]:+.6e},{mmean[2]:+.6e})  "
                    f"E={E_total:.6e}  max|projH|={max_projH:.4e}  "
                    f"iters={n_iter}  {reason}  {elapsed_i:.2f}s",
                    flush=True,
                )

        if xdmf is not None:
            xdmf.close()

        return results



# Backward-friendly aliases.
LaBonteBBMinimizerCPU = LaBonteBBMinimizerCPUMPI

__all__ = [
    "EnergyMinimizer",
    "MinimizerContext",
    "LaBonteBBMinimizerCPUMPI",
    "LaBonteBBMinimizerCPU",
]
