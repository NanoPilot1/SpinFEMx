"""
Minimizer_GPU.py

Single-file GPU energy-minimization module for the micromagnetic code.

This file exposes the GPU LaBonte-BB energy minimizer with a
Minimizer.py-like API:

    minimizer = EnergyMinimizerGPU(mesh, Ms)
    minimizer.add_exchange(Aex)
    y, stats, elapsed = minimizer.minimize(m0_array, method="labonte", tol=1.0)

Design
------
`EnergyMinimizerGPU` is the public driver.  It builds the GPU EffectiveField
internally from the same interaction-style API used by the CPU minimizer:
`add_exchange`, `add_demag`, `add_anisotropy`, `add_dmi_bulk`, etc.

`LaBonteBBMinimizerGPU` is the internal algorithm class.  It requires a GPU
EffectiveField exposing `m_gpu`, `H_eff_gpu`, `compute_H_eff_vec`,
`compute_Energy_minimize_gpu`, `local_size`, `local_dofs`, and `comm`.

Important limitation
--------------------
This module preserves the limitation of the original GPU solvers: single-rank GPU
execution only. If MPI size is greater than 1, the solver classes raise an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from contextlib import contextmanager
from typing import Any, Optional
import sys

import numpy as np
import cupy as cp
from petsc4py import PETSc
from dolfinx import io
import adios4dolfinx as ad



_CAYLEY_KERNEL = cp.ElementwiseKernel(
    "float64 mx, float64 my, float64 mz, "
    "float64 dx, float64 dy, float64 dz, "
    "float64 alpha",
    "float64 yx, float64 yy, float64 yz",
    """
    double px = my*dz - mz*dy;
    double py = mz*dx - mx*dz;
    double pz = mx*dy - my*dx;
    double cx = 0.5*alpha*px;
    double cy = 0.5*alpha*py;
    double cz = 0.5*alpha*pz;
    double bx = mx + cy*mz - cz*my;
    double by = my + cz*mx - cx*mz;
    double bz = mz + cx*my - cy*mx;
    double cbx = cy*bz - cz*by;
    double cby = cz*bx - cx*bz;
    double cbz = cx*by - cy*bx;
    double cdb = cx*bx + cy*by + cz*bz;
    double c2 = cx*cx + cy*cy + cz*cz;
    double inv = 1.0 / (1.0 + c2);
    yx = (bx + cbx + cx*cdb) * inv;
    yy = (by + cby + cy*cdb) * inv;
    yz = (bz + cbz + cz*cdb) * inv;
    """,
    name="cayley_step",
)


_PROJDIR_KERNEL = cp.ElementwiseKernel(
    "float64 mx, float64 my, float64 mz, "
    "float64 hx, float64 hy, float64 hz",
    "float64 dx, float64 dy, float64 dz",
    """
    double mdh = mx*hx + my*hy + mz*hz;
    dx = hx - mdh*mx;
    dy = hy - mdh*my;
    dz = hz - mdh*mz;
    """,
    name="projected_direction",
)


_NORM3_MAX_REDUCE = cp.ReductionKernel(
    "float64 sx, float64 sy, float64 sz",
    "float64 out",
    "sqrt(sx*sx + sy*sy + sz*sz)",
    "max(a, b)",
    "out = a",
    "0.0",
    name="norm3_max_reduce",
)


# -----------------------------------------------------------------------------
# Shared PETSc/CuPy helpers
# -----------------------------------------------------------------------------

def _set_vec_cuda(vec: PETSc.Vec, block_size: Optional[int] = None):
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


def _dup_cuda_vec(template: PETSc.Vec, block_size: Optional[int] = None):
    out = template.duplicate()
    _set_vec_cuda(out, block_size=block_size)
    out.zeroEntries()
    return out


def _vec_to_cupy(vec: PETSc.Vec, mode: str = "rw"):
    return cp.from_dlpack(vec.toDLPack(mode))


class _TeeStdout:
    """Duplicate stdout into multiple streams."""

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
def _stdout_tee(log_path, enabled: bool = True):
    """Duplicate stdout to console and a log file.

    The GPU minimizer is intentionally single-rank only, so there is no
    rank-0 branching here.
    """
    if (not enabled) or log_path is None:
        yield
        return

    old_stdout = sys.stdout
    with open(log_path, "a", buffering=1) as f:
        sys.stdout = _TeeStdout(old_stdout, f)
        try:
            yield
        finally:
            sys.stdout = old_stdout


class LaBonteBBMinimizerGPU:
    """
    LaBonte/Exl-type steepest descent minimizer for micromagnetics.

    Single-rank GPU implementation.

    Main ingredients:
    - projected effective field,
    - norm-preserving curvilinear update,
    - Barzilai-Borwein step length,
    - nonmonotone energy acceptance,
    - stagnation and repeated-rejection safeguards.
    """

    def __init__(self, effective_field):
        self.hef = effective_field
        self.comm = effective_field.comm

        if self.comm.size != 1:
            raise RuntimeError(
                "LaBonteBBMinimizerGPU is implemented for single-rank GPU execution only."
            )

        self.local_size = effective_field.local_size
        self.local_dofs = effective_field.local_dofs

        self.m_old = _dup_cuda_vec(self.hef.m_gpu, block_size=3)
        self.m_trial = _dup_cuda_vec(self.hef.m_gpu, block_size=3)

        self.H = _dup_cuda_vec(self.hef.H_eff_gpu, block_size=3)

        # Projected descent direction.
        self.d = _dup_cuda_vec(self.hef.m_gpu, block_size=3)

        # Projected energy gradient, stored as -d.
        self.g = _dup_cuda_vec(self.hef.m_gpu, block_size=3)
        self.g_old = _dup_cuda_vec(self.hef.m_gpu, block_size=3)

        # Barzilai-Borwein work vectors.
        self.s_vec = _dup_cuda_vec(self.hef.m_gpu, block_size=3)
        self.y_vec = _dup_cuda_vec(self.hef.m_gpu, block_size=3)

        self.last_energy = None
        self.last_max_grad = None
        self.last_iterations = 0
        self.reused_h_eff_count = 0
        self.recomputed_h_eff_count = 0

        # State for the adaptive BB steplength selection used in GPSS
        # (Loris/Bertero/De Mol/Zanella/Zanni), which is the more elaborate
        # BB strategy referenced by Exl et al.  It is reset at the beginning
        # of each minimize(...) call.
        self.bb_tau = 0.5
        self.bb_Malpha = 2
        self.bb_tau_min = 1e-12
        self.bb_tau_max = 1e12
        self.bb_alpha2_history = []
        self.bb_adaptive_alpha1_count = 0
        self.bb_adaptive_alpha2_count = 0
        self.bb_adaptive_alpha_max_count = 0
        self.bb_last_rule = "init"
        self.bb_last_alpha1 = None
        self.bb_last_alpha2 = None

    # ------------------------------------------------------------
    # Basic GPU vector helpers
    # ------------------------------------------------------------
    def _normalize(self, vec):
        arr_all = _vec_to_cupy(vec, "rw")
        arr = arr_all[: self.local_size].reshape((-1, 3))

        nrm = cp.sqrt(cp.sum(arr * arr, axis=1))
        nrm = cp.maximum(nrm, 1e-30)

        arr[:, 0] /= nrm
        arr[:, 1] /= nrm
        arr[:, 2] /= nrm

        if arr_all.size > self.local_size:
            arr_all[self.local_size :] = 0.0

    def _dot(self, a_vec, b_vec):
        a = _vec_to_cupy(a_vec, "r")[: self.local_size]
        b = _vec_to_cupy(b_vec, "r")[: self.local_size]

        return float(cp.dot(a, b).item())

    def _max_norm3(self, vec):
        arr_all = _vec_to_cupy(vec, "r")
        arr = arr_all[: self.local_size].reshape((-1, 3))

        if arr.shape[0] == 0:
            return 0.0

        # Single fused reduction kernel: sqrt(|s|^2) max over rows.
        return float(_NORM3_MAX_REDUCE(arr[:, 0], arr[:, 1], arr[:, 2]).item())

    def _mean_m_single_rank(self, m_vec):
        """
        Compute <mx>, <my>, <mz> for single-rank GPU execution.
        """

        m_all = _vec_to_cupy(m_vec, "r")
        m = m_all[: self.local_size].reshape((-1, 3))

        if m.shape[0] == 0:
            return np.zeros(3, dtype=np.float64)

        return cp.mean(m, axis=0).get().astype(np.float64)


    def _energy_at(self, m_vec):
        """
        Energy evaluation used by the minimizer.

        Prefer the GPU/lumped energy if available. Fall back to the exact
        CPU/UFL energy only if the fast minimization energy is not implemented.
        """

        if hasattr(self.hef, "compute_Energy_minimize_gpu"):
            return float(self.hef.compute_Energy_minimize_gpu(m_vec))

        # Fallback: exact but slower CPU/UFL path.  This route does not
        # populate H_eff_gpu, so any minimizer-side reuse cache must be
        # invalidated explicitly.
        m_vec.copy(self.hef.m_gpu)
        if hasattr(self.hef, "_h_eff_valid_for_m_gpu"):
            self.hef._h_eff_valid_for_m_gpu = False
        if hasattr(self.hef, "_h_eff_gpu_filled_by_energy"):
            self.hef._h_eff_gpu_filled_by_energy = False
        return float(self.hef.compute_Energy())

    # ------------------------------------------------------------
    # Projected field / projected gradient
    # ------------------------------------------------------------
    def _compute_projected_direction(self, m_vec):
        """
        Compute the projected effective field:

            d = H_eff - (m · H_eff) m

        This is tangent to the unit sphere.

        Also stores:

            g = -d

        because BB formulas are usually written using the gradient.
        """

        self.hef.compute_H_eff_vec(m_vec, self.H)
        self._compute_projected_direction_from_H(m_vec)

    def _compute_projected_direction_reuse(self, m_vec):
        """
        Reuse `self.hef.H_eff_gpu` as the effective field at `m_vec`, instead
        of paying for another compute_H_eff_vec() call.

        Safe to call iff the caller knows that H_eff_gpu was just refreshed
        for `m_vec` (typically right after a successful line-search trial).
        The EffectiveField sets `_h_eff_valid_for_m_gpu = True` whenever it
        refreshes the buffer for `m_gpu`.

        Falls back transparently to the slow path if the flag is not set or
        if the EffectiveField does not implement it.
        """

        h_eff_gpu = getattr(self.hef, "H_eff_gpu", None)
        valid = bool(getattr(self.hef, "_h_eff_valid_for_m_gpu", False))

        if h_eff_gpu is None or not valid or m_vec is not self.hef.m_gpu:
            self.recomputed_h_eff_count += 1
            self._compute_projected_direction(m_vec)
            return

        # Copy H_eff_gpu into our local self.H buffer (still on device) so
        # the rest of the algorithm sees a self-owned buffer.
        self.reused_h_eff_count += 1
        h_eff_gpu.copy(self.H)
        self._compute_projected_direction_from_H(m_vec)

    def _compute_projected_direction_from_H(self, m_vec):
        """
        Shared back-end: given self.H = H_eff(m_vec), compute
        d = H - (m·H)m and g = -d, all in a single fused kernel.
        """

        m_all = _vec_to_cupy(m_vec, "r")
        h_all = _vec_to_cupy(self.H, "r")
        d_all = _vec_to_cupy(self.d, "rw")
        g_all = _vec_to_cupy(self.g, "rw")

        m = m_all[: self.local_size].reshape((-1, 3))
        h = h_all[: self.local_size].reshape((-1, 3))
        d = d_all[: self.local_size].reshape((-1, 3))
        g = g_all[: self.local_size].reshape((-1, 3))

        _PROJDIR_KERNEL(
            m[:, 0], m[:, 1], m[:, 2],
            h[:, 0], h[:, 1], h[:, 2],
            d[:, 0], d[:, 1], d[:, 2],
        )

        cp.negative(d[:, 0], out=g[:, 0])
        cp.negative(d[:, 1], out=g[:, 1])
        cp.negative(d[:, 2], out=g[:, 2])

        if d_all.size > self.local_size:
            d_all[self.local_size:] = 0.0
            g_all[self.local_size:] = 0.0

    # ------------------------------------------------------------
    # Curvilinear update on the sphere
    # ------------------------------------------------------------
    def _curvilinear_step(self, m_vec, d_vec, alpha, out_vec):
        """
        Norm-preserving Cayley-type update.

        For small alpha:

            m_new = m + alpha*d + O(alpha^2)

        d should be tangent to m.

        """

        m_all = _vec_to_cupy(m_vec, "r")
        d_all = _vec_to_cupy(d_vec, "r")
        out_all = _vec_to_cupy(out_vec, "rw")

        m = m_all[: self.local_size].reshape((-1, 3))
        d = d_all[: self.local_size].reshape((-1, 3))
        y = out_all[: self.local_size].reshape((-1, 3))

        _CAYLEY_KERNEL(
            m[:, 0], m[:, 1], m[:, 2],
            d[:, 0], d[:, 1], d[:, 2],
            float(alpha),
            y[:, 0], y[:, 1], y[:, 2],
        )

        if out_all.size > self.local_size:
            out_all[self.local_size:] = 0.0

    # ------------------------------------------------------------
    # Barzilai-Borwein step
    # ------------------------------------------------------------
    def _bb_step(self, alpha_old, alpha_min, alpha_max, variant="alternate", iteration=0):
        """
        Select the next Barzilai-Borwein steplength.

        Available variants
        ------------------
        "bb1":
            alpha_BB1 = (s^T s) / (s^T y)

        "bb2":
            alpha_BB2 = (s^T y) / (y^T y)

        "alternate":
            Alternate BB1 and BB2 by iteration parity.

        "adaptive":
            Adaptive BB alternation used by the GPSS rule of
            Loris, Bertero, De Mol, Zanella and Zanni, i.e. the
            steplength-selection strategy referenced by Exl et al.

            If s^T y <= 0, use alpha_max. Otherwise compute the clamped
            BB1 and BB2 steps. If alpha_BB2 / alpha_BB1 <= tau_k, choose
            the minimum recent BB2 value over the last M_alpha + 1 values
            and shrink tau_k by 0.9. Otherwise choose BB1 and enlarge
            tau_k by 1.1.

        where:
            s = m_n - m_{n-1}
            y = g_n - g_{n-1}


        """

        s = _vec_to_cupy(self.s_vec, "r")[: self.local_size]
        y = _vec_to_cupy(self.y_vec, "r")[: self.local_size]

        vals = cp.empty(3, dtype=cp.float64)
        vals[0] = cp.dot(s, y)  # sty
        vals[1] = cp.dot(s, s)  # sts
        vals[2] = cp.dot(y, y)  # yty

        sty, sts, yty = vals.get()
        sty = float(sty)
        sts = float(sts)
        yty = float(yty)

        variant_key = str(variant).lower().replace("-", "_")

        if sts < 1e-300 or yty < 1e-300:
            self.bb_last_rule = "reuse_old_degenerate"
            return min(max(float(alpha_old), alpha_min), alpha_max)

        if variant_key in ("adaptive", "gpss", "loris", "exl", "loris_zanni"):
            if sty <= 1e-300:
                self.bb_last_alpha1 = None
                self.bb_last_alpha2 = None
                self.bb_last_rule = "adaptive_alpha_max_nonpositive_sty"
                self.bb_adaptive_alpha_max_count += 1
                return float(alpha_max)

            alpha1 = sts / sty
            alpha2 = sty / yty

            alpha1 = min(max(float(alpha1), alpha_min), alpha_max)
            alpha2 = min(max(float(alpha2), alpha_min), alpha_max)

            self.bb_last_alpha1 = float(alpha1)
            self.bb_last_alpha2 = float(alpha2)

            self.bb_alpha2_history.append(float(alpha2))
            max_hist = max(int(self.bb_Malpha) + 1, 1)
            if len(self.bb_alpha2_history) > max_hist:
                self.bb_alpha2_history = self.bb_alpha2_history[-max_hist:]

            ratio = alpha2 / max(alpha1, 1e-300)
            if ratio <= self.bb_tau:
                alpha = min(self.bb_alpha2_history)
                self.bb_tau *= 0.9
                self.bb_last_rule = "adaptive_recent_min_bb2"
                self.bb_adaptive_alpha2_count += 1
            else:
                alpha = alpha1
                self.bb_tau *= 1.1
                self.bb_last_rule = "adaptive_bb1"
                self.bb_adaptive_alpha1_count += 1

            # The paper does not impose explicit bounds on tau_k, but clamping
            # avoids pathological floating-point growth/underflow in very long
            # hysteresis loops without changing the intended rule in practice.
            self.bb_tau = min(max(float(self.bb_tau), self.bb_tau_min), self.bb_tau_max)
            return min(max(float(alpha), alpha_min), alpha_max)

        if abs(sty) < 1e-300:
            self.bb_last_rule = "reuse_old_small_sty"
            return min(max(float(alpha_old), alpha_min), alpha_max)

        alpha1 = sts / sty
        alpha2 = sty / yty

        self.bb_last_alpha1 = float(alpha1)
        self.bb_last_alpha2 = float(alpha2)

        if variant_key == "bb1":
            alpha = alpha1
            self.bb_last_rule = "bb1"
        elif variant_key == "bb2":
            alpha = alpha2
            self.bb_last_rule = "bb2"

        # Backward-compatible safeguard: use abs and clamp.

        alpha = abs(alpha)
        alpha = min(max(alpha, alpha_min), alpha_max)

        return float(alpha)

    # ------------------------------------------------------------
    # Main minimization loop
    # ------------------------------------------------------------
    def minimize(
        self,
        max_iter=5000,
        tol=1.0,

        alpha0=5e-11,
        alpha_min=1e-13,
        alpha_max=1e-8,

        bb_variant="adaptive",
        bb_adaptive_tau0=0.5,
        bb_adaptive_Malpha=2,
        bb_adaptive_tau_min=1e-12,
        bb_adaptive_tau_max=1e12,

        nonmonotone_window=10,
        max_backtracking=10,
        shrink=0.5,
        print_every=100,

        energy_accept_rtol=1e-10,
        energy_accept_atol=1e-24,
        energy_stagnation_rtol=1e-10,
        energy_stagnation_atol=1e-24,
        stagnation_window=30,
        max_rejected_at_alpha_min=30,

        alpha_restart=5e-11,
        max_alpha_restarts=3,
        restart_residual_factor=50.0,
        stagnation_residual_stop=50,
    ):
        """
        Run LaBonte/Exl steepest descent minimization.

        Stopping criteria:
        - max projected field < tol
        - repeated step rejection at alpha_min
        - energy stagnation over a moving window only when the projected
          residual is already below a prescribed residual threshold
        - otherwise, stagnation triggers alpha restarts before the algorithm
          continues the line search.

        BB variants:
        - "bb1": always use BB1.
        - "bb2": always use BB2.
        - "alternate": alternate BB1/BB2.
        - "adaptive"/"gpss"/"loris"/"exl": adaptive GPSS rule
          used by Loris et al. and referenced by Exl et al.
        """

        # Reset adaptive BB state for this minimization run.  This matters for
        # hysteresis, because each field step calls minimize(...) separately.
        self.bb_tau = min(max(float(bb_adaptive_tau0), float(bb_adaptive_tau_min)), float(bb_adaptive_tau_max))
        self.bb_Malpha = max(int(bb_adaptive_Malpha), 0)
        self.bb_tau_min = float(bb_adaptive_tau_min)
        self.bb_tau_max = float(bb_adaptive_tau_max)
        self.bb_alpha2_history = []
        self.bb_adaptive_alpha1_count = 0
        self.bb_adaptive_alpha2_count = 0
        self.bb_adaptive_alpha_max_count = 0
        self.bb_last_rule = "init"
        self.bb_last_alpha1 = None
        self.bb_last_alpha2 = None

        self._normalize(self.hef.m_gpu)

        E = self._energy_at(self.hef.m_gpu)
        energy_history = [E]

        alpha = float(alpha0)
        alpha = min(max(alpha, alpha_min), alpha_max)

        rejected_at_alpha_min = 0
        alpha_restarts = 0
        stop_reason = "max_iter"

        # Initial projected direction.
        self._compute_projected_direction(self.hef.m_gpu)

        for it in range(max_iter):
            self.last_iterations = it

            max_g = self._max_norm3(self.d)
            self.last_max_grad = max_g

            if it == 0 or it % print_every == 0:
                m_mean = self._mean_m_single_rank(self.hef.m_gpu)

                print(
                    f"[LaBonte-BB] it={it:06d} "
                    f"E={E:.12e} "
                    f"max|projH|={max_g:.6e} "
                    f"alpha={alpha:.6e} "
                    f"<m>=({m_mean[0]:+.6e}, {m_mean[1]:+.6e}, {m_mean[2]:+.6e})",
                    flush=True,
                )

            # 1. Normal convergence criterion.
            if max_g < tol:
                stop_reason = "projected_field_tol"
                break

            # 2. Energy stagnation criterion.
            if len(energy_history) > stagnation_window:
                E_old = energy_history[-stagnation_window]
                E_new = energy_history[-1]
                dE = abs(E_new - E_old)
                scale = max(abs(E_old), abs(E_new), 1e-300)

                if dE <= energy_stagnation_atol + energy_stagnation_rtol * scale:
                    # Match the CPU logic: stagnation is accepted as a
                    # convergence condition only when the projected residual is
                    # already sufficiently small.
                    if max_g <= restart_residual_factor * tol:
                        stop_reason = "energy_stagnation_low_residual"
                        print(
                            f"[LaBonte-BB] Stopping by energy stagnation: "
                            f"dE={dE:.3e}, E={E_new:.12e}, "
                            f"max|projH|={max_g:.6e}",
                            flush=True,
                        )
                        break

                    # If residual is still high, try restarting alpha.
                    if alpha_restarts < max_alpha_restarts:
                        alpha = min(max(alpha_restart, alpha_min), alpha_max)
                        alpha_restarts += 1
                        rejected_at_alpha_min = 0
                        energy_history = [E]
                        self._compute_projected_direction(self.hef.m_gpu)
                        print(
                            f"[LaBonte-BB] Energy stagnated but residual is still high. "
                            f"Restarting alpha: alpha={alpha:.3e}, "
                            f"restart={alpha_restarts}/{max_alpha_restarts}, "
                            f"max|projH|={max_g:.6e}",
                            flush=True,
                        )
                        continue

                    # After restarts are exhausted, do not stop merely because
                    # the energy is flat.  Stop only below a residual threshold,
                    # following the CPU minimizer safeguard.
                    if max_g < stagnation_residual_stop:
                        stop_reason = "energy_stagnation_high_residual"
                        print(
                            f"[LaBonte-BB] Stopping by energy stagnation with residual "
                            f"below threshold: dE={dE:.3e}, E={E_new:.12e}, "
                            f"max|projH|={max_g:.6e}, "
                            f"threshold={stagnation_residual_stop:.6e}",
                            flush=True,
                        )
                        break

                    # Residual is still too high; ignore the stagnation event and
                    # continue with the normal line-search update.

            # Save current state and gradient.
            self.hef.m_gpu.copy(self.m_old)
            self.g.copy(self.g_old)

            E_ref = max(energy_history[-nonmonotone_window:])

            accepted = False
            alpha_try = alpha
            E_trial = np.nan

            for _ in range(max_backtracking + 1):
                self._curvilinear_step(self.m_old, self.d, alpha_try, self.m_trial)

                E_trial = self._energy_at(self.m_trial)

                # Same acceptance rule as the CPU minimizer: use a tolerance
                # scaled by both reference and trial energies.
                scale = max(abs(E_ref), abs(E_trial), 1e-300)
                accept_limit = E_ref + energy_accept_atol + energy_accept_rtol * scale
                if E_trial <= accept_limit:
                    self.m_trial.copy(self.hef.m_gpu)

                    # _energy_at(self.m_trial) has just evaluated E(m_trial).
                    # In the fast GPU backend, that same call also filled
                    # hef.H_eff_gpu with H_eff(m_trial).  Since m_trial is now
                    # copied into hef.m_gpu, the cached H_eff_gpu is valid for
                    # the accepted state and can be reused below.
                    if getattr(self.hef, "_h_eff_gpu_filled_by_energy", False):
                        self.hef._h_eff_valid_for_m_gpu = True
                    else:
                        self.hef._h_eff_valid_for_m_gpu = False

                    E = float(E_trial)
                    energy_history.append(E)
                    accepted = True
                    rejected_at_alpha_min = 0
                    break

                alpha_try *= shrink

                if alpha_try < alpha_min:
                    alpha_try = alpha_min

            if not accepted:
                # Restore the accepted state.  This is essential when the energy
                # fallback path copied the trial vector into hef.m_gpu.
                self.m_old.copy(self.hef.m_gpu)

                if hasattr(self.hef, "_h_eff_valid_for_m_gpu"):
                    self.hef._h_eff_valid_for_m_gpu = False
                if hasattr(self.hef, "_h_eff_gpu_filled_by_energy"):
                    self.hef._h_eff_gpu_filled_by_energy = False

                rejected_at_alpha_min += 1
                alpha = alpha_min

                print(
                    f"[LaBonte-BB] step rejected at it={it}; "
                    f"new alpha={alpha:.3e}; "
                    f"rejected_at_alpha_min={rejected_at_alpha_min}",
                    flush=True,
                )

                if rejected_at_alpha_min >= max_rejected_at_alpha_min:
                    stop_reason = "repeated_rejection_at_alpha_min"
                    print(
                        f"[LaBonte-BB] Stopping: repeated rejection at alpha_min. "
                        f"alpha={alpha:.3e}, "
                        f"max|projH|={max_g:.6e}, "
                        f"E={E:.12e}",
                        flush=True,
                    )
                    break

                self._compute_projected_direction(self.hef.m_gpu)
                continue

            # Compute new projected direction at accepted state.
            # _energy_at(self.m_trial) inside the line search populated
            # self.hef.H_eff_gpu with H_eff(m_trial).  After accepting the
            # trial, m_trial has been copied into self.hef.m_gpu and the cache
            # was explicitly promoted to valid, so this should reuse H_eff_gpu
            # instead of recomputing a full effective field, including demag.
            self._compute_projected_direction_reuse(self.hef.m_gpu)

            # s = m_new - m_old
            self.hef.m_gpu.copy(self.s_vec)
            self.s_vec.axpy(-1.0, self.m_old)

            # y = g_new - g_old
            self.g.copy(self.y_vec)
            self.y_vec.axpy(-1.0, self.g_old)

            alpha = self._bb_step(
                alpha_old=alpha_try,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                variant=bb_variant,
                iteration=it,
            )

            alpha = min(max(alpha, alpha_min), alpha_max)

        # Match the CPU finalization: normalize, recompute projected
        # direction, then recompute final energy and final residual.
        self._normalize(self.hef.m_gpu)
        self._compute_projected_direction(self.hef.m_gpu)
        self.last_energy = self._energy_at(self.hef.m_gpu)
        self.last_max_grad = self._max_norm3(self.d)

        m_mean = self._mean_m_single_rank(self.hef.m_gpu)
        print(
            f"[LaBonte-BB] finished: "
            f"it={self.last_iterations}, "
            f"E={self.last_energy:.12e}, "
            f"max|projH|={self.last_max_grad:.6e}, "
            f"reason={stop_reason}, "
            f"<m>=({m_mean[0]:+.6e}, {m_mean[1]:+.6e}, {m_mean[2]:+.6e})",
            flush=True,
        )

        converged = stop_reason in [
            "projected_field_tol",
            "energy_stagnation_low_residual",
        ]

        return {
            "iterations": int(self.last_iterations),
            "energy": float(self.last_energy),
            "max_projected_field": float(self.last_max_grad),
            "alpha": float(alpha),
            "bb_variant": str(bb_variant),
            "bb_last_rule": str(self.bb_last_rule),
            "bb_last_alpha1": None if self.bb_last_alpha1 is None else float(self.bb_last_alpha1),
            "bb_last_alpha2": None if self.bb_last_alpha2 is None else float(self.bb_last_alpha2),
            "bb_adaptive_tau": float(self.bb_tau),
            "bb_adaptive_Malpha": int(self.bb_Malpha),
            "bb_adaptive_alpha1_count": int(self.bb_adaptive_alpha1_count),
            "bb_adaptive_alpha2_count": int(self.bb_adaptive_alpha2_count),
            "bb_adaptive_alpha_max_count": int(self.bb_adaptive_alpha_max_count),
            "alpha_restarts": int(alpha_restarts),
            "rejected_at_alpha_min": int(rejected_at_alpha_min),
            "energy_history_len": int(len(energy_history)),
            "h_eff_reused_count": int(self.reused_h_eff_count),
            "h_eff_recomputed_after_accept_count": int(self.recomputed_h_eff_count),
            "stop_reason": stop_reason,
            "converged": bool(converged),
        }




@dataclass
class MinimizerGPUContext:
    """
    Compatibility object.

    The GPU minimizers do not use Jacobian-vector products or preconditioners,
    but keeping these counters makes the return signature look similar to the
    CPU minimizer / PETSc-TS workflow.
    """

    calls: int = 0
    callsPre: int = 0


class EnergyMinimizerGPU:
    """
    GPU energy-minimization driver with the same public role as the CPU
    EnergyMinimizer class.

    This class owns the minimization workflow.  LLG_GPU is kept for time
    integration and hysteresis workflows; direct energy relaxation should use
    this class instead.

    Only the LaBonte-BB minimizer is exposed here.  Projected SD has been
    removed from this module to avoid two competing minimization APIs.
    """

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

        if self.comm.size != 1:
            raise RuntimeError(
                "EnergyMinimizerGPU currently supports single-rank GPU execution only."
            )

        self.Ms = float(Ms)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.do_precess = float(do_precess)

        # Physical coefficients / fields.
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

        # Enabled flags.
        self._has_exchange = False
        self._has_demag = False
        self._has_anisotropy = False
        self._has_dmi_bulk = False
        self._has_dmi_int = False
        self._has_cubic = False

        self._demag_method = "fmm"
        self._demag_kwargs: dict[str, Any] = {}

        self.hef = None
        self.ctx = MinimizerGPUContext()
        self.y: Optional[PETSc.Vec] = None
        self.last_algorithm: Optional[Any] = None

    # ------------------------------------------------------------------
    # Micromagnetic interactions.
    # ------------------------------------------------------------------
    def add_exchange(self, Aex: float):
        self._Aex = float(Aex)
        self._has_exchange = True

    def add_demag(self, method: str = "fmm", **kwargs):
        self._has_demag = True
        self._demag_method = str(method)
        self._demag_kwargs = dict(kwargs)

    def add_anisotropy(self, Ku: float, n_vec):
        self._Ku = float(Ku)
        self._n_ani = n_vec
        self._has_anisotropy = True

    def add_uniaxial_anisotropy(self, Ku: float, n_vec):
        self.add_anisotropy(Ku, n_vec)

    def add_dmi_bulk(self, D_bulk: float):
        self._D_bulk = float(D_bulk)
        self._has_dmi_bulk = True

    def add_dmi_interfacial(self, D_int: float, n0_vec):
        self._D_int = float(D_int)
        self._n0_int = n0_vec
        self._has_dmi_int = True

    def add_external_field(self, H0_vec=None, H_time_func=None):
        self._H0_vec = H0_vec
        self._H_time_func = H_time_func

        if self.hef is not None and H0_vec is not None:
            self.hef.set_uniform_field(*np.asarray(H0_vec, dtype=np.float64).reshape(3))

    def set_uniform_field(self, Hx: float, Hy: float, Hz: float):
        self._H0_vec = np.array([Hx, Hy, Hz], dtype=np.float64)
        if self.hef is not None:
            self.hef.set_uniform_field(Hx, Hy, Hz)

    def add_cubic_anisotropy(self, Kc1: float, u1_vec, u2_vec):
        self._Kc1 = float(Kc1)
        self._u1_cub = u1_vec
        self._u2_cub = u2_vec
        self._has_cubic = True

    # ------------------------------------------------------------------
    # Effective-field construction.
    # ------------------------------------------------------------------
    def _build_effective_field(self):
        # Lazy import avoids a circular import: llg_module_GPU may import this
        # module locally inside hysteresis, while this driver needs EffectiveField.
        try:
            from .llg_module_GPU import EffectiveField
        except ImportError:  # useful when running this file as a local script
            from llg_module_GPU import EffectiveField

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

    # ------------------------------------------------------------------
    # Initial/final state helpers.
    # ------------------------------------------------------------------
    def _load_initial_state(self, m0_array):
        if self.hef is None:
            self._build_effective_field()
        if m0_array is not None:
            self.hef.set_m_from_cpu(m0_array)

    def _copy_solution_vec(self):
        y = _dup_cuda_vec(self.hef.m_gpu, block_size=3)
        self.hef.m_gpu.copy(y)
        self.y = y
        return y

    def _prepare_function_for_io(self, function_name: str = "m", mesh_name: str = "Grid"):
        self.hef.sync_m_to_function()

        try:
            self.mesh.name = mesh_name
        except Exception:
            pass

        try:
            self.hef.m.name = function_name
        except Exception:
            pass

    def _write_final_outputs(
        self,
        output_path: Path,
        write_xdmf: bool,
        save_final_state: bool,
        xdmf_name: str,
        bp_name: str,
        function_name: str = "m",
        time: float = 0.0,
    ):
        info = {"xdmf_path": None, "bp_path": None}

        self._prepare_function_for_io(function_name=function_name, mesh_name="Grid")

        if write_xdmf:
            xdmf_path = output_path / xdmf_name
            with io.XDMFFile(self.comm, str(xdmf_path), "w") as xdmf:
                xdmf.write_mesh(self.mesh)
                try:
                    xdmf.write_function(self.hef.m, float(time))
                except TypeError:
                    xdmf.write_function(self.hef.m)
            info["xdmf_path"] = str(xdmf_path)

        if save_final_state:
            bp_path = output_path / bp_name
            ad.write_mesh(bp_path, self.mesh)
            ad.write_function(bp_path, self.hef.m, time=float(time), name=function_name)
            info["bp_path"] = str(bp_path)

        return info

    # ------------------------------------------------------------------
    # Public minimization driver: LaBonte-BB only.
    # ------------------------------------------------------------------
    def minimize(
        self,
        m0_array=None,
        method: str = "labonte",
        max_iter: int = 3000,
        tol: float = 1.0,
        output_dir: str | Path | None = "output_minimize_gpu",
        save_final_state: bool = True,
        write_xdmf: bool = True,
        xdmf_name: str | None = None,
        bp_name: str | None = None,
        log_name: str | None = None,
        write_log: bool = True,
        print_every: int = 25,
        return_context: bool = False,
        **kwargs,
    ):
        """
        Minimize the micromagnetic energy on GPU using LaBonte-BB only.

        The public API mirrors the CPU EnergyMinimizer style: construct the
        minimizer, attach interactions, and call minimize.  No LLG_GPU
        object is required.
        """
        if self.hef is None:
            self._build_effective_field()

        if m0_array is not None:
            self._load_initial_state(m0_array)

        method_key = str(method).lower().replace("-", "_")
        if method_key not in ("labonte", "labonte_bb", "bb", "minimize"):
            raise ValueError(
                f"Unsupported GPU minimization method {method!r}. Use method='labonte'."
            )

        default_xdmf_name = "Minimize_LaBonte_GPU.xdmf"
        default_bp_name = "Minimize_LaBonte_GPU.bp"
        default_log_name = "minimize_labonte_gpu_log.txt"

        if xdmf_name is None:
            xdmf_name = default_xdmf_name
        if bp_name is None:
            bp_name = default_bp_name
        if log_name is None:
            log_name = default_log_name

        output_path = Path(output_dir) if output_dir is not None else None
        log_path = None

        if output_path is not None:
            output_path.mkdir(parents=True, exist_ok=True)

            if write_log:
                log_path = output_path / log_name
                with open(log_path, "w") as f:
                    f.write("# GPU energy minimization log: LaBonte-BB\n")
                    f.write(f"method                  {method_key}\n")
                    f.write(f"max_iter                {int(max_iter)}\n")
                    f.write(f"tol                     {float(tol):.16e}\n")
                    f.write(f"mpi_size                {int(self.comm.size)}\n")
                    f.write("# monitor output\n")

        t0 = perf_counter()

        alg = LaBonteBBMinimizerGPU(self.hef)

        with _stdout_tee(log_path, enabled=write_log):
            stats = alg.minimize(
                max_iter=max_iter,
                tol=tol,
                print_every=print_every,
                **kwargs,
            )

        elapsed = perf_counter() - t0
        self.last_algorithm = alg
        self.ctx = MinimizerGPUContext(calls=0, callsPre=0)

        # Synchronize final GPU magnetization to the host-side DOLFINx Function.
        self.hef.sync_m_to_function()
        y = self._copy_solution_vec()

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

        if isinstance(stats, dict):
            stats.setdefault("method", method_key)
            stats.setdefault("method_tag", "LaBonte-BB-GPU")
            stats.setdefault("elapsed", float(elapsed))
            stats.setdefault("mpi_size", int(self.comm.size))
            stats.setdefault("backend", "gpu")
            stats.setdefault("output_dir", str(output_path) if output_path is not None else None)
            stats.setdefault("log_path", str(log_path) if log_path is not None else None)
            stats.setdefault("xdmf_path", output_info.get("xdmf_path"))
            stats.setdefault("bp_path", output_info.get("bp_path"))

        if write_log and log_path is not None:
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
            return y, self.ctx, elapsed, stats

        return y, stats, elapsed


    # ------------------------------------------------------------------
    # Quasistatic hysteresis loop via energy minimization.
    # ------------------------------------------------------------------
    def hysteresis(
        self,
        m0_array,
        H_steps,
        method: str = "labonte",
        max_iter: int = 3000,
        tol: float = 1.0,
        output_dir: str | Path = "hyst_minimize_gpu",
        write_xdmf_per_step: bool = True,
        write_bp_series: bool = False,
        xdmf_name: str = "Hysteresis_GPU.xdmf",
        bp_name: str = "Hysteresis_GPU.bp",
        log_name: str = "hysteresis_gpu_log.txt",
        write_xdmf_series: bool = False,
        print_every: int = 25,
        **kwargs,
    ):
        """
        Compute a quasistatic hysteresis loop on GPU using energy minimization.

        This mirrors the CPU EnergyMinimizer.hysteresis(...) workflow, but keeps
        the GPU single-rank design.  At each field step, the external field is
        updated and the current magnetization is relaxed with self.minimize(...).

        Parameters
        ----------
        m0_array:
            Initial magnetization in the local GPU/DOLFINx layout accepted by
            self._load_initial_state.

        H_steps:
            Iterable of external-field values (Hx, Hy, Hz) in A/m.

        method:
            Kept for API symmetry.  Only "labonte" and its aliases are
            accepted by self.minimize.

        max_iter, tol, print_every:
            Forwarded to self.minimize at every field step.

        output_dir:
            Root directory for per-step files and hysteresis log.

        write_xdmf_per_step:
            If True, writes one XDMF file per field step: m_00000.xdmf, ...

        write_bp_series:
            If True, writes all relaxed states into a single ADIOS/BP series.

        write_xdmf_series:
            If True, writes all relaxed states into a single XDMF time series.

        kwargs:
            Additional LaBonte-BB parameters forwarded to self.minimize
            such as alpha0, alpha_min, alpha_max,
            bb_variant="adaptive", bb_adaptive_tau0,
            bb_adaptive_Malpha, nonmonotone_window,
            max_backtracking and shrink.

        """
        H_steps = np.asarray(list(H_steps), dtype=np.float64).reshape((-1, 3))

        if self.comm.size != 1:
            raise RuntimeError(
                "EnergyMinimizerGPU.hysteresis currently supports single-rank "
                "GPU execution only."
            )

        # Build EffectiveField once and load the initial state once.  Subsequent
        # field steps continue from the relaxed state of the previous step.
        if self.hef is None:
            self._build_effective_field()
        self._load_initial_state(m0_array)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        log_path = output_path / log_name
        with open(log_path, "w") as f:
            f.write(
                "# step Hx Hy Hz  <mx> <my> <mz>  "
                "E_total  max_projH  iterations  reason  elapsed  "
                "h_eff_reused  h_eff_recomputed\n"
            )

        self._prepare_function_for_io(function_name="m", mesh_name="Grid")

        # Optional XDMF time series.
        xdmf = None
        if write_xdmf_series:
            xdmf_path = output_path / xdmf_name
            xdmf = io.XDMFFile(self.comm, str(xdmf_path), "w")
            xdmf.write_mesh(self.mesh)

        # Optional ADIOS/BP series.
        bp_path = output_path / bp_name
        if write_bp_series:
            try:
                ad.write_mesh(bp_path, self.mesh)
            except Exception:
                try:
                    ad.write_mesh(str(bp_path), self.mesh)
                except Exception as exc:
                    print(
                        "[WARN] Could not initialize ADIOS/BP hysteresis series. "
                        f"Disabling write_bp_series. Error: {repr(exc)}",
                        flush=True,
                    )
                    write_bp_series = False

        def _mean_m_gpu():
            m_all = _vec_to_cupy(self.hef.m_gpu, "r")
            m = m_all[: self.hef.local_size].reshape((-1, 3))
            if m.shape[0] == 0:
                return np.zeros(3, dtype=np.float64)
            return cp.mean(m, axis=0).get().astype(np.float64)

        def _sync_function_for_io():
            self._prepare_function_for_io(function_name="m", mesh_name="Grid")

        def _write_xdmf_function(xdmf_file, m_fun, time_value=None):
            if time_value is None:
                try:
                    xdmf_file.write_function(m_fun)
                except TypeError:
                    xdmf_file.write_function(m_fun, 0.0)
            else:
                try:
                    xdmf_file.write_function(m_fun, float(time_value))
                except TypeError:
                    xdmf_file.write_function(m_fun)

        def _write_bp_function(path, m_fun, time_value):

            attempts = (
                lambda: ad.write_function(path, m_fun, time=float(time_value), name="m"),
                lambda: ad.write_function(str(path), m_fun, time=float(time_value), name="m"),
                lambda: ad.write_function(m_fun, path, time=float(time_value), name="m"),
                lambda: ad.write_function(m_fun, str(path), time=float(time_value), name="m"),
            )
            last_error = None
            for attempt in attempts:
                try:
                    attempt()
                    return True
                except Exception as exc:
                    last_error = exc
            print(
                "[WARN] Could not write ADIOS/BP hysteresis function at "
                f"step time={time_value}. Error: {repr(last_error)}",
                flush=True,
            )
            return False

        results = []
        mu0 = 4.0 * np.pi * 1e-7

        for i, (Hx, Hy, Hz) in enumerate(H_steps):
            self.set_uniform_field(float(Hx), float(Hy), float(Hz))

            # Continue from the current state.  Do not reload m0_array after the
            # first step; this is the key continuation mechanism of hysteresis.
            _, stats_i, elapsed_i = self.minimize(
                m0_array=None,
                method=method,
                max_iter=max_iter,
                tol=tol,
                output_dir=None,
                save_final_state=False,
                write_xdmf=False,
                write_log=False,
                print_every=print_every,
                **kwargs,
            )

            _sync_function_for_io()

            if write_xdmf_series and xdmf is not None:
                _write_xdmf_function(xdmf, self.hef.m, time_value=float(i))

            if write_xdmf_per_step:
                fname = output_path / f"m_{i:05d}.xdmf"
                with io.XDMFFile(self.comm, str(fname), "w") as xf:
                    xf.write_mesh(self.mesh)
                    _write_xdmf_function(xf, self.hef.m)

            if write_bp_series:
                ok = _write_bp_function(bp_path, self.hef.m, time_value=float(i))
                if not ok:
                    write_bp_series = False

            mmean = _mean_m_gpu()
            E_total = float(stats_i.get("energy", float("nan")))
            max_projH = float(
                stats_i.get("max_projected_field", stats_i.get("max_projH", float("nan")))
            )
            n_iter = int(stats_i.get("iterations", 0))
            reason = str(stats_i.get("stop_reason", ""))
            h_reuse = int(stats_i.get("h_eff_reused_count", 0))
            h_recomp = int(stats_i.get("h_eff_recomputed_after_accept_count", 0))

            entry = {
                "step": int(i),
                "H": (float(Hx), float(Hy), float(Hz)),
                "H_mT": (
                    float(Hx * mu0 * 1e3),
                    float(Hy * mu0 * 1e3),
                    float(Hz * mu0 * 1e3),
                ),
                "m_mean": (float(mmean[0]), float(mmean[1]), float(mmean[2])),
                "energy": E_total,
                "max_projected_field": max_projH,
                "iterations": n_iter,
                "stop_reason": reason,
                "elapsed": float(elapsed_i),
                "h_eff_reused_count": h_reuse,
                "h_eff_recomputed_after_accept_count": h_recomp,
                "stats": stats_i,
            }
            results.append(entry)

            with open(log_path, "a") as f:
                f.write(
                    f"{i:d} {Hx:.6e} {Hy:.6e} {Hz:.6e} "
                    f"{mmean[0]:.6e} {mmean[1]:.6e} {mmean[2]:.6e} "
                    f"{E_total:.6e} {max_projH:.6e} {n_iter:d} "
                    f"{reason} {elapsed_i:.6f} {h_reuse:d} {h_recomp:d}\n"
                )

            print(
                f"[HYST-MIN-GPU] i={i:05d}  "
                f"H(mT)=({Hx*mu0*1e3:+.4e},{Hy*mu0*1e3:+.4e},{Hz*mu0*1e3:+.4e})  "
                f"<m>=({mmean[0]:+.6e},{mmean[1]:+.6e},{mmean[2]:+.6e})  "
                f"E={E_total:.6e}  max|projH|={max_projH:.4e}  "
                f"iters={n_iter}  {reason}  {elapsed_i:.2f}s  "
                f"reuse={h_reuse} recomp={h_recomp}",
                flush=True,
            )

        if xdmf is not None:
            xdmf.close()

        return results


__all__ = [
    "EnergyMinimizerGPU",
    "MinimizerGPUContext",
    "LaBonteBBMinimizerGPU",
]

