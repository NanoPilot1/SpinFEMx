
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demag_Lindholm_HTool_MPI_CPU.py

CPU + MPI + PETSc MATHTOOL implementation of a FEM/BEM demagnetizing field
using a Lindholm double-layer boundary operator.

Main design:
  - FEM solves remain standard DOLFINx/PETSc CPU MPI.
  - The dense boundary operator B is NOT assembled in production.
  - B is represented as PETSc MATHTOOL using a block callback A(I,J).
  - The callback evaluates the nonsymmetric Laplace double-layer operator.
  - The Htool MatMult is CPU-side. MATHTOOL has no device/GPU support.
  - Boundary vectors of length Nb are distributed for MATHTOOL MatMult.
  - The geometry data are currently replicated on all ranks.

"""

from __future__ import annotations

import os
import sys
import math
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from mpi4py import MPI

import petsc4py




_APP_OPTIONS_WITH_VALUE = {
    "--mesh",
    "--mesh-name",
    "--serial-mesh",
    "--ms",
    "--volume-scale",
    "--m0x",
    "--m0y",
    "--m0z",
    "--map-tol",
    "--ksp-rtol-u1",
    "--ksp-rtol-u2",
    "--incident-policy",
    "--htool-epsilon",
    "--htool-eta",
    "--htool-max-leaf-size",
    "--cache-dir",
    "--cache-tag",
    "--timing-repeats",
}

_APP_FLAGS = {
    "--validate-htool",
    "--view-htool",
}


def split_petsc_and_app_argv(argv):
    petsc_argv = [argv[0]]
    app_argv = [argv[0]]

    i = 1
    while i < len(argv):
        a = argv[i]

        if a in _APP_OPTIONS_WITH_VALUE:
            app_argv.append(a)
            if i + 1 < len(argv):
                app_argv.append(argv[i + 1])
                i += 2
            else:
                i += 1

        elif a in _APP_FLAGS:
            app_argv.append(a)
            i += 1

        else:
            petsc_argv.append(a)
            # PETSc options may have a value. Keep the following non-option token.
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                petsc_argv.append(argv[i + 1])
                i += 2
            else:
                i += 1

    return petsc_argv, app_argv


PETSC_ARGV, APP_ARGV = split_petsc_and_app_argv(sys.argv)
petsc4py.init(PETSC_ARGV)

from petsc4py import PETSc
import re
import dolfinx
from dolfinx import fem, io


# Patch XDMFFile so meshes remember the file they came from.
# This lets DemagField_Lindholm(mesh, V, V1, ...) infer the serial mesh path
# without passing serial_mesh_path explicitly.

_ORIGINAL_XDMFFile = dolfinx.io.XDMFFile


class XDMFFileWithMeshPath:
    def __init__(self, comm, filename, mode, *args, **kwargs):
        self.filename = str(filename)
        self._xdmf = _ORIGINAL_XDMFFile(comm, filename, mode, *args, **kwargs)

    def __enter__(self):
        self._xdmf.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._xdmf.__exit__(exc_type, exc_value, traceback)

    def read_mesh(self, *args, **kwargs):
        mesh = self._xdmf.read_mesh(*args, **kwargs)
        mesh._serial_mesh_path = self.filename
        return mesh

    def __getattr__(self, name):
        return getattr(self._xdmf, name)


dolfinx.io.XDMFFile = XDMFFileWithMeshPath
io.XDMFFile = XDMFFileWithMeshPath

from dolfinx.mesh import exterior_facet_indices
from dolfinx.fem.petsc import apply_lifting, set_bc
import ufl

from scipy.spatial import cKDTree
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from numba import njit, prange


# -----------------------------------------------------------------------------
# Thread defaults
# -----------------------------------------------------------------------------
def _enforce_thread_defaults():
    # Keep BLAS/MKL from oversubscribing under MPI.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
    # Do not force NUMBA_NUM_THREADS here if the user already set it.
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")


def _format_bytes(nbytes: float) -> str:
    nbytes = float(nbytes)
    if nbytes < 1024:
        return f"{nbytes:.0f} B"
    kib = nbytes / 1024.0
    if kib < 1024:
        return f"{kib:.2f} KiB"
    mib = kib / 1024.0
    if mib < 1024:
        return f"{mib:.2f} MiB"
    gib = mib / 1024.0
    return f"{gib:.2f} GiB"


def _balanced_range(N: int, size: int, rank: int):
    base = N // size
    rem = N % size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def _safe_array_nbytes(x):
    try:
        return int(x.nbytes)
    except Exception:
        return 0


# Lindholm kernel primitives

@njit(fastmath=True)
def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(fastmath=True)
def _norm(a):
    return math.sqrt(_dot(a, a))


@njit(fastmath=True)
def _det3(a, b, c):
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


@njit(fastmath=True)
def solid_angle_with_atan2(x0, p0, p1, p2):
    tiny = 1e-300

    r0 = p0 - x0
    r1 = p1 - x0
    r2 = p2 - x0

    n0 = _norm(r0) + tiny
    n1 = _norm(r1) + tiny
    n2 = _norm(r2) + tiny

    det = _det3(r0, r1, r2)

    d01 = _dot(r0, r1)
    d12 = _dot(r1, r2)
    d20 = _dot(r2, r0)

    denom = n0 * n1 * n2 + d01 * n2 + d12 * n0 + d20 * n1
    return 2.0 * math.atan2(det, denom)


@njit(fastmath=True)
def _P_log1p(ri, rj, s):
    tiny = 1e-300
    A = ri + rj
    denom = A - s
    if denom < tiny:
        denom = tiny
    return math.log1p((2.0 * s) / denom)


@njit(fastmath=True)
def lindholm_weights_precomp(
    x0,
    p0,
    p1,
    p2,
    n_unit,
    area,
    s_edge,
    eta_vecs,
    gamma_mat,
):
    tiny = 1e-300

    rho0 = p0 - x0
    rho1 = p1 - x0
    rho2 = p2 - x0

    r0 = _norm(rho0) + tiny
    r1 = _norm(rho1) + tiny
    r2 = _norm(rho2) + tiny

    h = _dot(n_unit, rho0)

    eta0 = _dot(eta_vecs[0], rho0)
    eta1 = _dot(eta_vecs[1], rho1)
    eta2 = _dot(eta_vecs[2], rho2)

    s0 = s_edge[0]
    s1 = s_edge[1]
    s2 = s_edge[2]

    P0 = _P_log1p(r0, r1, s0)
    P1 = _P_log1p(r1, r2, s1)
    P2 = _P_log1p(r2, r0, s2)

    Omega = solid_angle_with_atan2(x0, p0, p1, p2)

    gP0 = gamma_mat[0, 0] * P0 + gamma_mat[0, 1] * P1 + gamma_mat[0, 2] * P2
    gP1 = gamma_mat[1, 0] * P0 + gamma_mat[1, 1] * P1 + gamma_mat[1, 2] * P2
    gP2 = gamma_mat[2, 0] * P0 + gamma_mat[2, 1] * P1 + gamma_mat[2, 2] * P2

    A = area + tiny
    c = 1.0 / (8.0 * math.pi * A)

    # Opposite-edge convention.
    w0 = s1 * c * (eta1 * Omega - h * gP0)
    w1 = s2 * c * (eta2 * Omega - h * gP1)
    w2 = s0 * c * (eta0 * Omega - h * gP2)

    return w0, w1, w2



# Serial boundary data construction

def _read_serial_mesh_any_format(meshfile, mesh_name="Grid"):
    meshfile = str(meshfile)
    suffix = Path(meshfile).suffix.lower()

    if suffix == ".bp":
        import adios4dolfinx as ad
        return ad.read_mesh(Path(meshfile), MPI.COMM_SELF)

    with io.XDMFFile(MPI.COMM_SELF, meshfile, "r") as xdmf:
        try:
            return xdmf.read_mesh(name=mesh_name)
        except Exception:
            return xdmf.read_mesh()


def build_boundary_data_serial(mesh_ser):
    tdim = mesh_ser.topology.dim
    fdim = tdim - 1

    mesh_ser.topology.create_connectivity(fdim, 0)
    mesh_ser.topology.create_connectivity(fdim, tdim)
    mesh_ser.topology.create_connectivity(tdim, 0)

    f2v = mesh_ser.topology.connectivity(fdim, 0)
    f2c = mesh_ser.topology.connectivity(fdim, tdim)
    c2v = mesh_ser.topology.connectivity(tdim, 0)

    X = mesh_ser.geometry.x
    bfacets = exterior_facet_indices(mesh_ser.topology)

    tris = []
    normals = []

    for f in bfacets:
        vs = f2v.links(f)
        v0, v1, v2 = int(vs[0]), int(vs[1]), int(vs[2])

        p0 = X[v0]
        p1 = X[v1]
        p2 = X[v2]

        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn == 0.0:
            continue
        n = n / nn

        cell = int(f2c.links(f)[0])
        cc = X[c2v.links(cell)].mean(axis=0)
        fc = (p0 + p1 + p2) / 3.0

        if np.dot(fc - cc, n) < 0.0:
            v1, v2 = v2, v1
            p1 = X[v1]
            p2 = X[v2]
            n = np.cross(p1 - p0, p2 - p0)
            n /= (np.linalg.norm(n) + 1e-30)

        tris.append((v0, v1, v2))
        normals.append(n)

    tris = np.asarray(tris, dtype=np.int64)
    ntri = np.asarray(normals, dtype=np.float64)

    bverts = np.unique(tris.ravel())
    Nb = bverts.size
    Xb = X[bverts].astype(np.float64)

    bmap = -np.ones(X.shape[0], dtype=np.int64)
    bmap[bverts] = np.arange(Nb, dtype=np.int64)
    tri_lid = bmap[tris].astype(np.int32)

    tri_pts = X[tris].astype(np.float64)

    cross = np.cross(
        tri_pts[:, 1, :] - tri_pts[:, 0, :],
        tri_pts[:, 2, :] - tri_pts[:, 0, :],
    )
    area = (0.5 * np.linalg.norm(cross, axis=1)).astype(np.float64)

    e0 = tri_pts[:, 1, :] - tri_pts[:, 0, :]
    e1 = tri_pts[:, 2, :] - tri_pts[:, 1, :]
    e2 = tri_pts[:, 0, :] - tri_pts[:, 2, :]

    s0 = np.linalg.norm(e0, axis=1) + 1e-30
    s1 = np.linalg.norm(e1, axis=1) + 1e-30
    s2 = np.linalg.norm(e2, axis=1) + 1e-30

    s_edge = np.stack([s0, s1, s2], axis=1).astype(np.float64)

    xi0 = (e0 / s0[:, None]).astype(np.float64)
    xi1 = (e1 / s1[:, None]).astype(np.float64)
    xi2 = (e2 / s2[:, None]).astype(np.float64)
    xi = np.stack([xi0, xi1, xi2], axis=1)

    eta = np.zeros_like(xi)
    eta[:, 0, :] = np.cross(ntri, xi[:, 0, :])
    eta[:, 1, :] = np.cross(ntri, xi[:, 1, :])
    eta[:, 2, :] = np.cross(ntri, xi[:, 2, :])

    gamma = np.zeros((xi.shape[0], 3, 3), dtype=np.float64)
    for i in range(3):
        ip1 = (i + 1) % 3
        for j in range(3):
            gamma[:, i, j] = np.einsum("ij,ij->i", xi[:, ip1, :], xi[:, j, :])

    # Solid-angle correction per boundary vertex.
    c2v_ser = mesh_ser.topology.connectivity(tdim, 0)
    imap = mesh_ser.topology.index_map(tdim)
    ncell = imap.size_local

    omega_sum = np.zeros(Nb, dtype=np.float64)
    for c in range(ncell):
        vs = c2v_ser.links(c)
        pts = X[vs]

        for j in range(4):
            lid = bmap[int(vs[j])]
            if lid < 0:
                continue

            P = pts[j].astype(np.float64)
            T1 = pts[(j + 1) % 4].astype(np.float64)
            T2 = pts[(j + 2) % 4].astype(np.float64)
            T3 = pts[(j + 3) % 4].astype(np.float64)

            omega_sum[lid] += abs(float(solid_angle_with_atan2(P, T1, T2, T3)))

    return Xb, tri_lid, tri_pts, ntri, area, s_edge, eta, gamma, omega_sum



# Boundary data + parallel maps

class DoubleLayerBoundaryDataMPI:
    """
    Builds/broadcasts the serial boundary geometry and maps local owned scalar
    FEM boundary DOFs to global boundary indices.
    """

    def __init__(
        self,
        mesh_par,
        V1_par,
        serial_mesh_path,
        mesh_name="Grid",
        map_tol=1e-10,
        incident_policy="skip",
    ):
        self.mesh = mesh_par
        self.V1_par = V1_par
        self.comm = mesh_par.comm
        self.rank = self.comm.rank
        self.map_tol = float(map_tol)
        self.incident_policy = 0 if str(incident_policy).lower().startswith("skip") else 1

        if serial_mesh_path is None:
            raise ValueError("serial_mesh_path must be provided for robust MPI execution.")

        if self.rank == 0:
            mesh_ser = _read_serial_mesh_any_format(serial_mesh_path, mesh_name=mesh_name)
            data = build_boundary_data_serial(mesh_ser)
        else:
            data = None

        self._bcast_boundary_data(data)
        self._build_parallel_boundary_maps()

        # Unique global boundary rows corresponding to owned scalar boundary dofs.
        self.rows = np.unique(self.bidx_owned).astype(np.int32)
        self.nrows = int(self.rows.size)

        self.rowpos_for_bdof = np.searchsorted(self.rows, self.bidx_owned).astype(np.int32)

        # Collision diagnostic.
        cloc = np.zeros(self.Nb, dtype=np.int32)
        np.add.at(cloc, self.bidx_owned, 1)

        self._count_global = np.empty_like(cloc)
        self.comm.Allreduce(cloc, self._count_global, op=MPI.SUM)

        self.has_collisions = not (
            int(self._count_global.min()) == 1 and int(self._count_global.max()) == 1
        )


    def _bcast_ndarray(self, arr, root=0):
        if self.rank == root:
            arr = np.ascontiguousarray(arr)
            shape = np.array(arr.shape, dtype=np.int64)
            dtype = arr.dtype.str
        else:
            shape = None
            dtype = None
            arr = None

        shape = self.comm.bcast(shape, root=root)
        dtype = self.comm.bcast(dtype, root=root)

        if self.rank != root:
            arr = np.empty(tuple(shape.tolist()), dtype=np.dtype(dtype))

        self.comm.Bcast(arr, root=root)
        return arr

    def _bcast_boundary_data(self, data):
        if self.rank == 0:
            Xb, tri_lid, tri_pts, ntri, area, s_edge, eta, gamma, omega_sum = data
        else:
            Xb = tri_lid = tri_pts = ntri = area = s_edge = eta = gamma = omega_sum = None

        self.Xb = self._bcast_ndarray(Xb, root=0)
        self.tri_lid = self._bcast_ndarray(tri_lid, root=0)
        self.tri_pts = self._bcast_ndarray(tri_pts, root=0)
        self.ntri = self._bcast_ndarray(ntri, root=0)
        self.area = self._bcast_ndarray(area, root=0)
        self.s_edge = self._bcast_ndarray(s_edge, root=0)
        self.eta = self._bcast_ndarray(eta, root=0)
        self.gamma = self._bcast_ndarray(gamma, root=0)
        self.omega_sum = self._bcast_ndarray(omega_sum, root=0)

        self.Nb = int(self.Xb.shape[0])
        self.Nt = int(self.tri_lid.shape[0])

        # Exact diagonal / jump term of the double-layer operator.
        # HTool is used only for the off-diagonal contribution.
        # This improves robustness because the singular jump term is no longer
        # represented through the compressed H-matrix.
        self.diag_jump = (self.omega_sum / (4.0 * math.pi) - 1.0).astype(np.float64)

    def _build_parallel_boundary_maps(self):
        mesh = self.mesh
        V1 = self.V1_par
        tdim = mesh.topology.dim
        fdim = tdim - 1

        mesh.topology.create_connectivity(fdim, tdim)
        mesh.topology.create_connectivity(fdim, 0)
        mesh.topology.create_connectivity(tdim, 0)

        bfacets = exterior_facet_indices(mesh.topology)
        bdofs_all = fem.locate_dofs_topological(V1, fdim, bfacets)
        bdofs_all = np.unique(bdofs_all)

        imap = V1.dofmap.index_map
        nloc = imap.size_local

        self.bdofs_owned = bdofs_all[bdofs_all < nloc].astype(np.int32)

        coords = V1.tabulate_dof_coordinates()
        coords_owned = coords[self.bdofs_owned]

        tree = cKDTree(self.Xb)
        d, idx = tree.query(coords_owned, k=1, workers=1)

        dmax = float(np.max(d)) if d.size else 0.0
        dmax_g = self.comm.allreduce(dmax, op=MPI.MAX)
        if self.rank == 0 and dmax_g > self.map_tol:
            print(
                f"[WARN] dmax mapping DOF_boundary->Xb = {dmax_g:g} > tol={self.map_tol:g}",
                flush=True,
            )

        self.bidx_owned = idx.astype(np.int32)



# For validation

@njit(parallel=True, fastmath=True)
def assemble_B_rows_owned_dense(
    Xb,
    tri_lid,
    tri_pts,
    ntri,
    area,
    s_edge,
    eta,
    gamma,
    omega_sum,
    rows,
    incident_policy,
):
    Nb = Xb.shape[0]
    Nt = tri_lid.shape[0]
    nrows = rows.shape[0]

    Bloc = np.zeros((nrows, Nb), dtype=np.float64)
    inv4pi = 1.0 / (4.0 * math.pi)

    for ii in prange(nrows):
        gi = rows[ii]
        Bloc[ii, gi] = omega_sum[gi] * inv4pi - 1.0

    for ii in prange(nrows):
        gi = rows[ii]
        x0 = Xb[gi]

        for t in range(Nt):
            a = tri_lid[t, 0]
            b = tri_lid[t, 1]
            c = tri_lid[t, 2]

            if incident_policy == 0:
                if gi == a or gi == b or gi == c:
                    continue

            w0, w1, w2 = lindholm_weights_precomp(
                x0,
                tri_pts[t, 0],
                tri_pts[t, 1],
                tri_pts[t, 2],
                ntri[t],
                area[t],
                s_edge[t],
                eta[t],
                gamma[t],
            )

            Bloc[ii, a] += w0
            Bloc[ii, b] += w1
            Bloc[ii, c] += w2

    return Bloc



# Htool block callback

def build_node_incident_triangle_lists(tri_lid, Nb):
    counts = np.zeros(Nb, dtype=np.int32)

    for t in range(tri_lid.shape[0]):
        counts[tri_lid[t, 0]] += 1
        counts[tri_lid[t, 1]] += 1
        counts[tri_lid[t, 2]] += 1

    ptr = np.zeros(Nb + 1, dtype=np.int32)
    ptr[1:] = np.cumsum(counts)

    incident_tri = np.empty(ptr[-1], dtype=np.int32)
    incident_loc = np.empty(ptr[-1], dtype=np.int32)

    cursor = ptr[:-1].copy()
    for t in range(tri_lid.shape[0]):
        for loc in range(3):
            j = tri_lid[t, loc]
            p = cursor[j]
            incident_tri[p] = t
            incident_loc[p] = loc
            cursor[j] += 1

    return ptr, incident_tri, incident_loc


@njit(parallel=True, fastmath=True, nogil=True)
def _fill_lindholm_block_incident_numba(
    Xb,
    tri_lid,
    tri_pts,
    ntri,
    area,
    s_edge,
    eta,
    gamma,
    omega_sum,
    incident_ptr,
    incident_tri,
    incident_loc,
    rows,
    cols,
    incident_policy,
    out,
):
    M = rows.shape[0]
    N = cols.shape[0]
    inv4pi = 1.0 / (4.0 * math.pi)

    for ii in prange(M):
        for jj in range(N):
            out[ii, jj] = 0.0

    for ii in prange(M):
        gi = rows[ii]
        x0 = Xb[gi]

        for jj in range(N):
            gj = cols[jj]

            # Diagonal / jump term is applied exactly outside HTool.
            # Here the compressed matrix represents only the off-diagonal part.
            start = incident_ptr[gj]
            end = incident_ptr[gj + 1]

            acc = 0.0

            for p in range(start, end):
                t = incident_tri[p]
                loc = incident_loc[p]

                a = tri_lid[t, 0]
                b = tri_lid[t, 1]
                c = tri_lid[t, 2]

                if incident_policy == 0:
                    if gi == a or gi == b or gi == c:
                        continue

                w0, w1, w2 = lindholm_weights_precomp(
                    x0,
                    tri_pts[t, 0],
                    tri_pts[t, 1],
                    tri_pts[t, 2],
                    ntri[t],
                    area[t],
                    s_edge[t],
                    eta[t],
                    gamma[t],
                )

                if loc == 0:
                    acc += w0
                elif loc == 1:
                    acc += w1
                else:
                    acc += w2

            out[ii, jj] += acc


class LindholmHtoolKernelContextMPI:
    def __init__(self, bd: DoubleLayerBoundaryDataMPI, debug=False):
        self.Xb = np.ascontiguousarray(bd.Xb, dtype=np.float64)
        self.tri_lid = np.ascontiguousarray(bd.tri_lid, dtype=np.int32)
        self.tri_pts = np.ascontiguousarray(bd.tri_pts, dtype=np.float64)
        self.ntri = np.ascontiguousarray(bd.ntri, dtype=np.float64)
        self.area = np.ascontiguousarray(bd.area, dtype=np.float64)
        self.s_edge = np.ascontiguousarray(bd.s_edge, dtype=np.float64)
        self.eta = np.ascontiguousarray(bd.eta, dtype=np.float64)
        self.gamma = np.ascontiguousarray(bd.gamma, dtype=np.float64)
        self.omega_sum = np.ascontiguousarray(bd.omega_sum, dtype=np.float64)
        self.incident_policy = int(bd.incident_policy)

        self.incident_ptr, self.incident_tri, self.incident_loc = (
            build_node_incident_triangle_lists(self.tri_lid, bd.Nb)
        )

        self.call_count = 0
        self.debug = bool(debug)


def lindholm_htool_kernel_mpi(sdim, M, N, rows, cols, v, ctx):
    ctx.call_count += 1

    rows_i = np.asarray(rows, dtype=np.int32)
    cols_i = np.asarray(cols, dtype=np.int32)

    _fill_lindholm_block_incident_numba(
        ctx.Xb,
        ctx.tri_lid,
        ctx.tri_pts,
        ctx.ntri,
        ctx.area,
        ctx.s_edge,
        ctx.eta,
        ctx.gamma,
        ctx.omega_sum,
        ctx.incident_ptr,
        ctx.incident_tri,
        ctx.incident_loc,
        rows_i,
        cols_i,
        ctx.incident_policy,
        v,
    )



# Nullspace helpers

def _cell_components(mesh):
    tdim = mesh.topology.dim
    fdim = tdim - 1

    mesh.topology.create_connectivity(tdim, fdim)
    mesh.topology.create_connectivity(fdim, tdim)

    c2f = mesh.topology.connectivity(tdim, fdim)
    f2c = mesh.topology.connectivity(fdim, tdim)

    imap = mesh.topology.index_map(tdim)
    ncell = imap.size_local + imap.num_ghosts

    rows = []
    cols = []

    for c in range(ncell):
        for f in c2f.links(c):
            for c2 in f2c.links(f):
                if c2 != c:
                    rows.append(c)
                    cols.append(c2)

    A = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(ncell, ncell),
    )

    ncomp, labels = connected_components(A, directed=False)
    return ncomp, labels


def build_nullspace_per_component(mesh, V1, A_u1):
    tdim = mesh.topology.dim
    fdim = tdim - 1

    mesh.topology.create_connectivity(tdim, fdim)
    mesh.topology.create_connectivity(fdim, tdim)
    mesh.topology.create_connectivity(tdim, 0)
    mesh.topology.create_connectivity(0, tdim)

    c2v = mesh.topology.connectivity(tdim, 0)
    ncomp, cell_labels = _cell_components(mesh)

    comp_vertices = [set() for _ in range(ncomp)]
    for c, comp in enumerate(cell_labels):
        comp_vertices[comp].update(c2v.links(c).tolist())

    imap = V1.dofmap.index_map
    nloc = imap.size_local

    z_vecs = []
    for k in range(ncomp):
        verts = np.array(sorted(comp_vertices[k]), dtype=np.int32)
        dofs = fem.locate_dofs_topological(V1, 0, verts)
        dofs = dofs[dofs < nloc]

        z = A_u1.createVecRight()
        z.set(0.0)
        z.array[dofs] = 1.0
        z.assemble()

        nrm = z.norm()
        if nrm > 0.0:
            z.scale(1.0 / nrm)

        z_vecs.append(z)

    ns = PETSc.NullSpace().create(vectors=z_vecs, comm=mesh.comm)

    ok = True
    try:
        ok = ns.test(A_u1)
    except Exception:
        ok = False

    if not ok:
        ns = PETSc.NullSpace().create(constant=True, comm=mesh.comm)

    A_u1.setNullSpace(ns)
    A_u1.setNearNullSpace(ns)

    return ns



# Main class

class DemagField_Lindholm_HTool_MPI:
    def __init__(
        self,
        mesh,
        V,
        V1,
        Ms,
        serial_mesh_path=None,
        mesh_name="Grid",
        map_tol=1e-10,
        ksp_rtol_u1=1e-6,
        ksp_rtol_u2=1e-6,
        incident_policy="skip",
        volume_scale=1e27,
        htool_epsilon=1e-6,
        htool_eta=2.0,
        htool_max_leaf_size=64,
        recompress_bool = 0,
        Compressor_Algorithm = "fullaca",
        validate_htool=False,
        view_htool=False,
        htool_kernel_debug=False,
        memory_report=False,
    ):
        _enforce_thread_defaults()

        self.mesh = mesh
        self.V = V
        self.V1 = V1
        self.Ms = float(Ms)
        self.mu0 = 4.0 * math.pi * 1e-7
        self.comm = mesh.comm
        self.rank = self.comm.rank
        self.size = self.comm.size
        self.volume_scale = float(volume_scale)

        if serial_mesh_path is None:
            serial_mesh_path = os.environ.get("DEMAG_SERIAL_MESH_PATH", None)

        if serial_mesh_path is None:
            serial_mesh_path = getattr(mesh, "_serial_mesh_path", None)

        if serial_mesh_path is None:
            if self.comm.size > 1:
                raise RuntimeError(
                    "serial_mesh_path could not be inferred in MPI. "
                    "Read the mesh with the patched dolfinx.io.XDMFFile, "
                    "set DEMAG_SERIAL_MESH_PATH, or pass serial_mesh_path explicitly."
                )

            raise RuntimeError(
                "serial_mesh_path could not be inferred. "
                "Pass serial_mesh_path explicitly or read the mesh with patched XDMFFile."
            )

        self.serial_mesh_path = str(serial_mesh_path)

        self.last_its_u1 = -1
        self.last_its_u2 = -1

        tdim = mesh.topology.dim
        fdim = tdim - 1

        # ---------------- u1 ----------------
        self.u1_sol = fem.Function(V1)

        v1 = ufl.TestFunction(V1)
        u1 = ufl.TrialFunction(V1)

        a_u1 = fem.form(ufl.inner(ufl.grad(v1), ufl.grad(u1)) * ufl.dx)
        self.A_u1 = fem.petsc.assemble_matrix(a_u1)
        self.A_u1.assemble()

        self.ns_u1 = build_nullspace_per_component(mesh, V1, self.A_u1)

        self.ksp_u1 = PETSc.KSP().create(self.comm)
        self.ksp_u1.setType("cg")
        self.ksp_u1.setOperators(self.A_u1)
        self.ksp_u1.setTolerances(rtol=ksp_rtol_u1, atol=1e-12, max_it=2000)


        imap = self.V1.dofmap.index_map
        bs = self.V1.dofmap.index_map_bs

        self.n_u1_dofs_global = imap.size_global * bs


        if self.n_u1_dofs_global<60000:

            self.ksp_u1.setType("preonly")
            pc1 = self.ksp_u1.getPC()
            pc1.setType("lu")
            pc1.setReusePreconditioner(True)
            
        else:

            pc1 = self.ksp_u1.getPC()
            pc1.setType("gamg")
            pc1.setReusePreconditioner(True)

        self.ksp_u1.setFromOptions()

        v2 = ufl.TestFunction(V1)
        m_trial = ufl.TrialFunction(V)

        Div_form = fem.form(self.Ms * ufl.inner(ufl.grad(v2), m_trial) * ufl.dx)
        self.Div = fem.petsc.assemble_matrix(Div_form)
        self.Div.assemble()

        self.b_u1 = self.Div.createVecLeft()

        # ---------------- u2 ----------------
        self.u2_on_boundary_parallel = fem.Function(V1)




        # ---------------- u2 ----------------
        self.u2_sol = fem.Function(V1)

        mesh.topology.create_connectivity(fdim, tdim)
        mesh.topology.create_connectivity(fdim, 0)

        boundary_facets = exterior_facet_indices(mesh.topology)
        boundary_dofs = fem.locate_dofs_topological(V1, fdim, boundary_facets)
        boundary_dofs = np.unique(boundary_dofs).astype(np.int32)

        imap1 = V1.dofmap.index_map
        nloc1 = imap1.size_local
        self.boundary_dofs = boundary_dofs
        self.boundary_dofs_owned = boundary_dofs[boundary_dofs < nloc1].astype(np.int32)

        zero = fem.Function(V1)
        zero.x.array[:] = 0.0
        zero.x.scatter_forward()

        bc_zero = fem.dirichletbc(zero, self.boundary_dofs)

        uu = ufl.TrialFunction(V1)
        vv = ufl.TestFunction(V1)

        self._a_u2_form = fem.form(ufl.inner(ufl.grad(uu), ufl.grad(vv)) * ufl.dx)

        A_u2_raw = fem.petsc.assemble_matrix(self._a_u2_form)
        A_u2_raw.assemble()

        A_u2_bc = fem.petsc.assemble_matrix(self._a_u2_form, bcs=[bc_zero])
        A_u2_bc.assemble()

        self.A_u2_raw = A_u2_raw
        self.A_u2 = A_u2_bc

        self.g_u2_fun = fem.Function(V1, name="g_u2")
        self.g_u2_fun.x.array[:] = 0.0
        self.g_u2_fun.x.scatter_forward()

        self.g_u2 = self.g_u2_fun.x.petsc_vec
        self.b_u2 = self.A_u2.createVecLeft()

        self.ksp_u2 = PETSc.KSP().create(self.comm)
        self.ksp_u2.setType("cg")
        self.ksp_u2.setOperators(self.A_u2)
        self.ksp_u2.setTolerances(rtol=ksp_rtol_u2, atol=1e-12, max_it=12000)

        if self.n_u1_dofs_global<60000:

            self.ksp_u2.setType("preonly")
            pc2 = self.ksp_u2.getPC()
            pc2.setType("lu")
            pc2.setReusePreconditioner(True)

        else:
            pc2 = self.ksp_u2.getPC()
            pc2.setType("gamg")
            pc2.setReusePreconditioner(True)

        self.ksp_u2.setFromOptions()
        self.last_its_u2 = -1

        self.opB = DoubleLayerBoundaryDataMPI(
            mesh_par=mesh,
            V1_par=V1,
            serial_mesh_path=serial_mesh_path,
            mesh_name=mesh_name,
            map_tol=map_tol,
            incident_policy=incident_policy,
        )

        self.Nb = int(self.opB.Nb)
        self.Nt = int(self.opB.Nt)

        self.dense_B_theoretical_bytes = (
            int(self.Nb) * int(self.Nb) * np.dtype(np.float64).itemsize
        )

        # Global boundary index ownership for the PETSc boundary vector.
        self.bstart, self.bend = _balanced_range(self.Nb, self.size, self.rank)
        self.nbloc = self.bend - self.bstart

        self.coords_local = np.ascontiguousarray(
            self.opB.Xb[self.bstart:self.bend],
            dtype=PETSc.RealType,
        )

        self.kernel_ctx = LindholmHtoolKernelContextMPI(
            self.opB,
            debug=htool_kernel_debug,
        )

        # petsc4py uses sizes in the form:
        #     [[m_local, M_global], [n_local, N_global]]
        # Rows and columns are distributed with the same balanced ownership.
        self.B_htool = PETSc.Mat().createHtoolFromKernel(
            [[self.nbloc, self.Nb], [self.nbloc, self.Nb]],
            3,
            self.coords_local,
            self.coords_local,
            lindholm_htool_kernel_mpi,
            self.kernel_ctx,
            comm=self.comm,
        )

        # HTool parameters
        try:
            self.B_htool.setHtoolEpsilon(float(htool_epsilon))
        except Exception:
            PETSc.Options()["mat_htool_epsilon"] = float(htool_epsilon)

        try:
            self.B_htool.setHtoolEta(float(htool_eta))
        except Exception:
            PETSc.Options()["mat_htool_eta"] = float(htool_eta)

        try:
            self.B_htool.setHtoolMaxClusterLeafSize(int(htool_max_leaf_size))
        except Exception:
            PETSc.Options()["mat_htool_max_cluster_leaf_size"] = int(htool_max_leaf_size)

        try:
            self.B_htool.setHtoolCompressorType(Compressor_Algorithm)
        except Exception:
            PETSc.Options()["mat_htool_compressor"] = Compressor_Algorithm

        try:
            self.B_htool.useHtoolRecompression(int(recompress_bool))
        except Exception:
            PETSc.Options()["mat_htool_recompression"] = int(recompress_bool)

        self.B_htool.setFromOptions()

        t_h0 = perf_counter()
        self.B_htool.assemble()
        self.comm.Barrier()
        t_h1 = perf_counter()

        self.htool_build_time = t_h1 - t_h0


        self._print_minimal_compression_report(
            build_time=self.htool_build_time,
            dense_bytes=self.dense_B_theoretical_bytes,
        )

        if view_htool:
            self.B_htool.view(PETSc.Viewer.STDOUT(self.comm))

        self.xb_vec, self.yb_vec = self.B_htool.createVecs()

        self.xb_lo, self.xb_hi = self.xb_vec.getOwnershipRange()
        self.yb_lo, self.yb_hi = self.yb_vec.getOwnershipRange()

        self.xb_nloc = self.xb_hi - self.xb_lo
        self.yb_nloc = self.yb_hi - self.yb_lo

        self._yb_counts = np.array(
            self.comm.allgather(self.yb_nloc),
            dtype=np.int32,
        )

        self._yb_displs = np.zeros_like(self._yb_counts)
        self._yb_displs[1:] = np.cumsum(self._yb_counts[:-1])

        self._y_global = np.empty(self.Nb, dtype=np.float64)
        self._x_global = np.zeros(self.Nb, dtype=np.float64)

        if int(np.sum(self._yb_counts)) != self.Nb:
            raise RuntimeError(
                f"Distributed yb_vec sizes do not sum to Nb: "
                f"sum={int(np.sum(self._yb_counts))}, Nb={self.Nb}"
            )

        if validate_htool:
            self.validate_htool_against_dense()

        self.H_d = fem.Function(V)

        one = fem.Function(V1)
        one.x.array.fill(1.0)

        vsc = ufl.TestFunction(V1)
        mass_form = fem.form(one * vsc * ufl.dx)

        vol_nodes = fem.Function(V1)
        fem.petsc.assemble_vector(vol_nodes.x.petsc_vec, mass_form)
        vol_nodes.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )
        vol_nodes.x.scatter_forward()

        nloc = V1.dofmap.index_map.size_local
        self.inv_vol = (1.0 / vol_nodes.x.array[:nloc]).astype(np.float64)

        vvec = ufl.TestFunction(V)
        phi = ufl.TrialFunction(V1)

        G_form = fem.form(ufl.inner(vvec, -ufl.grad(phi)) * ufl.dx)
        self.G = fem.petsc.assemble_matrix(G_form)
        self.G.assemble()


    def _set_boundary_x_from_u1(self):
        """
        Build the distributed PETSc boundary vector xb_vec from u1 on the boundary.

        Safer MPI version:
        1. Build replicated global x_full over boundary indices.
        2. Fill only the locally owned part of xb_vec using Vec.getArray().
        """

        x_full = self._x_global
        x_full.fill(0.0)

        vals = self.u1_sol.x.array[self.opB.bdofs_owned].astype(
            np.float64,
            copy=False,
        )

        if self.opB.bidx_owned.size > 0:
            x_full[self.opB.bidx_owned] = vals

        self.comm.Allreduce(MPI.IN_PLACE, x_full, op=MPI.SUM)

        if self.opB.has_collisions:
            if np.any(self.opB._count_global == 0):
                raise RuntimeError(
                    "count_global has zeros: inconsistent boundary mapping."
                )
            x_full /= self.opB._count_global.astype(np.float64)

        lo, hi = self.xb_vec.getOwnershipRange()
        xloc = self.xb_vec.getArray(readonly=False)

        expected = hi - lo
        if xloc.size != expected:
            raise RuntimeError(
                f"xb_vec local array has wrong size on rank {self.rank}: "
                f"xloc.size={xloc.size}, ownership range=[{lo},{hi}), "
                f"expected={expected}"
            )

        xloc[:] = x_full[lo:hi]


    def _gather_yb_global(self):

        """
        Gather distributed yb_vec into self._y_global on all ranks.
        """

        yloc = self.yb_vec.getArray(readonly=True).copy()

        self.comm.Allgatherv(
            [yloc, MPI.DOUBLE],
            [self._y_global, self._yb_counts, self._yb_displs, MPI.DOUBLE],
        )




    def _print_minimal_compression_report(self, build_time, dense_bytes):
        """
        Minimal HTool report
        """

        compression_ratio = None


        token = self.comm.bcast(
            f"{os.getpid()}_{int(perf_counter() * 1e9)}" if self.rank == 0 else None,
            root=0,
        )

        tmp_dir = Path(os.environ.get("DEMAG_TMP_DIR", "/tmp"))
        tmp_dir.mkdir(parents=True, exist_ok=True)

        view_path = tmp_dir / f"__htool_view_{token}.txt"

        viewer = PETSc.Viewer().createASCII(
            str(view_path),
            mode="w",
            comm=self.comm,
        )

        self.B_htool.view(viewer)

        try:
            viewer.flush()
        except Exception:
            pass

        try:
            viewer.destroy()
        except Exception:
            pass

        self.comm.Barrier()

        if self.rank == 0:
            try:
                txt = view_path.read_text(errors="ignore")
                match = re.search(r"compression ratio:\s*([0-9eE+\-.]+)", txt)
                if match is not None:
                    compression_ratio = float(match.group(1))
            finally:
                try:
                    view_path.unlink()
                except Exception:
                    pass

        compression_ratio = self.comm.bcast(compression_ratio, root=0)

        if self.rank == 0:
            print("")
            print("==== HTool double-layer compression summary ====")
            print(f"Double-layer build time          : {build_time:.6e} s")
            print(f"Dense B theoretical memory       : {_format_bytes(dense_bytes)}")

            if compression_ratio is not None and compression_ratio > 0.0:
                compressed_est = dense_bytes / compression_ratio
                print(f"HTool compression ratio          : {compression_ratio:.6f}")
                print(f"HTool compressed memory estimate : {_format_bytes(compressed_est)}")
            else:
                print("HTool compression ratio          : unavailable")
                print("HTool compressed memory estimate : unavailable")

            print("================================================")
            print("", flush=True)


    def apply_B_to_u1_boundary(self):
        self._set_boundary_x_from_u1()

        self.B_htool.mult(self.xb_vec, self.yb_vec)
        self.comm.Barrier()

        self._gather_yb_global()

        self._y_global += self.opB.diag_jump * self._x_global

        self.g_u2_fun.x.array[:] = 0.0

        if self.opB.bdofs_owned.size > 0:
            self.g_u2_fun.x.array[self.opB.bdofs_owned] = (
                self._y_global[self.opB.bidx_owned]
            )

        self.g_u2_fun.x.scatter_forward()


    def solve_u1(self, m: fem.Function):
        self.Div.mult(m.x.petsc_vec, self.b_u1)

        try:
            self.ns_u1.remove(self.b_u1)
        except Exception:
            pass

        self.ksp_u1.solve(self.b_u1, self.u1_sol.x.petsc_vec)
        self.last_its_u1 = self.ksp_u1.getIterationNumber()

        try:
            self.ns_u1.remove(self.u1_sol.x.petsc_vec)
        except Exception:
            pass

        self.u1_sol.x.scatter_forward()
        return self.u1_sol



    def solve_u2(self):
        self.apply_B_to_u1_boundary()


        self.A_u2_raw.mult(self.g_u2, self.b_u2)
        self.b_u2.scale(-1.0)

        b_arr = self.b_u2.getArray(readonly=False)
        g_arr = self.g_u2_fun.x.array

        if self.boundary_dofs_owned.size > 0:
            b_arr[self.boundary_dofs_owned] = g_arr[self.boundary_dofs_owned]




        self.ksp_u2.solve(self.b_u2, self.u2_sol.x.petsc_vec)
        self.last_its_u2 = self.ksp_u2.getIterationNumber()

        self.u2_sol.x.scatter_forward()


        return self.u2_sol


    def compute(self, m: fem.Function):
        self.solve_u1(m)
        self.solve_u2()

        self.G.mult(self.u1_sol.x.petsc_vec, self.H_d.x.petsc_vec)
        self.G.multAdd(
            self.u2_sol.x.petsc_vec,
            self.H_d.x.petsc_vec,
            self.H_d.x.petsc_vec,
        )

        hd_loc = self.H_d.x.petsc_vec.array.reshape(-1, 3)
        hd_loc *= self.inv_vol[:, None]

        self.H_d.x.scatter_forward()
        return self.H_d

    def Energy(self, m: fem.Function):
        """
        Return local demagnetizing energy contribution on this MPI rank.

        The global reduction is intentionally not done here because the LLG monitor
        gathers and sums all local energy contributions.
        """
        self.H_d.x.scatter_forward()
        dE = ufl.inner(m, self.H_d) * ufl.dx(domain=self.mesh)
        local = fem.assemble_scalar(fem.form(dE))

        return -0.5 * self.mu0 * self.Ms * local / self.volume_scale


    def Energy_global(self, m: fem.Function):
        """
        Return globally reduced demagnetizing energy.

        """
        E_local = self.Energy(m)
        return self.comm.allreduce(E_local, op=MPI.SUM)


    # Validation

    def validate_htool_against_dense(self, seed=1234):

        rng = np.random.default_rng(seed + self.rank)

        # Same x on all ranks for deterministic comparison.
        if self.rank == 0:
            x = rng.standard_normal(self.Nb).astype(np.float64)
        else:
            x = np.empty(self.Nb, dtype=np.float64)
        self.comm.Bcast(x, root=0)

        # Set x into distributed PETSc Vec.
        self.xb_vec.set(0.0)
        lo, hi = self.xb_vec.getOwnershipRange()
        local_idx = np.arange(lo, hi, dtype=PETSc.IntType)
        self.xb_vec.setValues(local_idx, x[lo:hi], addv=PETSc.InsertMode.INSERT_VALUES)
        self.xb_vec.assemble()

        self.B_htool.mult(self.xb_vec, self.yb_vec)
        self._gather_yb_global()
        self._y_global += self.opB.diag_jump * x

        # Dense local rows for this rank's geometric row set.
        Bloc_local = assemble_B_rows_owned_dense(
            self.opB.Xb,
            self.opB.tri_lid,
            self.opB.tri_pts,
            self.opB.ntri,
            self.opB.area,
            self.opB.s_edge,
            self.opB.eta,
            self.opB.gamma,
            self.opB.omega_sum,
            self.opB.rows,
            self.opB.incident_policy,
        )

        y_dense_rows = Bloc_local @ x
        y_htool_rows = self._y_global[self.opB.rows]

        num_local = np.linalg.norm(y_dense_rows - y_htool_rows) ** 2
        den_local = np.linalg.norm(y_dense_rows) ** 2

        num = self.comm.allreduce(num_local, op=MPI.SUM)
        den = self.comm.allreduce(den_local, op=MPI.SUM)

        relerr = math.sqrt(num) / max(math.sqrt(den), 1e-300)

        if self.rank == 0:
            print(f"[HTool validation] relative error vs dense owned rows: {relerr:.12e}", flush=True)

        return relerr





# Main

def parse_args():
    parser = argparse.ArgumentParser(
        description="CPU MPI demag field with Lindholm double-layer operator compressed by PETSc MATHTOOL."
    )

    parser.add_argument("--mesh", required=True)
    parser.add_argument("--mesh-name", default="Grid")
    parser.add_argument(
        "--serial-mesh",
        default=None,
        help=(
            "Mesh file read by rank 0 with MPI.COMM_SELF to build boundary data. "
            "Defaults to --mesh. Passing this avoids fragile parallel temp mesh writes."
        ),
    )
    parser.add_argument("--ms", type=float, default=1.0)
    parser.add_argument("--volume-scale", type=float, default=1e27)

    parser.add_argument("--m0x", type=float, default=0.0)
    parser.add_argument("--m0y", type=float, default=0.0)
    parser.add_argument("--m0z", type=float, default=1.0)

    parser.add_argument("--map-tol", type=float, default=1e-10)
    parser.add_argument("--ksp-rtol-u1", type=float, default=1e-6)
    parser.add_argument("--ksp-rtol-u2", type=float, default=1e-6)
    parser.add_argument("--incident-policy", default="skip", choices=["skip", "include"])

    parser.add_argument("--htool-epsilon", type=float, default=1e-6)
    parser.add_argument("--htool-eta", type=float, default=2.0)
    parser.add_argument("--htool-max-leaf-size", type=int, default=64)

    parser.add_argument("--cache-dir", default=None)  # kept for CLI compatibility; unused here
    parser.add_argument("--cache-tag", default=None)  # kept for CLI compatibility; unused here

    parser.add_argument("--validate-htool", action="store_true")
    parser.add_argument("--view-htool", action="store_true")
    parser.add_argument("--timing-repeats", type=int, default=3)

    return parser.parse_args(APP_ARGV[1:])


def main():
    from time import perf_counter

    comm = MPI.COMM_WORLD
    rank = comm.rank

    with dolfinx.io.XDMFFile(comm, "Cylinder.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
    V1 = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    m = fem.Function(V)

    npts = mesh.geometry.x.shape[0]
    m0 = np.zeros((npts, 3), dtype=np.float64)
    m0[:, 2] = 1.0
    m.x.array[:] = m0.ravel()
    m.x.scatter_forward()

    t0 = perf_counter()
    demag = DemagField_Lindholm_HTool_MPI(
        mesh,
        V,
        V1,
        Ms=1.0,
        volume_scale=1e27,
    )
    comm.Barrier()
    init_time = perf_counter() - t0

    if rank == 0:
        print("Initialization time:", init_time)

    comm.Barrier()
    t0 = perf_counter()
    demag.compute(m)
    comm.Barrier()
    first_time = perf_counter() - t0

    if rank == 0:
        print("First compute time:", first_time)
        if hasattr(demag, "last_its_u1"):
            print("u1 iterations:", demag.last_its_u1)
        if hasattr(demag, "last_its_u2"):
            print("u2 iterations:", demag.last_its_u2)

    comm.Barrier()
    t0 = perf_counter()
    for _ in range(3):
        demag.compute(m)
    comm.Barrier()
    loop_time = perf_counter() - t0

    hdsol = demag.H_d
    hdsol.x.scatter_forward()

    E_total = demag.Energy(m)

    owned = V.dofmap.index_map_bs * V.dofmap.index_map.size_local
    H_owned = hdsol.x.array[:owned].reshape((-1, 3))

    local_sum = H_owned.sum(axis=0)
    local_min = H_owned.min(axis=0) if H_owned.size else np.array([np.inf, np.inf, np.inf])
    local_max = H_owned.max(axis=0) if H_owned.size else np.array([-np.inf, -np.inf, -np.inf])
    local_l2 = float(np.sum(H_owned * H_owned))
    local_count = H_owned.shape[0]

    global_sum = np.zeros(3, dtype=np.float64)
    global_min = np.zeros(3, dtype=np.float64)
    global_max = np.zeros(3, dtype=np.float64)

    comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
    comm.Allreduce(local_min, global_min, op=MPI.MIN)
    comm.Allreduce(local_max, global_max, op=MPI.MAX)

    global_l2 = comm.allreduce(local_l2, op=MPI.SUM)
    global_count = comm.allreduce(local_count, op=MPI.SUM)

    avg = global_sum / max(global_count, 1)

    if rank == 0:
        print("Total time for 3 compute calls:", loop_time)
        print("Average compute time:", loop_time / 3.0)
        print("Demag energy:", E_total)
        print("Average field:", avg)
        print("Field minima:  ", global_min)
        print("Field maxima:  ", global_max)
        print("Field L2 norm: ", math.sqrt(global_l2))


if __name__ == "__main__":
    main()
