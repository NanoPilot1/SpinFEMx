import os
import math
import uuid
from pathlib import Path

import numpy as np
from mpi4py import MPI

import dolfinx
from dolfinx import fem, io
try:
    from dolfinx.mesh import exterior_facet_indices
except Exception:
    from dolfinx.mesh import exterior_facet_indices  # type: ignore

from dolfinx.fem.petsc import apply_lifting, set_bc
import ufl
from petsc4py import PETSc

from scipy.spatial import cKDTree
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from numba import njit, prange

try:
    from .Lindholm_kernel import lindholm_weights_precomp, solid_angle_with_atan2
except ImportError:
    from Lindholm_kernel import lindholm_weights_precomp, solid_angle_with_atan2

# -----------------------------------------------------------------------------
# Patch XDMFFile so meshes remember the file they came from
# -----------------------------------------------------------------------------
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

# This implementation follows the analytic formulation of the double-layer operator by Lindholm (1984). Results were validated by comparing demagnetizing energy against MuMax3/nmag.
# In particular, we use the section 2.2.4 of the thesis tittled Micromagnetic simulations of three dimensional core-shell nanostructures by A. Knittel


# -----------------------------------------------------------------------------
# Thread defaults (safe)
# -----------------------------------------------------------------------------
def _enforce_thread_defaults():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")


# -----------------------------------------------------------------------------
# NUMBA: assembly of owned rows of B (dense)
# -----------------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def assemble_B_rows_owned(
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

    # diagonal: c_i - 1  (c_i = Omega_i/4pi)
    for ii in prange(nrows):
        gi = rows[ii]
        Bloc[ii, gi] = omega_sum[gi] * inv4pi - 1.0

    # off-diagonal
    for ii in prange(nrows):
        gi = rows[ii]
        x0 = Xb[gi]

        for t in range(Nt):
            a = tri_lid[t, 0]
            b = tri_lid[t, 1]
            c = tri_lid[t, 2]

            # Skip "self" panels: triangles incident to the observation vertex gi.
            # These require a principal-value (singular) treatment. This implementation does not
            # evaluate the finite-part of those incident panels; we only add the jump term via
            # the solid-angle correction (Omega/(4*pi) - 1).
            # For typical micromagnetic meshes (often h <= l_ex) this omission is usually small,
            # but can be inaccurate on coarse meshes or near non-smooth boundary features.

            if incident_policy == 0:
                if gi == a or gi == b or gi == c:
                    continue

            p0 = tri_pts[t, 0]
            p1 = tri_pts[t, 1]
            p2 = tri_pts[t, 2]

            w0, w1, w2 = lindholm_weights_precomp(
                x0,
                p0, p1, p2,
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


# -----------------------------------------------------------------------------
# MPI: write mesh in disk ( ADIOS2 .bp)
# -----------------------------------------------------------------------------
def ensure_serial_meshfile_from_parallel_mesh(mesh_par, out_dir=None, prefer="bp"):
    comm = mesh_par.comm
    rank = comm.rank

    if out_dir is None:
        out_dir = os.environ.get("DEMAG_TMP_DIR", "/tmp")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = os.environ.get("DEMAG_SERIAL_TAG", None)
    if tag is None:
        tag = uuid.uuid4().hex

    # Attempt 1: ADIOS2 (.bp)
    if prefer.lower() == "bp":
        try:
            import adios4dolfinx as ad
            bp_path = out_dir / f"__demag_serial_{tag}.bp"
            try:
                ad.write_mesh(mesh_par, str(bp_path))
            except Exception:
                ad.write_mesh(str(bp_path), mesh_par)
            comm.Barrier()
            return str(bp_path)
        except Exception as e:
            if rank == 0:
                print(f"[WARN] ADIOS write_mesh failed: {e}", flush=True)
            comm.Barrier()

    # Attempt 2: XDMF/HDF5
    xdmf_path = out_dir / f"__demag_serial_{tag}.xdmf"
    with io.XDMFFile(comm, str(xdmf_path), "w") as xdmf:
        xdmf.write_mesh(mesh_par)
    comm.Barrier()
    return str(xdmf_path)


def _read_serial_mesh_any_format(meshfile):
    meshfile = str(meshfile)
    suffix = Path(meshfile).suffix.lower()

    if suffix == ".bp":
        import adios4dolfinx as ad
        return ad.read_mesh(Path(meshfile), MPI.COMM_SELF)

    with io.XDMFFile(MPI.COMM_SELF, meshfile, "r") as xdmf:
        try:
            return xdmf.read_mesh(name="Grid")
        except Exception:
            return xdmf.read_mesh()


# -----------------------------------------------------------------------------
# Serial: build edge data (geom + omega_sum)
# -----------------------------------------------------------------------------
def build_boundary_data_serial(mesh_ser):
    tdim = mesh_ser.topology.dim
    fdim = tdim - 1

    mesh_ser.topology.create_connectivity(fdim, 0)    # facet->vertex
    mesh_ser.topology.create_connectivity(fdim, tdim) # facet->cell
    mesh_ser.topology.create_connectivity(tdim, 0)    # cell->vertex

    f2v = mesh_ser.topology.connectivity(fdim, 0)
    f2c = mesh_ser.topology.connectivity(fdim, tdim)
    c2v = mesh_ser.topology.connectivity(tdim, 0)

    X = mesh_ser.geometry.x
    bfacets = exterior_facet_indices(mesh_ser.topology)

    tris = []
    normals = []

    # Outward-facing triangles (outward normal)
    for f in bfacets:
        vs = f2v.links(f)
        v0, v1, v2 = int(vs[0]), int(vs[1]), int(vs[2])
        p0, p1, p2 = X[v0], X[v1], X[v2]

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
            p1, p2 = X[v1], X[v2]
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

    # area of each triangle
    cross = np.cross(tri_pts[:, 1, :] - tri_pts[:, 0, :], tri_pts[:, 2, :] - tri_pts[:, 0, :])
    area = (0.5 * np.linalg.norm(cross, axis=1)).astype(np.float64)

    # Edges
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

    # eta_i = n x xi_i
    eta = np.zeros_like(xi)
    eta[:, 0, :] = np.cross(ntri, xi[:, 0, :])
    eta[:, 1, :] = np.cross(ntri, xi[:, 1, :])
    eta[:, 2, :] = np.cross(ntri, xi[:, 2, :])

    # gamma_{ij} = xi_{i+1} @  xi_j
    gamma = np.zeros((xi.shape[0], 3, 3), dtype=np.float64)
    for i in range(3):
        ip1 = (i + 1) % 3
        for j in range(3):
            gamma[:, i, j] = np.einsum("ij,ij->i", xi[:, ip1, :], xi[:, j, :])

    # omega_sum per vertex from tetrahedra (always positive)
    mesh_ser.topology.create_connectivity(tdim, 0)
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


# -----------------------------------------------------------------------------
# Double-layer operator (B) on owned rows
# -----------------------------------------------------------------------------
class DoubleLayerDenseOwnedRowsMPI:
    def __init__(
        self,
        mesh_par,
        V1_par,
        serial_mesh_path=None,
        cache_dir=None,
        cache_prefix="Bcache_lindholm_ownedrows",
        cache_tag=None,
        map_tol=1e-10,
        prefer="bp",
        incident_policy="skip",
    ):
        self.mesh = mesh_par
        self.V1_par = V1_par
        self.comm = mesh_par.comm
        self.rank = self.comm.rank

        self.map_tol = float(map_tol)
        self.cache_prefix = str(cache_prefix)
        self.incident_policy = 0 if str(incident_policy).lower().startswith("skip") else 1

        if cache_dir is None:
            cache_dir = os.environ.get("DEMAG_CACHE_DIR", os.environ.get("DEMAG_TMP_DIR", "/tmp"))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if serial_mesh_path is None:
            serial_mesh_path = os.environ.get("DEMAG_SERIAL_MESH_PATH", None)

        if serial_mesh_path is None:
            serial_mesh_path = getattr(mesh_par, "_serial_mesh_path", None)

        if serial_mesh_path is None:
            if self.comm.size > 1:
                raise RuntimeError(
                    "serial_mesh_path is required when running with MPI. "
                    "Either pass serial_mesh_path=..., set DEMAG_SERIAL_MESH_PATH, "
                    "or attach mesh._serial_mesh_path before constructing DemagField_Lindholm. "
                    "Automatic parallel-to-serial mesh writing is fragile and may hang."
                )

            serial_mesh_path = ensure_serial_meshfile_from_parallel_mesh(
                mesh_par,
                prefer=prefer,
            )

        self.serial_mesh_path = str(serial_mesh_path)

        if cache_tag is None:
            cache_tag = Path(self.serial_mesh_path).stem
        self.cache_tag = str(cache_tag)

        # rank0: read full serial mesh and build data
        if self.rank == 0:
            mesh_ser = _read_serial_mesh_any_format(self.serial_mesh_path)
            data = build_boundary_data_serial(mesh_ser)
        else:
            data = None

        self._bcast_boundary_data(data)

        # map owned edge DOFs -> global edge index
        self._build_parallel_boundary_maps()

        self.rows = np.unique(self.bidx_owned).astype(np.int32)
        self.nrows = int(self.rows.size)

        self.rowpos_for_bdof = np.searchsorted(self.rows, self.bidx_owned).astype(np.int32)

        # detect collisions (one-time)
        cloc = np.zeros(self.Nb, dtype=np.int32)
        np.add.at(cloc, self.bidx_owned, 1)
        self._count_global = np.empty_like(cloc)
        self.comm.Allreduce(cloc, self._count_global, op=MPI.SUM)
        self.has_collisions = not (int(self._count_global.min()) == 1 and int(self._count_global.max()) == 1)

        if self.rank == 0:
            cmin, cmax = int(self._count_global.min()), int(self._count_global.max())
            print(f"[Lindholm method] Nb={self.Nb}, Nt={self.Nt}, count_global min/max={cmin}/{cmax}", flush=True)

        # buffers 
        self._x = np.zeros(self.Nb, dtype=np.float64)
        self._y_rows = np.empty(self.nrows, dtype=np.float64)

        self.Bloc = None

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
            print(f"[WARN] dmax mapping DOF_edge->Xb = {dmax_g:g} > tol={self.map_tol:g}", flush=True)

        self.bidx_owned = idx.astype(np.int32)

    def assemble(self, force=False):
        fn = self.cache_dir / f"{self.cache_prefix}_{self.cache_tag}_rank{self.rank}_Nb{self.Nb}_nrows{self.nrows}.npy"

        if (not force) and fn.exists():
            self.Bloc = np.load(str(fn))
            if self.Bloc.shape != (self.nrows, self.Nb):
                raise RuntimeError(f"Cache Block unexpected shape: {self.Bloc.shape} vs {(self.nrows, self.Nb)}")
            self.comm.Barrier()
            if self.rank == 0:
                print("[Lindholm method] Block loaded from cache.", flush=True)
            return self.Bloc

        self.Bloc = assemble_B_rows_owned(
            self.Xb,
            self.tri_lid,
            self.tri_pts,
            self.ntri,
            self.area,
            self.s_edge,
            self.eta,
            self.gamma,
            self.omega_sum,
            self.rows,
            self.incident_policy,
        )
        np.save(str(fn), self.Bloc)
        self.comm.Barrier()
        if self.rank == 0:
            print("[Lindholm method] Owned-rows assembly completed and cached.", flush=True)
        return self.Bloc

    def apply_from_u1_to_boundary_function(self, u1_fun: fem.Function, u2_on_boundary_parallel: fem.Function):
        if self.Bloc is None:
            raise RuntimeError("First call opB.assemble().")

        # x_global in Nb
        self._x.fill(0.0)
        vals = u1_fun.x.array[self.bdofs_owned].astype(np.float64, copy=False)
        self._x[self.bidx_owned] = vals

        self.comm.Allreduce(MPI.IN_PLACE, self._x, op=MPI.SUM)

        if self.has_collisions:
            if np.any(self._count_global == 0):
                raise RuntimeError("count_global has zeros: inconsistent edge mapping.")
            self._x /= self._count_global.astype(np.float64)

        # y_rows = Bloc @ x
        np.matmul(self.Bloc, self._x, out=self._y_rows)

        # set u2 to owned edge DOFs
        u2_on_boundary_parallel.x.array[self.bdofs_owned] = self._y_rows[self.rowpos_for_bdof]
        u2_on_boundary_parallel.x.scatter_forward()

        return self._y_rows


# -----------------------------------------------------------------------------
# Nullspace
# -----------------------------------------------------------------------------
def _cell_components(mesh):
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(tdim, fdim)
    mesh.topology.create_connectivity(fdim, tdim)

    c2f = mesh.topology.connectivity(tdim, fdim)
    f2c = mesh.topology.connectivity(fdim, tdim)

    imap = mesh.topology.index_map(tdim)
    ncell = imap.size_local + imap.num_ghosts

    rows, cols = [], []
    for c in range(ncell):
        for f in c2f.links(c):
            for c2 in f2c.links(f):
                if c2 != c:
                    rows.append(c)
                    cols.append(c2)

    A = sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(ncell, ncell))
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
        if nrm > 0:
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


# -----------------------------------------------------------------------------
# Demag field (FEM + BEM)
# -----------------------------------------------------------------------------
class DemagField_Lindholm:
    def __init__(
        self,
        mesh,
        V,
        V1,
        Ms,
        serial_mesh_path=None,
        prefer_serial_write="bp",
        cache_dir=None,
        cache_prefix="Bcache_lindholm_ownedrows",
        cache_tag=None,
        map_tol=1e-10,
        ksp_rtol_u1=1e-6,
        ksp_rtol_u2=1e-6,
        incident_policy="skip",
        volume_scale=1e27,  # our mesh is in meters
    ):
        _enforce_thread_defaults()

        self.mesh = mesh
        self.V = V
        self.V1 = V1
        self.Ms = float(Ms)
        self.mu0 = 4.0 * math.pi * 1e-7
        self.comm = mesh.comm
        self.rank = self.comm.rank
        self.volume_scale = float(volume_scale)



        tdim = mesh.topology.dim
        fdim = tdim - 1

        # ---------------- u1 ----------------
        self.u1_sol = fem.Function(V1)

        imap = self.V1.dofmap.index_map
        bs = self.V1.dofmap.index_map_bs

        self.n_u1_dofs_global = imap.size_global * bs

        

        v = ufl.TestFunction(V1)
        u = ufl.TrialFunction(V1)
        a_u1 = fem.form(ufl.inner(ufl.grad(v), ufl.grad(u)) * ufl.dx)
        self.A_u1 = fem.petsc.assemble_matrix(a_u1)
        self.A_u1.assemble()

        self.ns_u1 = build_nullspace_per_component(mesh, V1, self.A_u1)

        self.ksp_u1 = PETSc.KSP().create(self.comm)
        self.ksp_u1.setType("cg")
        self.ksp_u1.setOperators(self.A_u1)
        self.ksp_u1.setTolerances(rtol=ksp_rtol_u1, atol=1e-12, max_it=2000)

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
        Div_form = self.Ms * ufl.inner(ufl.grad(v2), m_trial) * ufl.dx
        self.Div = fem.petsc.assemble_matrix(fem.form(Div_form))
        self.Div.assemble()
        self.b_u1 = self.Div.createVecLeft()

        # ---------------- u2 ----------------
        # ---------------- u2 ----------------
        self.u2_on_boundary_parallel = fem.Function(V1, name="g_u2")
        self.u2_sol = fem.Function(V1)

        mesh.topology.create_connectivity(fdim, tdim)
        mesh.topology.create_connectivity(fdim, 0)

        boundary_facets = exterior_facet_indices(mesh.topology)
        boundary_dofs = fem.locate_dofs_topological(V1, fdim, boundary_facets)
        boundary_dofs = np.unique(boundary_dofs).astype(np.int32)

        self.boundary_dofs = boundary_dofs

        imap1 = V1.dofmap.index_map
        nloc1 = imap1.size_local
        self.boundary_dofs_owned = boundary_dofs[boundary_dofs < nloc1].astype(np.int32)

        zero = fem.Function(V1)
        zero.x.array[:] = 0.0
        zero.x.scatter_forward()

        bc_zero = fem.dirichletbc(zero, self.boundary_dofs)

        uu = ufl.TrialFunction(V1)
        vv = ufl.TestFunction(V1)

        self._a_u2_form = fem.form(ufl.inner(ufl.grad(uu), ufl.grad(vv)) * ufl.dx)

        self.A_u2_raw = fem.petsc.assemble_matrix(self._a_u2_form)
        self.A_u2_raw.assemble()

        self.A_u2 = fem.petsc.assemble_matrix(self._a_u2_form, bcs=[bc_zero])
        self.A_u2.assemble()

        self.b_u2 = self.A_u2.createVecLeft()
        self.g_u2 = self.u2_on_boundary_parallel.x.petsc_vec

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


        # ---------------- B ----------------
        self.opB = DoubleLayerDenseOwnedRowsMPI(
            mesh_par=mesh,
            V1_par=V1,
            serial_mesh_path=serial_mesh_path,
            cache_dir=cache_dir,
            cache_prefix=cache_prefix,
            cache_tag=cache_tag,
            map_tol=map_tol,
            prefer=prefer_serial_write,
            incident_policy=incident_policy,
        )
        self.opB.assemble(force=False)

        # ---------------- Hd = -grad(u1+u2) ----------------
        self.H_d = fem.Function(V)

        one = fem.Function(V1)
        one.x.array.fill(1.0)
        vsc = ufl.TestFunction(V1)
        mass_form = fem.form(one * vsc * ufl.dx)

        vol_nodes = fem.Function(V1)
        fem.petsc.assemble_vector(vol_nodes.x.petsc_vec, mass_form)
        vol_nodes.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        vol_nodes.x.scatter_forward()

        nloc = V1.dofmap.index_map.size_local
        self.inv_vol = (1.0 / vol_nodes.x.array[:nloc]).astype(np.float64)

        vvec = ufl.TestFunction(V)
        phi = ufl.TrialFunction(V1)
        G_form = fem.form(ufl.inner(vvec, -ufl.grad(phi)) * ufl.dx)
        self.G = fem.petsc.assemble_matrix(G_form)
        self.G.assemble()

    def solve_u1(self, m: fem.Function):
        self.Div.mult(m.x.petsc_vec, self.b_u1)
        try:
            self.ns_u1.remove(self.b_u1)
        except Exception:
            pass

        self.ksp_u1.solve(self.b_u1, self.u1_sol.x.petsc_vec)

        try:
            self.ns_u1.remove(self.u1_sol.x.petsc_vec)
        except Exception:
            pass

        return self.u1_sol

    def solve_u2(self):

        self.opB.apply_from_u1_to_boundary_function(
            self.u1_sol,
            self.u2_on_boundary_parallel,
        )

        self.u2_on_boundary_parallel.x.scatter_forward()

        # b = -A_raw * g
        self.A_u2_raw.mult(self.g_u2, self.b_u2)
        self.b_u2.scale(-1.0)

        b_arr = self.b_u2.getArray(readonly=False)
        g_arr = self.u2_on_boundary_parallel.x.array

        if self.boundary_dofs_owned.size > 0:
            b_arr[self.boundary_dofs_owned] = g_arr[self.boundary_dofs_owned]

        self.ksp_u2.solve(self.b_u2, self.u2_sol.x.petsc_vec)
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

        return self.H_d

    def Energy(self, m: fem.Function):
        self.H_d.x.scatter_forward()
        dE = ufl.inner(m, self.H_d) * ufl.dx(domain=self.mesh)
        energy = fem.assemble_scalar(fem.form(dE))

        return -0.5 * self.mu0 * self.Ms * energy / self.volume_scale



if __name__ == "__main__":
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
    demag = DemagField_Lindholm(
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

    comm.Barrier()
    t0 = perf_counter()
    for _ in range(3):
        demag.compute(m)
    comm.Barrier()
    loop_time = perf_counter() - t0

    hdsol = demag.H_d
    hdsol.x.scatter_forward()

    E_local = demag.Energy(m)
    E_total = comm.allreduce(E_local, op=MPI.SUM)

    owned = V.dofmap.index_map_bs * V.dofmap.index_map.size_local
    H_owned = hdsol.x.array[:owned].reshape((-1, 3))

    local_sum = H_owned.sum(axis=0)
    local_min = H_owned.min(axis=0)
    local_max = H_owned.max(axis=0)
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