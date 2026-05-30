from mpi4py import MPI
import numpy as np

from pathlib import Path
from time import perf_counter

from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import apply_lifting, set_bc, create_vector
from petsc4py import PETSc

import ufl

from scipy.spatial import cKDTree
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from bempp_cl.api.external import fenicsx
import bempp_cl.api

from dolfinx import mesh as dmesh

from scipy.sparse.csgraph import connected_components





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

        z = A_u1.createVecRight()
        z.set(0.0)

        dofs = dofs[dofs < nloc]  
        z.array[dofs] = 1.0

        z.assemble()

        nrm = z.norm()
        if nrm > 0:
            z.scale(1.0 / nrm)

        z_vecs.append(z)

    ns = PETSc.NullSpace().create(vectors=z_vecs, comm=mesh.comm)

    assert ns.test(A_u1), "Nullspace fails test(A). Probable: non-global components in MPI."

    A_u1.setNullSpace(ns)
    A_u1.setNearNullSpace(ns)

    return ns




class DemagField_bempp:
    def __init__(self, domain_mesh, V, V1, Ms):
        self.mesh = domain_mesh
        self.mu0 = 4 * np.pi * 1e-7
        self.Ms = Ms

        self.V1 = V1
        self.V = V


        self.start, self.end = self.V.dofmap.index_map.local_range
        self.start_V1, self.end_V1 = self.V1.dofmap.index_map.local_range
        self.comm =  self.mesh.comm
        self.rank =  self.comm.Get_rank()

        with io.XDMFFile(self.mesh.comm, "tmp_mesh.xdmf", "w") as xdmf:
             xdmf.write_mesh(self.mesh)

        num_owned_nodes = self.mesh.geometry.index_map().size_local
        coords_owned =    self.mesh.geometry.x[:num_owned_nodes]
        coords_all =  self.mesh.comm.gather(coords_owned, root=0)

        Parallel_mesh =  self.mesh.comm.gather( (self.rank, self.mesh.geometry.x), root=0)

        '''
        In the current version of Bempp-cl, MPI parallelization is not supported. Therefore, in order to use it together with FEniCSx under MPI, the original mesh must be loaded with special care taken regarding index consistency.

        All operations related to Bempp will be executed on rank 0.

        The double-layer operator will be stored as a .npy file to allow reuse in subsequent simulations. All subsequent simulations must be run using the same number of MPI cores.

        '''
        if self.rank == 0:
 
            with io.XDMFFile(MPI.COMM_SELF, "tmp_mesh.xdmf", "r") as xdmf:
                try:
                    # Try to read the mesh name
                    serial_mesh = xdmf.read_mesh(name="Grid")
                except Exception:
                    # if empty, read the first avaible mesh
                    serial_mesh = xdmf.read_mesh()
            tdim = serial_mesh.topology.dim
            fdim = tdim - 1


            serial_mesh.topology.create_connectivity(fdim, 0)
            serial_mesh.topology.create_connectivity(fdim, tdim)

            print("Setting double-layer operator", flush=True)

            t0 = perf_counter()

            self.V1_serial = fem.functionspace(serial_mesh, ("Lagrange", 1))
            self.trace_space, self.trace_matrix = fenicsx.fenics_to_bempp_trace_data(self.V1_serial)
            self.trace_matrix = sp.csr_matrix(self.trace_matrix)
            self.bempp_space = self.trace_space

            bempp_cl.api.DEFAULT_PRECISION = "double"
            bempp_cl.api.DEFAULT_DEVICE_INTERFACE = "numba"
            bempp_cl.api.DEFAULT_DEVICE_TYPE = "cpu"

            bempp_cl.api.GLOBAL_PARAMETERS.quadrature.regular = 5
            bempp_cl.api.GLOBAL_PARAMETERS.quadrature.singular = 5

            
            #self.dlp = bempp_cl.api.operators.boundary.laplace.double_layer(self.bempp_space, self.bempp_space, self.bempp_space)

            self.boundary_facets = dmesh.exterior_facet_indices(serial_mesh.topology)
            self.boundary_dofs_serial = fem.locate_dofs_topological(self.V1_serial, serial_mesh.topology.dim - 1, self.boundary_facets)

            self.u1_serial = fem.Function(self.V1_serial)
            self.u1_on_boundary = fem.Function(self.V1_serial)

            #self.angles_per_point = compute_solid_angles(serial_mesh, self.boundary_dofs_serial)
            
            self.u1_boundary_vals = np.zeros(self.V1_serial.dofmap.index_map.size_local)
            self.u2_corrected_serial = np.zeros_like(self.u1_boundary_vals)



            self.trace_T        = self.trace_matrix.T.tocsr()
            Nbnd                = self.trace_matrix.shape[0]
            


            dlp_op = bempp_cl.api.operators.boundary.laplace.double_layer(self.trace_space, self.trace_space, self.trace_space,assembler="dense", device_interface="numba")
            mass_op = bempp_cl.api.operators.boundary.sparse.identity(self.trace_space, self.trace_space, self.trace_space)

            
            

            #if Path("dlp_dense.npy").exists():
            #    self.dlp_dense = np.load("dlp_dense.npy", mmap_mode=None)  # o mmap_mode="r"
            #else:
            A = dlp_op.weak_form().A
            self.dlp_dense = np.ascontiguousarray(A, dtype=np.float64)
            #np.save("dlp_dense.npy", self.dlp_dense)


            t1 = perf_counter()

            print(f"taking: {t1-t0:.2f} s", flush=True)


            M = mass_op.weak_form().A          
            M = M.tocsc()      


            if M.nnz == M.shape[0] and np.all(M.indices == np.arange(M.shape[0])):
                self.mass_is_diag = True
                self.mass_inv_diag = 1.0 / M.diagonal().astype(np.float64)

                #print("diagonal" , flush=True )
            else:
                self.mass_is_diag = False
                self.mass_splu = splu(M)        

                #print("sparse" , flush=True )

            t1 = perf_counter()


            self._tmp = np.empty(self.dlp_dense.shape[0], dtype=np.float64)

            self._phi_all = np.empty(len(self.u1_serial.x.array), dtype=np.float64)        
            self._phi_serial = np.empty(len(self.u1_serial.x.array), dtype=np.float64)

            

            coords_all = np.vstack(coords_all)
            coords_serial = self.V1_serial.tabulate_dof_coordinates()
            tree = cKDTree(coords_all)
            dists, self.indices = tree.query(coords_serial, workers=10)
            assert np.all(dists < 1e-12), "!The mesh are not equivalent!"



            serial_node = serial_mesh.geometry.x
            Parallel_mesh.sort(key=lambda x: x[0])
            Parallel_mesh = [x[1] for x in Parallel_mesh]
  
            index_map = np.empty(len(Parallel_mesh), dtype=object)

            tree = cKDTree(serial_node)
            self.index_map = np.empty(len(Parallel_mesh), dtype=object)

            for i in range(len(Parallel_mesh)):
                geo_i = Parallel_mesh[i]  # shape (M_i, 3)
                _, idxs = tree.query(geo_i, distance_upper_bound=1e-8)

                
                idxs[idxs >= len(serial_node)] = -1
                self.index_map[i] = idxs
                if idxs.any() >= len(serial_node):
                    print("error")


        else:

            self.index_map = None

        self.index_map = self.mesh.comm.bcast(self.index_map, root=0)




        if self.rank==0:
                self.u2_on_boundary = fem.Function(self.V1_serial)


        ###### u1-potential
        ###### u1-potential
        ###### u1-potential         
        self.u1_sol = fem.Function(self.V1)

        v1 = ufl.TestFunction(self.V1)
        u1 = ufl.TrialFunction(self.V1)
        a_u1 = fem.form(ufl.inner(ufl.grad(v1), ufl.grad(u1)) * ufl.dx)
        self.A_u1 = fem.petsc.assemble_matrix(a_u1)
        self.A_u1.assemble()

        self.ns_u1 = build_nullspace_per_component(self.mesh, self.V1, self.A_u1)



        self.A_u1.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)
        self.A_u1.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        self.A_u1.setOption(PETSc.Mat.Option.SPD, True)

        self.ksp_u1 = PETSc.KSP().create(self.mesh.comm)
        self.ksp_u1.setType("cg")
        self.ksp_u1.setOperators(self.A_u1)    
        self.ksp_u1.setTolerances(rtol=1e-6, atol=1e-6, max_it=1000)


        
        self.ksp_u1.setFromOptions()    
        pc = self.ksp_u1.getPC()
        pc.setType("gamg")
        pc.setReusePreconditioner(True)         


        v11    = ufl.TestFunction(self.V1)
        mvec  = ufl.TrialFunction(self.V)         

        

        Div_form = self.Ms * ufl.inner(ufl.grad(v11), mvec) * ufl.dx
        self.Div = fem.petsc.assemble_matrix(fem.form(Div_form))
        self.Div.assemble()

        self.b_u1 = self.Div.createVecLeft()  
        

        ###### u2-potential
        ###### u2-potential
        ###### u2-potential

        self.tdim = self.mesh.topology.dim
        self.fdim = self.tdim - 1
                 
        self.mesh.topology.create_connectivity(self.fdim, 0)
        self.mesh.topology.create_connectivity(self.fdim, self.tdim)

        boundary_facets_parallel = dmesh.exterior_facet_indices(self.mesh.topology)
        self.boundary_dofs_parallel = fem.locate_dofs_topological( self.V1, self.fdim, boundary_facets_parallel)


        self.u2_on_boundary_parallel = fem.Function(self.V1)


        self.uu = ufl.TrialFunction(self.V1 )
        self.vv = ufl.TestFunction(self.V1 )

        self.a_u2 = ufl.inner(ufl.grad(self.uu), ufl.grad(self.vv)) * ufl.dx
        self.L_u2 = fem.Constant(self.mesh, 0.0) * self.vv * ufl.dx  

        self.bc_u2 = fem.dirichletbc(self.u2_on_boundary_parallel, self.boundary_dofs_parallel)
        self.u2_sol = fem.Function(self.V1)

        ###### test
        ###### test
        ###### test

        a_u2_form = fem.form(self.a_u2)
        L_u2_form = fem.form(self.L_u2)

        self.A_u2 = fem.petsc.assemble_matrix(a_u2_form, bcs=[self.bc_u2])
        self.A_u2.assemble()

        self.b_u2 = create_vector(L_u2_form)

        self.ksp_u2 = PETSc.KSP().create(self.mesh.comm)
        self.ksp_u2.setType("cg")
        self.ksp_u2.setOperators(self.A_u2)
        self.ksp_u2.setTolerances(rtol=1e-6, atol=1e-6, max_it=10000)


        pc = self.ksp_u2.getPC()
        pc.setType("gamg")
        pc.setReusePreconditioner(True)   
        self.ksp_u2.setFromOptions()

        self._a_u2_form = a_u2_form
        self._L_u2_form = L_u2_form



        ######## Hd-field
        ######## Hd-field
        ######## Hd-field

        self.total_potential =   fem.Function(self.V1)
        self.H_d = fem.Function(self.V)

        self.vv = ufl.TestFunction(self.V)

        self.v_scalar = ufl.TestFunction(self.V1)
        self.one_scalar = fem.Function(self.V1)
        self.one_scalar.x.array.fill(1.0)

        mass_form = self.one_scalar * self.v_scalar * ufl.dx
        self.vol_nodes = fem.Function(self.V1)

        fem.petsc.assemble_vector(self.vol_nodes.x.petsc_vec, fem.form(mass_form))
        self.vol_nodes.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        self.vol_nodes.x.scatter_forward()
        self.vol = self.vol_nodes.x.array[:self.vol_nodes.function_space.dofmap.index_map.size_local]
        self.dim = self.mesh.geometry.dim
        self.inv_vol = (1.0 / self.vol).astype(np.float64) 


        v_vec   = ufl.TestFunction(self.V)
        phi_tri = ufl.TrialFunction(self.V1)
        G_form  = ufl.inner(v_vec, -ufl.grad(phi_tri)) * ufl.dx

        self.G = fem.petsc.assemble_matrix(fem.form(G_form))
        self.G.assemble()




    def solve_u1(self, m: fem.Function):
        self.Div.mult(m.x.petsc_vec, self.b_u1)
        self.ns_u1.remove(self.b_u1)

        self.ksp_u1.solve(self.b_u1, self.u1_sol.x.petsc_vec)

        self.ns_u1.remove(self.u1_sol.x.petsc_vec)

        self.u1_sol.x.scatter_forward()
        return self.u1_sol







    def solve_u2(self):
        comm = self.mesh.comm
        rank = self.rank

        phi_local =  self.u1_sol.x.array[:self.end_V1- self.start_V1]
        phi_vals_all =  self.mesh.comm.gather(phi_local, root=0)

        if self.rank == 0:



            pos = 0
            for arr in phi_vals_all:
                n = arr.size
                self._phi_all[pos:pos+n] = arr
                pos += n

            np.take(self._phi_all, self.indices, out=self._phi_serial)
            phi_serial_vals = self._phi_serial


            ###########################################################

            #u1_mean = np.mean(phi_serial_vals)

            #phi_serial_vals -= u1_mean
  
            u1_coeffs = self.trace_matrix.dot( phi_serial_vals )        

            np.matmul(self.dlp_dense, u1_coeffs, out=self._tmp)

            tmp = self._tmp  

            if self.mass_is_diag:
                u2_coeffs = self.mass_inv_diag * tmp
            else:
                u2_coeffs = self.mass_splu.solve(tmp)

            u2_fem = self.trace_T.dot(u2_coeffs)
            correction = -0.5 * phi_serial_vals[self.boundary_dofs_serial]

            
            self.u2_corrected_serial[:] = 0
            self.u2_corrected_serial[self.boundary_dofs_serial] = u2_fem[self.boundary_dofs_serial] + correction 

            u2_vals_parallel_ordered = self.u2_corrected_serial
        else:
            u2_vals_parallel_ordered = None
            #u1_mean =  None

        u2_vals_parallel_ordered = self.mesh.comm.bcast(u2_vals_parallel_ordered, root=0)
        #u1_mean = self.mesh.comm.bcast(u1_mean, root=0)

        #self.u1_sol.x.array[:] -= u1_mean
        #self.u1_sol.x.scatter_forward()



        self.u2_on_boundary_parallel.x.array[:] = 0
        self.u2_on_boundary_parallel.x.array[self.boundary_dofs_parallel] = u2_vals_parallel_ordered[self.index_map[self.rank]][self.boundary_dofs_parallel]

        self.u2_on_boundary_parallel.x.scatter_forward()
 

        with self.b_u2.localForm() as loc:
            loc.set(0.0)
        fem.petsc.assemble_vector(self.b_u2, self._L_u2_form)  

        apply_lifting(self.b_u2, [self._a_u2_form], bcs=[[self.bc_u2]])
        self.b_u2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b_u2, [self.bc_u2])

        self.ksp_u2.solve(self.b_u2, self.u2_sol.x.petsc_vec)
        self.u2_sol.x.scatter_forward()
        return self.u2_sol




    def compute(self, m):
        self.solve_u1(m)
        self.solve_u2()

        self.G.mult(self.u1_sol.x.petsc_vec, self.H_d.x.petsc_vec)
        self.G.multAdd(self.u2_sol.x.petsc_vec, self.H_d.x.petsc_vec, self.H_d.x.petsc_vec)

        hd_loc = self.H_d.x.petsc_vec.array.reshape(-1, 3)  

        hd_loc *= self.inv_vol[:, None]

        self.H_d.x.scatter_forward()
        return self.H_d


    def Energy(self, m):
        dx = ufl.dx(domain=self.mesh)  
        dE = ufl.inner(m, self.H_d) * dx
        energy = fem.assemble_scalar(fem.form(dE))
        return -0.5 * self.mu0 * self.Ms * energy / 1e27

