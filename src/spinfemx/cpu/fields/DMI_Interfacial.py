import dolfinx
import ufl
import numpy as np
from mpi4py import MPI
from dolfinx import fem
from petsc4py import PETSc
#
from dolfinx.fem.petsc import assemble_matrix, assemble_vector


class DMIInterfacial:



    def __init__(self, mesh, V, V1, D, n0, Ms, Nodal_V):
        self.mesh = mesh
        self.Ms = Ms
        self.mu_0 = 4 * np.pi * 1e-7
        self.D = D
        self.V = V
        self.V1 = V1
        self.dim = mesh.geometry.dim
        self.n = fem.Function(self.V)
        self.n.x.array[:] = n0

        # Buffers
        self.H_DMI = fem.Function(V)
        self.rhs_func = fem.Function(self.V)


        self.n0, self.n1, self.n2 = ufl.split(self.n) 

        self.v  = ufl.TestFunction(V)
        self.u  = ufl.TrialFunction(V)

        n      = self.n                       
        div_m  = ufl.div(self.u)              # \nabla \cdot u  
        n_dot_u = ufl.dot(n, self.u)          # n\cdot u


        H_D = div_m * n - ufl.grad(n_dot_u)

        #H_D = div_m * n - (ufl.dot(n, ufl.grad(self.u)) + ufl.dot(ufl.grad(n), self.u))



        normal = ufl.FacetNormal(mesh)

        # We include the boundary term
        boundary_term = ufl.cross(ufl.cross(n, normal), self.u)

        # Bilinear Form:
        Kform = ufl.inner(H_D, self.v) * ufl.dx + 0.5 * ufl.inner(boundary_term, self.v) * ufl.ds

        self.K  = fem.petsc.assemble_matrix(fem.form(Kform), bcs=[])
        self.K.assemble()  

        prefactor = fem.Function(self.V)
        prefactor.x.array[:] = -2 * self.D / (self.mu_0 * self.Ms)/1e-9 / Nodal_V[:]
        self.K.diagonalScale(prefactor.x.petsc_vec, None)


    def compute(self, m):

        self.K.mult(m.x.petsc_vec, self.H_DMI.x.petsc_vec)     # K is local, we do not update the ghost in this part
        #self.H_DMI.x.petsc_vec.ghostUpdate( addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
        return self.H_DMI


    def Energy(self, m):
        self.H_DMI.x.petsc_vec.ghostUpdate( addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
        dE = ufl.inner(m, self.H_DMI) * ufl.dx
        energy = -0.5 * self.mu_0 * self.Ms * fem.assemble_scalar(fem.form(dE)) * 1e-27
        return energy

