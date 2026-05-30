import ufl
import numpy as np
from dolfinx import fem
from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_matrix

class DMIBULK:
    def __init__(self, mesh, V, V1, D, Ms,  Nodal_V):         
        mesh_in_nm = True
        self.mu0   = 4*np.pi*1e-7
        self.M_s = Ms
        self.D = D


        self.mesh, self.V = mesh, V
        self.v  = ufl.TestFunction(V)
        self.u  = ufl.TrialFunction(V)
        n       = ufl.FacetNormal(mesh)

        Kform   = (- ufl.inner(ufl.curl(self.u), self.v) * ufl.dx +0.5*ufl.inner(ufl.cross(n, self.u), self.v) * ufl.ds)
        self.K  = fem.petsc.assemble_matrix(fem.form(Kform), bcs=[])
        self.K.assemble()                                   

         
        self.H_DMI = fem.Function(V)

        prefactor = fem.Function(self.V)
        prefactor.x.array[:] = 2*self.D/(self.mu0*Ms) * 1e9 /Nodal_V[:]
        self.K.diagonalScale(prefactor.x.petsc_vec, None)



    def compute(self, m):
        #self.H_DMI.x.petsc_vec.set(0.0)
        self.K.mult(m.x.petsc_vec, self.H_DMI.x.petsc_vec ) # K is local, we do not update the ghost in this part
        #self.H_DMI.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,     mode=PETSc.ScatterMode.FORWARD)    
    
        return self.H_DMI



    def Energy(self, m):
        self.H_DMI.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,     mode=PETSc.ScatterMode.FORWARD)    
        dE = ufl.inner(m, self.H_DMI) * ufl.dx
        energy = -0.5 * self.mu0 * self.M_s *fem.assemble_scalar(fem.form(dE))
        #curl_m = ufl.curl(m)  
        #dE = self.D * ufl.inner(m, curl_m) * ufl.dx
        #E_dmi = fem.assemble_scalar(fem.form(dE))
        return energy  *1e-27

