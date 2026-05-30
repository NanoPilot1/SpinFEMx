import ufl
import numpy as np
from dolfinx import fem
from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_matrix


class ZhangLi:
    def __init__(self, mesh,  V,  Jdir, VolN):

        '''
         Jdir: Normalized vector

        '''

        self.mesh = mesh

        self.V =  V
        self.v = ufl.TestFunction(self.V)
        self.Jdir= fem.Function(self.V)

        self.Jdir.x.array[:] = Jdir[:]
        self.Jdir.x.scatter_forward()

        self.ZhangLi = fem.Function(self.V)

        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        grad_m = ufl.grad(self.u)

        directional_derivative = ufl.dot(ufl.grad(self.u), self.Jdir)


        advection_form = ufl.inner(directional_derivative, self.v) * ufl.dx
 
        self.K_J = fem.petsc.assemble_matrix(fem.form(advection_form))
        self.K_J.assemble()

        prefactor = fem.Function(self.V)
        prefactor.x.array[:] = 1 /VolN[:]
        self.K_J.diagonalScale(prefactor.x.petsc_vec, None)


    # --------------------------------------------------------------------------

    def compute(self, m):
        #self.ZhangLi.x.petsc_vec.set(0.0)
        self.K_J.mult(m.x.petsc_vec, self.ZhangLi.x.petsc_vec)
        #self.ZhangLi.x.petsc_vec.ghostUpdate( addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
        return self.ZhangLi

