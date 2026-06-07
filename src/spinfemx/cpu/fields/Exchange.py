import ufl
import numpy as np
from dolfinx import fem
from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_matrix


class ExchangeField:
    def __init__(self, mesh, V,  A, Ms, Nodal_V):
        """
        A: Exchange constant (J/m)
        Ms: is the  saturation magnetization in units of A/m.
        """
        self.mesh = mesh
        self.A = A
        self.M_s = Ms
        self.mu_0 = 4 * np.pi * 1e-7
        self.V =   V
        self.v = ufl.TestFunction(self.V)
 
        self.m_trial = ufl.TrialFunction(self.V)
        self.a = ufl.inner(ufl.grad(self.m_trial), ufl.grad(self.v)) * ufl.dx
        self.K = assemble_matrix(fem.form(self.a))
        self.K.assemble()

        self.H_exch = fem.Function(self.V)
        prefactor = fem.Function(self.V)
        prefactor.x.array[:] = -2 * self.A /Nodal_V[:]/ (self.mu_0 * self.M_s) / 1e-18
        self.K.diagonalScale(prefactor.x.petsc_vec, None)

        #.diagonalScale(self.exchange_field.prefactor, None)

    def compute(self, m):

        #self.H_exch.x.petsc_vec.set(0.0)
        self.K.mult(m.x.petsc_vec, self.H_exch.x.petsc_vec ) # K is local, we do not update the ghost in this part
        #self.H_exch.x.petsc_vec.ghostUpdate( addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)


        return self.H_exch

    def Energy(self, m):
        #dE = ufl.dot(m,self.H_exch ) * ufl.dx
        #energy =-1/2*self.mu_0 * self.M_s*fem.assemble_scalar(fem.form(dE))*1e-27
        energy = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(m), ufl.grad(m)) * ufl.dx))*self.A*1e-9

        return energy




