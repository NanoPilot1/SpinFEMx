

import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc

"""
GPU bulk Dzyaloshinskii-Moriya interaction field.

The bulk DMI operator is assembled with DOLFINx, scaled by the lumped nodal
volume, and converted to a PETSc AIJCUSPARSE matrix for CUDA execution.

Conventions
-----------
- Mesh coordinates are assumed to be in nm.
- compute(m_gpu) expects a PETSc.Vec CUDA vector.
- Energy(m_fun) expects a dolfinx.fem.Function on the host.
"""



class DMIBULK:

    """
    Bulk DMI effective field.

    The weak form follows the bulk DMI operator

        Kform = -integrate curl(u) cdot v dx + 1/2 integrate (n times u) cdot v ds,

    and the resulting matrix is scaled to produce the effective field

        H_DMI = 2D/(mu0 Ms) * M_lump^{-1} K m.

    D      : DMI bulk constant J/m^2.
    Ms     : Saturation magnetization A/m.
    VolN   : Nodal volume nm^3.

    """


    def __init__(self, mesh, V, V1, D, Ms, Nodal_V):

        self.mesh = mesh
        self.V = V
        self.V1 = V1
        self.D = float(D)
        self.M_s = float(Ms)
        self.mu0 = 4.0 * np.pi * 1e-7

        self.v = ufl.TestFunction(V)
        self.u = ufl.TrialFunction(V)

        n = ufl.FacetNormal(mesh)
        Kform = (
            -ufl.inner(ufl.curl(self.u), self.v) * ufl.dx
            + 0.5 * ufl.inner(ufl.cross(n, self.u ), self.v) * ufl.ds
        )

        K_cpu = fem.petsc.assemble_matrix(fem.form(Kform), bcs=[])
        K_cpu.assemble()

        prefactor = fem.Function(self.V)
        prefactor.x.array[:] = (2.0 * self.D / (self.mu0 * self.M_s) * 1e9 / Nodal_V[:])


        K_cpu.diagonalScale(prefactor.x.petsc_vec, None)

        self.K = K_cpu.convert(PETSc.Mat.Type.AIJCUSPARSE)
        self.K.bindToCPU(False)

        self.H_DMI = fem.Function(self.V)

        self.h_gpu = self.K.createVecLeft()
        self.h_gpu.setType(PETSc.Vec.Type.CUDA)
        self.h_gpu.bindToCPU(False)

    def compute(self, m_gpu):
        """
        m_gpu should be PETSc.Vec CUDA.
        """
        self.K.mult(m_gpu, self.h_gpu)
        self.h_gpu.copy(self.H_DMI.x.petsc_vec)
        return self.H_DMI

    def Energy(self, m):
        """
        m should be fem.Function CPU/host.
        """
        dE = ufl.inner(m, self.H_DMI) * ufl.dx
        energy = - 0.5 * self.mu0 * self.M_s * fem.assemble_scalar(fem.form(dE))
        return float(energy * 1e-27)



