import ufl
import numpy as np
from dolfinx import fem
from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_matrix


"""
GPU exchange field for finite-element micromagnetics.

The exchange operator is assembled with DOLFINx on the host, scaled by the
lumped nodal volume, and converted to a PETSc AIJCUSPARSE matrix. The field
evaluation runs on a CUDA PETSc vector.

Conventions
-----------
- Mesh coordinates are assumed to be in nm.
- compute(m_gpu) expects a PETSc.Vec CUDA vector.
- Energy(m_fun) expects a dolfinx.fem.Function on the host.
"""


class ExchangeField:

    """
    Exchange effective field using a GPU PETSc sparse matrix.

    The field is computed as

        H_exch = 2A/(mu0 Ms) * laplacian(m),

    with the finite-element stiffness matrix scaled by the lumped nodal
    volume. Since the mesh coordinates are in nm, the field scaling includes
    the required unit conversion.


    A      : Exchange stifness J/m.
    Ms     : Saturation magnetization A/m.
    VolN   : Nodal volume nm^3.


    """


    def __init__(self, mesh, V, A, Ms, Nodal_V):

        self.mesh = mesh
        self.V = V
        self.A = float(A)
        self.M_s = float(Ms)
        self.mu_0 = 4.0 * np.pi * 1e-7

        v = ufl.TestFunction(V)
        u = ufl.TrialFunction(V)

        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

        K_cpu = assemble_matrix(fem.form(a))
        K_cpu.assemble()

        prefactor = fem.Function(V)
        prefactor.x.array[:] = (-2.0 * self.A / (self.mu_0 * self.M_s) / Nodal_V[:] / 1e-18)
        prefactor.x.scatter_forward()

        K_cpu.diagonalScale(prefactor.x.petsc_vec, None)

        self.K = K_cpu.convert(PETSc.Mat.Type.AIJCUSPARSE)
        self.K.bindToCPU(False)

        self.H_exch = fem.Function(V)

        self.h_gpu = self.K.createVecLeft()
        self.h_gpu.setType(PETSc.Vec.Type.CUDA)
        self.h_gpu.bindToCPU(False)

    def compute(self, m_gpu):
        """
        m_gpu should be PETSc.Vec CUDA.
        """
        self.K.mult(m_gpu, self.h_gpu)
        self.h_gpu.copy(self.H_exch.x.petsc_vec)

        return self.H_exch

    def Energy(self, m):
        """
        m should be fem.Function CPU/host.
        """
        dE= ufl.inner(ufl.grad(m), ufl.grad(m)) * ufl.dx
        energy = self.A * fem.assemble_scalar(fem.form(dE)) * 1e-9

        return float(energy)
