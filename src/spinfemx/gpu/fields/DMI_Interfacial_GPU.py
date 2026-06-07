import dolfinx
import ufl
import numpy as np
from dolfinx import fem
from petsc4py import PETSc
#
from dolfinx.fem.petsc import assemble_matrix, assemble_vector



class DMIInterfacial:

    """
    Interfacial DMI effective field.

    The effective field follows the interfacial DMI form

        H_DMI = -2D/(mu0 Ms) [ n div(m) - grad(m cdot n) ],

    where n is the fixed interface normal field. A boundary contribution is
    included in the weak form before converting the operator to a CUDA PETSc
    matrix.

    n0     : interface normal vector
    D      : DMi interfacial constant J/m^2
    Ms     : Saturation magnetization A/m.
    VolN   : Nodal volume nm^3.


    """

    def __init__(self, mesh, V, V1, D, n0, Ms, Nodal_V):

        self.mesh = mesh
        self.V = V
        self.V1 = V1
        self.D = float(D)
        self.Ms = float(Ms)
        self.mu_0 = 4.0 * np.pi * 1e-7
        self.dim = mesh.geometry.dim

        self.n = fem.Function(self.V)
        self.n.x.array[:] = n0
        self.n.x.scatter_forward()

        self.H_DMI = fem.Function(self.V)

        self.v = ufl.TestFunction(V)
        self.u = ufl.TrialFunction(V)

        n = self.n
        div_u = ufl.div(self.u)
        n_dot_u = ufl.dot(n, self.u)

        H_D = div_u * n - ufl.grad(n_dot_u)

        normal = ufl.FacetNormal(mesh)
        boundary_term = ufl.cross(ufl.cross(n, normal), self.u)

        Kform = (
            ufl.inner(H_D, self.v) * ufl.dx
            + 0.5 * ufl.inner(boundary_term, self.v) * ufl.ds
        )


        K_cpu = fem.petsc.assemble_matrix(fem.form(Kform), bcs=[])
        K_cpu.assemble()


        prefactor = fem.Function(self.V)
        prefactor.x.array[:] = (
            -2.0 * self.D / (self.mu_0 * self.Ms) / 1e-9 / Nodal_V[:]
        )
        prefactor.x.scatter_forward()

        K_cpu.diagonalScale(prefactor.x.petsc_vec, None)


        self.K = K_cpu.convert(PETSc.Mat.Type.AIJCUSPARSE)
        self.K.bindToCPU(False)

        self.m_gpu = self.K.createVecRight()
        self.h_gpu = self.K.createVecLeft()

        self.m_gpu.setType(PETSc.Vec.Type.CUDA)
        self.h_gpu.setType(PETSc.Vec.Type.CUDA)
        self.m_gpu.bindToCPU(False)
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
        energy = -0.5 * self.mu_0 * self.Ms * fem.assemble_scalar(fem.form(dE))
        return float(energy * 1e-27)
