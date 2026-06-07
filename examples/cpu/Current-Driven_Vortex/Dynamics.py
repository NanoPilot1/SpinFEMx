"""
Current-driven vortex in a square film

This simulation is composed of two parts:

(i) Relaxation: We relax a vortex state in a nanopillar of 100 x 100 x 10 nm^3. The magnetic parameters are typical of permalloy.

(ii) Dynamics (this file): We use the relaxed state as input and include spin-transfer torque effects. In this case,
     we apply a constant current density of 1e12 A/m^2 along the x-axis. We assume a polarization of 1.0 and a
     non-adiabaticity parameter of 0.05.


Model:

  - Exchange
  - Magnetostatics (demagnetizing field)


This script performs energy relaxation by disabling precession (do_precess = 0) and using a high damping value (alpha = 1.0)
for faster convergence.

Threading recommendations:

  export OMP_NUM_THREADS=1
  export NUMBA_NUM_THREADS=4   # affects numba-based kernels (e.g., bempp-cl)

Run:

  mpiexec -n 2 python Dynamics.py

"""

from mpi4py import MPI
import numpy as np
import dolfinx
from spinfemx.cpu import LLG_STT
from pathlib import Path
import adios4dolfinx as ad



# 1.  Load mesh (assumed to be in meters):

fname = Path("relax/Relax.bp")
mesh  = ad.read_mesh(fname, MPI.COMM_WORLD)
mesh._serial_mesh_path = str(fname)

V     = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))

# 2.  Initial state.

m = dolfinx.fem.Function(V, name="m")
ad.read_function(fname, m, time=0.0, name="m")

m.x.scatter_forward()

m0_array = m.x.array


# 3.  Unit direction field for the current density

xyz = mesh.geometry.x
n= mesh.geometry.x.shape[0]

Je = np.zeros((n, 3))

for i in range(n):
    x = xyz[i,0]
    y = xyz[i,1]
    z = xyz[i,2]


    Je[i,0] = 1.0
    Je[i,1] = 0.0
    Je[i,2] = 0.0

Je_Array = Je.flatten()
# 4. Material parameters (typical for Permalloy)

Ms = 8.0e5
Aex = 13.0e-12

# 5. Build solver and interactions


llg = LLG_STT(mesh, Ms = Ms, gamma=2.211e5, alpha=0.1, do_precess=1)

llg.add_exchange(Aex=Aex )
llg.add_demag(method="lindholm")
#llg.add_demag(method="bempp")
#llg.add_demag(method="fmm")

llg.add_current(Jmagnitude=1e12, Jdir_vec=Je_Array , P=1.0, beta=0.05)

# 6. Time stepping setup (relaxation with stopping criterion)


t0 = 0.0                  # Initial time of the simulation
t_final = 8e-9            # Final time of the simulation If the stopping_dmdt is not reached
dt_init = 1.0e-14         # Initial time step

dt_print = 5.e-11          # simulated-time interval between solver log outputs (monitoring)
dt_snap  = 1.0e-10         # simulated-time interval between saved magnetization snapshots (XDMF)



y_final, ctx, elapsed = llg.solve(
    m0_array,
    t0,
    t_final,
    dt_init,
    dt_save=dt_print,
    dt_snap=dt_snap,
    output_dir="STT2",
    ts_rtol=1.0e-6,
    ts_atol=1.0e-6,
    monitor_fn=None
)
