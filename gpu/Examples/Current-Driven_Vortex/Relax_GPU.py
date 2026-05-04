"""
Current-driven vortex in a square film

This simulation is composed of two parts:

(i) Relaxation (this file): We relax a vortex state in a nanopillar of 100 x 100 x 10 nm^3. The magnetic parameters are typical of permalloy.

(ii) Dynamics: We use the relaxed state as input and include spin-transfer torque effects. In this case,
     we apply a constant current density of 1e12 A/m^2 along the x-axis. We assume a polarization of 1.0 and a
     non-adiabaticity parameter of 0.05.


Model:

  - Exchange
  - Magnetostatics (demagnetizing field)


This script performs energy relaxation by disabling precession (do_precess = 0) and using a high damping value (alpha = 1.0)
for faster convergence.

"""


from micromagnetic_gpu import LLG_GPU
from mpi4py import MPI
import dolfinx
import numpy as np
from pathlib import Path
import adios4dolfinx as ad


comm = MPI.COMM_WORLD

# 1.  Load mesh (assumed to be in meters):

Nx, Ny, Nz = 50, 50, 5
L, B, H = 100,100,10
mesh = dolfinx.mesh.create_box(MPI.COMM_WORLD, [np.array([-L/2,-B/2,-H/2]), np.array([L/2, B/2, H/2])], [Nx,Ny,Nz])

# 2.  Initial state.

xyz = mesh.geometry.x
n = xyz.shape[0]
m0 = np.zeros((n, 3))

r_core = 10

for i in range(n):
    x = xyz[i,0]
    y = xyz[i,1]
    z = xyz[i,2]

    if x*x+y*y<=r_core*r_core:

        m0[i,0] = 0
        m0[i,1] = 0
        m0[i,2] = 1.0   

    else:

        m0[i,0] = -y
        m0[i,1] = x
        m0[i,2] = 0 

    norma = np.sqrt(m0[i,0]*m0[i,0]+m0[i,1]*m0[i,1]+m0[i,2]*m0[i,2])

    m0[i,0] = m0[i,0]/norma
    m0[i,1] = m0[i,1]/norma
    m0[i,2] = m0[i,2]/norma




m0_array = m0.flatten()


# 3. Material parameters (typical for Permalloy)

Ms = 8.0e5
Aex = 13.0e-12

# 4. Build solver and interactions

llg = LLG_GPU(mesh, Ms = Ms, gamma=2.211e5, alpha=0.5, do_precess=0)

llg.add_exchange(Aex=Aex)
#llg.add_demag(method="lindholm_gpu")
llg.add_demag(method="fmm")

# 5. Time stepping setup (relaxation with stopping criterion)

t0 = 0.0                  # Initial time of the simulation
t_final = 2e-9            # Final time of the simulation If the stopping_dmdt is not reached
dt_init = 1.0e-14         # Initial time step

dt_print = 1.e-11          # simulated-time interval between solver log outputs (monitoring)
dt_snap  = 1.0e-10         # simulated-time interval between saved magnetization snapshots (XDMF)


y, ctx, elapsed = llg.relax(
    m0_array,
    t0,
    t_final,
    dt_init,
    dt_save=dt_print,
    dt_snap=dt_snap,
    output_dir="relax_fmm_gpu",
    #output_dir="relax_lindholm_gpu",
    ts_rtol=1.0e-6,
    ts_atol=1.0e-6,
    save_final_state=True, # writes a checkpoint (.bp) for reuse in subsequent simulations
    stopping_dmdt=0.1     # stopping criterion
)


if mesh.comm.rank == 0:
    print(f"Tiempo wall-clock ts.solve : {elapsed:.3f} s")
