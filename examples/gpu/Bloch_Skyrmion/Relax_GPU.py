"""
Relaxation of a Bloch skyrmion in a disk.

Model:
  - Exchange
  - Bulk Dzyaloshinskii-Moriya interaction (DMI)
  - Uniaxial anisotropy (easy axis along +z)
  - Magnetostatics (demagnetizing field)

This script performs energy relaxation by disabling precession (do_precess = 0) and using a high damping value (alpha = 1.0)
for faster convergence.

We consider that the bloch skyrmion is  stabilized in a disk with R = 50 nm and h = 4 nm.

"""

from mpi4py import MPI
import numpy as np
from spinfemx.gpu import LLG_GPU, load_mesh_xdmf


comm = MPI.COMM_WORLD

# 1.  Load mesh (assumed to be in meters):
    
mesh = load_mesh_xdmf("disk.xdmf",comm=MPI.COMM_WORLD,name="Grid",)

xyz = mesh.geometry.x
n = xyz.shape[0]

# 2.  Initial state.

m0 = np.zeros((n, 3))
uaxis = np.zeros((n, 3))

for i in range(n):
    x, y, z = xyz[i]
    if x**2 + y**2 <= 20*20:
        m0[i] = [0.0, 0.0, 1.0]
    else:
        m0[i] = [0,0,-1]

    m0[i] /= np.linalg.norm(m0[i])
    uaxis[i] = [0,0,1] 


m0_array = m0.flatten()
uaxis = uaxis.flatten()

# 3. Material parameters 

Ms = 3.84e5                 # A/m
Aex = 8.78e-12              # J/m
D_bulk = 1.5e-3             # J/m^2
Ku = 2.5e5                  # J/m^3

# 4. Build solver and interactions

LLG = LLG_GPU(mesh, Ms, gamma=2.211e5,alpha=1.0,)


LLG.add_exchange(Aex=Aex)
LLG.add_demag(method="lindholm_gpu")

LLG.add_anisotropy(Ku, uaxis)
LLG.add_dmi_bulk(D_bulk)
 
 # 5. Time stepping setup (relaxation with stopping criterion)

t0 = 0.0                  # Initial time of the simulation
t_final = 10e-9           # Final time of the simulation If the stopping_dmdt is not reached
dt_init = 1.0e-14         # Initial time step

dt_print = 1.e-10         # simulated-time interval between solver log outputs (monitoring)
dt_snap  = 1.0e-10         # simulated-time interval between saved magnetization snapshots (XDMF)



y, ctx, elapsed = LLG.relax(
    m0_array,
    t0,
    t_final,
    dt_init,
    dt_save=dt_print,
    dt_snap=dt_snap,
    output_dir="relax_gpu",
    ts_rtol=1.0e-6,
    ts_atol=1.0e-6,
    save_final_state=True,  # writes a checkpoint (.bp) for reuse in subsequent simulations
    stopping_dmdt=0.01,     # stopping criterion

)


if mesh.comm.rank == 0:
    print(f"Tiempo wall-clock ts.solve : {elapsed:.3f} s")
