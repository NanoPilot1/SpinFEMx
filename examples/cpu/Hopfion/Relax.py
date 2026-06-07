"""
Relaxation of a Hopfion in a cylinder.

Model:
  - Exchange
  - Bulk Dzyaloshinskii-Moriya interaction (DMI)
  - Uniaxial anisotropy (easy axis along +z)
  - Magnetostatics (demagnetizing field)

This script performs energy relaxation by disabling precession (do_precess = 0) and using a high damping value (alpha = 1.0)
for faster convergence.

We consider that the Hopfion is  stabilized in a disk with R = 100 nm and h = 100 nm.

Threading recommendations (avoid oversubscription when using MPI):

  export OMP_NUM_THREADS=1

Run:
  mpiexec -n 6 python Relax.py

"""


from mpi4py import MPI
import numpy as np
import dolfinx
from spinfemx.cpu import LLG, load_mesh_xdmf
from pathlib import Path
import adios4dolfinx as ad


comm = MPI.COMM_WORLD

# 1.  Load mesh (assumed to be in meters):

mesh = load_mesh_xdmf("Cylinder.xdmf", comm=MPI.COMM_WORLD,name="Grid",)

# 2.  Initial state.

xyz = mesh.geometry.x
dimension = mesh.geometry.x.shape[0]

mag_init = np.zeros((dimension, 3))
uaxis = np.zeros((dimension, 3))
alpha = np.pi/2
m0 = -1
u = 1
nu = 1
rmaj = 50
rmin = 45

for i in range(dimension):
    x, y, z = xyz[i]

    z = z-50

    rho = np.sqrt(z**2+(np.sqrt(x**2+y**2)-rmaj)**2)
    theta = np.arctan2(y,x)
    phi = np.arctan2(z,(np.sqrt(x**2+y**2)-rmaj))

    if rho <= rmin:
        mag_init[i,0] = np.sin(np.pi*rho/rmin) * np.cos(nu*theta - u*phi + alpha)
        mag_init[i,1] = np.sin(np.pi*rho/rmin) * np.sin(nu*theta - u*phi + alpha)
        mag_init[i,2] = m0*np.cos(np.pi*rho/rmin)
    else:
        mag_init[i,0] = 0.0
        mag_init[i,1] = 0.0
        mag_init[i,2] = -m0

    norma = np.sqrt(mag_init[i,0]**2 + mag_init[i,1]**2 + mag_init[i,2]**2)

    mag_init[i,0] = mag_init[i,0]/norma
    mag_init[i,1] = mag_init[i,1]/norma
    mag_init[i,2] = mag_init[i,2]/norma


    if z>49 or z<-49:
        uaxis[i] = [0,0,1] 
    else:
        uaxis[i] = [0,0,0] 
    

m0_array = mag_init.flatten()
uaxis = uaxis.flatten()


# 3. Material parameters (typical for FeGe)

Ms = 3.84e5
Aex = 8.78e-12
Kus = 1.0e7
Dbulk = 1.58e-3

# 4. Build solver and interactions

llg = LLG(mesh, Ms = Ms, gamma=2.211e5, alpha=1.0, do_precess=0)

llg.add_exchange(Aex=Aex)
llg.add_demag(method="lindholm")
llg.add_anisotropy(Kus, uaxis)
llg.add_dmi_bulk(Dbulk)
#llg.add_demag( method="htool",htool_epsilon=1e-7, htool_eta=2, htool_max_leaf_size=64, recompress_bool = 1)



# 5. Time stepping setup (relaxation with stopping criterion)

t0 = 0.0                  # Initial time of the simulation
t_final = 10e-9           # Final time of the simulation If the stopping_dmdt is not reached
dt_init = 1.0e-14         # Initial time step

dt_print = 2.e-12         # simulated-time interval between solver log outputs (monitoring)
dt_snap  = 1.0e-10         # simulated-time interval between saved magnetization snapshots (XDMF)

y, ctx, elapsed = llg.relax(
    m0_array,
    t0,
    t_final,
    dt_init,
    dt_save=dt_print,
    dt_snap=dt_snap,
    output_dir="relax",
    ts_rtol=1.0e-6,
    ts_atol=1.0e-6,
    save_final_state=True, # writes a checkpoint (.bp) for reuse in subsequent simulations
    stopping_dmdt=0.1,     # stopping criterion
    pc_python = False,     # Since the anisotropy is strongly localized in a small region, the current Python preconditioner produces poor performance.
)


if mesh.comm.rank == 0:
    print(f"Tiempo wall-clock ts.solve : {elapsed:.3f} s")

