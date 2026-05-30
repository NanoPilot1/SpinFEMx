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

Threading recommendations (avoid oversubscription when using MPI):

  export OMP_NUM_THREADS=1
  export NUMBA_NUM_THREADS=1   # affects numba-based kernels (e.g., bempp-cl)

Run:
  mpiexec -n 2 python Relax.py

"""
from fenicsx_micromagnetics.cpu import LLG, load_mesh_xdmf
from mpi4py import MPI
import dolfinx
import numpy as np

comm = MPI.COMM_WORLD



# 1.  Load mesh (assumed to be in meters):

mesh = load_mesh_xdmf("disk.xdmf")

# 2.  Initial state and anisotropy easy axis (considered along z-axis).

xyz = mesh.geometry.x
n = xyz.shape[0]

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

llg = LLG(mesh, Ms = Ms, gamma=2.211e5, alpha=1.0, do_precess=0)

llg.add_exchange(Aex=Aex)
llg.add_demag(method="lindholm")
llg.add_anisotropy(Ku, uaxis)
llg.add_dmi_bulk(D_bulk)

# 5. Time stepping setup (relaxation with stopping criterion)

t0 = 0.0                  # Initial time of the simulation
t_final = 5e-9            # Final time of the simulation If the stopping_dmdt is not reached
dt_init = 1.0e-14         # Initial time step

dt_print = 2.e-10          # simulated-time interval between solver log outputs (monitoring)
dt_snap  = 5.0e-10         # simulated-time interval between saved magnetization snapshots (XDMF)

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
    stopping_dmdt=0.01,     # stopping criterion
)


if mesh.comm.rank == 0:
    print(f"time taken to converge ts.solve : {elapsed:.3f} s")

