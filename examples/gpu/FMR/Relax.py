"""
Relaxation of in plane state in a disk.

Model:
  - Exchange
  - Magnetostatics (demagnetizing field)
  - Zeeman

This script performs energy relaxation by disabling precession (do_precess = 0) and using a high damping value (alpha = 1.0)
for faster convergence.

We consider that the uniform in plane state is stabilized in a disk of R = 50 nm and L = 4 nm

"""
from spinfemx.gpu import LLG_GPU, load_mesh_xdmf
from mpi4py import MPI
import dolfinx
import numpy as np

comm = MPI.COMM_WORLD

# 1.  Load mesh (assumed to be in meters):

mesh = load_mesh_xdmf("disk.xdmf")

# 2.  Initial state and constant external magnetic field along x-axis.

xyz = mesh.geometry.x
n = xyz.shape[0]

m0 = np.zeros((n, 3))
Hext = np.zeros((n, 3))


for i in range(n):
    m0[i] = [1,0,0]

    m0[i] /= np.linalg.norm(m0[i])

    Hext[i,0] = 0.1/(4*np.pi*1e-7)
    Hext[i,1] = 0.
    Hext[i,2] = 0.

Hext = Hext.flatten()

m0_array = m0.flatten()


# 3. Material parameters 

Ms = 8.0e5                 # A/m
Aex = 13.0e-12              # J/m

# 4. Build solver and interactions

llg = LLG_GPU(mesh, Ms = Ms, gamma=2.211e5, alpha=1.0, do_precess=0)

llg.add_exchange(Aex=Aex)
llg.add_demag(
    method="lindholm_gpu",
    boundary_backend="hmatrix", # or "dense"
    hmatrix_cache_path="./hmatrix_cache/disk_eps1e-5_eta2_leaf64.npz", # Reuse the same hmatrix for all the simulations, since the geometry is the same.
)
llg.add_external_field(H0_vec=Hext)

# 5. Time stepping setup (relaxation with stopping criterion)

t0 = 0.0                  # Initial time of the simulation
t_final = 5e-9            # Final time of the simulation If the stopping_dmdt is not reached
dt_init = 1.0e-14         # Initial time step

dt_print = 1.e-10          # simulated-time interval between solver log outputs (monitoring)
dt_snap  = 1.0e-9         # simulated-time interval between saved magnetization snapshots (XDMF)

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
    stopping_dmdt=0.01     # stopping criterion
)


if mesh.comm.rank == 0:
    print(f"time taken to converge ts.solve : {elapsed:.3f} s")

