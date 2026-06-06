"""
Relaxation of a Bloch skyrmion in a nanotube of external radius R = 50 nm and thickness t = 10 nm.

Model:

  - Exchange
  - Bulk Dzyaloshinskii-Moriya interaction (DMI)
  - Uniaxial anisotropy (easy axis along rho axis)
  - Magnetostatic interaction.

"""


from mpi4py import MPI
import numpy as np
import dolfinx
from dolfinx import fem
from fenicsx_micromagnetics.gpu import LLG_GPU, load_mesh_xdmf

comm = MPI.COMM_WORLD

  
# 1.  Load mesh (assumed to be in meters):

mesh = load_mesh_xdmf("Tube.xdmf",comm=MPI.COMM_WORLD,name="Grid",)

# 2.  Initial state and anisotropy easy axis (considered along of rho-axis).

xyz = mesh.geometry.x
n = xyz.shape[0]

m0 = np.zeros((n, 3))
uaxis = np.zeros((n, 3))

for i in range(n):
    x, y, z = xyz[i]
    if x**2 + (z-150)**2 <= 15*15 and y>0:
        m0[i] = [-x, -y, 0]
    else:
        m0[i] = [x,y,0]

    m0[i] /= np.linalg.norm(m0[i])
    axis = [x,y,0] 
    axis  /= np.linalg.norm(axis)
    uaxis[i] = axis

m0_array = m0.flatten()
uaxis = uaxis.flatten()

# 3. Material parameters (The magnetic parameters were obtained from https://arxiv.org/pdf/1812.11767)

Ms = 1.1e5                 # A/m
Aex = 8.78e-12              # J/m
D_bulk = 1.2e-3             # J/m^2
Ku = 2.0e5                  # J/m^3


llg = LLG_GPU(mesh, Ms = Ms, gamma=2.211e5, alpha=1.0, do_precess=0)

llg.add_exchange(Aex=Aex)
llg.add_anisotropy(Ku, uaxis)
llg.add_dmi_bulk(D_bulk)
llg.add_demag(
    method="lindholm_gpu",
    boundary_backend="hmatrix",                                    
    hmatrix_cache_path="./hmatrix_cache/disk_eps1e-5_eta2_leaf64.npz", # reuse the same hmatrix for different simulations with the same mesh.
)


# 5. Time stepping setup (relaxation with stopping criterion)

t0 = 0.0                  # Initial time of the simulation
t_final = 10e-9           # Final time of the simulation If the stopping_dmdt is not reached
dt_init = 1.0e-14         # Initial time step

dt_print = 1.e-10         # simulated-time interval between solver log outputs (monitoring)
dt_snap  = 5.0e-10        # simulated-time interval between saved magnetization snapshots (XDMF)

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
    stopping_dmdt=1.0,     # stopping criterion
)


if mesh.comm.rank == 0:
    print(f"Tiempo wall-clock ts.solve : {elapsed:.3f} s")

