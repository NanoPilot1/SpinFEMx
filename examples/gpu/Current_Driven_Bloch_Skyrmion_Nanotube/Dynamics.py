"""
Current-driven bloch skyrmion in a nanotube of external radius R = 50 nm and thickness t = 10 nm.

Model:

  - Exchange
  - Bulk Dzyaloshinskii-Moriya interaction (DMI)
  - Uniaxial anisotropy (easy axis along rho axis)
  - Magnetostatic interaction.

"""

from mpi4py import MPI
import numpy as np
import dolfinx
from spinfemx.gpu import LLG_STT_GPU, load_mesh_xdmf
from pathlib import Path
import adios4dolfinx as ad



# 1.  Load mesh (assumed to be in meters):

fname = Path("relax/Relax.bp")
mesh  = ad.read_mesh(fname, MPI.COMM_WORLD)
mesh._serial_mesh_path = str(fname)

mesh  = ad.read_mesh(fname, MPI.COMM_WORLD)
V     = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))

# 2.  Initial state.

m = dolfinx.fem.Function(V, name="m")
ad.read_function(fname, m, time=0.0, name="m")

m.x.scatter_forward()

m0_array = m.x.array


# 3.  Anisotropy easy axis and electric current direction

xyz = mesh.geometry.x
n= mesh.geometry.x.shape[0]

uaxis = np.zeros((n, 3))
Je = np.zeros((n, 3))

for i in range(n):
    x = xyz[i,0]
    y = xyz[i,1]
    z = xyz[i,2]

    axis = [x,y,0] 
    axis  /= np.linalg.norm(axis)
    uaxis[i] = axis

    Je[i,0] = 0.0
    Je[i,1] = 0.0
    Je[i,2] = 1.0

Je_Array = Je.flatten()

# 4. Material parameters 

Ms = 1.1e5                  # A/m
Aex = 8.78e-12              # J/m
D_bulk = 1.2e-3             # J/m^2
Ku = 2.0e5                  # J/m^3


# 5. Build solver and interactions


llg = LLG_STT_GPU(mesh, Ms = Ms, gamma=2.211e5, alpha=0.1, do_precess=1)

llg.add_exchange(Aex=Aex)
llg.add_anisotropy(Ku, uaxis)
llg.add_dmi_bulk(D_bulk)
llg.add_demag(
    method="lindholm_gpu",
    boundary_backend="hmatrix",
    hmatrix_cache_path="./hmatrix_cache/disk_eps1e-5_eta2_leaf64.npz",
)


llg.add_current(Jmagnitude= -5e12, Jdir_vec=Je_Array , P=0.5, beta=0.5)

# 6. Time stepping setup (relaxation with stopping criterion)


t0 = 0.0                  # Initial time of the simulation
t_final = 0.5e-9          # Final time of the simulation If the stopping_dmdt is not reached
dt_init = 1.0e-14         # Initial time step

dt_print = 2.00e-11         # simulated-time interval between solver log outputs (monitoring)
dt_snap  = 2.00e-11         # simulated-time interval between saved magnetization snapshots (XDMF)



y_final, ctx, elapsed = llg.solve(
    m0_array,
    t0,
    t_final,
    dt_init,
    dt_save=dt_print,
    dt_snap=dt_snap,
    output_dir="STT",
    ts_rtol=1.0e-6,
    ts_atol=1.0e-6,
    monitor_fn=None,
)
