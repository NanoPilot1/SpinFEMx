"""
Relaxation of a Neel skyrmion in a disk.

Model:
  - Exchange
  - Interfacial Dzyaloshinskii-Moriya interaction (DMI)
  - Uniaxial anisotropy (easy axis along +z)
  - Magnetostatic interaction.

This script performs energy relaxation using the minimize

We consider that the Neel skyrmion is  stabilized in a nanotube with R = 40 nm and t = 1.5 nm.

Threading recommendations (avoid oversubscription when using MPI):

  export OMP_NUM_THREADS=1


Run:

  mpiexec -n <core_number> python Relax.py

"""


from mpi4py import MPI
import numpy as np
import dolfinx
from dolfinx import fem
from spinfemx.cpu import LLG, load_mesh_xdmf

comm = MPI.COMM_WORLD

    

# 1.  Load mesh (assumed to be in meters):

mesh = load_mesh_xdmf("tube.xdmf",comm=MPI.COMM_WORLD,name="Grid",)

# 2.  Initial state, anisotropy easy axis and DMI (considered along of rho-axis).

xyz = mesh.geometry.x
n = xyz.shape[0]

m0 = np.zeros((n, 3))
uaxis = np.zeros((n, 3))
Daxis = np.zeros((n, 3))

for i in range(n):
    x, y, z = xyz[i]
    if x**2 + (z-150)**2 <= 10*10 and y>0:
        m0[i] = [-x, -y, 0]
    else:
        m0[i] = [x,y,0]
    axis = [x,y,0] 
    axis  /= np.linalg.norm(axis)

    m0[i] /= np.linalg.norm(m0[i])
    uaxis[i] = axis
    Daxis[i] = axis

m0_array = m0.flatten()
uaxis = uaxis.flatten()
Daxis = uaxis.flatten()

# 3. Material parameters (i,e 10.1103/PhysRevB.105.054425)

Ms = 1.09817e6                # A/m
Aex = 1.6e-11             # J/m
D_int = 2.6e-3             # J/m^2
Ku = 1.4e6                # J/m^3


llg = LLG(mesh, Ms = Ms, gamma=2.211e5, alpha=1.0, do_precess=0)

llg.add_exchange(Aex=Aex)
llg.add_anisotropy(Ku, uaxis)
llg.add_dmi_interfacial(D_int, Daxis)
llg.add_demag(method="htool",htool_epsilon=1e-7,htool_eta=2,htool_max_leaf_size=64,recompress_bool = 1)


# 5. Time stepping setup (relaxation with stopping criterion)

t0 = 0.0                  # Initial time of the simulation
t_final = 10e-9            # Final time of the simulation If the stopping_dmdt is not reached
dt_init = 1.0e-14         # Initial time step

dt_print = 5.e-11         # simulated-time interval between solver log outputs (monitoring)
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
    stopping_dmdt=1.0,     # stopping criterion
)


if mesh.comm.rank == 0:
    print(f"Tiempo wall-clock ts.solve : {elapsed:.3f} s")

