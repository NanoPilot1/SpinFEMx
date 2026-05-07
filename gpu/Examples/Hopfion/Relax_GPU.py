"""
Hopfion stabilization in a cylindrical nanopillar.

This script follows the setup described in J. Appl. Phys. 138, 043907 (2025). It relaxes a hopfion state in a cylindrical nanostructure with radius 100 nm
and height 100 nm. The magnetic parameters are chosen to be representative of FeGe.

Model:

  - Exchange
  - Magnetostatics (demagnetizing field)
  - Bulk DMI
  - Uniaxial Anisotropy

This script performs energy relaxation by disabling precession (do_precess = 0) and using a high damping value (alpha = 1.0)
for faster convergence.

"""

from mpi4py import MPI
import numpy as np
import dolfinx
from dolfinx import fem

from micromagnetic_gpu import LLG_GPU

comm = MPI.COMM_WORLD

# 1.  Load mesh (assumed to be in meters):
    
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "Cylinder.xdmf", "r") as xdmf:
     mesh = xdmf.read_mesh(name="Grid")


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


    if z>48 or z<-48:
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

llg = LLG_GPU(mesh, Ms, gamma=2.211e5, alpha=1.0, do_precess=0)

llg.add_exchange(Aex=Aex)
llg.add_demag(method="fmm")

llg.add_anisotropy(Kus, uaxis)
llg.add_dmi_bulk(Dbulk)
 
 # 5. Time stepping setup (relaxation with stopping criterion)

t0 = 0.0
t_final = 10e-9
dt_init = 1.0e-15

dt_print = 1.e-11         # simulated-time interval between solver log outputs (monitoring)
dt_snap  = 2.0e-10        # simulated-time interval between saved magnetization snapshots (XDMF)

y, ctx, elapsed = llg.relax(
    m0_array,
    t0,
    t_final,
    dt_init,
    dt_save=dt_print,
    dt_snap=dt_snap,
    output_dir="relax_GPU",
    ts_rtol=1.0e-6,
    ts_atol=1.0e-6,
    save_final_state=True, # writes a checkpoint (.bp) for reuse in subsequent simulations
    stopping_dmdt=1        # stopping criterion
)


if mesh.comm.rank == 0:
    print(f"Tiempo wall-clock ts.solve : {elapsed:.3f} s")

