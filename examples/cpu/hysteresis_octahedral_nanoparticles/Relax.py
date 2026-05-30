"""
Hysteresis curve of two interacting nanoparticles with octaedral truncaded shape

Model:

  - Exchange
  - Cubic anisotropy (easy axis along +z)
  - Magnetostatics (demagnetizing field)
  - Zeeman 

This script performs energy relaxation by disabling precession (do_precess = 0) and using a high damping value (alpha = 1.0)
for faster convergence.

Threading recommendations (avoid oversubscription when using MPI):

  export OMP_NUM_THREADS=1
  export NUMBA_NUM_THREADS=4   # affects numba-based kernels (e.g., bempp-cl)

Run:

  mpiexec -n 1 python Relax.py

"""


from mpi4py import MPI
import numpy as np
import dolfinx
from dolfinx import fem
from pathlib import Path
from fenicsx_micromagnetics.cpu import LLG, load_mesh_xdmf


comm = MPI.COMM_WORLD

mesh = load_mesh_xdmf("chain.xdmf")

# 2.  Initial state, anisotropy cubic axis.

xyz = mesh.geometry.x

n = xyz.shape[0]

m0 = np.zeros((n, 3))
u1= np.zeros((n, 3))
u2= np.zeros((n, 3))


for i in range(n):
    x, y, z = xyz[i]

    m0[i] = [1.0, 0.0, 0.1]
    m0[i] /= np.linalg.norm(m0[i])
    u1[i] = [0,1,1] 
    u2[i] = [0,1,-1] 

m0_array = m0.flatten()
u1 = u1.flatten()
u2 = u2.flatten()


# 3. Material parameters 

Ms = 4.8e5
Aex = 13.3e-12
Kc1 = 30e3

# 4. Build solver and interactions

llg = LLG(mesh, Ms, gamma=2.211e5, alpha=1.0, do_precess=0)

llg.add_exchange(Aex=Aex )
llg.add_demag(method="lindholm")
llg.add_cubic_anisotropy(Kc1=Kc1, u1_vec=u1, u2_vec=u2)



# 5. Magnetic field steps from 0.2 T to -0.2 T

mu0 = 4*np.pi*1e-7
Hz1 = np.linspace(0.2/mu0, -0.2/mu0, 201)   # [0.2T, -0.2 T
Hz2 = np.linspace(-0.2/mu0, 0.2/mu0, 201)[1:]  # [-0.2T, 0.2T]

H_steps = []
for Hz in Hz1:
    H_steps.append((float(Hz), 0.0,float(Hz)*0.01))
for Hz in Hz2:
    H_steps.append((float(Hz), 0.0,float(Hz)*0.01))

# 6. Time stepping setup (relaxation with stopping criterion)


t_final_per_step = 40e-9
dt_init = 1.0e-14
stopping_dmdt = 0.01

results = llg.hysteresis(
    m0_array=m0_array,
    H_steps=H_steps,
    t_final_per_step=t_final_per_step,
    dt_init=dt_init,
    output_dir="hyst_out",
    ts_rtol=1e-6,
    ts_atol=1e-6,
    stopping_dmdt=stopping_dmdt,
    check_every_stop=5,
    stop_print=False,
)

