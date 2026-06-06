"""

This script reproduces the relaxed Néel-skyrmion configuration used as the initial state for the dynamics shown in Fig. S10(b) of the paper
"Spin-Transfer Torque on Curved Surfaces: A Generalized Thiele Formalism."

A Neel skyrmion is relaxed on a torus with major radius R = 180 nm, minor radius r = 50 nm, and thickness h = 2 nm.

The energy functional includes:

- Exchange interaction
- Interfacial Dzyaloshinskii-Moriya interaction (DMI)
- Uniaxial anisotropy

To accelerate convergence toward a local energy minimum, precession is disabled
(do_precess = 0) and the damping parameter is set to alpha = 1.0.

The resulting relaxed state is used as the initial configuration for a separate dynamics script.
configuration on the torus.

"""


from mpi4py import MPI
import numpy as np
import dolfinx
from dolfinx import fem
from fenicsx_micromagnetics.gpu import LLG_GPU, load_mesh_xdmf

comm = MPI.COMM_WORLD

    
# 1.  Load mesh (assumed to be in meters):

mesh = load_mesh_xdmf("Torus.xdmf",comm=MPI.COMM_WORLD,name="Grid",)

# 2.  Initial state, anisotropy easy axis and DMI (considered along of rho-axis).

xyz = mesh.geometry.x
n = xyz.shape[0]

m0 = np.zeros((n, 3))
uvec = np.zeros((n, 3))

R = 180

for i in range(n):
    x = xyz[i,0]
    y = xyz[i,1]
    z = xyz[i,2]

    r = np.sqrt(x*x+y*y+z*z)
    phi = np.arctan2(y,x)
    theta = np.arccos(z/r)
    r0 = np.sqrt(r*r+R*R-2*r*R*np.sin(theta))

    if y>R and x*x+z*z<=16*16:

        m0[i,0] = 0.
        m0[i,1] = -1.0
        m0[i,2] = 0.
    else:
        m0[i,0] =  np.cos(phi)*(-R+r*np.sin(theta))/(np.sqrt(r*r+R*R-2*r*R*np.sin(theta)))
        m0[i,1] =  np.sin(phi)*(-R+r*np.sin(theta))/(np.sqrt(r*r+R*R-2*r*R*np.sin(theta)))
        m0[i,2] =  r*np.cos(theta)/(np.sqrt(r*r+R*R-2*r*R*np.sin(theta)))

    norma = np.sqrt(m0[i,0]*m0[i,0]+m0[i,1]*m0[i,1]+m0[i,2]*m0[i,2])

    m0[i,0] = m0[i,0]/norma
    m0[i,1] = m0[i,1]/norma
    m0[i,2] = m0[i,2]/norma

    uvec[i,0] = np.cos(phi)*(-R+r*np.sin(theta))/(np.sqrt(r*r+R*R-2*r*R*np.sin(theta)))
    uvec[i,1] = np.sin(phi)*(-R+r*np.sin(theta))/(np.sqrt(r*r+R*R-2*r*R*np.sin(theta)))
    uvec[i,2] = r*np.cos(theta)/(np.sqrt(r*r+R*R-2*r*R*np.sin(theta)));

    normavec = np.sqrt(uvec[i,0]*uvec[i,0]+uvec[i,1]*uvec[i,1]+uvec[i,2]*uvec[i,2])

    uvec[i,0] = uvec[i,0]/normavec
    uvec[i,1] = uvec[i,1]/normavec
    uvec[i,2] = uvec[i,2]/normavec

m0_array = m0.flatten()
uaxis = uvec.flatten()

# 3. Material parameters 

Ms = 1.09817e6
Aex = 1.6e-11
Dint = 2.8e-3
Ku = 5.9e5


# 4. Build solver and interactions

llg = LLG_GPU(mesh, Ms = Ms, gamma=2.211e5, alpha=1.0, do_precess=0)

llg.add_exchange(Aex=Aex)
llg.add_anisotropy(Ku, uaxis)
llg.add_dmi_interfacial(Dint, uaxis)


# 5. Time stepping setup (relaxation with stopping criterion)

t0 = 0.0                  # Initial time of the simulation
t_final = 20e-9           # Final time of the simulation If the stopping_dmdt is not reached
dt_init = 1.0e-14         # Initial time step

dt_print = 1.e-9          # simulated-time interval between solver log outputs (monitoring)
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
    stopping_dmdt=0.1,     # stopping criterion
)


if mesh.comm.rank == 0:
    print(f"Tiempo wall-clock ts.solve : {elapsed:.3f} s")

