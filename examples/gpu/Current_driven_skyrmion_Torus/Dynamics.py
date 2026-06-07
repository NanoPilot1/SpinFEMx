"""
This script reproduces the current-driven dynamics of a Néel skyrmion on a torus, as shown
in Fig. S10(b) of the paper "Spin-Transfer Torque on Curved Surfaces: A Generalized Thiele Formalism."

The simulation starts from the relaxed Néel-skyrmion configuration obtained with the
corresponding relaxation script. The torus has major radius R = 180 nm, minor radius
r = 50 nm, and thickness h = 2 nm.

The energy functional includes:

- Exchange interaction
- Interfacial Dzyaloshinskii-Moriya interaction (DMI)
- Uniaxial anisotropy

The skyrmion dynamics are driven by an electric current density J = 5 x 10^11 A/m^2 applied along the azimuthal phi-direction of the torus. 

The simulation evolves the magnetization in time to analyze the current-induced motion of the skyrmion on the curved surface.
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


# 3.  Anisotropy and electric current directions, considered along of rho-axis and phi-direction, respectively.

xyz = mesh.geometry.x
n= mesh.geometry.x.shape[0]

Jvec = np.zeros((n,3))
uvec = np.zeros((n,3))

xyz = mesh.geometry.x


m0 = m.x.array[:]

R = 180

for i in range(n):
    x = xyz[i,0]
    y = xyz[i,1]
    z = xyz[i,2]

    r = np.sqrt(x*x+y*y+z*z)
    phi = np.arctan2(y,x)
    theta = np.arccos(z/r)
    r0 = np.sqrt(r*r+R*R-2*r*R*np.sin(theta))


    uvec[i,0] = np.cos(phi)*(-R+r*np.sin(theta))/(np.sqrt(r*r+R*R-2*r*R*np.sin(theta)))
    uvec[i,1] = np.sin(phi)*(-R+r*np.sin(theta))/(np.sqrt(r*r+R*R-2*r*R*np.sin(theta)))
    uvec[i,2] = r*np.cos(theta)/(np.sqrt(r*r+R*R-2*r*R*np.sin(theta)));

    normavec = np.sqrt(uvec[i,0]*uvec[i,0]+uvec[i,1]*uvec[i,1]+uvec[i,2]*uvec[i,2])

    uvec[i,0] = uvec[i,0]/normavec
    uvec[i,1] = uvec[i,1]/normavec
    uvec[i,2] = uvec[i,2]/normavec

    Jvec[i,0] = np.sin(phi)
    Jvec[i,1] = -np.cos(phi)
    Jvec[i,2] = 0.


uvec = uvec.flatten()
Jvec = Jvec.flatten()

# 4. Material parameters (typical for Permalloy)

Ms = 1.09817e6    # A/m
Aex = 1.6e-11     # J/m
Ku = 5.9e5        # J/m^3
Dint = 2.8e-3     # J/m^2
Je = 5e11         # A/m^2

llg = LLG_STT_GPU(mesh, Ms, gamma=2.211e5, alpha=0.02, do_precess=1) 

llg.add_exchange(Aex= Aex)
llg.add_anisotropy(Ku, uvec )
llg.add_dmi_interfacial(Dint, uvec )
llg.add_current(Jmagnitude=Je , Jdir_vec=Jvec , P=0.5, beta=0.5)

# The values ​​of alpha, beta, and P were obtained from the curvature-induced effects on the movement of skyrmions along curved nanotubes.

# 6. Time stepping setup (relaxation with stopping criterion)


t0 = 0.0                  # Initial time of the simulation
t_final = 90e-9           # Final time of the simulation 
dt_init = 1.0e-14         # Initial time step

dt_print = 5.00e-10        # simulated-time interval between solver log outputs (monitoring)
dt_snap  = 2.00e-9         # simulated-time interval between saved magnetization snapshots (XDMF)


y_final, ctx, elapsed = llg.solve(
    m0_array,
    t0,
    t_final,
    dt_init,
    dt_save=dt_print,
    dt_snap=dt_snap,
    output_dir="STT_Skyrmion",
    ts_rtol=1.0e-6,
    ts_atol=1.0e-6,
    monitor_fn=None,
)
