from mpi4py import MPI
from pathlib import Path
import numpy as np
import dolfinx
from dolfinx import fem

from fenicsx_micromagnetics.cpu import LLG, load_mesh_xdmf
import adios4dolfinx as ad

# 1.  Load mesh (assumed to be in meters):

fname = Path("relax/Relax.bp")
mesh  = ad.read_mesh(fname, MPI.COMM_WORLD)
mesh._serial_mesh_path = str(fname)

V     = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))

# 2.  Initial state.

m = fem.Function(V, name="m")
ad.read_function(fname, m, time=0.0, name="m")

m.x.scatter_forward()  

# 3.  set time dependent magnetic field that consider a uniform magnetic field of 0.1 T along the x-direction and a sinc pulse along of y-direction

xyz = mesh.geometry.x
n = mesh.geometry.x.shape[0]

m0 = np.zeros((n, 3))
uaxis = np.zeros((n, 3))

def make_time_field_sinc():

    def H_time_func(t):
        fmax = 50.0e9
        t0 = 1/fmax
        H_amp = 0.001/(4*np.pi*1e-7)
        arg = 2*(t - t0) *fmax  # np.sinc included the \pi factor
        Hy = H_amp * np.sinc(arg)
        return np.array([0.1/(4*np.pi*1e-7), Hy, 0.], dtype=np.float64)

    return H_time_func


m0_array = m.x.array

# 4. Material parameters (typical for Permalloy)

Ms = 8.0e5                
Aex = 13.0e-12        

# 5. Build solver and interactions

llg = LLG(mesh, Ms=Ms, gamma=2.211e5, alpha=0.01, do_precess=1)

llg.add_exchange(Aex=Aex)
llg.add_demag(method="lindholm")
H_time_func = make_time_field_sinc()
llg.add_external_field(H0_vec=None, H_time_func=H_time_func)

# 6. Time stepping setup (relaxation with stopping criterion)

t0 = 0.0          # Initial time of the simulation
t_final = 10.0e-9 # Final time of the simulation If the stopping_dmdt is not reached
dt_init = 1.0e-15 # Initial time step

dt_print = 1.e-11    # simulated-time interval between solver log outputs (monitoring)
dt_snap  = 10.0e-9   # simulated-time interval between saved magnetization snapshots (XDMF)

y, ctx, elapsed = llg.relax(
    m0_array,
    t0,
    t_final,
    dt_init,
    dt_save=dt_print,
    dt_snap=dt_snap,
    output_dir="dynamic",
    ts_rtol=1e-7,  # we increase the tolerances to obtain a little more clean spectrum
    ts_atol=1e-7,  # we increase the tolerances to obtain a little more clean spectrum
    save_final_state=False,
    stopping_dmdt=0.0001 # We use a very low value.
)

if mesh.comm.rank == 0:
    print(f"Tiempo wall-clock ts.solve : {elapsed:.3f} s")

