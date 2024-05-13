"""
Visualisation of the snapshots 
"""
# Basic Environment
import numpy as np 
import h5py 
import matplotlib.pyplot as plt
from matplotlib import animation
from utils import plt_rc_setup 
from utils.plot import colorplate as cc 
import os 
import time
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Unlock the h5py file

font_dict = {'size':20,'weight':"bold"}
fig_kw = {'bbox_inches':'tight',"dpi":300}
clb_kw = {"format":'%.2f','orientation':'horizontal','shrink':0.9,"pad":0.15,"aspect":20}
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument("--start","-s",default=1,type=int)
parser.add_argument("--end","-e",default=26,type=int)
args = parser.parse_args()

FBase           = args.start # Start base 
NoBase          = args.end # End base
base_dir        = os.getcwd() + "/"

datafile        = f'data/u_{FBase}_to_{NoBase}.hdf5'
save_fig_path   = base_dir + "Figs/"

# Open the datafile
with h5py.File(base_dir + datafile, 'r') as f:
  u_keras   = np.array(f['u'][:],dtype=np.float32)
#   nt        = int(f['nt'][()])
  Nx        = int(f['nx'][()])

  if NoBase == 26:
    Nz        = int(f['nz'][()])
  else:
    ny        = int(f['ny'][()])

#   u_mean    = f['mean'][:]
#   u_std     = f['std'][:]



print(f"The {FBase}-{NoBase} dataset has been loaded, the shape is {u_keras.shape}")
# Examine the nan value in dataset
anyNan  = np.sum(np.isnan(u_keras))
print(f"Check If there is nan before processing: {anyNan}")
u_keras = np.nan_to_num(u_keras)
anyNan  = np.sum(np.isnan(u_keras))
print(f"Check If there is nan after processing: {anyNan}")

# Reshape the data and add new axis 
# u_keras = np.transpose( u_keras)
U = u_keras 
print(f"The data now has shape of {U.shape}")

Nt,Nx,Nz        = U.shape
NInterval = 100 
Nvideo    = int(Nt/NInterval) -1
print(f"There will be {Nvideo} videos to be made")

print(f"Number of grid points x = {Nx}, z = {Nz}")
xx,zz       = np.mgrid[-1:5:1j*Nx,-1.5:1.5:1j*Nz]
print(f"Mesh grid for x has shape = {xx.shape}, for z has shape = {zz.shape}")

xb = np.array([-0.125, -0.125, 0.125, 0.125, -0.125])
yb = np.array([-0.125, 0.125, 0.125, -0.125, -0.125])

idx = 0

fig,axs = plt.subplots(figsize=(6,4))
clb = axs.contourf(xx,zz,U[idx,:,:],
                    levels =200, 
                    cmap = "YlGnBu_r"
                )

cb0 = fig.colorbar(clb,ax = axs,**clb_kw)
cb0.ax.locator_params(nbins = 4)

axs.set_xlabel(r'$x/c$',font_dict)
axs.set_ylabel(r'$z/c$',font_dict)
axs.set_xticks(ticks=[])
axs.set_yticks(ticks=[])

axs.set_aspect('equal')
axs.fill(xb, yb, c = 'w',zorder =3)
axs.plot(xb, yb, c = 'k', lw = 1, zorder = 5)
fig.savefig(save_fig_path+f'SnapShot_{idx}.jpg',**fig_kw)