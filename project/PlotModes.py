"""
Visualise the spatial and temporal modes and compare with POD modes 
using the ran mode only!! 
@yuningw 
"""

import h5py
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
import cmocean as cmo
from matplotlib import cm
import seaborn as sns
from scipy import signal
import os 
# Defalut 
cmp = "RdBu"
# cmp = "YlGnBu_r"
plt.set_cmap(cmp)
plt.rc("font",  family      = "serif")
plt.rc("font",  size        = 22)
plt.rc("axes",  labelsize   = 16, linewidth     = 2)
plt.rc("legend",fontsize    = 12, handletextpad = 0.3)
plt.rc("xtick", labelsize   = 14)
plt.rc("ytick", labelsize   = 14)


class line_beta2:
    red = "#D23918" # luoshenzhu
    blue = "#2E59A7" # qunqing
    yellow = "#E5A84B" # huanghe liuli
    cyan = "#5DA39D" # er lv
    black = "#151D29" # lanjian
font_dict = {"weight":"bold","size":22}


# Corrdinate 
Nx, Ny  =   192, 96 
x       =   np.linspace(-1, 5, Nx)
y       =   np.linspace(-1.5, 1.5, Ny)
y, x    =   np.meshgrid(y, x)
x       =   x[:192, :96]
y       =   y[:192, :96]
xb      =   np.array([-0.125, -0.125, 0.125, 0.125, -0.125])
yb      =   np.array([-0.125, 0.125, 0.125, -0.125, -0.125])
print(f"Generate the spatial coordinate, shape of X and Y are {x.shape, y.shape}")




latent_dim  =   10 

base_dir    =   os.getcwd() + '/'

load_pod    =   base_dir + "08_POD/"
case_pod    =   f"POD-m{latent_dim}-n25999"

with np.load( load_pod +  case_pod + ".npz") as pod_file:
    # Load spatial modes
    U_pod   =   pod_file['modes']
    # Load temporal modes
    V_pod   =   pod_file["vh"] * np.sqrt(25999)




#########################################
## Spatial modes analysis 
#########################################




print(f"The spatial modes has shape of {U_pod.shape} while the temporal modes has shape of {V_pod.shape}")

U_pod           =   U_pod.reshape(latent_dim,Ny, Nx)
U_min, U_max    =   U_pod.min(), U_pod.max()

fig, axs    =   plt.subplots(2, latent_dim//2, sharex= True, sharey= True, figsize= (2.5*latent_dim,4))
axs         =   axs.flatten()
for ind, ax in enumerate(axs):
    ax.contourf(x,y, U_pod[ind, :,:].T,
                levels = 200, 
                vmin = U_min, vmax = U_max
                )
    ax.fill(xb, yb, c = 'w',zorder =3)
    ax.plot(xb, yb, c = 'k', lw = 1, zorder = 5)
    ax.set_aspect('equal')
    ax.set_title(f"Mode {ind+1}")
    (ax.set_xlabel(r"${x/h}$") if ind >= latent_dim//2 else None)
    ax.set_aspect('equal')

plt.subplots_adjust(hspace= 0.3)
axs[0].set_ylabel(r"${z/h}$")
axs[5].set_ylabel(r"${z/h}$")
    
plt.savefig("POD_SpModes.jpg",bbox_inches='tight')

###################################
# Spectral analysis
###################################

# V_pod   =   V_pod.reshape(-1, latent_dim)
print(V_pod.shape)

fig, axs    =   plt.subplots(2, latent_dim//2, sharey= True, figsize= (3.5 * latent_dim, 6))
axs         =   axs.flatten()
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text = r"$St$ = "+"{:5f}".format(xmax)

    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

fs      =   1
tfreq   =   0.005
for ind, ax in enumerate(axs):
    f, Pxx_den  = signal.welch(V_pod[ind,:], fs, nperseg=4096)
    f           /= tfreq
    ax.plot(f, Pxx_den,lw =2.5, c = line_beta2.black)
    ax.set_xlim([0,1])
    (axs[ind].set_xlabel(r'${St}$') if ind >=latent_dim//2 else None)
    ax.set_title(f"Mode {ind+1}")
    annot_max(f,Pxx_den,ax = ax)
    ax.set_yticks([1000])
plt.subplots_adjust(hspace= 0.5, wspace = 0.3)
plt.savefig("POD_TempMode.jpg", bbox_inches = "tight")


# plt.show()


