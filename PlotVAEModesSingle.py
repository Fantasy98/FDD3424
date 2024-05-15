"""
Visualise the spatial and temporal modes and compare with POD modes 
using the ran mode only!! 
@yuningw 
"""

import h5py
import numpy as np
import os 
import matplotlib.pyplot as plt
from tqdm import tqdm
import cmocean as cmo
from matplotlib import cm
import seaborn as sns
from scipy import signal
from utils.configs import VAE_custom as cfg 
# Defalut 
cmp = "RdBu"
# cmp = "YlGnBu_r"
plt.set_cmap(cmp)
plt.rc("font",  family      = "serif")
plt.rc("font",  size        = 14)
plt.rc("axes",  labelsize   = 16, linewidth     = 2)
plt.rc("legend",fontsize    = 12, handletextpad = 0.3)
plt.rc("xtick", labelsize   = 14)
plt.rc("ytick", labelsize   = 14)


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



print("#"*30)
print("Load the VAE modes")

latent_dim  =   cfg.latent_dim

base_dir        =   os.getcwd() + '/'
mode_dir        =   base_dir  + "latent_modes/"
pod_dir         =   base_dir  + "pod_modes/"
save_spmode_fig   =   base_dir  + "Figs/SpModes/"
save_tpmode_fig   =   base_dir  + "Figs/TempModes/"


vae_type    =   ["v35" , "v4",  "v45", "v5", "v55"]

vae_order = {}
for i, vae in enumerate(vae_type): 
    vae_order[vae] = i+1


param_dict  = {}
param_dict['v35']   = [ 0.0001, 0.001, 0.005, 0.01]
param_dict['v4']    = [ 0.0001, 0.001, 0.005, 0.01]
param_dict['v45']   = [ 0.0001, 0.001, 0.005, 0.01]
param_dict['v5']    = [ 0.001,  0.0025, 0.005,0.01]
param_dict['v55']   = [ 0.01]

plot_dict           = {}
plot_dict['v35']    = [ r"$1 \times 10^{-4}$", r"$1 \times 10^{-3}$", r"$5 \times 10^{-3}$", r"$1 \times 10^{-2}$"]
plot_dict['v4']    = [ r"$1 \times 10^{-4}$", r"$1 \times 10^{-3}$", r"$5 \times 10^{-3}$", r"$1 \times 10^{-2}$"]
plot_dict['v45']    = [ r"$1 \times 10^{-4}$", r"$1 \times 10^{-3}$", r"$5 \times 10^{-3}$",  r"$1 \times 10^{-2}$"]
plot_dict['v5']    = [ r"$1 \times 10^{-3}$", r"$2.5 \times 10^{-3}$", r"$5 \times 10^{-3}$",  r"$1 \times 10^{-2}$"]
plot_dict['v55']    = [r"$1 \times 10^{-2}$"]



Spatial_Modes       =   []
Temporal_Modes      =   []


for beta in param_dict[cfg.model]:
    fileID      =    f"{cfg.model}_{25999}n_{cfg.latent_dim}d_{int(beta*10000)}e-4beta_"+\
                    f"{cfg.block_type}conv_{len(cfg.filters)}Nf_{cfg.filters[-1]}Fdim_{cfg.linear_dim}Ldim"+\
                    f"{cfg.act_conv}convact_{cfg.act_linear}_" +\
                    f"{int(cfg.lr *1e5)}e-5LR_{int(cfg.w_decay*1e5)}e-5Wd"+\
                    f"{cfg.batch_size}bs_{cfg.epoch}epoch_{cfg.earlystop}ES_{cfg.patience}P"
    file_path   =   mode_dir + "Rank_Mode_" + fileID + ".npz"

    with np.load(file_path) as d:
        print(f"INFO: Loaded the case:\n{file_path}")
        print(f"The keys in the file:\n{d.keys()}")
        rank             =   d['ranks']
        print(f"The rank is loaded: {rank}")
        spatialmodes     =   d['modes']
        temporalmodes    =   d["z_mean"]   
        print(f"The spatial modes has shape = {spatialmodes.shape}")
        print(f"The temporal modes has shape = {temporalmodes.shape}")
        Spatial_Modes.append(spatialmodes[rank,:,:])
        Temporal_Modes.append(temporalmodes[:, rank].T)

print(f"All VAE modes are loaded")

print(f"#"*30)
print(f"Load the POD modes  with {latent_dim} modes")
 
case_pod    =   f"POD-m{latent_dim}-n25999"

with np.load( pod_dir +  case_pod + ".npz") as pod_file:
    # Load spatial modes
    U_pod   =   pod_file['modes']
    # Load temporal modes
    V_pod   =   pod_file["vh"] * np.sqrt(25999-1)

Spatial_Modes.append(U_pod)
Temporal_Modes.append(V_pod)
print(f"The POD modes have been loaded, spatial mode: {U_pod.shape}, temporal mode {V_pod.shape}")


###################################
# Spatial mode
###################################

print("#"*30)
print("Plot the spatial modes of VAE and POD")


fs      =   1
tfreq   =   0.005
latent_dim =  1

fig, axs    =   plt.subplots(   len(Temporal_Modes) , latent_dim, 
                                sharex= True,sharey=True,
                                figsize= (  4*len(Temporal_Modes), 2*len(Temporal_Modes) ))

for i  in range(len(Temporal_Modes)):
    for j in range(latent_dim):
        #########
        # Spatial modes
        #########
        axs[i].contourf(x, y, Spatial_Modes[i][0,:,:].T, 
                                levels = 100,
                                vmin =  Spatial_Modes[i].min(),
                                vmax =  Spatial_Modes[i].max(),
                                )
        axs[i].set_aspect("equal")
        axs[i].fill(xb, yb, c = 'w',zorder =3)
        axs[i].plot(xb, yb, c = 'k', lw = 1, zorder = 5)
        axs[-1].set_xlabel("x/h",fontsize = 18)
        axs[i].set_ylabel("z/h",fontsize = 18)
        if i != len(Temporal_Modes)-1:
            axs[i].set_title(f"Arch {vae_order[cfg.model]}, " + r"$\beta$" +" = " + plot_dict[cfg.model][i] )
            # axs[i].annotate( r"${\beta}$" + " = " + plot_dict[cfg.model][i], xy=(0.5, 1.25), xytext=(0, 5),
            #                         xycoords='axes fraction', textcoords='offset points',
            #                          ha='center', va='baseline', fontsize = 12)
            
        else:
            axs[i].set_title(f"POD")
plt.subplots_adjust(hspace=0.5, wspace=0.01)
plt.savefig(save_spmode_fig + f"{cfg.model}_Spatialmode_1st.jpg", bbox_inches="tight")
# plt.subplots_adjust(hspace=0.01)



#########
# Temporal modes
#########
fs      =   1
tfreq   =   0.005

figsize = (0.6*len(Temporal_Modes),1.6*len(Temporal_Modes))
if cfg.model== "v55":
    figsize = (4,3)

fig, axs    =   plt.subplots(   len(Temporal_Modes) , latent_dim, 
                                sharex= True,sharey=True,
                                figsize= figsize)

for i  in range(len(Temporal_Modes)):
    for j in range(latent_dim):
        f, Pxx_den  = signal.welch(Temporal_Modes[i][0,:], fs, nperseg=4096)
        f           /= tfreq

        if i != len(Temporal_Modes)-1:
            axs[i].plot(f, Pxx_den,lw =2.5, c = line_beta2.blue)
            axs[i].set_title(f"Arch {vae_order[cfg.model]}, " + r"$\beta$" +" = " + plot_dict[cfg.model][i] )
        
        else:
            axs[i].plot(f, Pxx_den,lw =2.5, c = line_beta2.black)
            axs[i].set_title(f"POD")
            axs[i].set_yticks([1000])
        axs[i].set_xlim([0,1])

        axs[-1].set_xlabel(r'${St}$')
        annot_max(f,Pxx_den,ax = axs[i])
plt.subplots_adjust(hspace=0.5, wspace=0.01)
plt.savefig(save_tpmode_fig +  f"{cfg.model}_Temporalmode_1st.jpg", bbox_inches="tight")

