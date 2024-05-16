"""
Visualise the Poincare Map as joint-PDF 
@yuningw
"""

import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from utils.configs import VAE_custom as args
import matplotlib as mpl
import os
import shutil
import pandas as pd
import argparse

#Plot config
red         = "#D12920" #yinzhu
blue        = "#2E59A7" # qunqing
gray        = "#DFE0D9" # ermuyu

plt.rc('font', family = 'serif')
plt.rc('axes', labelsize = 20, linewidth = 1.5)
plt.rc('font', size = 18)
plt.rc('legend', fontsize = 12, handletextpad=0.3)
plt.rc('xtick', labelsize = 21)
plt.rc('ytick', labelsize = 21)
font_dict = {"weight":"bold","size":20}


baseDir     =   os.getcwd() + '/'
save_fig_to =   baseDir + "Figs/Phase_Space/"

plot_cfg  = {'ls':'none','marker':'o',"markersize":0.5, 'lw':2.5,'c':red}
plot_scatter_cfg  = {'cmap':'rainbow'}

################################################################
# Load Data
###############################################################
print("#"*30)
print("Plotting the PDF for effect of beta:")

num_fields      =   25999
latent_dim      =   10
Epoch           =   300
vae_type        =   f"v5"
batch_size      =   128
earlystop       =   False
patience        =   0
if vae_type == 'v5':
    betas           =   [0.005]
elif vae_type == 'v45':
    betas           =   [5e-4,10e-4, 25e-4, 50e-4]
elif vae_type == 'v4':
    betas           =   [0.001, 50e-4, 100e-4, 500e-4]
elif vae_type == 'v35':
    betas           =   [5e-4, 0.001, 50e-4, 100e-4]

#compute detR
modes = []
ranks = []
for beta in betas:
    filesID   =  f"Rank_Mode_{vae_type}_{int( num_fields )}n_{latent_dim}d_{int(beta*10000)}e-4beta_"+\
                f"{args.block_type}conv_{len(args.filters)}Nf_{args.filters[-1]}Fdim_{args.linear_dim}Ldim"+\
                f"{args.act_conv}convact_{args.act_linear}_" +\
                f"{int(args.lr *1e5)}e-5LR_{int(args.w_decay*1e5)}e-5Wd"+\
                f"{batch_size}bs_{Epoch}epoch_{earlystop}ES_{patience}P"
    modes_filepath = baseDir+ "latent_modes/"+filesID+".npz"
    print(f"Loading case: \n{filesID}")
    d = np.load(modes_filepath)
    z_mean = np.array(d["z_mean"])
    z_var = np.array(d["z_var"])
    
    # z_mean = z_mean + np.exp(z_var*0.5) * np.random.random(size=z_var.shape)
    orders = np.array(d['ranks'])
    corr_matrix_latent = abs(np.corrcoef(z_mean.T))
    detR = np.linalg.det(corr_matrix_latent)
    print(f"In order to confirm the case ,we confirm the detR is {np.round(detR,4)}")
    modes.append(z_mean)
    ranks.append(orders)

load_pod    =   baseDir + "pod_modes/"
case_pod    =   f"POD-m{latent_dim}-n25999"

with np.load( load_pod +  case_pod + ".npz") as pod_file:
    # Load spatial modes
    U_pod   =   pod_file['modes']
    # Load temporal modes
    V_pod   =   pod_file["vh"] * np.sqrt(25999)
    pod     =   pod_file["s"]

pod = V_pod.T
vae_mode = modes[0][:,ranks[0]]
print(f"VAE Mode = {vae_mode.shape}, POD Mode = {pod.shape}")


################################################################
# Poincare Map 
###############################################################
from utils.chaotic import Intersection , PDF

## Basic setup for Pmap
planeNo      = 0 
postive_dir  = True
lim_val      = 2.5 # Limitation of x and y bound when compute joint pdf 
grid_val     = 50  # Mesh grid number for plot Pmap
Pmap_Info    = f"PMAP_{planeNo}P_{postive_dir}pos_{lim_val}lim_{grid_val}grid_"

InterSec_pred = Intersection(vae_mode,
                            planeNo=planeNo,postive_dir=postive_dir)
print(f"The intersection of Prediction has shape of {InterSec_pred.shape}")

InterSec_test = Intersection(pod,
                            planeNo=planeNo,postive_dir=postive_dir)
print(f"The intersection of Test Data has shape of {InterSec_test.shape}")

# Plot all the Poincare map for each section
Nmodes   =  10 
fig, axs = plt.subplots(Nmodes,Nmodes, 
                        figsize=(3.5* Nmodes, 3.5*Nmodes),
                        sharex=True,sharey=True)

for i in range(0,Nmodes):
        for j in range(0,Nmodes):
            if(i==j or j==planeNo or i==planeNo or j>i):
                axs[i,j].set_visible(False)
                continue
           
            xx,yy, pdf_test      = PDF(  
                                        InterSecX= InterSec_test[:,i],
                                        InterSecY= InterSec_test[:,j],
                                        xmin=-lim_val,xmax=lim_val,
                                        ymin=-lim_val,ymax=lim_val,
                                        x_grid=grid_val,y_grid=grid_val,
                                    )
                  

            xx,yy, pdf_pred      = PDF(   
                                        InterSecX= InterSec_pred[:,i],
                                        InterSecY= InterSec_pred[:,j],
                                        xmin=-lim_val,xmax=lim_val,
                                        ymin=-lim_val, ymax=lim_val,
                                        x_grid=grid_val,y_grid=grid_val,
                                    )


            
            axs[i,j].contour(xx,yy,pdf_test,colors=blue)
            axs[i,j].contour(xx,yy,pdf_pred,colors=red)
            axs[i,j].set_xlim(-lim_val,lim_val)
            axs[i,j].set_xlabel(f"z{i+1}",fontsize='large')
            axs[i,j].set_ylabel(f"z{j+1}",fontsize='large')
            axs[i,j].set_aspect('equal',"box")
            axs[i,j].grid(visible=True,markevery=1,color='gainsboro', zorder=1)
plt.subplots_adjust(wspace= 0.1)
plt.savefig( f"Figs/Phase_Space/Pmap_VAE_POD.jpg", bbox_inches="tight")



fig, axs = plt.subplots(1,1, 
                        figsize=(5,5),
                        sharex=True,sharey=True)

xx,yy, pdf_test      = PDF(  
                                        InterSecX= InterSec_test[:,4],
                                        InterSecY= InterSec_test[:,6],
                                        xmin=-lim_val,xmax=lim_val,
                                        ymin=-lim_val,ymax=lim_val,
                                        x_grid=grid_val,y_grid=grid_val,
                                    )
                  

xx,yy, pdf_pred      = PDF(   
                                        InterSecX= InterSec_pred[:,4],
                                        InterSecY= InterSec_pred[:,6],
                                        xmin=-lim_val,xmax=lim_val,
                                        ymin=-lim_val, ymax=lim_val,
                                        x_grid=grid_val,y_grid=grid_val,
                                    )


            
axs.contour(xx,yy,pdf_test,colors=blue)
axs.contour(xx,yy,pdf_pred,colors=red)
axs.set_xlim(-lim_val,lim_val)
axs.set_xlabel(f"z{4+1}",fontsize='large')
axs.set_ylabel(f"z{6+1}",fontsize='large')
axs.set_aspect('equal',"box")
axs.grid(visible=True,markevery=1,color='gainsboro', zorder=1)
plt.subplots_adjust(wspace= 0.1)
plt.savefig( f"Figs/Phase_Space/Pmap_VAE_POD_5V7.jpg", bbox_inches="tight")


