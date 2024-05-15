"""
Visualisation of the phase space 
yuningw
"""
#%%
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
# Effect of beta
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
    betas           =   [0.001, 0.0025, 0.005, 0.01]
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

#%%

# Create a figure for plot 
elev,azim = 15, 30
fig, axs = plt.subplots(
                        1, 
                        len(modes),
                        # sharex=True, sharey= True,
                        figsize =(5*len(modes),12),
                        subplot_kw={'projection':'3d'},
                        )
#Plot PDF 1x4 vs Beta
axs = axs.flatten()
for j, ax in enumerate(axs):
    z_mean  =   modes[j]
    rank    =   ranks[j]
    
    xyz = z_mean[:,rank][:,:3]
    axs[j].plot(
                xyz[:,0],
                xyz[:,1],
                xyz[:,2],
                **plot_cfg
                )
    
    # For Label of each pannel
    if betas[j] < 0.01: 
        ax.set_title(r"$\beta$ "+f"= {np.round(betas[j]*1000,1)}" + r"$\times$ " + r"$10^3$" )
    
    elif betas[j] >= 0.01:
        ax.set_title(r"$\beta$ "+f"= {np.round(betas[j]*100,1)}" + r"$\times$ " + r"$10^2$" )
    
    axs[j].view_init(elev, azim)
fig.savefig(save_fig_to +"Phase_Space_"+vae_type+"_n"+str(num_fields)+'_m'+str(latent_dim)+'_b_all'+"_epoch" + str(Epoch) + '.jpg',
            bbox_inches = "tight",
            dpi=300)
print("The Fig has been saved")
print("#"*30)

#%%


##############
# Further Visualisation using t-SNE
##############

from sklearn.manifold import TSNE
setup_dict = {
            "n_components":3,
            'learning_rate':'auto',
            'init':"pca",
            "verbose":1,
            'n_jobs':16}
j = 2
z_mean  =   modes[j]
rank    =   ranks[j]
z_mean = z_mean[:,rank]
xyz = TSNE(**setup_dict).fit_transform(z_mean)
print(f"Fit End for {betas[j]*100}, shape = {xyz.shape}")
#%%
#####
# Load POD 
#####

base_dir    =   os.getcwd() + '/'

load_pod    =   base_dir + "pod_modes/"
case_pod    =   f"POD-m{latent_dim}-n25999"

with np.load( load_pod +  case_pod + ".npz") as pod_file:
    # Load spatial modes
    U_pod   =   pod_file['modes']
    # Load temporal modes
    V_pod   =   pod_file["vh"] * np.sqrt(25999)
    pod     =   pod_file["s"]

pod = V_pod.T
print(pod.shape)

#%%
fig, axs = plt.subplots(
                        1, 
                        1,
                        # sharex=True, sharey= True,
                        figsize =(6,6),
                        subplot_kw={'projection':'3d'},
                        )

t = np.linspace(10,30,len(xyz))
axs.scatter(
                xs = xyz[:,0],
                ys = xyz[:,1],
                zs = xyz[:,2],
                c =  t,
                s = 10,
                marker = 'o',
                cmap='RdBu_r'
                )
axs.set_xlabel("t-SNE 1",labelpad = 15)
axs.set_ylabel("t-SNE 2",labelpad = 15)
axs.set_zlabel("t-SNE 3",labelpad = 15)
# For Label of each pannel
if betas[j] < 0.01: 
    axs.set_title(r"$\beta$ "+f"= {np.round(betas[j]*1000,1)}" + r"$\times$ " + r"$10^3$" )

elif betas[j] >= 0.01:
    axs.set_title(r"$\beta$ "+f"= {np.round(betas[j]*100,1)}" + r"$\times$ " + r"$10^2$" )

elev, azim = 15,30

axs.view_init(elev, azim)
fig.savefig(save_fig_to +"Phase_Space_Manifold_"+vae_type+"_n"+str(num_fields)+'_m'+str(latent_dim)+'_b_all'+"_epoch" + str(Epoch) + '.jpg',
            bbox_inches = "tight",
            dpi=300)
print("The Fig has been saved")
print("#"*30)

# %%
plot_cfg  = {'ls':'none','marker':'o',"markersize":1.5, 'lw':2.5,'c':red}
plot_cfg2  = {'ls':'none','marker':'o',"markersize":1.5, 'lw':2.5,'c':blue}

icount  = 1
for i in range(0, len(xyz),100):
    fig, axs = plt.subplots(
                        1, 
                        1,
                        # sharex=True, sharey= True,
                        figsize =(6,6),
                        subplot_kw={'projection':'3d'},
                        )

    axs.plot(
                    xyz[:i,0],
                    xyz[:i,1],
                    xyz[:i,2],
                    **plot_cfg
                    )
    axs.set_xlabel("t-SNE 1",labelpad = 15)
    axs.set_ylabel("t-SNE 2",labelpad = 15)
    axs.set_zlabel("t-SNE 3",labelpad = 15)
    axs.set_xlim(xyz[:,0].min(),xyz[:,0].max())
    axs.set_ylim(xyz[:,1].min(),xyz[:,1].max())
    axs.set_zlim(xyz[:,2].min(),xyz[:,2].max())
    # For Label of each pannel
    if betas[j] < 0.01: 
        axs.set_title(r"$\beta$ "+f"= {np.round(betas[j]*1000,1)}" + r"$\times$ " + r"$10^3$" )

    elif betas[j] >= 0.01:
        axs.set_title(r"$\beta$ "+f"= {np.round(betas[j]*100,1)}" + r"$\times$ " + r"$10^2$" )

    elev, azim = 15,30

    axs.view_init(elev, azim)
    lenString = len(str(xyz.shape[0]))
    lenil     = len(str(icount))
    nofile = "0"*(( lenString- lenil)) + str(icount) 
    icount +=1 
    fig.savefig(f"Figs/Animation/Manifold_{nofile}.jpg",
                dpi=300)
    plt.clf()
    plt.close()


# %%
plot_cfg1  = {'ls':'none','marker':'o',"markersize":1.5, 'lw':2.5,'c':blue}
plot_cfg2  = {'ls':'none','marker':'o',"markersize":1.5, 'lw':2.5,'c':red}

icount  = 1
for i in range(0, len(xyz),100):
    fig, axs = plt.subplots(
                        1, 
                        2,
                        sharex=True, sharey= True,
                        figsize =(10,6),
                        subplot_kw={'projection':'3d'},
                        )
    axs[0].set_title('POD')
    axs[1].set_title(r"$\beta$"+"-VAE")
    axs[0].plot(
                    pod[:i,0],
                    pod[:i,1],
                    pod[:i,2],
                    **plot_cfg1
                    )
    axs[1].plot(
                    z_mean[:i,0],
                    z_mean[:i,1],
                    z_mean[:i,2],
                    **plot_cfg2
                    )
    for i in range(2):
        axs[i].set_xlabel(r"$z_1$",labelpad = 15)
        axs[i].set_ylabel(r"$z_2$",labelpad = 15)
        axs[i].set_zlabel(r"$z_3$",labelpad = 15)
        axs[i].set_xlim(min(pod[:,0].min(),z_mean[:,0].min()),max(pod[:,0].max(),z_mean[:,0].max()) )
        axs[i].set_ylim(min(pod[:,1].min(),z_mean[:,1].min()),max(pod[:,1].max(),z_mean[:,1].max()) )
        axs[i].set_zlim(min(pod[:,2].min(),z_mean[:,2].min()),max(pod[:,2].max(),z_mean[:,2].max()) )
        elev, azim = 15,30

        axs[i].view_init(elev, azim)
    lenString = len(str(xyz.shape[0]))
    lenil     = len(str(icount))
    nofile = "0"*(( lenString- lenil)) + str(icount) 
    icount +=1 
    # axs.legend(['POD',r"$\beta$"+"-VAE"])
    fig.savefig(f"Figs/Animation/VAE_VS_POD_{nofile}.jpg",
                dpi=300)
    plt.clf()
    plt.close()

# %%
