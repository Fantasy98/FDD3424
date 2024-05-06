"""
Plot the Correlation Matrix for latent variables
@Author yuningw
"""


import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils.configs import VAE_custom as args
import matplotlib as mpl
import matplotlib.colors as colors
import os
import pandas as pd
import argparse

"""
Note that to plot all the figures, please use:
    
    python PlotPdf.py -b -d -n

"""

parser      =   argparse.ArgumentParser("To specify which Correlation Matrix to plot")
parser.add_argument("--beta",   "-b",action="store_true",help="Plot the effect of beta")
parser.add_argument("--dim",    "-d",action="store_true",help="Plot the effect of latent dimension")
parser.add_argument("--nfields","-n",action="store_true",help="Plot the effect of number of training data")
Args        =   parser.parse_args()


#Plot config
colorlist= ["#A64036","#F0C2A2","#4182A4","#354E6B"]
colorlist.reverse()
china_color = colors.LinearSegmentedColormap.from_list('china_color',colorlist, N=100)
plt.register_cmap(cmap=china_color)
plt.set_cmap(china_color)

font_dict = {"weight":"bold","size":22}
plt.rc("font",family = "serif")
plt.rc("font",size = 14)
plt.rc("axes",labelsize = 16, linewidth = 2)
plt.rc("legend",fontsize= 12, handletextpad = 0.3)
plt.rc("xtick",labelsize = 14)
plt.rc("ytick",labelsize = 14)

baseDir     =   "/scratch/yuningw/beta_vae_cylinder/"
save_matrix_to =   baseDir + "10_Post_Figs/Corr/"

################################################################
# Effect of beta
###############################################################

if Args.beta:
    print("#"*30)
    print("Plotting the Correlation Matrix for effect of beta:")

    num_fields      =   25999
    latent_dim      =   10
    betas           =   [0.001, 0.0025, 0.005, 0.01]
    Epoch           =   300
    vae_type        =   "v5"
    batch_size      =   128
    earlystop       =   False
    patience        =   0

    #compute detR
    Mats = []; detRs = []
    for beta in betas:

        filesID   =  f"{vae_type}_{int( num_fields )}n_{latent_dim}d_{int(beta*10000)}e-4beta_"+\
                    f"{args.block_type}conv_{len(args.filters)}Nf_{args.filters[-1]}Fdim_{args.linear_dim}Ldim"+\
                    f"{args.act_conv}convact_{args.act_linear}_" +\
                    f"{int(args.lr *1e5)}e-5LR_{int(args.w_decay*1e5)}e-5Wd"+\
                    f"{batch_size}bs_{Epoch}epoch_{earlystop}ES_{patience}P"


        modes_filepath = baseDir+ "03_Mode/"+filesID +"modes"+ ".npz"

        print(f"Loading case: \n{filesID}")

        d       =   np.load(modes_filepath)
        z_mean  =   np.array(d["z_mean"])

        corr_matrix_latent = abs(np.corrcoef(z_mean.T))
        detR    =   np.linalg.det(corr_matrix_latent)
        print(f"In order to confirm the case ,we confirm the detR is {np.round(detR,4)}")
        Mats.append(corr_matrix_latent)
        detRs.append(detR)

    betas_label = [r"$1\times10^{-3}$", r"$2.5\times10^{-3}$",r"$5\times10^{-3}$", r"$1\times10^{-2}$"  ]

    fig, axs    = plt.subplots(1,4,figsize=(12, 8),sharey=True)
    axs         = axs.flatten()
    for ind, ax in enumerate(axs):
        corr_matrix_latent = Mats[ind]
        cb                 = ax.imshow(corr_matrix_latent)
        ax.set_title(r"${\beta}$"+ " = " + betas_label[ind]+"\n" + r"${{\rm det}_{\mathbf{R}}}$" +f" = {np.round(100*detRs[ind],2)}")
        ax.set_xticks(range(latent_dim))
        ax.set_yticks(range(latent_dim))
        ax.set_xticklabels(range(1,latent_dim+1))
        ax.set_yticklabels(range(1,latent_dim+1))
        ax.set_xlabel(r"$z_i$",fontdict = font_dict )
    axs[0].set_ylabel(r"$z_i$",fontdict = font_dict )
    cax = fig.add_axes([axs[3].get_position().x1+0.03,axs[3].get_position().y0,0.02,0.25])
    cbar = fig.colorbar(cb, cax=cax)
    cbar.ax.locator_params(nbins = 5,tight=True)
    # plt.tight_layout()
    plt.savefig(save_matrix_to +"Corr_"+vae_type+"_n"+str(num_fields)+'_m'+str(latent_dim)+'_b_all'+"_epoch" + str(Epoch),
                bbox_inches = "tight",
                dpi=300)

 
    print("The Fig has been saved")
    print("#"*30)



################################################################
# Effect of latent dim
###############################################################
if Args.dim:
    print("#"*30)
    print("Plotting the PDF for effect of Latent dimension:")

    num_fields      =   25999
    latent_dims     =   [10,15,20]
    beta            =   0.0025
    Epoch           =   300
    vae_type        =   "v5"
    batch_size      =   128
    earlystop       =   False
    patience        =   0

    #compute detR
    Mats = [] ; detRs = []
    for latent_dim in latent_dims:

        filesID   =  f"{vae_type}_{int( num_fields )}n_{latent_dim}d_{int(beta*10000)}e-4beta_"+\
                    f"{args.block_type}conv_{len(args.filters)}Nf_{args.filters[-1]}Fdim_{args.linear_dim}Ldim"+\
                    f"{args.act_conv}convact_{args.act_linear}_" +\
                    f"{int(args.lr *1e5)}e-5LR_{int(args.w_decay*1e5)}e-5Wd"+\
                    f"{batch_size}bs_{Epoch}epoch_{earlystop}ES_{patience}P"


        modes_filepath = baseDir+ "03_Mode/"+filesID +"modes"+ ".npz"

        print(f"Loading case: \n{filesID}")
        d       =   np.load(modes_filepath)
        z_mean  =   np.array(d["z_mean"])

        corr_matrix_latent = abs(np.corrcoef(z_mean.T))
        detR    =   np.linalg.det(corr_matrix_latent)
        print(f"In order to confirm the case ,we confirm the detR is {np.round(detR,4)}")
        Mats.append(corr_matrix_latent)
        detRs.append(detR)

    

    fig, axs    = plt.subplots(1,len(Mats),figsize=(14, 8))
    axs         = axs.flatten()
    for ind, ax in enumerate(axs):
        corr_matrix_latent = Mats[ind]
        cb                 = ax.imshow(corr_matrix_latent)
        ax.set_aspect("equal")
        ax.set_title(r"$d$"+ " = " + str(latent_dims[ind])+"\n" + r"${{\rm det}_{\mathbf{R}}}$" +f" = {np.round(100*detRs[ind],2)}")
        ax.set_xticks(range(latent_dims[ind]))
        ax.set_yticks(range(latent_dims[ind]))
        ax.set_xticklabels(range(1,latent_dims[ind]+1))
        ax.set_yticklabels(range(1,latent_dims[ind]+1))
        ax.set_xlabel(r"$z_i$",fontdict = font_dict )
        
    axs[0].set_ylabel(r"$z_i$",fontdict = font_dict )
    cax = fig.add_axes([axs[-1].get_position().x1+0.03,axs[-1].get_position().y0,0.02,0.25])
    cbar = fig.colorbar(cb, cax=cax)
    cbar.ax.locator_params(nbins = 5,tight=True)
    
    plt.savefig( save_matrix_to +"Corr_" + vae_type + f"_n_{int(num_fields)}_m_all_b_{int(beta*10000)}e-4_Epoch_{Epoch}",
                bbox_inches = "tight",
                dpi=300)

    # plt.savefig(save_pdf_to + "10_Post_Figs/PDF/"+"PDF_"+vae_type+"_n"+str(num_fields)+'_m'+str(latent_dim)+'_b_all'+"_epoch" + str(Epoch) + ".pdf",
    #             bbox_inches = "tight",
    #              dpi=300)
    print("The Fig has been saved")
    print("#"*30)



    
################################################################
# Effect of Nfields
###############################################################
if Args.nfields:

    print("#"*30)
    print("Plotting the PDF for effect of Nfield:")

    Num_fields      =   [int(25999*0.25), int(25999*0.5), int(25999*0.75), int(25999*1)]
    latent_dim      =   10
    beta            =   0.0025
    Epoch           =   300
    vae_type        =   "v5"
    batch_size      =   128
    earlystop       =   False
    patience        =   0

    #compute detR
    Mats = []; detRs = []
    for num_fields in Num_fields:

        filesID   =  f"{vae_type}_{int( num_fields )}n_{latent_dim}d_{int(beta*10000)}e-4beta_"+\
                    f"{args.block_type}conv_{len(args.filters)}Nf_{args.filters[-1]}Fdim_{args.linear_dim}Ldim"+\
                    f"{args.act_conv}convact_{args.act_linear}_" +\
                    f"{int(args.lr *1e5)}e-5LR_{int(args.w_decay*1e5)}e-5Wd"+\
                    f"{batch_size}bs_{Epoch}epoch_{earlystop}ES_{patience}P"


        modes_filepath = baseDir+ "03_Mode/"+filesID +"modes"+ ".npz"

        print(f"Loading case: \n{filesID}")

        d       =   np.load(modes_filepath)
        z_mean  =   np.array(d["z_mean"])

        corr_matrix_latent = abs(np.corrcoef(z_mean.T))
        detR    =   np.linalg.det(corr_matrix_latent)
        print(f"In order to confirm the case ,we confirm the detR is {np.round(detR,4)}")
        Mats.append(corr_matrix_latent)
        detRs.append(detR)

    

    fig, axs    = plt.subplots(1,len(Mats),figsize=(14, 8))
    axs         = axs.flatten()
    for ind, ax in enumerate(axs):
        corr_matrix_latent = Mats[ind]
        cb                 = ax.imshow(corr_matrix_latent)
        ax.set_aspect("equal")
        ax.set_title(r"$N_{\rm fields}$"+ " = " + str(Num_fields[ind]+1)+"\n" + r"${{\rm det}_{\mathbf{R}}}$" +f" = {np.round(100*detRs[ind],2)}")
        ax.set_xticks(range(latent_dim))
        ax.set_yticks(range(latent_dim))
        ax.set_xticklabels(range(1,latent_dim+1))
        ax.set_yticklabels(range(1,latent_dim+1))
        ax.set_xlabel(r"$z_i$",fontdict = font_dict )
        
    axs[0].set_ylabel(r"$z_i$",fontdict = font_dict )
    cax = fig.add_axes([axs[-1].get_position().x1+0.03,axs[-1].get_position().y0,0.02,0.25])
    cbar = fig.colorbar(cb, cax=cax)
    cbar.ax.locator_params(nbins = 5,tight=True)
    
    plt.savefig( save_matrix_to +"Corr_" + vae_type + f"_n_all_m_{latent_dim}_b_{int(beta*10000)}e-4_Epoch_{Epoch}",
                bbox_inches = "tight",
                dpi=300)

    print("The Fig has been saved")
    print("#"*30)


