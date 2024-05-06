"""
PDF Plot for latent variables analysis
@yuninw
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

"""
Note that to plot all the figures, please use:
    
    python PlotPdf.py -b -d -n

"""

parser      =   argparse.ArgumentParser("To specify which PDF to plot")
parser.add_argument("--beta",   "-b",action="store_true",help="Plot the effect of beta")
parser.add_argument("--dim",    "-d",action="store_true",help="Plot the effect of latent dimension")
parser.add_argument("--nfields","-n",action="store_true",help="Plot the effect of number of training data")
parser.add_argument("--vae",    "-v",default=5,type= int,help="The type of vae")
Args        =   parser.parse_args()
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
save_pdf_to =   baseDir + "Figs/PDF/"
# Arguments for FileID
# We first investigate the effect of beta



################################################################
# Effect of beta
###############################################################

if Args.beta:
    print("#"*30)
    print("Plotting the PDF for effect of beta:")

    num_fields      =   25999
    latent_dim      =   10
    betas           =   [0.001, 0.0025, 0.005, 0.01]
    Epoch           =   300
    vae_type        =   f"v{Args.vae}"
    batch_size      =   128
    earlystop       =   False
    patience        =   0

    #compute detR
    modes = []
    for beta in betas:

        filesID   =  f"{vae_type}_{int( num_fields )}n_{latent_dim}d_{int(beta*10000)}e-4beta_"+\
                    f"{args.block_type}conv_{len(args.filters)}Nf_{args.filters[-1]}Fdim_{args.linear_dim}Ldim"+\
                    f"{args.act_conv}convact_{args.act_linear}_" +\
                    f"{int(args.lr *1e5)}e-5LR_{int(args.w_decay*1e5)}e-5Wd"+\
                    f"{batch_size}bs_{Epoch}epoch_{earlystop}ES_{patience}P"


        modes_filepath = baseDir+ "03_Mode/"+filesID +"modes"+ ".npz"

        print(f"Loading case: \n{filesID}")

        d = np.load(modes_filepath)
        z_mean = np.array(d["z_mean"])

        corr_matrix_latent = abs(np.corrcoef(z_mean.T))
        detR = np.linalg.det(corr_matrix_latent)
        print(f"In order to confirm the case ,we confirm the detR is {np.round(detR,4)}")
        modes.append(z_mean)

    # Compute the standard deviation
    Inds = []
    for m in modes:
        std = np.std(m,axis=0)
        ind = std > 0.5
        Inds.append(ind)

    # Create a figure for plot 
    fig, axs = plt.subplots(1, len(modes),
                            sharex=True, sharey= True,
                            figsize =(4*len(modes),3)
                            )

    #Plot PDF 1x4 vs Beta
    axs = axs.flatten()
    for j, ax in enumerate(axs):
        z_mean  =   modes[j]
        kdes    =   []
        for i in range(z_mean.shape[1]):
            ri      = z_mean[:,i]
            xx      = np.linspace(-2.5,2.5,z_mean.shape[0])
            pdf     = gaussian_kde(ri)
            kdes.append(pdf(xx))
        kdes    =   np.array(kdes)
        inds    =   Inds[j]
        # generate the PDF through gaussain
        for i in range(z_mean.shape[1]):
            if inds[i]:
                ax.plot(xx,         kdes[i]/kdes[i].max(),  zorder= 3,  c = blue)
                ax.fill_between(xx, kdes[i]/kdes[i].max(),  zorder= 3,  color = blue,   alpha =0.15)
            else:
                ax.plot(xx,         kdes[i]/kdes[i].max(),  zorder= 5,  c = red)
        n_noise = inds.shape[0] -  np.count_nonzero(inds)
        
        ax.set_xlim(-2.5,2.5)
        ax.set_ylim(0,1.2)
        ax.set_xlabel(r"$z_i$",fontdict = font_dict)

        # For Label of each pannel
        if betas[j] < 0.01: 
            ax.set_title(r"$\beta$ "+f"= {np.round(betas[j]*1000,1)}" + r"$\times$ " + r"$10^3$" )
        
        elif betas[j] >= 0.01:
            ax.set_title(r"$\beta$ "+f"= {np.round(betas[j]*100,1)}" + r"$\times$ " + r"$10^2$" )
        print(f"For beta =  {betas[j]}, there are {n_noise} noises")
    axs[0].set_ylabel("PDF")
        

    plt.savefig(save_pdf_to +"PDF_"+vae_type+"_n"+str(num_fields)+'_m'+str(latent_dim)+'_b_all'+"_epoch" + str(Epoch),
                bbox_inches = "tight",
                dpi=300)

    # plt.savefig(save_pdf_to + "10_Post_Figs/PDF/"+"PDF_"+vae_type+"_n"+str(num_fields)+'_m'+str(latent_dim)+'_b_all'+"_epoch" + str(Epoch) + ".pdf",
    #             bbox_inches = "tight",
    #              dpi=300)
    print("The Fig has been saved")
    print("#"*30)




################################################################
# Effect of latent dim
###############################################################
if Args.dim:
    print("#"*30)
    print("Plotting the PDF for effect of Latent dimension:")

    num_fields      =   25999
    latent_dims     =   [10,15,20,25]
    beta            =   0.0025
    Epoch           =   300
    vae_type        =   f"v{Args.vae}"
    batch_size      =   128
    earlystop       =   False
    patience        =   0

    #compute detR
    modes = []
    for latent_dim in latent_dims:

        filesID   =  f"{vae_type}_{int( num_fields )}n_{latent_dim}d_{int(beta*10000)}e-4beta_"+\
                    f"{args.block_type}conv_{len(args.filters)}Nf_{args.filters[-1]}Fdim_{args.linear_dim}Ldim"+\
                    f"{args.act_conv}convact_{args.act_linear}_" +\
                    f"{int(args.lr *1e5)}e-5LR_{int(args.w_decay*1e5)}e-5Wd"+\
                    f"{batch_size}bs_{Epoch}epoch_{earlystop}ES_{patience}P"


        modes_filepath = baseDir+ "03_Mode/"+filesID +"modes"+ ".npz"

        print(f"Loading case: \n{filesID}")

        d = np.load(modes_filepath)
        z_mean = np.array(d["z_mean"])

        corr_matrix_latent = abs(np.corrcoef(z_mean.T))
        detR = np.linalg.det(corr_matrix_latent)
        print(f"In order to confirm the case ,we confirm the detR is {np.round(detR,4)}")
        modes.append(z_mean)

    # Compute the standard deviation
    Inds = []
    for m in modes:
        std = np.std(m,axis=0)
        ind = std > 0.5
        Inds.append(ind)

    # Create a figure for plot 
    fig, axs = plt.subplots(1, len(modes),
                            sharex=True, sharey= True,
                            figsize =(4*len(modes),3)
                            )

    #Plot PDF 1x4 vs Beta
    axs = axs.flatten()
    for j, ax in enumerate(axs):
        z_mean  =   modes[j]
        kdes    =   []
        for i in range(z_mean.shape[1]):
            ri      = z_mean[:,i]
            xx      = np.linspace(-2.5,2.5,z_mean.shape[0])
            pdf     = gaussian_kde(ri)
            kdes.append(pdf(xx))
        kdes    =   np.array(kdes)
        inds    =   Inds[j]
        # generate the PDF through gaussain
        for i in range(z_mean.shape[1]):
            if inds[i]:
                ax.plot(xx,         kdes[i]/kdes[i].max(),  zorder= 3,  c = blue)
                ax.fill_between(xx, kdes[i]/kdes[i].max(),  zorder= 3,  color = blue,   alpha =0.15)
            else:
                ax.plot(xx,         kdes[i]/kdes[i].max(),  zorder= 5,  c = red)
        n_noise = inds.shape[0] -  np.count_nonzero(inds)
        
        ax.set_xlim(-2.5,2.5)
        ax.set_ylim(0,1.2)

        # For Label of each pannel
        ax.set_title(r"d" + " = " + f"{latent_dims[j]}",font_dict)
        ax.set_xlabel(r"$z_i$",fontdict = font_dict)


        print(f"For latent dim =  {latent_dims[j]}, there are {n_noise} noises")
        
    axs[0].set_ylabel("PDF")


    plt.savefig(save_pdf_to +"PDF_" + vae_type + f"_n_{int(num_fields)}_m_all_b_{int(beta*10000)}e-4_Epoch_{Epoch}",
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
    vae_type        =   f"v{Args.vae}"
    batch_size      =   128
    earlystop       =   False
    patience        =   0

    #compute detR
    modes = []
    for num_fields in Num_fields:

        filesID   =  f"{vae_type}_{int( num_fields )}n_{latent_dim}d_{int(beta*10000)}e-4beta_"+\
                    f"{args.block_type}conv_{len(args.filters)}Nf_{args.filters[-1]}Fdim_{args.linear_dim}Ldim"+\
                    f"{args.act_conv}convact_{args.act_linear}_" +\
                    f"{int(args.lr *1e5)}e-5LR_{int(args.w_decay*1e5)}e-5Wd"+\
                    f"{batch_size}bs_{Epoch}epoch_{earlystop}ES_{patience}P"


        modes_filepath = baseDir+ "03_Mode/"+filesID +"modes"+ ".npz"

        print(f"Loading case: \n{filesID}")

        d = np.load(modes_filepath)
        z_mean = np.array(d["z_mean"])

        corr_matrix_latent = abs(np.corrcoef(z_mean.T))
        detR = np.linalg.det(corr_matrix_latent)
        print(f"In order to confirm the case ,we confirm the detR is {np.round(detR,4)}")
        modes.append(z_mean)

    # Compute the standard deviation
    Inds = []
    for m in modes:
        std = np.std(m,axis=0)
        ind = std > 0.5
        Inds.append(ind)

    # Create a figure for plot 
    fig, axs = plt.subplots(1, len(modes),
                            sharex=True, sharey= True,
                            figsize =(4*len(modes),3)
                            )

    #Plot PDF 1x4 vs Beta
    axs = axs.flatten()
    for j, ax in enumerate(axs):
        z_mean  =   modes[j]
        kdes    =   []
        for i in range(z_mean.shape[1]):
            ri      = z_mean[:,i]
            xx      = np.linspace(-2.5,2.5,z_mean.shape[0])
            pdf     = gaussian_kde(ri)
            kdes.append(pdf(xx))
        kdes    =   np.array(kdes)
        inds    =   Inds[j]
        # generate the PDF through gaussain
        for i in range(z_mean.shape[1]):
            if inds[i]:
                ax.plot(xx,         kdes[i]/kdes[i].max(),  zorder= 3,  c = blue)
                ax.fill_between(xx, kdes[i]/kdes[i].max(),  zorder= 3,  color = blue,   alpha =0.15)
            else:
                ax.plot(xx,         kdes[i]/kdes[i].max(),  zorder= 5,  c = red)
        n_noise = inds.shape[0] -  np.count_nonzero(inds)
        
        ax.set_xlim(-2.5,2.5)
        ax.set_ylim(0,1.2)

        # For Label of each pannel
        ax.set_title(r"$N_{\rm fields}$" + " = " + f"{Num_fields[j]+1}",font_dict)
        ax.set_xlabel(r"$z_i$",fontdict = font_dict)
        
        print(f"For Nfields =  {Num_fields[j]}, there are {n_noise} noises")
        
    axs[0].set_ylabel("PDF")

    plt.savefig(save_pdf_to +"PDF_" + vae_type + f"_n_all_m_{latent_dim}_b_{int(beta*10000)}e-4_Epoch_{Epoch}",
                bbox_inches = "tight",
                dpi=300)

    print("The Fig has been saved")
    print("#"*30)


