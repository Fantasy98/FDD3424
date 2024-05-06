"""
Line Plot for parameter analysis 
@yuningw
"""
# Environment 
import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils.configs import VAE_custom as  args, Name_Costum_VAE
import os
import pandas as pd
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Unlock the h5py file



parser      =   argparse.ArgumentParser("To specify which Line to plot")
parser.add_argument("--vae",    "-v",default=5,type= int,help="The type of vae")
Args        =   parser.parse_args()

plt.rc("font",family = "serif")
plt.rc("font",size = 14)
plt.rc("axes",labelsize = 16, linewidth = 2)
plt.rc("legend",fontsize= 12, handletextpad = 0.3)
plt.rc("xtick",labelsize = 14)
plt.rc("ytick",labelsize = 14)

class line_beta2:
    red = "#D23918" # luoshenzhu
    blue = "#2E59A7" # qunqing
    yellow = "#E5A84B" # huanghe liuli
    cyan = "#5DA39D" # er lv
    black = "#151D29" # lanjian
font_dict = {"weight":"bold","size":22}

#load CSV

base_dir        =   os.getcwd()
base_dir        += "/"
save_line_path  =   "10_Post_Figs/Lines/" 

df = pd.read_csv(base_dir+"vae_results4Beta_copy.csv")

print(f"INFO: The CSV file has been loaded, {df.info()}")

vae_type    =   args.model
betas       =   [ 0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.05]
Latent_Dims  =   [ 10, 15, 20, 25, 30]
Num_Fields  =   [int(25999*0.25), int(25999*0.5), int(25999*0.75), int(25999*1)]
Num_Field  =   25999
Epochs      =   300
Latent_Dim  =   args.latent_dim
beta        =   0.0001

################################################################
# Effect of beta
###############################################################


print("#"*30)
print("Investigate the Effect of Beta")
# Use condition plot for investigate the effect of beta :

# vae_type    =   ["v35" , "v4",  "v45", "v5", "v55"]
vae_type    =   ["v4", "v5"]

df_cs  = []

for ind, vae_tp  in enumerate(vae_type): 
  df_c =   df[  (df["Type"]         ==  vae_type[ind])&
          (df["N_field"]       ==  Num_Field) &
          (df["latent_dim"]   ==  Latent_Dim) &
          (df["Epoch"]        ==  Epochs)&
          (df["beta"]         <=  betas[-1]) & 
          (df["beta"]         >=  betas[0])&
          (df["E_k"]         >=  90.0)
           ]
  df_c = df_c.sort_values(by= ["beta"],ascending=True)
  df_cs.append(df_c)
  print(f"The filtered DataFrame is:\n{df_c.head()}")






ls = [ "-", "--", "-.", ":", "-"]
mk = ["o", "v", "s", "P", "X"]

# Betas = [   
#         [ 0.0001, 0.0005, 0.001, 0.0025, 0.005],
#         [ 0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01], 
#         [0.0001, 0.0005, 0.001, 0.0025, 0.005,  0.01], 
#         [ 0.001, 0.0025, 0.005,0.01],
#         [ 0.01]
         
#                        ]


Betas = [   
       
        [ 0.001, 0.0025, 0.005,  0.01], 
        [ 0.001, 0.0025, 0.005,  0.01], 
         
                       ]


###############################################################


print("#"*30)
feature1 = "E_k"
feature2 = "R"

print(f"INFO: Plotting results on {feature1} and {feature2}")

# The line plots setup 
fig, ax = plt.subplots(1,1,sharex=True,figsize = (8,6))
ax2 = ax.twinx()

# The color used for two axis and labels
ax.tick_params("y",colors = line_beta2.blue) ;ax.yaxis.label.set_color(line_beta2.blue)
ax2.tick_params("y",colors = line_beta2.red) ;ax2.yaxis.label.set_color(line_beta2.red)
ax2.spines["left"].set_color(line_beta2.blue);ax2.spines["right"].set_color(line_beta2.red)
for ind, df_c in enumerate(df_cs):
  print(Betas[ind])
  ax.semilogx(Betas[ind], df_c[feature1],
          linestyle = ls[ind], marker = mk[ind],
          c= line_beta2.blue, lw = 2, markersize = 8.5)
  ax2.semilogx(Betas[ind], 100* df_c[feature2],
           linestyle = ls[ind], marker = mk[ind],
           c= line_beta2.red, lw = 2, markersize = 8.5)
  print(f"At case {ind}")

ax.set_ylim(90, 100);
ax2.set_ylim(96, 100);
#print(np.linspace(0.0001, 0.1, 3))
ax.set_xticks(np.array([0.001, 0.005,0.01]))
ax.set_xticklabels(np.array([0.001,0.005, 0.01]))

ax.set_ylabel(r"$E_{k} (\%)$",fontdict = font_dict)
ax.set_xlabel(r"$\beta$",fontdict = font_dict)
ax2.set_ylabel(r"${\rm det}_{\mathbf{R}}$",fontdict = font_dict)

legend =plt.legend(["Arch1", "Arch2", "Arch3", "Arch4", "Arch5"], loc = "upper right", ncol = 1, bbox_to_anchor = (1, 1), frameon = False)

for text in legend.get_texts():
  text.set_color('black')
for handle in legend.legendHandles:
    if isinstance(handle, plt.Line2D):  # Check if it's a Line2D handle (line in the legend)
        handle.set_color('black')
    else:
        handle.set_markerfacecolor('black')
        handle.set_markeredgecolor('black')

plt.savefig(save_line_path+ f"{feature1}_{feature2}_n_{Num_Field}_m_{Latent_Dim}_b_all_rec_det.pdf", bbox_inches = "tight", dpi = 1000)


################################################################################

print("#"*30)
feature1 = "kl_loss"
feature2 = "R"

print(f"INFO: Plotting results on {feature1} and {feature2}")

fig, ax = plt.subplots(1,1,sharex=True,figsize =(8,6))
ax2 = ax.twinx()

# The color used for two axis and labels
ax.tick_params("y",colors = line_beta2.cyan) ;ax.yaxis.label.set_color(line_beta2.cyan)
ax2.tick_params("y",colors = line_beta2.red) ;ax2.yaxis.label.set_color(line_beta2.red)
ax2.spines["left"].set_color(line_beta2.cyan);ax2.spines["right"].set_color(line_beta2.red)


for ind, df_c in enumerate(df_cs):
  print(Betas[ind])
  ax.semilogx(Betas[ind], df_c[feature1],
          linestyle = ls[ind], marker = mk[ind],
          c= line_beta2.cyan, lw = 2, markersize = 8.5)
  ax2.semilogx(Betas[ind], 100* df_c[feature2],
           linestyle = ls[ind], marker = mk[ind],
           c= line_beta2.red, lw = 2, markersize = 8.5)
  print(f"At case {ind}")

ax.set_ylim(1, 5);
ax2.set_ylim(96, 100);
#print(np.linspace(0.0001, 0.1, 3))
ax.set_xticks(np.array([ 0.001,0.005, 0.01]))
ax.set_xticklabels(np.array([ 0.001,0.005, 0.01]))

ax.set_ylabel(r"$D_{KL}$",fontdict = font_dict)
ax.set_xlabel(r"$\beta$",fontdict = font_dict)
ax2.set_ylabel(r"${\rm det}_{\mathbf{R}}$",fontdict = font_dict)

legend =plt.legend(["Arch1", "Arch2", "Arch3", "Arch4", "Arch5"], loc = "upper right", ncol = 1, bbox_to_anchor = (1, 1), frameon = False)

for text in legend.get_texts():
  text.set_color('black')
for handle in legend.legendHandles:
    if isinstance(handle, plt.Line2D):  # Check if it's a Line2D handle (line in the legend)
        handle.set_color('black')
    else:
        handle.set_markerfacecolor('black')
        handle.set_markeredgecolor('black')

plt.savefig(save_line_path+ f"{feature1}_{feature2}_all_n_{Num_Field}_m_{Latent_Dim}_b_all_rec_det.pdf", bbox_inches = "tight", dpi = 1000)


################################################################


print("#"*30)
feature1 = "E_k"
feature2 = "kl_loss"

print(f"INFO: Plotting results on {feature1} and {feature2}")

fig, ax = plt.subplots(1,1,sharex=True,figsize = (8, 6))
ax2 = ax.twinx()

# The color used for two axis and labels
ax.tick_params("y",colors = line_beta2.blue)
ax.yaxis.label.set_color(line_beta2.blue)
ax2.tick_params("y",colors = line_beta2.cyan)
ax2.yaxis.label.set_color(line_beta2.cyan)
ax2.spines["left"].set_color(line_beta2.blue)
ax2.spines["right"].set_color(line_beta2.cyan)


for ind, df_c in enumerate(df_cs):
  print(Betas[ind])
  ax.semilogx(Betas[ind], df_c[feature1],
          linestyle = ls[ind], marker = mk[ind],
          c= line_beta2.blue, lw = 2, markersize = 8.5)
  ax2.semilogx(Betas[ind], df_c[feature2],
           linestyle = ls[ind], marker = mk[ind],
           c= line_beta2.cyan, lw = 2, markersize = 8.5)
  print(f"At case {ind}")

ax.set_ylim(90, 100);
ax2.set_ylim(1, 5);
#print(np.linspace(0.0001, 0.1, 3))
ax.set_xticks(np.array([ 0.001,0.005, 0.01]))
ax.set_xticklabels(np.array([ 0.001,0.005, 0.01]))

ax.set_ylabel(r"$E_k (\%)$",fontdict = font_dict)
ax.set_xlabel(r"$\beta$",fontdict = font_dict)
ax2.set_ylabel(r"$D_{KL}$",fontdict = font_dict)

legend =plt.legend(["Arch1", "Arch2", "Arch3", "Arch4", "Arch5"], loc = "upper right", ncol = 1, bbox_to_anchor = (1, 1), frameon = False)

for text in legend.get_texts():
  text.set_color('black')
for handle in legend.legendHandles:
    if isinstance(handle, plt.Line2D):  # Check if it's a Line2D handle (line in the legend)
        handle.set_color('black')
    else:
        handle.set_markerfacecolor('black')
        handle.set_markeredgecolor('black')

plt.savefig(save_line_path+ f"{feature1}_{feature2}_all_n_{Num_Field}_m_{Latent_Dim}_b_all_rec_det.pdf", bbox_inches = "tight", dpi = 1000)


################################################################