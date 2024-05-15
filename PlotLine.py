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

base_dir        =   os.getcwd() + '/'
save_line_path  =   "Figs/Lines/" 

df = pd.read_csv(base_dir+"vae_results.csv")

print(f"INFO: The CSV file has been loaded, {df.info()}")



################################################################
# Effect of beta
###############################################################


print("#"*30)
print("Investigate the Effect of Beta")
# Use condition plot for investigate the effect of beta :
vae_type    =   f"v{Args.vae}"
betas       =   [0.001,0.0025,0.005,0.01]
Num_Fields  =   25999
Epochs      =   300
Latent_Dim  =   10

df_c = df[  (df["Type"]         ==  vae_type)&
            (df["Nfield"]       ==  Num_Fields) &
            (df["latent_dim"]   ==  Latent_Dim) &
            (df["Epoch"]        ==  Epochs)&
            (df["beta"]         <=  betas[-1]) & 
            (df["beta"]         >=  betas[0])
             ]
df_c = df_c.drop_duplicates(["beta"])
df_c = df_c.sort_values(by= ["beta"],ascending=True)
print(f"The filtered DataFrame is:\n{df_c.head()}")
# The line plots setup 
fig, ax = plt.subplots(1,1,sharex=True,figsize = (6,6))
ax2 = ax.twinx()

# The color used for two axis and labels
ax.tick_params("y",colors = line_beta2.blue)
ax.yaxis.label.set_color(line_beta2.blue)
ax2.tick_params("y",colors = line_beta2.red)
ax2.yaxis.label.set_color(line_beta2.red)
ax2.spines["left"].set_color(line_beta2.blue)
ax2.spines["right"].set_color(line_beta2.red)



if vae_type == "v4":
  mk = "o"; ls = "-"
elif vae_type == "v45":
  mk = "s"; ls = "-."
elif vae_type== "v5":
  mk = "v"; ls = "--"

ax.plot(betas, df_c["E_k"],
        linestyle = ls, marker = mk,
        c= line_beta2.blue, lw = 2, markersize = 8.5)

ax2.plot(betas, 100* df_c["R"],
         linestyle = ls, marker = mk,
         c= line_beta2.red, lw = 2, markersize = 8.5)

ax.set_ylim(90, 100);
ax2.set_ylim(90, 100);

ax.set_xticklabels(betas)
ax.set_xticks(betas)
ax.set_ylabel(r"$E_{k} (\%)$",fontdict = font_dict)
ax.set_xlabel(r"$\beta$",fontdict = font_dict)
ax2.set_ylabel(r"${\rm det}_{\mathbf{R}}$",fontdict = font_dict)
plt.savefig(save_line_path+ f"{vae_type}_n_{Num_Fields}_m{Latent_Dim}_b_all_rec_det.pdf", bbox_inches = "tight", dpi = 300)



################################################################
# Effect of Latent Dimension
###############################################################

print("#"*30)
print("Investigate the Effect of Latent Dimension")
vae_type    =   f"v{Args.vae}"
betas       =   0.0025
Num_Fields  =   25999
Epochs      =   300
Latent_Dim  =   [10,15,20,25]
df = pd.read_csv(base_dir+"vae_results.csv")
df_c = df[
            (df["Nfield"]       ==  Num_Fields) &
            (df["latent_dim"]   >=  Latent_Dim[0]) &(df["latent_dim"]  <=  Latent_Dim[-1])&
            (df["Epoch"]        ==  Epochs)&
            (df["beta"]         ==  betas)
             ]
df_c = df_c.drop_duplicates(["latent_dim"])
df_c = df_c.sort_values(by= ["latent_dim"],ascending=True)
print(f"The filtered DataFrame is:\n{df_c.head()}")
# The line plots setup 
fig, ax = plt.subplots(1,1,sharex=True,figsize = (6,6))
ax2 = ax.twinx()

# The color used for two axis and labels
ax.tick_params("y",colors = line_beta2.blue)
ax.yaxis.label.set_color(line_beta2.blue)
ax2.tick_params("y",colors = line_beta2.red)
ax2.yaxis.label.set_color(line_beta2.red)
ax2.spines["left"].set_color(line_beta2.blue)
ax2.spines["right"].set_color(line_beta2.red)



if vae_type == "v4":
  mk = "o"; ls = "-"
else:
  mk = "v"; ls = "--"

ax.plot(Latent_Dim, df_c["E_k"],
        linestyle = ls, marker = mk,
        c= line_beta2.blue, lw = 2, markersize = 8.5)

ax2.plot(Latent_Dim, 100* df_c["R"],
         linestyle = ls, marker = mk,
         c= line_beta2.red, lw = 2, markersize = 8.5)

ax.set_ylim(90, 100);
ax2.set_ylim(0, 100);

ax.set_xticklabels(Latent_Dim)
ax.set_xticks(Latent_Dim)
ax.set_ylabel(r"$E_{k} (\%)$",fontdict = font_dict)
ax.set_xlabel(r"$d$",fontdict = font_dict)
ax2.set_ylabel(r"${\rm det}_{\mathbf{R}}$",fontdict = font_dict)
plt.savefig(save_line_path+ f"{vae_type}_n_{Num_Fields}_m_all_b_{betas*10000}e-4_rec_det.pdf", bbox_inches = "tight", dpi = 300)


################################################################
# Effect of Nfield 
###############################################################

print("#"*30)
print("Investigate the Effect of Number of fields for training")
vae_type    =   f"v{Args.vae}"
betas       =   0.0025
Num_Fields  =   [int(25999*0.25), int(25999*0.5), int(25999*0.75), int(25999*1)]
Epochs      =   300
Latent_Dim  =   10
df = pd.read_csv(base_dir+"vae_results.csv")
df_c = df[
            (df["Nfield"]       >=  Num_Fields[0]) & (df["Nfield"]       <=  Num_Fields[-1]) & 
            (df["latent_dim"]   ==  Latent_Dim) &
            (df["Epoch"]        ==  Epochs)&
            (df["beta"]         ==  betas)
             ]
df_c = df_c.drop_duplicates(["Nfield"])
df_c = df_c.sort_values(by= ["Nfield"],ascending=True)

print(f"The filtered DataFrame is:\n{df_c.head()}")
# The line plots setup 
fig, ax = plt.subplots(1,1,sharex=True,figsize = (6,6))
ax2 = ax.twinx()

# The color used for two axis and labels
ax.tick_params("y",colors = line_beta2.blue)
ax.yaxis.label.set_color(line_beta2.blue)
ax2.tick_params("y",colors = line_beta2.red)
ax2.yaxis.label.set_color(line_beta2.red)
ax2.spines["left"].set_color(line_beta2.blue)
ax2.spines["right"].set_color(line_beta2.red)


Num_Fields  =   [int(26000*0.25), int(26000*0.5), int(26000*0.75), int(26000*1)]

if vae_type == "v4":
  mk = "o"; ls = "-"
else:
  mk = "v"; ls = "--"



ax.plot(Num_Fields, df_c["E_k"],
        linestyle = ls, marker = mk,
        c= line_beta2.blue, lw = 2, markersize = 8.5)

ax2.plot(Num_Fields, 100* df_c["R"],
         linestyle = ls, marker = mk,
         c= line_beta2.red, lw = 2, markersize = 8.5)

ax.set_ylim(0, 100);
ax2.set_ylim(70, 100);

ax.set_xticklabels(Num_Fields)
ax.set_xticks(Num_Fields)
ax.set_ylabel(r"$E_{k} (\%)$",fontdict = font_dict)
ax.set_xlabel(r"$N_{\rm fields}$",fontdict = font_dict)
ax2.set_ylabel(r"${\rm det}_{\mathbf{R}}$",fontdict = font_dict)
plt.savefig(save_line_path+ f"{vae_type}_n_all_m_{Latent_Dim}_b_{betas*10000}e-4_rec_det.pdf", bbox_inches = "tight", dpi = 300)



plt.show()