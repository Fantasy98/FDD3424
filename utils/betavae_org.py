"""
The original architectures of Beta-VAE model used in the Paper
"""

import torch 
from torch import nn 
import torch.nn.functional as F
import numpy as np 

#########################################
## Conv Blocks modules for up and down sampling
#########################################
class DownBlock(nn.Module):
    def __init__(self,in_channel,out_channel,knsize,pool_size) -> None:
        super(DownBlock,self).__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size=knsize,stride=2,padding=1)
        self.act  = nn.ELU()
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        print("conv has been initialized")
    def forward(self,x):
        return  self.act((self.conv(x)))

class UpBlock(nn.Module):
    def __init__(self,in_channel,out_channel,knsize,up_size) -> None:
        super(UpBlock,self).__init__()
        self.conv = nn.ConvTranspose2d(in_channel,out_channel,knsize,stride = 2,output_padding=1,padding=1)
        self.act = nn.ELU()
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        print("conv has been initialized")
    def forward(self,x):
        x = self.conv(x)
        x = self.act(x)
        return x 


###########################################
## Arch1 of VAE
###########################################
class encoder(nn.Module):
    def __init__(self,zdim,knsize) -> None:
        super(encoder,self).__init__()
        self.knsize = knsize
        self.zdim = zdim
        self.down1 = DownBlock(1,16,3,2)
        self.down2 = DownBlock(16,32,3,2)
        self.down3 = DownBlock(32,64,3,2)
        self.down4 = DownBlock(64,128,3,2)
        self.down5 = DownBlock(128,256,3,2)
        self.flat  = nn.Flatten()
        self.linear = nn.Linear(in_features=4608,
                                out_features= 128)
        self.act = nn.ELU()
        self.lin_mean = nn.Linear(in_features=128,out_features=zdim)
        self.lin_var = nn.Linear(in_features=128,out_features=zdim)
        
        nn.init.xavier_uniform_(self.lin_mean.weight)
        nn.init.xavier_uniform_(self.lin_var.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.lin_var.bias)
        nn.init.zeros_(self.lin_mean.bias)
        nn.init.zeros_(self.linear.bias)
        
    
    def forward(self,x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x  = self.flat(x)
        x = self.act(self.linear(x))
        z_mean = self.lin_mean(x)
        z_var = self.lin_var(x)

        return (z_mean,z_var)



class decoder(nn.Module):
    def __init__(self,zdim,knsize) -> None:
        super(decoder,self).__init__()
        self.zdim = zdim
        self.linear = nn.Linear(self.zdim,128)
        self.recover = nn.Linear(128,4608)
        self.act = nn.ELU()
        self.up1 = UpBlock(256,128,knsize,2)
        self.up2 = UpBlock(128,64,knsize,2)
        self.up3 = UpBlock(64,32,knsize,2)
        self.up4 = UpBlock(32,16,knsize,2)
        self.conv = nn.ConvTranspose2d(16,1,knsize,stride=2,padding=1,output_padding=1)
        
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        nn.init.xavier_normal_(self.linear.weight)        
        nn.init.xavier_normal_(self.recover.weight)
        nn.init.zeros_(self.linear.bias)        
        nn.init.zeros_(self.recover.bias)        

    def forward(self,x):
        x = self.act(self.linear(x))
        x = self.act(self.recover(x))
        x = x.reshape(x.size(0),256,3,6)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.conv(x)
        return x




class BetaVAE(nn.Module):
    """
    nn.Module for Beta-VAE architecture and models 
    Args:  
        zdim    :   The dimension of latent space
        knsize  :   The kernel size for Conv Layer
        beta    :   The value of regularisation term beta for kl-divergence 
    
    func: 
        rec_loss()        : Compute the reconstruction loss via MSE
        kl_loss()         : Compute the kl-divergence loss 
        vae_loss()        : Compute the total loss = E_rec + beta * Kl-div
        reparameterize()  : Implement reparameterisation trick
        forward()         : Forward propergation of the model

            
    """    
    def __init__(self,zdim,knsize,beta) -> None:
        super(BetaVAE,self).__init__()
        self.zdim = zdim
        self.beta = beta
        self.encoder = encoder(zdim,knsize)
        self.decoder = decoder(zdim,knsize)
        self.mse = nn.MSELoss()
    def rec_loss(self,pred,y):
        loss = self.mse(pred,y)
        return loss
    
    def kl_loss(self,z_mean,z_log_var):
        kl_loss = 1 + z_log_var - torch.square(z_mean)-torch.exp(z_log_var)
        kl_loss *= -0.5
        return torch.mean(kl_loss)
    
    def vae_loss(self,rec_loss,kl_loss):
        loss = rec_loss + self.beta * kl_loss
        return torch.mean(loss)
    
    def reparameterize(self,args):
        z_mean,z_log_sigma = args
        epsilon = torch.randn_like(z_log_sigma)
        return z_mean +  torch.exp(0.5*z_log_sigma) * epsilon
     
    def forward(self,x):
        z_mean,z_var = self.encoder(x)
        z_out = self.reparameterize((z_mean,z_var))
        out = self.decoder(z_out)
        return z_mean,z_var, out

###########################################
## Arch2 of VAE
###########################################
class encoder2(nn.Module):
    def __init__(self,zdim,knsize) -> None:
        super(encoder2,self).__init__()
        self.knsize = knsize
        self.zdim = zdim
        self.down1 = DownBlock(1,48,3,2)
        self.down2 = DownBlock(48,96,3,2)
        self.down3 = DownBlock(96,128,3,2)
        self.down4 = DownBlock(128,256,3,2)
        self.down5 = DownBlock(256,512,3,2)
        self.flat  = nn.Flatten()
        self.linear = nn.Linear(in_features=9216,
                                out_features= 128)
        self.act = nn.ELU()
        self.lin_mean = nn.Linear(in_features=128,out_features=zdim)
        self.lin_var = nn.Linear(in_features=128,out_features=zdim)
        
        nn.init.xavier_uniform_(self.lin_mean.weight)
        nn.init.xavier_uniform_(self.lin_var.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.lin_var.bias)
        nn.init.zeros_(self.lin_mean.bias)
        nn.init.zeros_(self.linear.bias)
        
    
    def forward(self,x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x  = self.flat(x)
        x = self.act(self.linear(x))
        z_mean = self.lin_mean(x)
        z_var = self.lin_var(x)

        return (z_mean,z_var)



class decoder2(nn.Module):
    def __init__(self,zdim,knsize) -> None:
        super(decoder2,self).__init__()
        self.zdim = zdim
        self.linear = nn.Linear(self.zdim,128)
        self.recover = nn.Linear(128,9216)
        self.act = nn.ELU()
        self.up1 = UpBlock(512,256,knsize,2)
        self.up2 = UpBlock(256,128,knsize,2)
        self.up3 = UpBlock(128,96,knsize,2)
        self.up4 = UpBlock(96,48,knsize,2)
        self.conv = nn.ConvTranspose2d(48,1,knsize,stride=2,padding=1,output_padding=1)
        
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        nn.init.xavier_normal_(self.linear.weight)        
        nn.init.xavier_normal_(self.recover.weight)
        nn.init.zeros_(self.linear.bias)        
        nn.init.zeros_(self.recover.bias)        

    def forward(self,x):
        x = self.act(self.linear(x))
        x = self.act(self.recover(x))
        x = x.reshape(x.size(0),512,3,6)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.conv(x)
        return x




class BetaVAE2(nn.Module):
    def __init__(self,zdim,knsize,beta) -> None:
        """
        nn.Module for Beta-VAE architecture and models 
        Args:  
            zdim    :   The dimension of latent space
            knsize  :   The kernel size for Conv Layer
            beta    :   The value of regularisation term beta for kl-divergence 
        
        func: 
            rec_loss()        : Compute the reconstruction loss via MSE
            kl_loss()         : Compute the kl-divergence loss 
            vae_loss()        : Compute the total loss = E_rec + beta * Kl-div
            reparameterize()  : Implement reparameterisation trick
            forward()         : Forward propergation of the model

            
        """
        super(BetaVAE2,self).__init__()
        self.zdim = zdim
        self.beta = beta
        self.encoder = encoder2(zdim,knsize)
        self.decoder = decoder2(zdim,knsize)
        self.mse = nn.MSELoss()
    def rec_loss(self,pred,y):
        loss = self.mse(pred,y)
        return loss
    
    def kl_loss(self,z_mean,z_log_var):
        kl_loss = 1 + z_log_var - torch.square(z_mean)-torch.exp(z_log_var)
        kl_loss *= -0.5
        return torch.mean(kl_loss)
    
    def vae_loss(self,rec_loss,kl_loss):
        loss = rec_loss + self.beta * kl_loss
        return torch.mean(loss)
    
    def reparameterize(self,args):
        z_mean,z_log_sigma = args
        epsilon = torch.randn_like(z_log_sigma)
        return z_mean +  torch.exp(0.5*z_log_sigma) * epsilon
     
    def forward(self,x):
        z_mean,z_var = self.encoder(x)
        z_out = self.reparameterize((z_mean,z_var))
        out = self.decoder(z_out)
        return z_mean,z_var, out