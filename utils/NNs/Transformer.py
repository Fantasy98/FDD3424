"""
Transformer models in torch.nn.Module
@author: Yuning Wang
"""
import torch
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm
from utils.NNs.attns import SelfAttn,ConvSelfAttn
from utils.NNs.Embedding import SineActivation, CosineActivation,PosEncoding
import torch.nn as nn 



class TransformerEncoderBlock(Module):
    def __init__(self,  d_input, d_model, 
                        nmode,num_head,
                        attn = "self",
                        embed = None,
                        is_res_attn = True,                
                        proj_dim = 0,
                        act_proj = "elu",
                        is_res_proj = True,
                        attn_dropout=1e-5, 
                        proj_dropout=1e-5
                        
                         ):
        
        """
        Transformer Encoder designed for time-series prediction

        Args:
            d_input         : (Int) The input dimension

            d_model         : (Int) The projection dimension
            
            nmode           : (Int) Number of mode

            num_head        : (Int) Number of heads
            
            is_res_attn     : (bool) If use residual connection between attion and LayerNorm

            proj_dim        : (Int) The projection dimension used in Feed-Forward Network
            
            act_proj        : (Str) Activation function used in Feed-Forward Network
            
            is_res_proj     : (Bool) If use residual connection for Feed-Forward Network
            
            attn_dropout    : (Float) Dropout Value for attention output

            proj_dropout    : (Float) Dropout Value for feed-forward output        
        """
        super(TransformerEncoderBlock,self).__init__()

        self.embed = embed
        
        ####################################
        ### Embedding type
        ####################################
        if embed == "sin":
            self.embed = SineActivation(nmode,nmode,d_model)
        if embed == "cos":
            self.embed = CosineActivation(nmode,nmode,d_model)
        if embed == "posenc":
            self.embed = PosEncoding(d_input,nmode,d_model)    
        
        if embed == None:
            try: d_model == d_input
            except: print("INFO: NO Embedding! \nThe input dimension should be same as the d_model")
            self.embed = None
        ####################################
        ### Self Attention
        ####################################            
        if attn == "self":
            self.attn = SelfAttn(d_model,num_head)
        if attn == "selfconv":
            self.attn = ConvSelfAttn(d_model, num_head)
        ####################################
        ### Attention Residual and layernorm
        ####################################            
        self.attn_drop = Dropout(p=attn_dropout)        
        self.res_attn = is_res_attn
        if is_res_attn:
            self.Lynorm_attn = LayerNorm(d_model)
            # self.Lynorm_attn = nn.BatchNorm1d(nmode,eps=1e-3,momentum=0.99)
        
        ####################################
        ### Feed Froward
        ####################################            
        self.proj_dim = proj_dim
        if self.proj_dim != 0 :
            self.ffd = nn.Sequential()
            
            ## Project and compress the space 
            if attn == "self":
                self.ffd.add_module(name="projection",module=Linear(d_model,proj_dim))

            if attn == "selfconv":
                self.ffd.add_module(name="projection",module=Linear(d_model,proj_dim))

            # ## We project and compress the time instead of space using Conv1D
            #     self.ffd.add_module(name="projection",module=nn.Conv1d(in_channels=d_input,
            #                                                        out_channels= proj_dim,
            #                                                        kernel_size= 1)
            #                                                          )
            
            if act_proj == "elu":
                self.ffd.add_module(name="act",module=nn.ELU())
            elif act_proj == "relu":
                self.ffd.add_module(name="act",module=nn.ReLU())
            elif act_proj == "gelu":
                self.ffd.add_module(name="act",module=nn.GELU())
            elif act_proj == "tanh":
                self.ffd.add_module(name="act",module=nn.Tanh())
            
            if attn == "self":
                self.ffd.add_module(name="compress",module=Linear(proj_dim,d_model))
           
            if attn == 'selfconv':
                self.ffd.add_module(name="compress",module=Linear(proj_dim,d_model))

            #     self.ffd.add_module(name="compress",module=nn.Conv1d(in_channels=proj_dim,
            #                                                         out_channels= d_input,
            #                                                         kernel_size= 1)
            
            #                                                         )
            
            nn.init.xavier_uniform_(self.ffd[0].weight)
            nn.init.xavier_uniform_(self.ffd[-1].weight)
            nn.init.zeros_(self.ffd[-1].bias)
            nn.init.zeros_(self.ffd[-1].bias)
                
        
        self.ffd_drop = Dropout(p=proj_dropout)

        self.res_proj = is_res_proj
        if is_res_proj:
            self.Lynorm_proj = LayerNorm(d_model)
            # self.Lynorm_proj = nn.BatchNorm1d(nmode,1e-3,momentum=0.99)
    def forward(self,x):
        B,N,C = x.shape

        if self.embed is not None:
            x_emb = self.embed(x)
        else:
            x_emb = x
        # print(x_emb.shape)
        x = self.attn(x_emb)
        x = self.attn_drop(x)
        if self.res_attn:
            # x = self.Lynorm_attn(x) + x_emb
            x = self.Lynorm_attn(x + x_emb) 
        
        if self.proj_dim !=0:
            x_ = self.ffd(x)
            x_ = self.ffd_drop(x_)

        if self.res_proj:
            # x = x + self.Lynorm_proj(x_)
            x =  self.Lynorm_proj(x+ x_)
            # x = x + x_
        return x
    

class Transformer2(nn.Module):
    def __init__(self,  d_input, d_model, 
                        nmode,num_head,
                        attn = "self",
                        embed = None,
                        
                        num_block = 1,
                        is_res_attn = False,                
                        proj_dim = 0,
                        act_proj = "elu",
                        is_res_proj = False,
                        attn_dropout=1e-6, 
                        proj_dropout=1e-6,
                        is_output = True, out_dim = 1, is_output_bias = True,out_act = "elu"):
        super(Transformer2,self).__init__()

        self.out_dim = out_dim
        self.transformer = nn.Sequential()
        self.transformer.add_module(name="embed_block", module= TransformerEncoderBlock(d_input, d_model, 
                                                                                        nmode,num_head,
                                                                                        attn,
                                                                                        embed,
                                                                                        is_res_attn,                
                                                                                        proj_dim,
                                                                                        act_proj,
                                                                                        is_res_proj,           
                                                                                        attn_dropout, 
                                                                                        proj_dropout) 
                                                                )
        embed = None
        if num_block > 1:
            for i in range(num_block-1):
                self.transformer.add_module(name= f"hidden_{i+1}",module=TransformerEncoderBlock(d_input, d_model, 
                                                                                                    nmode,num_head,
                                                                                                    attn,
                                                                                                    embed,
                                                                                                    is_res_attn,                
                                                                                                    proj_dim,
                                                                                                    act_proj,
                                                                                                    is_res_proj,           
                                                                                                    attn_dropout, 
                                                                                                    proj_dropout))
        self.is_output =  is_output
        if is_output:
            self.out = nn.Sequential()
            self.out.add_module("out_1",nn.Conv1d(d_input,out_dim,kernel_size=1))
            self.out.add_module("out_2",nn.Linear(d_model,nmode))
            
            nn.init.xavier_uniform_(self.out[0].weight)
            nn.init.xavier_uniform_(self.out[-1].weight)
            nn.init.zeros_(self.out[0].bias)
            nn.init.zeros_(self.out[-1].bias)
            if out_act == "elu":
                self.out_act = nn.ELU()
            elif out_act == "sigmoid":
                self.out_act = nn.Sigmoid()
            elif out_act == "tanh":
                self.out_act = nn.Tanh() 
            elif out_act == "relu":
                self.out_act = nn.ReLU()  
            else: self.out_act = None
        
    def forward(self,x):

        x = self.transformer(x)
        if self.is_output:
            if self.out_act is not None:
                return self.out_act(self.out(x))
            else:    
                return self.out(x)
        
        else:
            return x 
        

class ResTransformer(nn.Module):
    def __init__(self,  d_input, d_model, 
                        nmode,num_head,
                        attn = "self",
                        embed = None,
                        
                        num_block = 1,
                        is_res_attn = False,                
                        proj_dim = 0,
                        act_proj = "elu",
                        is_res_proj = False,
                        attn_dropout=1e-6, 
                        proj_dropout=1e-6,
                        is_output = True, out_dim = 1, is_output_bias = True,out_act = "elu"):
        super(ResTransformer,self).__init__()

        self.out_dim = out_dim

        if embed == "sin":
            self.embed_layer = SineActivation(nmode,nmode,d_model)
        if embed == "cos":
            self.embed_layer = CosineActivation(nmode,nmode,d_model)
        if embed == "posenc":
            self.embed_layer = PosEncoding(d_input,nmode,d_model)
        
        embed = None
        self.transformer = nn.Sequential()
        self.transformer.add_module(name="embed_block", module= TransformerEncoderBlock(d_input, d_model, 
                                                                                        nmode,num_head,
                                                                                        attn,
                                                                                        embed,
                                                                                        is_res_attn,                
                                                                                        proj_dim,
                                                                                        act_proj,
                                                                                        is_res_proj,           
                                                                                        attn_dropout, 
                                                                                        proj_dropout) 
                                                                )
        embed = None
        if num_block > 1:
            for i in range(num_block-1):
                self.transformer.add_module(name= f"hidden_{i+1}",module=TransformerEncoderBlock(d_input, d_model, 
                                                                                                    nmode,num_head,
                                                                                                    attn,
                                                                                                    embed,
                                                                                                    is_res_attn,                
                                                                                                    proj_dim,
                                                                                                    act_proj,
                                                                                                    is_res_proj,           
                                                                                                    attn_dropout, 
                                                                                                    proj_dropout))
        self.is_output =  is_output
        if is_output:
            self.out = nn.Sequential()
            self.out.add_module("out_1",nn.Conv1d(d_input,out_dim,kernel_size=1))
            self.out.add_module("out_2",nn.Linear(d_model,nmode))
            
            nn.init.xavier_uniform_(self.out[0].weight)
            nn.init.xavier_uniform_(self.out[-1].weight)
            nn.init.zeros_(self.out[0].bias)
            nn.init.zeros_(self.out[-1].bias)
            if out_act == "elu":
                self.out_act = nn.ELU()
            elif out_act == "sigmoid":
                self.out_act = nn.Sigmoid()
            elif out_act == "tanh":
                self.out_act = nn.Tanh() 
            elif out_act == "relu":
                self.out_act = nn.ReLU()  
            else: self.out_act = None
        
    def forward(self,x):

        x = self.embed_layer(x)

        x_ = x 
        for layer in self.transformer:
            r = x
            x = layer(x) + r 
        
        x = x_ + x

        if self.is_output:
            if self.out_act is not None:
                return self.out_act(self.out(x))
            else:    
                return self.out(x)
        
        else:
            return x 
        

