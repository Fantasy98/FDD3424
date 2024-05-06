import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot

torch.manual_seed(0)
np.random.seed(0)

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

#src = torch.rand((10, 32, 512)) # (S,N,E) 
#tgt = torch.rand((20, 32, 512)) # (T,N,E)
#out = transformer_model(src, tgt)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = 1 / (10000 ** ((2 * np.arange(d_model)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])

        pe = pe.unsqueeze(0) # [1,5000, d_model],so need seq-len <= 5000
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:,:x.size(1),:]
          


class TransAm2(nn.Module):
    def __init__(self,feature_size= 128,num_head=8,num_layers=1,dropout=0.1):
        super(TransAm2, self).__init__()
        self.model_type = 'Transformer'
        self.input_embedding  = nn.Linear(10,feature_size)
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_head, 
                                                        dropout=dropout,
                                                        # dim_feedforward=1024,
                                                        activation="gelu",
                                                        # batch_first=True,
                                                        )
        self.transformer_encoder = nn.TransformerEncoder(
                                                        self.encoder_layer, 
                                                         num_layers=num_layers,
                                                        #   norm=nn.LayerNorm,
                                                          )
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        # nn.init.xavier_uniform_(self.decoder.weight)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        src_ = self.input_embedding(src) # linear transformation before positional embedding
        src = self.pos_encoder(src_)
        output = self.transformer_encoder(src)#, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class TransAm2(nn.Module):
    def __init__(self,feature_size= 128,num_head=8,num_layers=1,dropout=0.1):
        super(TransAm2, self).__init__()
        self.model_type = 'Transformer'
        self.input_embedding  = nn.Linear(10,feature_size)
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_head, 
                                                        dropout=dropout,
                                                        # dim_feedforward=1024,
                                                        activation="gelu",
                                                        # batch_first=True,
                                                        )
        self.transformer_encoder = nn.TransformerEncoder(
                                                        self.encoder_layer, 
                                                         num_layers=num_layers,
                                                        #   norm=nn.LayerNorm,
                                                          )
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        # nn.init.xavier_uniform_(self.decoder.weight)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        src_ = self.input_embedding(src) # linear transformation before positional embedding
        src = self.pos_encoder(src_)
        output = self.transformer_encoder(src)#, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TransAm_res(nn.Module):
    def __init__(self,feature_size= 128,num_head=8,num_layers=1,dropout=0.1):
        super(TransAm_res, self).__init__()
        self.model_type = 'Transformer'
        self.input_embedding  = nn.Linear(10,feature_size)
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_head, 
                                                        dropout=dropout,
                                                        # dim_feedforward=1024,
                                                        activation="gelu",
                                                        batch_first=True,
                                                        )
        self.transformer_encoder = nn.TransformerEncoder(
                                                        self.encoder_layer, 
                                                         num_layers=num_layers,
                                                          norm=nn.LayerNorm((10,feature_size)),
                                                          )
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        # nn.init.xavier_uniform_(self.decoder.weight)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        src = self.input_embedding(src) # linear transformation before positional embedding
        src_ = self.pos_encoder(src)
        output = self.transformer_encoder(src_) + src_ #, self.src_mask)
        output = self.decoder(output) 
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask