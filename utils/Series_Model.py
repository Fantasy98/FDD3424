import torch 
from torch import nn 

class LSTM_model(nn.Module):
    def __init__(self,num_layer,feature_size = 128) -> None:
        super(LSTM_model,self).__init__()
        self.num_layer = num_layer
        self.feature_size = feature_size
        self.input_embedding = nn.Linear(10,feature_size)

        self.lstms = nn.LSTM(   input_size =feature_size,
                                hidden_size=feature_size,
                                num_layers = num_layer,
                                batch_first = True) 

        self.output =  nn.Linear(feature_size,1)

    def forward(self,x,hidden,cell):
        x = self.input_embedding(x)
        x,(hidden,cell) = self.lstms(x,(hidden,cell))
    
        x = self.output(x)

        return x 
   
    def init_hidden(self,batch_size):
        hidden = torch.zeros(self.num_layer,batch_size, self.feature_size)
        cell = torch.zeros(self.num_layer,batch_size,self.feature_size)
        return hidden, cell
