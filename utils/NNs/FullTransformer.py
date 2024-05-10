"""
Transformer included Encoder and Decoder 
@ Author yuningw
"""
import  torch 
from    torch                 import  nn
from    utils.NNs.layers      import  EncoderLayer, DecoderLayer 
from    utils.NNs.Embedding   import  SineActivation, CosineActivation, PosEncoding, TimeSpaceEmbedding


class Transformer(nn.Module):
    def __init__(self, 
                 d_input,d_output,n_mode,d_model, 
                 num_heads,num_layers,d_ff, 
                 embed_type,
                 act_proj,
                 act_out,
                 dropout = 1e-5,):
        """
        The model for transformer architecture
        Args:
            d_input     :   (Int) The size of the input sequence    
            
            d_output    :   (Int) The size of the output sequence    
            
            n_mode      :   (Int) The number of the modes in the sequnce    
            
            d_model     :   (Int) The embedding dimension
            
            num_heads   :   (Int) The number of heads 
            
            d_ff        :   (Int) The dimension of project in FFD network
            
            embed_type  :   (Str) The type of embedding method 
            
            act_proj    :   (Str) The activation function used in the FFD    
            
            act_out     :   (Str) The activation function used for OutPut    

            dropout     :   (Float) The dropout ratio used for prevent model from overfitting

        """

        super(Transformer, self).__init__()
        
        # Embedding
        if embed_type == "sin":     self.encoder_embedding =   SineActivation(n_mode,   n_mode, d_model) ;  self.decoder_embedding = SineActivation(n_mode, n_mode, d_model)
        if embed_type == "cos":     self.encoder_embedding =   CosineActivation(n_mode, n_mode, d_model) ;  self.decoder_embedding = CosineActivation(n_mode, n_mode, d_model)
        if embed_type == "posenc":  self.encoder_embedding =   PosEncoding(d_input, n_mode, d_model) ;  self.decoder_embedding = PosEncoding(d_input, n_mode, d_model)
        if embed_type == "time":  self.encoder_embedding =   TimeSpaceEmbedding(d_input, n_mode, d_model, d_model) ;  self.decoder_embedding = TimeSpaceEmbedding(d_input, n_mode, d_model,d_model)
        
        # Encoder and Decoder 
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout,act_proj) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout,act_proj) for _ in range(num_layers)])

        # OutPut 
        self.fc = nn.Linear(d_model, n_mode)
        self.fn = nn.Conv1d(d_input,d_output,kernel_size=1)

        # Activation for output
        if act_out == "elu"         :   self.act_out = nn.ELU()
        elif act_out == "sigmoid"   :   self.act_out = nn.Sigmoid()
        elif act_out == "tanh"      :   self.act_out = nn.Tanh() 
        elif act_out == "relu"      :   self.act_out = nn.ReLU()  
        else                        :   self.act_out = nn.Identity()
        
        # DropOut
        self.dropout = nn.Dropout(dropout)

        # Init a trainable target, following the idea of inserting non-linear Force into HDMD

        self.target  = nn.Parameter(data= torch.randn(size=(1, d_output, n_mode), requires_grad=True))

    def generate_mask(self, src, tgt):
        """
        A function for generate down-triangle matirx as mask

        Args:
            src     :   torch.Tensor as input

            tgt     :   torch.Tensor as target

        Returns:
            src_mask    : The 

        """

        seq_length  = tgt.size(1)
        tgt_mask    = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(tgt.device)
        src_mask    = None

        return src_mask, tgt_mask
    
    def forward(self, src):

        B            = src.shape[0]
        tgt          = self.target.repeat(B,1,1)
        

        src_embedded = self.dropout(self.encoder_embedding(src))

        # Experiment, add non-linear force to src instead of learn it only Sep17th yuningw 21:53
        tgt_embedded = self.dropout(self.decoder_embedding(src))


        enc_output = src_embedded
        
        # src_mask, tgt_mask = self.generate_mask(src_embedded, tgt_embedded)
        src_mask, tgt_mask = None, None
        
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded.transpose(-1,-2)
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        # output = self.fn(dec_output)
        output = self.fc(dec_output)
        
        return self.act_out(output)

if __name__ == "__main__":
    class cfg:
        """
        A class of configuration of Transformer predictor 
        """

        target      = "POD" # POD or VAE

        in_dim      = 32
        d_model     = 96

        next_step   = 1
        if target == "VAE":
            nmode       = 20  # Should be consistent as the modes
        elif target == "POD":
            nmode       = 10  # Choose from [10, 15, 20] 


        num_head    = 4
        attn_type   = "selfconv" # self or selfconv

        embed       = "posenc" # sin / cos/ posenc
        num_block   = 2    # Number of layer 

        is_res_attn = True
        is_res_proj = True
        proj_dim    = 128

        act_proj    = "relu"
        is_output   = True
        out_act     = None

        Epoch       = 100
        Batch_size  = 256
        lr          = 1e-3

        train_split = 0.8 
        num_train   = 135000

        early_stop  = True

        if early_stop == True:
            patience  = 10
        else:
            patience  = 0 



    # Example for use the model
    model = Transformer( d_input          = cfg.in_dim, 
                        d_output          = cfg.next_step,
                        n_mode            = cfg.nmode,
                        d_model           = cfg.d_model, 
                        num_heads         = cfg.num_head,
                        num_layers        = cfg.num_block,
                        d_ff              = cfg.proj_dim,
                        embed_type        = cfg.embed,
                        act_out           = cfg.out_act,
                        act_proj          = cfg.act_proj,
                        dropout           = 1e-6,
                        )
    print(model.eval)

    src = torch.randn(size = (cfg.Batch_size, cfg.in_dim,    cfg.nmode)).float()
    tgt = torch.randn(size = (cfg.Batch_size, cfg.next_step, cfg.nmode)).float()

    p   = model(src)

    print(f"The forward prop success, the prediction has shape = {p.shape}")