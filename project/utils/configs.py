class VAE_config:
    model = "v3" # v2 for small architecture and v3 for large

    beta = 0.01 # The regularisation 
    latent_dim = 10 # 
    lr = 1e-3 
    epoch = 100
    
    test_split  = 0.8 # ratio of training data before train_val_split 
    train_split = 0.8 # Ratio of val and train data during training 
    batch_size = 128
    earlystop = True
    if earlystop == True:
        patience = 30
    else:
        patience = 0

class VAE_custom:
    model       = "v5" # Name Arch1 as v4 and v5 as Arch2 v6 for an Arch between v4 and v5 

    beta        = 0.005 # The regularisation [0.001, 0.0025, 0.005, 0.01]
    latent_dim  = 10 # Latent-space dimension 
    lr          = 1e-3  # Learning Rate
    # lr          = 5e-4  # Learning Rate
    w_decay     = 0     # Weight decay if needed
    epoch       = 300 # Training Epoch

    # Kernel Size of Conv 
    knsize      = 3
    # The filter used in Encoder and decoder
    # Arch 1 [1, 16, 32, 64, 128, 256]
    # Arch 2 [1, 48, 96, 128, 256, 512] # Corrected
    if model == "v35":
        filters     = [1, 8, 16, 32, 96, 128]
    
    elif model == "v4":
        filters     = [1, 16, 32, 64, 128, 256]
    
    elif model  == "v45": # v45 
        filters     = [1, 24, 48, 96, 192, 384] # An Arch between Arch1 and Arch2

    elif model == "v5":
        filters     = [1, 48, 96, 128, 256, 512]
    
    elif model == "v5old":
        filters     = [1, 48, 96, 128, 256, 512]
    
    
    elif model == "v55":
        filters     = [1, 72, 144, 160, 320, 640]
    
    ## Filters with 5 layers (For previous test)
    # filters     = [1, 32, 64, 128, 192, 256]
    # filters     = [1, 48, 96, 192, 256, 512]
    
    # Type of ConvBlock
    ## Could be useful for future work
    block_type  = "original" # "original", "bn_original", "deep", "res" "deepres"

    # Dimension of linear layer after flatten 
    linear_dim  = 128
    

    #activation fuction for conv layer and linear layer
    act_conv    = "elu"
    act_linear  = "elu"


    test_split  = 1 # ratio of training data before train_val_split  We use test_split = 1 
    
    if test_split == 1:
        train_split = 0.8 # Ratio of val and train data during training 
    else:
        train_split = 1 # If we do not use full dataset, all data will be used for training
    
    batch_size  = 128
    earlystop   = False
    if earlystop == True:
        patience = 50
    else:
        patience = 0






class Transformer_config:
    from utils.configs import VAE_custom

    model_type  = "TFSelf"

    in_dim      =   128 
    out_dim     =   128
    d_model     =   128
    time_proj   =   128

    next_step   =   1
    nmode       =   VAE_custom.latent_dim

    num_head    =   8
    attn_type   =   "self"

    embed       =   "time"
    num_block   =   4

    proj_dim    =   256

    act_proj    =   "relu"
    is_output   =   True
    out_act     =   None

    Epoch       =   3000
    Batch_size  =   128 # Beyond 128 then MMO
    lr          =   1e-3

    wdecay      =   0

    train_split =   0.85
    val_split   =   0.2 

    early_stop  =   True
    if early_stop:
        patience    = 30
    else:
        patience    = 0


class EasyAttn_config:
    from utils.configs import VAE_custom

    model_type  = "TFEasy"

    in_dim      =  128
    out_dim     =  128
    d_model     =  128
    time_proj   =  128

    next_step   =   1
    nmode       =   VAE_custom.latent_dim

    num_head    =   8
    attn_type   =   "easy"

    embed       =  "time"
    num_block   =   4

    proj_dim    =   256
    act_proj    =   "relu"
    is_output   =   True
    out_act     =   None

    Epoch       =   3000# 100 OR 300 OR 1000 OR 3000 
    Batch_size  =   128 # 128 or 256
    lr          =   1e-3

    wdecay      =   0

    train_split =   0.85
    val_split   =   0.2 

    early_stop  = True

    if early_stop:
        patience    = 30
    else:
        patience    = 0


class LSTM_config:
    from utils.configs import VAE_custom

    model_type  = "LSTM"

    in_dim      = 128
    d_model     = 128
    next_step   = 1
    nmode       = VAE_custom.latent_dim

    num_layer   = 4
    embed       = None
    
    hidden_size = 256

    is_output   = True
    out_act     = None

    Epoch       = 3000
    Batch_size  = 128
    lr          = 1e-3

    wdecay      = 0

    train_split = 0.85                                          
    val_split   = 0.2 
    num_train   = 135000

    early_stop =  True

    if early_stop == True:
        patience  = 30
    else:
        patience  = 0 




class ROM_config:

    from utils.configs import Transformer_config, LSTM_config 
    from utils.configs import VAE_custom

    ## Modal-decomposition config (mdcp)
    mdcp                =   VAE_custom
    
    ## Time-series prediction config (tssp)
    # predictor_model     =   "TFSelf"
    # predictor_model     =   "LSTM"
    predictor_model     =   "TFEasy"
    
    if predictor_model == "LSTM":
        tssp                =   LSTM_config
    elif predictor_model == "TFSelf":
        tssp                =   Transformer_config
    elif predictor_model ==  "TFEasy":
        tssp                =   EasyAttn_config

    if_pretrain         =   True
    froze_enc           =   True
    froze_dec           =   False

    W_latent_loss       =   1
    W_rec_loss          =   1

    save_predictor_only =   False


def Name_VAE(cfg, nt):
    """
    Name the original VAE model case
    Args:
        cfg: The class contains information of the case
        nt: Number of training data
    Returns:
        fileID: The name of the case
    """
    fileID = f"{cfg.model}_{int( nt*(cfg.train_split) )}n_{cfg.latent_dim}d_{int(cfg.beta*10000)}e-4_"+\
         f"{cfg.batch_size}bs_{cfg.epoch}epoch_{cfg.earlystop}ES_{cfg.patience}P"

    return fileID

def Name_Costum_VAE(cfg, nt):
    """
    Name the custum VAE model case
    Args:
        cfg: The class contains information of the case
        nt: Number of training data
    Returns:
        fileID: The name of the case
    """
    fileID =    f"{cfg.model}_{nt}n_{cfg.latent_dim}d_{int(cfg.beta*10000)}e-4beta_"+\
                f"{cfg.block_type}conv_{len(cfg.filters)}Nf_{cfg.filters[-1]}Fdim_{cfg.linear_dim}Ldim"+\
                f"{cfg.act_conv}convact_{cfg.act_linear}_" +\
                f"{int(cfg.lr *1e5)}e-5LR_{int(cfg.w_decay*1e5)}e-5Wd"+\
                f"{cfg.batch_size}bs_{cfg.epoch}epoch_{cfg.earlystop}ES_{cfg.patience}P"

    return fileID

def Name_ROM(cfg):

    """
    Name the case of Reduced-order model training

    Args:

        cfg         :  (Class) The class for training ROM

    Returns:

        case_name   :   (Str) The name of this case

    
    """
    
    if "TF" in cfg.tssp.model_type:
        case_name   =   f"VAE_{cfg.mdcp.model}_{cfg.mdcp.latent_dim}d_{int(cfg.mdcp.beta*10000)}e-4beta_"+\
                        f"{cfg.if_pretrain}pretrain_{cfg.froze_enc}encfrz_{cfg.froze_dec}decfrz_{cfg.W_latent_loss}Wlat_{cfg.W_rec_loss}Wrec_"+\
                        f"{cfg.tssp.model_type}_"+\
                        f"_{cfg.tssp.in_dim}in_{cfg.tssp.d_model}dmodel_{cfg.tssp.time_proj}_{cfg.tssp.next_step}next"+\
                        f"_{cfg.tssp.embed}emb_{cfg.tssp.num_head}h_{cfg.tssp.num_block}nb_{cfg.tssp.proj_dim}ff"+\
                        f"_{cfg.tssp.act_proj}act_{cfg.tssp.out_act}outact"+\
                        f"_{int(cfg.tssp.wdecay * 100000)}e-5wdeacy"+\
                        f"_{cfg.tssp.lr * 10000}e-4lr_{cfg.tssp.Batch_size}bs" +\
                        f"_{cfg.tssp.Epoch}Epoch_{cfg.tssp.early_stop}ES_{cfg.tssp.patience}P"
    
    elif cfg.tssp.model_type == "LSTM":
        case_name   =   f"VAE_{cfg.mdcp.model}_{cfg.mdcp.latent_dim}d_{int(cfg.mdcp.beta*10000)}e-4beta_"+\
                        f"{cfg.if_pretrain}pretrain_{cfg.froze_enc}encfrz_{cfg.froze_dec}decfrz_{cfg.W_latent_loss}Wlat_{cfg.W_rec_loss}Wrec_"+\
                        f"{cfg.tssp.model_type}_"+\
                        f"_{cfg.tssp.in_dim}in_{cfg.tssp.d_model}dmodel_{cfg.tssp.next_step}next_{cfg.tssp.nmode}dim"+\
                        f"_{cfg.tssp.embed}emb_{cfg.tssp.hidden_size}hideen_{cfg.tssp.num_layer}nlayer"+\
                        f"_{cfg.tssp.out_act}outact"+\
                        f"_{int(cfg.tssp.wdecay * 100000)}e-5wdeacy"+\
                        f"_{cfg.tssp.lr * 10000}e-4lr_{cfg.tssp.Batch_size}bs" +\
                        f"_{cfg.tssp.Epoch}Epoch_{cfg.tssp.early_stop}ES_{cfg.tssp.patience}P"

    return case_name


def Make_Transformer_Name(cfg):
    """
    A function to name the VAE checkpoint 

    Args: 
        cfg: A class of VAE model configuration
    
    Returns:
        name: A string for Transformer model
    """

    
    case_name = f"{cfg.model_type}"+\
                f"_{cfg.in_dim}in_{cfg.d_model}dmodel_{cfg.next_step}next_{cfg.nmode}dim"+\
                f"_{cfg.embed}emb_{cfg.num_head}h_{cfg.num_block}nb_{cfg.proj_dim}ff"+\
                f"_{cfg.act_proj}act_{cfg.out_act}outact"+\
                f"_{cfg.Epoch}Epoch_{cfg.early_stop}ES_{cfg.patience}P_{int(cfg.train_split*100)}train"
    
    return case_name

def Make_LSTM_Name(cfg):
    """
    A function to name the VAE checkpoint 

    Args: 
        cfg: A class of VAE model configuration
    
    Returns:
        name: A string for LSTM model
    """
    
    case_name = f"LSTM"+\
                f"_{cfg.in_dim}in_{cfg.d_model}dmodel_{cfg.next_step}next_{cfg.nmode}dim"+\
                f"_{cfg.embed}emb_{cfg.hidden_size}hideen_{cfg.num_layer}nlayer_"+\
                f"_{cfg.out_act}outact"+\
                f"_{cfg.Epoch}Epoch_{cfg.num_train}N_{cfg.early_stop}ES_{cfg.patience}P"
    
    return case_name

class Data_config:
    time_delay  = 16 
    step = 1 
    batch_size = 64 
    n_test = 1024 
    train_shuffle = True



