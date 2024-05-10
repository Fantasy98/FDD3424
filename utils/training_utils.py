import torch 
import numpy as np 

class EarlyStopper:
    """
    A class of early stopping schduler
    Args:
        patience:   The number of epoch before stopping the training loop 
        min_delta:  Residual of difference between val loss and train los
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False 


def fit(epochs,model,
        opt:torch.optim.Optimizer,
        opt_scheduler:list,
        t_dl, 
        v_dl, 
        device, 
        earlystop:bool,patience = 30 ):
    """
    A function for traing the beta-VAE neural networks and returns loss evolution history
    
    args:
        epochs:         A integer number of training epoch
        model:          The nn.Module object of model 
        opt:            Optimizer object
        opt_scheduler:  A list of learning rate scheduler
        t_dl:           DataLoader of training
        v_dl:           DataLoader of validation data
        device:         A string of training device on cpu or gpu
        early stop:     A bool variable, if use early stop or not 
        patience:       The number of patience

    returns:
        history:        A dict of loss evolution during training process
    
    """

    from tqdm import tqdm
    import torch
    history = {}
    history["tot_loss"] = []
    history["val_tot_loss"] = []
    history["kl_loss"] = []
    history["val_kl_loss"] = []
    history["rec_loss"] = []
    history["val_rec_loss"] = []

    model.to(device)
    print(f"INFO: The model is assigned to device: {device} ")
    
    if earlystop is True:
        early_stopper = EarlyStopper(patience=patience,min_delta=0)
        print("INFO: Early Stopping is going to be used!")

    if opt_scheduler is not None:
        print(f"INFO: The following schedulers are going to be used:")
        for sch in opt_scheduler:
            print(f"{sch.__class__}")


    print(f"Start training loop, totally {epochs} Epochs")

    for epoch in range(epochs):
        tot_loss_epoch = 0; kl_loss_epoch = 0; rec_loss_epoch = 0  
        val_tot_loss_epoch = 0; val_kl_loss_epoch = 0; val_rec_loss_epoch = 0  
        stp = 0  
        print(f"INFO: Training")
        model.train()
        for x, y in (t_dl):
            x = x.float().to(device)
            y = y.float().to(device)
            stp +=1 
            z_mean, z_var,pred = model(x)
            rec_loss = model.rec_loss(pred,y)
            kl_loss = model.kl_loss(z_mean,z_var)
            tot_loss = model.vae_loss(rec_loss,kl_loss)

            opt.zero_grad()
            tot_loss.backward()
            opt.step()

            tot_loss_epoch += tot_loss.item()
            kl_loss_epoch += kl_loss.item()
            rec_loss_epoch += rec_loss.item()

        history["tot_loss"].append(tot_loss_epoch/stp)
        history["kl_loss"].append(kl_loss_epoch/stp)
        history["rec_loss"].append(rec_loss_epoch/stp)
        print(  f"Epoch = {epoch}, "+\
                f"tot_loss = {tot_loss_epoch/stp},"+\
                f"rec_loss ={rec_loss_epoch/stp},"+\
                f" kl_loss={kl_loss_epoch/stp} ")

        if opt_scheduler is not None:
            lr_now = 0 
            for sch in opt_scheduler:
                sch.step()
                lr_now = sch.get_last_lr()
            print(f"INFO: Scheduler updated, LR = {lr_now} ")


        if v_dl != None:

            model.eval()
            print("INFO: Validating")
            stp =  0
            for x,y in (v_dl):
                stp +=1 
                x = x.float().to(device)
                y = y.float().to(device)
                z_mean, z_var,pred = model(x)
                rec_loss = model.rec_loss(pred,y)
                kl_loss = model.kl_loss(z_mean,z_var)
                tot_loss = model.vae_loss(rec_loss,kl_loss)

                val_tot_loss_epoch += tot_loss.item()
                val_kl_loss_epoch += kl_loss.item()
                val_rec_loss_epoch += rec_loss.item()
                


            history["val_tot_loss"].append(val_tot_loss_epoch/stp)
            history["val_kl_loss"].append(val_kl_loss_epoch/stp)
            history["val_rec_loss"].append(val_rec_loss_epoch/stp)
            if earlystop is True:
                    if early_stopper.early_stop(val_tot_loss_epoch/stp):
                        print("Early-stopp Triggered, Going to stop the training")
                        break
        print(  f"val_tot_loss = {val_tot_loss_epoch/stp},"+\
                f" val_rec_loss ={val_rec_loss_epoch/stp},"+\
                f" val_kl_loss={val_kl_loss_epoch/stp} ")
        # Evaluate if the kl-divergence exploded
        if tot_loss/stp == torch.nan:
            print("The loss exploded to be NAN, quit training")
            break
        
    return history


##########################################
#### For time-series prediction
##########################################




def fit_time(device,
        model,
        dl,
        loss_fn,
        Epoch,
        optimizer:torch.optim.Optimizer, 
        val_dl = None,
        scheduler:list= None,
        if_early_stop = True,patience = 10,
        ):
    
    """
    A function for training loop for temporal prediction

    Args: 
        device: the device for training, which should match the model
        model: The model to be trained
        dl: A dataloader for training
        loss_fn: Loss function
        Epochs: Number of epochs 
        optimizer: The optimizer object
        val_dl: The data for validation
        scheduler: A list of traning scheduler
        

    Returns:
        history: A dict contains training loss and validation loss (if have)

    """

    from tqdm import tqdm
    
    history = {}
    history["train_loss"] = []
    
    if val_dl:
        history["val_loss"] = []
    
    model.to(device)
    print(f"INFO: The model is assigned to device: {device} ")

    if scheduler is not None:
        print(f"INFO: The following schedulers are going to be used:")
        for sch in scheduler:
            print(f"{sch.__class__}")

    print(f"INFO: Training start")

    if if_early_stop: 
        early_stopper = EarlyStopper(patience=patience,min_delta=0)
        print("INFO: Early-Stopper prepared")

    for epoch in range(Epoch):
        #####
        #Training step
        #####
        model.train()
        loss_val = 0; num_batch = 0
        for batch in tqdm(dl):
            x, y = batch
            x = x.to(device).float(); y =y.to(device).float()
            optimizer.zero_grad()
            
            pred = model(x)
            loss = loss_fn(pred,y)
            loss.backward()
            optimizer.step()

            

            loss_val += loss.item()/x.shape[0]
            num_batch += 1

        history["train_loss"].append(loss_val/num_batch)

        if scheduler is not None:
            lr_now = 0 
            for sch in scheduler:
                sch.step()
                lr_now = sch.get_last_lr()
            print(f"INFO: Scheduler updated, LR = {lr_now} ")

        if val_dl:
        #####
        #Valdation step
        #####
            loss_val = 0 ; num_batch = 0 
            model.eval()
            for batch in tqdm(val_dl):
                x, y = batch
                x = x.to(device).float(); y =y.to(device).float()
                pred = model(x)
                loss = loss_fn(pred,y)
            
                loss_val += loss.item()/x.shape[0]
                num_batch += 1

            history["val_loss"].append(loss_val/num_batch)
        
        train_loss = history["train_loss"][-1]
        val_loss = history["val_loss"][-1]
        print(
              f"At Epoch = {epoch},\n"+\
              
              f"Train_loss = {train_loss}  "+\
              
              f"Val_loss = {val_loss}"          
              )
        
        if if_early_stop:
            if early_stopper.early_stop(loss_val/num_batch):
                print("Early-stopp Triggered, Going to stop the training")
                break
    return history
