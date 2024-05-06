"""
Script for supplemetary functions and object for the Reduced-order model 

"""
import numpy as np 
import torch
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

def Frozen_model(model:torch.nn.Module):
    """
    Function for freeze the learnable parameters in the architecture 

    Args:

        model   :   (torch.nn.Module) The target model to be frozen 
    
    Returns: 
    
        The nn.Module with frozen parameters
    
    """
    for name, child in model.named_children():
        print(f"Acting on the {name}")
        for param in child.parameters():
            param.requires_grad = False


    doublecheck = [param.requires_grad == True for param in model.parameters()]

    if True in doublecheck: 
        print(f"Error: There are parameters not been foren!")
        quit() 
    else:
        print(f"INFO: All parameters has been forzen!") 
        