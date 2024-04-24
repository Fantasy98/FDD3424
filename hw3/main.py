"""
Assignment 3

K-Layer Network with BatchNormalisation

@yuningw
Apr 16h, 2024 
"""
##########################################
## Environment and general setup 
##########################################
import numpy as np
import pickle
import os 
import numpy.linalg as LA
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd 
import time
import pathlib
import argparse
from matplotlib import ticker as ticker
# Dictionary for layer. cache, etc
from collections import OrderedDict
import unittest

# Parse Arguments 
parser = argparse.ArgumentParser()
parser.add_argument('-m',default=1,type=int,help='Choose which exercise to do 1,2,3,4,5')
args= parser.parse_args()

# Mkdir 
pathlib.Path('Figs/').mkdir(exist_ok=True)
pathlib.Path('data/').mkdir(exist_ok=True)
pathlib.Path('weights/').mkdir(exist_ok=True)
font_dict = {'size':20,'weight':'bold'}
fig_dict = {'bbox_inches':'tight','dpi':300}

# Setup the random seed
np.random.seed(400)

# SetUp For visualisation 
class colorplate:
    red = "#D23918" # luoshenzhu
    blue = "#2E59A7" # qunqing
    yellow = "#E5A84B" # huanghe liuli
    cyan = "#5DA39D" # er lv
    black = "#151D29" # lanjian
    gray    = "#DFE0D9" # ermuyu 
plt.rc("font",family = "serif")
plt.rc("font",size = 22)
plt.rc("axes",labelsize = 16, linewidth = 2)
plt.rc("legend",fontsize= 12, handletextpad = 0.3)
plt.rc("xtick",labelsize = 18)
plt.rc("ytick",labelsize = 18)


##########################################
## Dataloader
##########################################
def normalization(X):

    X_stdev = np.std(X, axis=1, keepdims=True)
    X_mean  = np.mean(X, axis=1, keepdims=True)
    X = (X - X_mean) / X_stdev

    return X


def load_batch(filename):
    """Loads a data batch

    Args:
        filename (str): filename of the data batch to be loaded

    Returns:
        X (np.ndarray): data matrix (D, N)
        Y (np.ndarray): one hot encoding of the labels (C, N)
        y (np.ndarray): vector containing the labels (N,)
    """
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')

        X = normalization((data_dict[b"data"]).T)
        y = np.array(data_dict[b"labels"])
        Y = (np.eye(10)[y]).T

    return X, Y, y



def unpickle(filename):
    """Unpickle the meta file"""

    with open(filename, 'rb') as f:
        file_dict = pickle.load(f, encoding='bytes')

    return file_dict




def dl_One_batch():
    """
    A dataloader for ONE batch for training 
    Returns:
        data (dict):   all the separate data sets
        labels (list): correct image labels
    """
    X_train, Y_train, y_train = \
        load_batch("data/data_batch_1")
    X_val, Y_val, y_val = \
        load_batch("data/data_batch_2")
    X_test, Y_test, y_test = \
        load_batch("data/test_batch")

    labels = unpickle(
        'data/batches.meta')[ b'label_names']

    data = {
        'X_train': X_train,'Y_train': Y_train,'y_train': y_train,
        'X_val': X_val,'Y_val': Y_val,'y_val': y_val,
        'X_test': X_test,'Y_test': Y_test,'y_test': y_test
    }

    return data, labels


def dl_Full_batch(val):
	"""
	A dataloader for all five batches

	Args:
		val: number of data used for valiation
	"""
	X_train1, Y_train1, y_train1 = load_batch("data/data_batch_1")
	X_train2, Y_train2, y_train2 = load_batch("data/data_batch_2")
	X_train3, Y_train3, y_train3 = load_batch("data/data_batch_3")
	X_train4, Y_train4, y_train4 = load_batch("data/data_batch_4")
	X_train5, Y_train5, y_train5 = load_batch("data/data_batch_5")
	
	X_train = np.concatenate((X_train1, X_train2, X_train3, X_train4, X_train5),axis=1)
	Y_train = np.concatenate((Y_train1, Y_train2, Y_train3, Y_train4, Y_train5),axis=1)
	y_train = np.concatenate((y_train1, y_train2, y_train3, y_train4, y_train5))
	
	X_val = X_train[:, -val:]
	Y_val = Y_train[:, -val:]
	y_val = y_train[-val:]
	X_train = X_train[:, :-val]
	Y_train = Y_train[:, :-val]
	y_train = y_train[:-val]

	X_test, Y_test, y_test = load_batch("data/test_batch")
	labels = unpickle('data/batches.meta')[ b'label_names']

	data = {'X_train': X_train,'Y_train': Y_train,'y_train': y_train,
        'X_val': X_val,'Y_val': Y_val,'y_val': y_val,
        'X_test': X_test,'Y_test': Y_test,'y_test': y_test}
    
	return data, labels

def make_layers_param(shapes, activations):
    """Create the layers of the network

    Args:
        shapes      (list): the shapes per layer as tuples
        activations (list): the activation functions per layer as strings
    Returns:
        layers:	 (dict) the shape and activation function of
        each layer
    """
    if len(shapes) != len(activations):
        raise RuntimeError('The size of shapes should equal the size of activations.')

    layers = OrderedDict([])

    for i, (shape, activation) in enumerate(zip(shapes, activations)):
        layers["layer%s" % i] = {"shape": shape, "activation": activation}

    return layers


#########################################
## I/O 
########################################

def name_case(n_s,eta_min,eta_max,
              batch_s,n_epochs,lamda,
              if_batch_norm,k, init_, stdev):
	case_name = f"{if_batch_norm}BN_W&B_{k}Layer_{init_}init_{stdev:.2e}dev"+\
                f"{batch_s}BS_{n_epochs}Epoch_"+\
                f"{n_s}NS_{lamda:.3e}Lambda_{eta_min:.3e}MINeta_{eta_max:.3e}MAXeta"
	return case_name

def save_as_mat(model,hist,acc,name):
	""" Used to transfer a python model to matlab """
	import scipy.io as sio
	sio.savemat("weights/HIST_" + name + '.mat',
			{
				'train_loss':np.array(hist['train_loss']),
				'train_cost':np.array(hist['train_cost']),
				'train_acc':np.array(hist['train_acc']),
				'val_loss':np.array(hist['val_loss']),
				'val_cost':np.array(hist['val_loss']),
				'val_acc':np.array(hist['val_acc']),
				"test_acc":acc
				})
	

#########################################
## Visualisation 
#########################################
def plot_loss(interval, loss,fig=None,axs=None,color=None,ls=None):
	if fig==None:
		fig, axs = plt.subplots(1,1,figsize=(6,4))
	
	if color == None: color = "r"
	if ls == None: ls = '-'
	axs.plot(interval, loss,ls,lw=2.5,c=color)
	axs.set_xlabel('Update Steps')
	axs.set_ylabel('Loss')
	return fig, axs 

def plot_hist(hist,n_start,n_interval,t=None):
	fig , axs  = plt.subplots(1,3,figsize=(24,6))
	
	if t == None:
		n_range = np.arange(len(hist['train_cost']))
	else:
		n_range = np.arange(t)

	fig, axs[0] = plot_loss(n_range[n_start:-1:n_interval], hist['train_cost'][n_start:-1:n_interval],fig,axs[0],color = colorplate.red,ls = '-')
	fig, axs[0] = plot_loss(n_range[n_start:-1:n_interval], hist['val_cost'][n_start:-1:n_interval],fig,axs[0],color =colorplate.blue, ls = '-')
	axs[0].set_ylabel('Cost',font_dict)

	fig, axs[1] = plot_loss(n_range[n_start:-1:n_interval],hist['train_loss'][n_start:-1:n_interval],fig,axs[1],color = colorplate.red,ls = '-')
	fig, axs[1] = plot_loss(n_range[n_start:-1:n_interval],hist['val_loss'][n_start:-1:n_interval],fig,axs[1],color =colorplate.blue, ls = '-')
	axs[1].set_ylabel('Loss',font_dict)
	
	fig, axs[2] = plot_loss(n_range[n_start:-1:n_interval],hist['train_acc'][n_start:-1:n_interval],fig,axs[2],color =colorplate.red, ls = '-')
	fig, axs[2] = plot_loss(n_range[n_start:-1:n_interval],hist['val_acc'][n_start:-1:n_interval],fig,axs[2],color =colorplate.blue, ls = '-')
	axs[2].set_ylabel('Accuracy',font_dict)
	
	for ax in axs:
		ax.legend(['Train',"Validation"],prop={'size':20})
	return fig, axs 



##########################################
## K-layer classifier 
##########################################
class mlpClassifier():
    """
    A MLP classifier with mini-batch gradient descent
    """
    def __init__(self, data, layers, alpha=0.8, 
                 if_batch_norm=False,
                 init_= 'he',
                 stdev = 1e-1):
        """
        Initialization of the model 
        Args:
        data        :   (dict) A dictionary contains the training, val and test data 
        layers	    :   (dict) A dictionary of hyperparameters and activation function to use 
        alpha	    :   (float) initial value for moving average 
        if_batch_norm  :    (bool) IF use BatchNorm in this class
        init_       : The method for initialization
        stdev       : (For he only)
        """

        # Get data by attribute 
        for ky,vl in data.items():
            setattr(self,ky,vl)

        self.layers = layers
        self.k      = len(layers)-1 # Number of layer 
        self.alpha  = alpha 
        self.if_batch_norm = if_batch_norm
        self.activation_funcs = {'relu':self._relu,"softmax":self._softmax}
        print(f"IF USE BN: {self.if_batch_norm}")

        self.W, self.b, self.gamma, self.beta, self.mu_av, self.var_av, \
        self.activations = [], [], [], [], [], [], []


        for layer in layers.values():
            for k, v in layer.items():
                if k == "shape":
                    # According to the parameter we generate and initialize the W&B
                    if init_ == 'he':
                        W, b, gamma, beta, mu_av, var_av = self._he_init(v,stdev)
                    elif init_ == 'xavier':
                        W, b, gamma, beta, mu_av, var_av = self._xavier_init(v)

                    self.W.append(W), self.b.append(b)
                    self.gamma.append(gamma), self.beta.append(beta)
                    self.mu_av.append(mu_av), self.var_av.append(var_av)
                elif k == "activation":
                    # We generate the activation like a tuple, but same utilities as dictionary
                    self.activations.append((v, self.activation_funcs[v]))

        if self.if_batch_norm:
            self.params = {"W": self.W, "b": self.b, "gamma": self.gamma,
                        "beta": self.beta}
        else:
            self.params = {"W": self.W, "b": self.b}


    # Without creating object, facilitate the speed
    @staticmethod
    def _he_init(d,stdev=1e-2):
        """He Kaiming initialization"""

        W      = np.random.normal(0, stdev, size=(d[0], d[1]))
        b      = np.zeros(d[0]).reshape(d[0], 1)
        gamma  = np.ones((d[0], 1))
        beta   = np.zeros((d[0], 1))
        mu_av  = np.zeros((d[0], 1))
        var_av = np.zeros((d[0], 1))

        return W, b, gamma, beta, mu_av, var_av
    
    @staticmethod
    def _xavier_init(d):
        """Xavier initialization"""

        W      = np.random.normal(0, 1/np.sqrt(d[0]), size=(d[0], d[1]))
        b      = np.zeros(d[0]).reshape(d[0], 1)
        gamma  = np.ones((d[0], 1))
        beta   = np.zeros((d[0], 1))
        mu_av  = np.zeros((d[0], 1))
        var_av = np.zeros((d[0], 1))

        return W, b, gamma, beta, mu_av, var_av
    
    @staticmethod
    def _softmax(x):
        s = np.exp(x - np.max(x, axis=0)) / \
                np.exp(x - np.max(x, axis=0)).sum(axis=0)
        return s


    @staticmethod
    def _relu(x):
        x[x<0] = 0
        return x


    def eval_gradients(self, grads_a, grads_n):
        """Maximum relative error between the analytical and numerical gradients

        Args:
            grads_a (np.ndarray): analytical gradients
            grads_n (np.ndarray): numerical gradients
        """
        num_layers = len(grads_a["W"])
        for l in range(num_layers):
            print(f"INFO: Checking layer:{l}")
            for key in grads_a:
                
                # |ga - gf|
                e = abs(grads_a[key][l].flat[:] - grads_n[key][l].flat[:])
                
                # max(|ga|, |gf|)
                s = np.asarray([max(abs(a), abs(b)) + 1e-10 for a,b in
                    zip(grads_a[key][l].flat[:], grads_n[key][l].flat[:])])
                
                max_rel_err = max(e / s)
                print("The relative error for layer %d %s: %.6g" %
                        (l+1, key, max_rel_err))
    

    def forward_prop(self, X, is_testing=False, is_training=False):
        """ Forward propagation of the model
        Args:
            X     (np.ndarray): data matrix (D, N_batch)
            is_testing  (bool): testing mode
            is_training (bool): training mode

        Returns:
            H          Hidden output for hidden layer without activaction values
            P          final outpur
            S          Hidden output with activation
            S_hat      normalized Hidden output with activation
            means      mean vectors
            variancess variance vectors
        """
        N = X.shape[1]
        s = np.copy(X)
        # IF use BN:
        if self.if_batch_norm:
            S, S_hat, means, variances, H = [], [], [], [], []
            for i, (W, b, gamma, beta, mu_av, var_av, activation) in enumerate(
                    zip(self.W, self.b, self.gamma, self.beta, self.mu_av,
                        self.var_av, self.activations)):
                
                H.append(s)
                s = W@s + b
               
                if i < self.k:
                    
                    S.append(s)
                    # For test mode we do not update the moving average
                    if is_testing:
                        s = (s - mu_av) / np.sqrt(var_av + \
                                np.finfo(np.float64).eps)

                    # For training and validation
                    else:
                        mu = np.mean(s, axis=1, keepdims=True)
                        means.append(mu)
                        var = np.var(s, axis=1, keepdims=True) * (N-1)/N
                        variances.append(var)

                        # During training we need to update the moving average
                        if is_training:
                            self.mu_av[i]  = self.alpha * mu_av + \
                                    (1-self.alpha) * mu
                            self.var_av[i] = self.alpha * var_av + \
                                    (1-self.alpha) * var

                        s = (s - mu) / np.sqrt(var + np.finfo(np.float64).eps)

                    S_hat.append(s)
                    s = activation[1](np.multiply(gamma, s) + beta)
                # For the last layer
                else: 
                    P = activation[1](s)
         
              
            return H, P, S, S_hat, means, variances
            

        else:
            # if not self.if_batch_norm:
            H = []
            for W, b, activation in zip(self.W, self.b, self.activations):
                if activation[0] == "relu":
                    s = activation[1](W@s + b)
                    H.append(s)
                if activation[0] == "softmax":
                    ps = activation[1](W@s + b)
            
            return H, ps
    

    def cost_func(self, X, Y, labda, is_testing=False):
        """Computes the cost function of the classifier using the cross-entropy loss

        Args:
            X     (np.ndarray): data matrix (D, N)
            Y     (np.ndarray): one-hot encoding labels matrix (C, N)
            labda (np.float64): regularization term
            is_testing  (bool): flag to indicate the testing phase

        Returns:
            lost (np.float64): current loss of the model
            cost (np.float64): current cost of the model
        """
        N = X.shape[1]

        if self.if_batch_norm:
            _, px, _, _, _, _ = self.forward_prop(X, is_testing=is_testing)
        else:
            _, px = self.forward_prop(X)

        # Part 1: Loss func
        loss = np.float64(1/N) * - np.sum(Y*np.log(px))

        # Part 2: L2 Regularisation
        squaredWeights = 0
        for W in self.W:
            squaredWeights += (np.sum(np.square(W)))
        cost = loss + labda * squaredWeights

        return loss, cost
    
    def compute_accuracy(self, X, y, is_testing=False):
        """ Assess the accuracy of the classifier"""
        if self.if_batch_norm:
            argMaxP = np.argmax(self.forward_prop(
                X, is_testing=is_testing)[1], axis=0)
        else:
            argMaxP = np.argmax(self.forward_prop(X)[1], axis=0)

        # Compute the proportion of TP sample
        acc = argMaxP.T[argMaxP == np.asarray(y)].shape[0] / X.shape[1]

        return acc

    def compute_gradients(self, X_batch, Y_batch, labda):
        """ Analytical gradients for W&B"""

        N = X_batch.shape[1] #  Batch size

        # NO Batch norm
        #-----------------------------------------------------
        if not self.if_batch_norm:
            grads = {"W": [], "b": []}
            for W, b in zip(self.W, self.b):
                grads["W"].append(np.zeros_like(W))
                grads["b"].append(np.zeros_like(b))

            # Forward pass
            H_batch, P_batch = self.forward_prop(X_batch)

            # Backward pass
            G_batch = - (Y_batch - P_batch)

            # for l = k, k-1, ..., 2
            for l in range(len(self.layers) - 1, 0, -1):
                grads["W"][l] = 1/N * G_batch@H_batch[l-1].T + 2 * labda * self.W[l]
                grads["b"][l] = np.reshape(1/N * G_batch@np.ones(N),
                                        (grads["b"][l].shape[0], 1))

                G_batch = self.W[l].T@G_batch
                H_batch[l-1][H_batch[l-1] <= 0] = 0
                G_batch = np.multiply(G_batch, H_batch[l-1] > 0)

            grads["W"][0] = 1/N * G_batch@X_batch.T + labda * self.W[0]
            grads["b"][0] = np.reshape(1/N * G_batch@np.ones(N), self.b[0].shape)
        

        # Batch Normalisation
        #-----------------------------------------------------        
        else:
            grads = {"W": [], "b": [], "gamma": [], "beta": []}

            for key in self.params:
                for par in self.params[key]:
                    grads[key].append(np.zeros_like(par))

            # Forward pass
            H_batch, P_batch, S_batch, S_hat_batch, means_batch, vars_batch = \
                    self.forward_prop(X_batch, is_training=True)

            # Backward pass
            G_batch = - (Y_batch - P_batch)

            # Update the last layer first as we do not have BN in this layer 
            grads["W"][self.k] = 1/N * G_batch@H_batch[self.k].T + \
                    2 * labda * self.W[self.k]
            
            grads["b"][self.k] = np.reshape(1/N * G_batch@np.ones(N),
                    (grads["b"][self.k].shape[0], 1))

            G_batch = self.W[self.k].T@G_batch
            H_batch[self.k][H_batch[self.k] <= 0] = 0
            G_batch = np.multiply(G_batch, H_batch[self.k] > 0)

            # for l = k-1, k-2, ..., 1
            for l in range(self.k - 1, -1, -1):
                grads["gamma"][l] = np.reshape(1/N * np.multiply(G_batch,
                    S_hat_batch[l])@np.ones(N), (grads["gamma"][l].shape[0], 1))
                grads["beta"][l]  = np.reshape(1/N * G_batch@np.ones(N),
                        (grads["beta"][l].shape[0], 1))

                G_batch = np.multiply(G_batch, self.gamma[l])

                G_batch = self.batchNorm_backprop(G_batch, S_batch[l],
                        means_batch[l], vars_batch[l])

                grads["W"][l] = 1/N * G_batch@H_batch[l].T + 2 * labda * self.W[l]

                grads["b"][l] = np.reshape(1/N * G_batch@np.ones(N),
                                        (grads["b"][l].shape[0], 1))
                if l > 0:
                    G_batch = self.W[l].T@G_batch
                    H_batch[l][H_batch[l] <= 0] = 0
                    G_batch = np.multiply(G_batch, H_batch[l] > 0)

        return grads


    def batchNorm_backprop(self, G_batch, S_batch, mean_batch, var_batch):
        """Computation of the batch normalization back pass
            Following Equations 31 - 37 in the instruction
        """
        N = G_batch.shape[1]
        sigma1 = np.power(var_batch + np.finfo(np.float64).eps, -0.5)
        sigma2 = np.power(var_batch + np.finfo(np.float64).eps, -1.5)

        G1 = np.multiply(G_batch, sigma1)
        G2 = np.multiply(G_batch, sigma2)

        D = S_batch - mean_batch

        c = np.sum(np.multiply(G2, D), axis=1, keepdims=True)

        G_batch = G1 - 1/N * np.sum(G1, axis=1, keepdims=True) - \
                1/N * np.multiply(D, c)

        return G_batch

    def compute_gradients_num(self, X_batch, Y_batch, size=2,
            labda=np.float64(0), h=np.float64(1e-5)):
        """Central difference for the gradients of W&B"""
        if self.if_batch_norm:
            grads = {"W": [], "b": [], "gamma": [], "beta": []}
        else:
            grads = {"W": [], "b": []}

        for j in range(len(self.b)):
            for key in self.params:
                grads[key].append(np.zeros(self.params[key][j].shape))
                for i in range(len(self.params[key][j].flatten())):
                    old_par = self.params[key][j].flat[i]
                    self.params[key][j].flat[i] = old_par + h
                    _, c2 = self.cost_func(X_batch, Y_batch, labda)
                    self.params[key][j].flat[i] = old_par - h
                    _, c3 = self.cost_func(X_batch, Y_batch, labda)
                    self.params[key][j].flat[i] = old_par
                    grads[key][j].flat[i] = (c2-c3) / (2*h)

        return grads
    
    def backward(self,X_batch,Y_batch, eta,labda):
        """Use the gradient to update the parameters"""

        grads = self.compute_gradients(X_batch, Y_batch, labda)

        for key in self.params:
            for par, grad in zip(self.params[key], grads[key]):
                par -= eta * grad


    def train(self, X, Y, lr_sch, labda, n_epochs,batch_s):
        """Mini-batch gradient descent"""

        lenX = X.shape[-1]
        train_batch_range = np.arange(0,lenX//batch_s)
        
        hist = {}; hist['train_cost'] = []; hist['val_cost'] = []
        hist['train_loss'] = []; hist['val_loss'] = []
        hist["train_acc"] = []; hist["val_acc"] = []
        
        for epoch in range(n_epochs):
            epst = time.time()
            # Shuffle the batch indicites 
            indices = np.random.permutation(lenX)
            X = X[:,indices]
            Y = Y[:,indices]
            
            for b in (train_batch_range):
                
                j_start = (b) * batch_s
                j_end = (b+1) * batch_s

                X_batch = X[:, j_start:j_end]
                Y_batch = Y[:, j_start:j_end]

                # Update the Param
                self.backward(X_batch,Y_batch, lr_sch.eta,labda)
                # Update the lr schedule
                lr_sch.update_lr()

                if lr_sch.t % batch_s == 0:
                    # Compute and record the loss func
                    loss_train, costs_train = self.cost_func(X, Y, labda)
                    hist['train_cost'].append(costs_train)
                    hist['train_loss'].append(loss_train)
                    
                    loss_val, costs_val     = self.cost_func(self.X_val, self.Y_val, labda)
                    hist['val_cost'].append(costs_val)
                    hist['val_loss'].append(loss_val)

                    acc_train = self.compute_accuracy(self.X_train, self.y_train)
                    acc_val = self.compute_accuracy(self.X_val, self.y_val)
                    hist["train_acc"].append(acc_train)
                    hist["val_acc"].append(acc_val)

            epet = time.time()
            epct = epet - epst
            print(f"\n Epoch ({epoch+1}/{n_epochs}), At Step =({lr_sch.t}/{n_epochs*lenX//batch_s}), Cost Time = {epct:.2f}s\n"+\
					f" Train Cost ={hist['train_cost'][-1]:.3f}, Val Cost ={hist['val_cost'][-1]:.3f}\n"+\
					f" Train Loss ={hist['train_loss'][-1]:.3f}, Val Loss ={hist['val_loss'][-1]:.3f}\n"+\
					f" Train Acc ={hist['train_acc'][-1]:.3f}, Val Acc ={hist['val_acc'][-1]:.3f}\n"+\
					f" The LR = {lr_sch.eta:.4e}")
        
        self.hist = hist
        return self.hist



class lr_scheduler:
	def __init__(self,eta_max,eta_min,n_s):
		"""
	cyclinal learning rate during training
	
	Args: 
		t: (int) Current Number of iteration 

		eta_min: Lower bound of learning rate 

		eta_max: Upper bound of learning rate 

		n_s		: How many epoch per cycle 
		"""

		self.eta_min = eta_min
		self.eta_max = eta_max
		self.n_s 	 = n_s
		self.eta 	 = eta_min
		self.hist    = []
		self.t 		 = 0
		print(f"INFO: LR scheduler:"+\
			f"\n eta_min={self.eta_min:.2e}, eta_max={self.eta_max:.2e}, n_s={self.n_s}"	)
	def update_lr(self):
		"""
		Update the LR
		
		"""
		# cycle = np.floor(1+self.t/(2*self.n_s))
		cycle = (self.t//(2*self.n_s))
		# x = abs(self.t / self.step_size - 2 * cycle + 1)
		
		if (2 * cycle * self.n_s <= self.t) and (self.t <= (2 * cycle + 1) * self.n_s):
			
			self.eta=self.eta_min+(self.t-2*cycle*self.n_s)/\
					self.n_s*(self.eta_max-self.eta_min)
		
		elif ((2 * cycle +1) * self.n_s <= self.t) and (self.t <= 2*( cycle + 1) * self.n_s) :
			
			self.eta=self.eta_max-(self.t-(2*cycle+1)*self.n_s)/\
					self.n_s*(self.eta_max-self.eta_min)
		
		self.hist.append(self.eta)
		self.t +=1


######
# The nittest class for debugging the code. 
######
class TestMethods(unittest.TestCase):
    def test_sizes(self):
        print("INFO: TEST The basic function utilities")
        data, labels = dl_One_batch()

        layers = make_layers_param(
                shapes=[(50, 3072), (10, 50)],
                activations=["relu", "softmax"])

        clf = mlpClassifier(data, layers)

        grads = clf.compute_gradients(clf.X_train, clf.Y_train, labda=0)

        # Assert for test if they are the same:
        self.assertEqual(clf.X_train.shape, (3072, 10000))
        self.assertEqual(clf.Y_train.shape, (10, 10000))
        self.assertEqual(np.shape(clf.y_train), (10000,))
        self.assertEqual(clf.W[0].shape, (50, 3072))
        self.assertEqual(clf.b[0].shape, (50, 1))
        self.assertEqual(clf.W[1].shape, (10, 50))
        self.assertEqual(clf.b[1].shape, (10, 1))
        self.assertEqual(clf.forward_prop(clf.X_train)[0][0].shape,
                        (50, 10000))
        self.assertEqual(clf.forward_prop(clf.X_train)[1].shape,
                        (10, 10000))
        self.assertAlmostEqual(sum(sum(clf.forward_prop(
            clf.X_train[:, 0].reshape((clf.X_train.shape[0], 1)))[1])), 1)
        self.assertIsInstance(clf.cost_func(clf.X_train, clf.Y_train,
            labda=0)[1], float)
        self.assertEqual(grads["W"][0].shape, clf.W[0].shape)
        self.assertEqual(grads["b"][0].shape, clf.b[0].shape)
        self.assertEqual(grads["W"][1].shape, clf.W[1].shape)
        self.assertEqual(grads["b"][1].shape, clf.b[1].shape)


    def test_grad_NOBN(self):
        data, labels = dl_One_batch()
        trunc = 10 
        bs    =  1
        layers = make_layers_param(
                shapes=[(50, trunc), (50, 50), (50, 50), (10, 50)],
                activations=["relu", "relu", "relu", "softmax"])

        clf = mlpClassifier(data,layers,if_batch_norm=False)

        grads_ana = clf.compute_gradients(
                clf.X_train[:trunc, :bs],
                clf.Y_train[:trunc, :bs],
                labda=0)

        grads_num = clf.compute_gradients_num(
                clf.X_train[:trunc, :bs],
                clf.Y_train[:trunc, :bs],
                labda=0,h=1e-5)

        clf.eval_gradients(grads_ana, grads_num)


    def test_grad_BN(self):

        trunc = 10
        bs    = 1
        data, labels = dl_One_batch()
        layers = make_layers_param(
                shapes=[(50, trunc), (50, 50), (10, 50)],
                activations=["relu", "relu", "softmax"])

        clf = mlpClassifier(data, layers=layers, if_batch_norm=True)

        grads_ana = clf.compute_gradients(
                clf.X_train[:trunc, :bs],
                clf.Y_train[:trunc, :bs],
                labda=0)

        grads_num = clf.compute_gradients_num(
                clf.X_train[:trunc, :bs],
                clf.Y_train[:trunc, :bs],
                labda=0,
                h=np.float64(1e-5))

        clf.eval_gradients(grads_ana, grads_num)


def run_net(    num_val=5000,
                shapes=[(50, 3072), (50, 50), (10, 50)], 
                activations=["relu", "relu", "softmax"],
                if_batch_norm=True,
                init_='he',stdev=1e-1,
                n_cycle=2,n_batch=100,
                n_s=2250,n_epoch=90,
                lamda=0.0,

              ):
    """
    Run the code for training, evaluation and visualisation
    """
    # Step 1: Load Data
    #-------------------------
    data, labels = dl_Full_batch(val=num_val)
    #-------------------------
    
    # Step 2: Parameter setup
    #------------------------------
    
    layers = make_layers_param(
                shapes=shapes,
                activations=activations)
    
    stdev = (stdev if init_ != 'xavier' else 0 )
    
    print(f"General INFO: n_s={n_s}, n_cycle = {n_cycle}, n_epoch={n_epoch}")
    lr_dict = {"n_s":n_s,"eta_min":1e-5,"eta_max":1e-1} 
    train_dict = {'batch_s':n_batch,'n_epochs':n_epoch}
    case_name = name_case(**train_dict,**lr_dict,lamda=lamda,
                          if_batch_norm=if_batch_norm,
                          k=len(layers),init_=init_,stdev=stdev)
    
    #-------------------------------

    # Step 3: Create Object 
    #--------------------------------
    lr_sch = lr_scheduler(**lr_dict)
    model = mlpClassifier(data,layers,if_batch_norm=if_batch_norm,init_=init_,stdev=stdev)
    #--------------------------------

    # Step 4: Training! 
    #--------------------------------
    print("\n"+"#"*30)
    print(f"Start:\t{case_name}")
    print(f"#"*30)
    
    hist = model.train(X=data['X_train'],
                Y=data['Y_train'],
                lr_sch=lr_sch, 
                labda = lamda,
                **train_dict)
    #---------------------------------

    # Step 5: Evaluate
    #---------------------------------
    acc_test = model.compute_accuracy(model.X_test,model.y_test)
    print(f"TEST Acc ={acc_test*100:.2f}%")
    #---------------------------------
	
    # Step 6: I/O
    #-------------------------------------
    fig,axs = plot_hist(hist,n_start=1,n_interval=1,t=None)
    fig.savefig(f'Figs/Loss_{case_name}.jpg',**fig_dict)
    save_as_mat(model,hist,acc_test,name=case_name)
    #-------------------------------------
    
    return acc_test

def lamda_coarse_search():
    """
    Search for the optimal L2 regularisation term for a 3-layer MLP with BN
    """
    l_min, l_max 	= -5, -1 
    n_search  		= 8
    
    acc_dict = {}
    acc_dict['lambda'] = np.zeros(shape=(n_search,)).astype(np.float64)
    acc_dict['acc'] = np.zeros(shape=(n_search,)).astype(np.float64)
    
    icount = 0
    for l in (range(n_search)):
        print(f'\nSearch At ({l+1}/{n_search})')
        l = l_min + (l_max - l_min)*np.random.rand()
        lamda_iter = 10**l
        print(f"Current Lambda = {lamda_iter:.3e}")

        model_config = dict(num_val=5000,
                        shapes=[(50, d), (50, 50), (K, 50)], 
                        activations=["relu", "relu", "softmax"],
                        if_batch_norm=True,
                        init_='he',stdev=1e-1,
                        n_cycle=2,n_batch=100,
                        n_s=2250,n_epoch=20,
                        lamda=lamda_iter)
        
        acc         = run_net(**model_config)

        acc_dict['lambda'][icount] = lamda_iter
        acc_dict['acc'][icount] = acc
        icount += 1
    
    df = pd.DataFrame(acc_dict)
    df.to_csv('Coarse_Search.csv')

def lamda_finer_search():
    """
    Search for the optimal L2 regularisation term for a 3-layer MLP with BN
    """
    l_min, l_max 	= -2.5, -2 
    n_search  		= 8
    
    acc_dict = {}
    acc_dict['lambda'] = np.zeros(shape=(n_search,)).astype(np.float64)
    acc_dict['acc'] = np.zeros(shape=(n_search,)).astype(np.float64)
    
    icount = 0
    for l in (range(n_search)):
        print(f'\nSearch At ({l+1}/{n_search})')
        l = l_min + (l_max - l_min)*np.random.rand()
        lamda_iter = 10**l
        print(f"Current Lambda = {lamda_iter:.3e}")

        model_config = dict(num_val=5000,
                        shapes=[(50, d), (50, 50), (K, 50)], 
                        activations=["relu", "relu", "softmax"],
                        if_batch_norm=True,
                        init_='he',stdev=1e-1,
                        n_cycle=2,n_batch=100,
                        n_s=2250,n_epoch=20,
                        lamda=lamda_iter)
        
        acc         = run_net(**model_config)

        acc_dict['lambda'][icount] = lamda_iter
        acc_dict['acc'][icount] = acc
        icount += 1
    
    df = pd.DataFrame(acc_dict)
    df.to_csv('Finer_Search.csv')

def sensitivity_study():
    """
    A sensitivity study on the initialisation stdev for Weight matrices
    
    On BN and No-BN case
    """
    # Std to use for initialisation
    sigs = [1e-1,1e-3,1e-4]
    # Get from the finer search 
    opt_lambda = 0.0032461338701209826

    acc_dict = {}
    acc_dict['sig'] = np.zeros(shape=(len(sigs),)).astype(np.float64)
    acc_dict['acc_noBN'] = np.zeros(shape=(len(sigs),)).astype(np.float64)
    acc_dict['acc_BN'] = np.zeros(shape=(len(sigs),)).astype(np.float64)
    
    icount = 0
    for il, sig in enumerate(sigs):
        print(f'\nSearch At ({il+1}/{len(sigs)})')
        print(f"Current Std = {sig:.3e}")

        model_config = dict(num_val=5000,
                        shapes=[(50, d), (50, 50), (K, 50)], 
                        activations=["relu", "relu", "softmax"],
                        if_batch_norm=True,
                        init_='he',stdev=sig,
                        n_cycle=2,n_batch=100,
                        n_s=2250,n_epoch=20,
                        lamda=opt_lambda)
        
        acc         = run_net(**model_config)

        acc_dict['sig'][icount] = sig
        acc_dict['acc_BN'][icount] = acc


        model_config = dict(num_val=5000,
                        shapes=[(50, d), (50, 50), (K, 50)], 
                        activations=["relu", "relu", "softmax"],
                        if_batch_norm=False,
                        init_='he',stdev=sig,
                        n_cycle=2,n_batch=100,
                        n_s=2250,n_epoch=20,
                        lamda=opt_lambda)
        
        acc         = run_net(**model_config)

        acc_dict['sig'][icount] = sig
        acc_dict['acc_noBN'][icount] = acc

        icount += 1
    
    df = pd.DataFrame(acc_dict)
    df.to_csv('Sensitivity_Study.csv')



if __name__ == '__main__':
    
    K = 10; d = 3072
    no_bn_3layer_config = dict(num_val=5000,
                        shapes=[(50, d), (50, 50), (K, 50)], 
                        activations=["relu", "relu", "softmax"],
                        if_batch_norm=False,
                        init_='he',stdev=1e-1,
                        n_cycle=2,n_batch=100,
                        n_s=2250,n_epoch=20,
                        lamda=0.005)

    bn_3layer_config = dict(num_val=5000,
                        shapes=[(50, d), (50, 50), (K, 50)], 
                        activations=["relu", "relu", "softmax"],
                        if_batch_norm=True,
                        init_='he',stdev=1e-1,
                        n_cycle=2,n_batch=100,
                        n_s=2250,n_epoch=20,
                        lamda=0.005)
    
    no_bn_9layer_config = dict(num_val=5000,
                        shapes=[(50, d), (30, 50), (20, 30), (20, 20), (10, 20),
                                (10, 10), (10, 10), (10, 10), (K, 10)],
                        activations=["relu", "relu", "relu", "relu", "relu", "relu",
                                "relu", "relu", "softmax"],
                        if_batch_norm=False,
                        init_='xavier',stdev=1e-1,
                        n_cycle=2,n_batch=100,
                        n_s=2250,n_epoch=20,
                        lamda=0.005)
    

    bn_9layer_config = dict(num_val=5000,
                        shapes=[(50, d), (30, 50), (20, 30), (20, 20), (10, 20),
                                (10, 10), (10, 10), (10, 10), (K, 10)],
                        activations=["relu", "relu", "relu", "relu", "relu", "relu",
                                "relu", "relu", "softmax"],
                        if_batch_norm=True,
                        init_='he',stdev=1e-1,
                        n_cycle=2,n_batch=100,
                        n_s=2250,n_epoch=20,
                        lamda=0.005)
        

    
    if args.m == 1:
        unittest.main()
    elif args.m == 2:
        test_acc_nobn = run_net(**no_bn_3layer_config)
        test_acc_bn   = run_net(**bn_3layer_config)

    elif args.m == 3:
        test_acc_nobn = run_net(**no_bn_9layer_config)
        test_acc_bn   = run_net(**bn_9layer_config)

    elif args.m == 4:
        lamda_coarse_search()
    
    elif args.m == 5:
        lamda_finer_search()
    
    elif args.m ==6:
        sensitivity_study()