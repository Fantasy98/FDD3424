"""
Assignment 4: Valia RNN for synthesize English text character by character 
@yuningw 
May 2nd
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
from matplotlib import ticker as ticker
# Dictionary for layer. cache, etc
from collections import OrderedDict
import unittest
import argparse 

# Parse Arguments 
parser = argparse.ArgumentParser()
parser.add_argument("-m","--mode",default=1,type=int,help='Choose which exercise to do 1,2,3,4,5')
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
def read_data(file_name):
    """read txt data"""

    with open(file_name,'r',encoding='utf8') as book :
        
        # Load all the data 
        book_data = book.read()

        # Split into char 
        book_char = list(set(book_data))
    
    book.close()

    data = {"book_data":book_data,'book_chars':book_char,
            'word_len':len(book_char),
            'char_to_idx':OrderedDict((c,idx) for idx,c in enumerate(book_char)),
            'idx_to_char':OrderedDict((idx,c) for idx,c in enumerate(book_char)),
            }
    return data

##########################################
## RNN object 
##########################################
class RNN: 
    """Implementation of conventional RNN """
    def __init__(self,data,m=100,eta=0.1,seq_length=25):
        """
        Initialisation of RNN 

        Args: 
            data    :   train, val, test data 

            m       :   Hidden Size 

            eta     :   Learning Rate 

            seq_length: The length of input sequence 

        """

        self.m, self.eta, self.N  = m, eta, seq_length

        for k, v in data.items():
            setattr(self,k,v)
        
        # W&B 
        self.U, self.V, self.W, self.b, self.c =  self._init_param(m = self.m, K = self.word_len)


        print(f"Summary: U = {self.U.shape}; V ={self.V.shape}; W = {self.W.shape}; b = {self.b.shape}, c = {self.c.shape}")

# Statsitic methods 
#----------------------------------------------------------
    @staticmethod
    def _init_param(m,K,sig=0.01):
        """
        Initialisation of W&B using He normalisation
        """
        
        b = np.zeros((m,1)).astype(np.float64)
        c = np.zeros((K,1)).astype(np.float64)

        U = np.random.normal(0,sig,size=(m,K)).astype(np.float64)
        V = np.random.normal(0,sig,size=(K,m)).astype(np.float64)
        W = np.random.normal(0,sig,size=(m,m)).astype(np.float64)

        
        return U,V,W, b, c
    

    @staticmethod
    def _softmax(x):
        s = np.exp(x - np.max(x,0)) / np.exp(x - np.max(x,0)).sum(0)
        return s 
    
    @staticmethod
    def _tanh(x):
        return np.tanh(x)

# Forward Prop
#----------------------------------------------------------
    def forward(self,h,x):
        """
        Args: 
            h   :   Hidden state 

            x   :   Input seqence 
        """

        a = self.W @ h + self.U @ x + self.b 
        h = self._tanh(a)
        o = self.V @ h  + self.c 
        p = self._softmax(o)
        return a,h,o,p


    def syn_txt(self,h,ix_start,n):
        """
        Synthesize the text for hidden state, similar to Time-Delay 
        
        ix_start  : location to start generate the text 
        
        n   : Length of txt to generate 

        """
        
        xnext     = np.zeros((self.word_len,1)).astype(np.float64)
        
        xnext[ix_start] = 1

        txt = ''

        for t in range  (n):
            _,h,_,p = self.forward(h,xnext)

            # Random choose an output according to the computed probability from the output 
            ix = np.random.choice(range(self.word_len), p = p.flat)

            xnext = np.zeros_like(xnext)
            
            # Mask by ONE
            xnext[ix] = 1

            # Get the text 
            txt+=self.idx_to_char[ix]
        
        return txt 


# Back Prop
#----------------------------------------------------------
    def compute_gradients(self,x,y,hprev):
        """
        Compute the gradient
        """
        
        n = len(x)
        loss = 0.0

        ## Prepare a dict for values appeared in forward pass 
        ad,xd,hd,od,pd = {}, {}, {}, {}, {}

        ## Locate the hidden output at the last 
        hd[-1] = np.copy(hprev)

        ## Step 1: Forward prop 
        #----------------------------
        for ti in range(n):
            xd[ti] = np.zeros((self.word_len,1))
            
            # One-Hot Encoding
            xd[ti][x[ti]] = 1 
            
            # The forward prop,
            ad[ti],hd[ti],od[ti],pd[ti] = self.forward(hd[ti-1], xd[ti])

            # Compute the Cross-entropy loss in accmulative way
            loss += -np.log(pd[ti][y[ti]][0])
        #----------------------------
        
        
        ## Step 2: Back Prop
        #----------------------------
        ## A dict for gradient 
        grads  = {
                    # For weight
                    "W": np.zeros_like(self.W).astype(np.float64),
                    "V": np.zeros_like(self.V).astype(np.float64),
                    "U": np.zeros_like(self.U).astype(np.float64),
                    "b": np.zeros_like(self.b).astype(np.float64),
                    "c": np.zeros_like(self.c).astype(np.float64),
                    # For passsing gradient 
                    "p": np.zeros_like(pd[0]).astype(np.float64),
                    "h": np.zeros_like(hd[0]).astype(np.float64),
                    "h_n": np.zeros_like(hd[0]).astype(np.float64),
                    "a": np.zeros_like(ad[0]).astype(np.float64),
                }
        
        for it in reversed(range(n)):

            grads['p']         = np.copy(pd[it])

            # Gradient for cross-entropy     
            grads["p"][y[it]] -= 1 

            # Back prop for V
            grads['V'] += grads['p']@hd[it].T 
            grads['c'] += grads['p']

            grads['h'] = self.V.T @ grads['p'] + grads['h_n']
            grads['a'] = np.multiply( grads['h'], (1 - np.square(hd[it]) ) )
            
            grads['U'] += grads['a'] @ xd[it].T
            grads['W'] += grads['a'] @ hd[it-1].T
            grads['b'] += grads['a']

            grads['h_n'] = self.W.T @ grads['a']
        

        ## Step 3: Remove the gradient NOT use 
        #----------------------------
        grads =  {k:grads[k] for k in grads if k not in ['p', 'h', 'h_n', 'a']}
        
        ## Step 4: Gradient Clipping any value higher 5 
        for grad in grads: 
            grads[grad] = np.clip(grads[grad],-5,5)

        
        # We make the outputs which becomes the input for next layer 
        h = hd[n-1]

        return grads, loss, h 


    def compute_gradient_numerical(self,x,y,hprev,h,num_comps = 20):

        rnn_params = {'W':self.W,'U':self.U,'V':self.V,
                    'b':self.b,'c':self.c}

        grads_num  =  {
                    # For weight
                    "W": np.zeros_like(self.W).astype(np.float64),
                    "V": np.zeros_like(self.V).astype(np.float64),
                    "U": np.zeros_like(self.U).astype(np.float64),
                    "b": np.zeros_like(self.b).astype(np.float64),
                    "c": np.zeros_like(self.c).astype(np.float64),
                    }
        
        for key in rnn_params:
            for i in range(num_comps):
                
                old_par                 = rnn_params[key].flat[i]
                
                # Forward
                rnn_params[key].flat[i] = old_par + h
                _,l1,_                  = self.compute_gradients(x,y,hprev)
                
                # Backward
                rnn_params[key].flat[i] = old_par - h
                _,l2,_                  = self.compute_gradients(x,y,hprev)
                

                rnn_params[key].flat[i] = old_par

                # Central difference 
                grads_num[key].flat[i]  = (l1-l2)/(2*h)

        return grads_num

    def relative_grad_error(self,x,y,hprev,num_comps = 20):

        grads_a, _,_ = self.compute_gradients(x,y,hprev)
        grads_n      = self.compute_gradient_numerical(x,y,hprev,
                                                    h = 1e-5,num_comps= num_comps)

        print(f"INFO: Checking Gradients computation ")

        for grad in grads_a:
            err     = np.abs(grads_a[grad].flat[:num_comps] - grads_n[grad].flat[:num_comps])

            diff    = np.asarray([ max(abs(a), abs(b)) + 1e-10 for a,b in 
                                zip(
                                    grads_a[grad].flat[:num_comps],
                                    grads_n[grad].flat[:num_comps],
                                    )
                                    ])
            
            err_l   = np.mean(err/diff)

            print(f"The mean relative error of {grad} = {err_l:.3e}")
        
#############################
##  Main Programme 
#############################
    
def main():

    #define counters 
    e,n,epoch = 0,0,0  

    # Epochs 
    Epochs = 10 

    # Loading data 

    data = read_data('goblet_book.txt')

    # RNN 
    model  = RNN(data)

    # Parameter for ADGRAD 

    rnn_params = {'W':model.W,"U":model.U,"V":model.V,
                "b":model.b,"c":model.c}
    
    cache_params = {'W':np.zeros_like(model.W).astype(np.float64),
                    "U":np.zeros_like(model.U).astype(np.float64),
                    "V":np.zeros_like(model.V).astype(np.float64),
                    "b":np.zeros_like(model.b).astype(np.float64),
                    "c":np.zeros_like(model.c).astype(np.float64)}

    loss_hist = []

    while epoch < Epochs: 
        # When it comes to the end of book we finish the epoch 
        if n == 0 or e >= (len(model.book_data) - model.N -1 ):
            if epoch != 0: print(f"INFO: Epoch = {epoch+1}/{Epochs} Complete!")

            # initialize the hidden variable 
            hprev = np.zeros((model.m,1)).astype(np.float64)
            # A counter for locating where we are in the book
            e = 0 
            epoch +=1 

        # Building shifting sequence, like DMD :-)
        x = [model.char_to_idx[char] for char in model.book_data[e:e+model.N]]
        y = [model.char_to_idx[char] for char in model.book_data[e+1:e+model.N+1]]

        
        grads, loss, hprev = model.compute_gradients(x,y,hprev)

        
        # Compuet the smooth loss
        if n == 0 and epoch ==1 : smooth_loss = loss 
        smooth_loss = 0.999 * smooth_loss + (1-0.999)*loss
        loss_hist.append(smooth_loss)

        # Compute gradient check 
        if n == 0 and epoch == 1: model.relative_grad_error(x,y,hprev)

        # Print Loss by verbose: 
        if n % 200 == 0: print(f'At Iteration {n}\t Smooth Loss = {smooth_loss:.3e}')

        # Print Syn text 
        if n % 500 == 0:

            txt_gen = model.syn_txt(hprev,x[0],200)

            print(f"\n At Iter={n}, Loss = {smooth_loss:.3e}; The Synthesized txt:\n {txt_gen}")

            with open(f"Epoch_{epoch}_Iter{n}.txt",'w') as w:
                w.write(txt_gen)
            w.close()
        
        ## Update Param: 
        for key in rnn_params:
            cache_params[key] += grads[key] * grads[key]
            rnn_params[key] -= model.eta / np.sqrt(cache_params[key] + np.finfo(float).eps) * grads[key]
        
        e += model.N 
        n += 1

    fig, axs = plt.subplots(1,1,figsize = (14,4))
    lenloss = len(loss_hist)
    n_range = np.arange(lenloss)
    
    axs.plot(n_range,loss_hist,c = colorplate.red, lw=3)
    axs.set_xlabel('Epoch', fontdict=font_dict)
    axs.set_ylabel('Smooth  Loss', fontdict=font_dict)
    
    fig.savefig('Loss_evo.jpg',**fig_dict)


if __name__ == '__main__':
    main()
