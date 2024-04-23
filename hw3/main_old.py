"""
Assignment 2

MLP with BatchNorm 

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
from tqdm import tqdm 
import time
import pathlib
import argparse
from matplotlib import ticker as ticker
# Parse Arguments 
parser = argparse.ArgumentParser()
parser.add_argument('-m',default=1,type=int,help='Choose which exercise to do 1,2,3,4,5')
parser.add_argument('-epoch',default=200,type=int,help='Number of epoch')
parser.add_argument('-batch',default=10,type=int,help='Batch size')
parser.add_argument('-lr',default=1e-3,type=float,help='learning rate')
parser.add_argument('-lamda',default=0,type=float,help='l2 regularisation')
args = parser.parse_args()

# Mkdir 
pathlib.Path('Figs/').mkdir(exist_ok=True)
pathlib.Path('data/').mkdir(exist_ok=True)
pathlib.Path('weights/').mkdir(exist_ok=True)
font_dict = {'size':20,'weight':'bold'}
fig_dict = {'bbox_inches':'tight','dpi':300}
# Setup the random seed
np.random.seed(400)
# Set the global variables
# global K, d, label

# For visualisation 
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

# Set up the parameter for training 
class GDparams:
	eta 	= args.lr		# [0.1,1e-3,1e-3,1e-3]
	n_batch = args.batch		# [100,100,100,100]
	n_epochs= args.epoch 		# [40,40,40,40]
	lamda 	= args.lamda 		# [0,0,0.1,1]

#-----------------------------------------------


##########################################
## Function from the assignments
##########################################

# I rename the function 
def readPickle(filename):
	""" Copied from the dataset website """
	# import pickle5 as pickle 
	with open('data/'+ filename, 'rb') as f:
		dict=pickle.load(f, encoding='bytes')
	f.close()
	return dict

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def montage(W,label):
	""" Display the image for each label in W """
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(2,5,figsize=(12,6))
	for i in range(2):
		for j in range(5):
			im  = W[i*5+j,:].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[i][j].imshow(sim, interpolation='nearest',cmap='RdBu')
			ax[i][j].set_title(f"y={str(5*i+j)}\n{str(label[5*i+j])}")
			ax[i][j].axis('off')
	return fig, ax 

def save_as_mat(model,hist,acc,name):
	""" Used to transfer a python model to matlab """
	import scipy.io as sio
	sio.savemat("HIST_" + name + '.mat',
			{
				'train_loss':np.array(hist['train_loss']),
				'train_cost':np.array(hist['train_cost']),
				'train_acc':np.array(hist['train_acc']),
				'val_loss':np.array(hist['val_loss']),
				'val_cost':np.array(hist['val_loss']),
				'val_acc':np.array(hist['val_acc']),
				"test_acc":acc
				})
	
	sio.savemat(name+'.mat',
			 	{**model.W_dict,**model.b_dict})

def name_case(n_s,eta_min,eta_max,n_batch,n_epochs,lamda):
	case_name = f"W&B_{n_batch}BS_{n_epochs}Epoch_{n_s}NS_{lamda:.3e}Lambda_{eta_min:.3e}MINeta_{eta_max:.3e}MAXeta"
	return case_name
#-----------------------------------------------



##########################################
## Functions from scratch 
## Yuningw 
##########################################

#----------------------
#	Data and pre-processing 
#-----------------------
#----------------------------------------------
def LoadBatch():
	"""
	Load Data from the binary file using Pickle 

	Returns: 
		X		: [d,n] 
		Y		: [1,n] 
		X_val	: [d,n] 
		Y_val	: [1,n] 
		X_test	: [d,n] 
		Y_test	: [1,n] 
	"""
	
	dt = readPickle("data_batch_1")
	X 		= np.array(dt[b'data']).astype(np.float64).T
	y 		= np.array(dt[b'labels']).astype(np.float64).flatten()
	print(f"TRAIN X: {X.shape}")
	print(f"TRAIN Y:{y.shape}, here are {len(np.unique(y))} Labels")
	
	dt = readPickle("data_batch_2")
	X_val 		= np.array(dt[b'data']).astype(np.float64).T
	y_val 		= np.array(dt[b'labels']).astype(np.float64).flatten()
	print(f"Val X: {X.shape}")
	print(f"Val Y:{y.shape}, here are {len(np.unique(y))} Labels")

	del dt 
	return X, y, X_val,y_val

def LoadAll():

	X = []; y = []
	for n in range(5):
		dt = readPickle(f"data_batch_{n+1}")
		X_ = np.array(dt[b'data']).astype(np.float64).T
		y_ = np.array(dt[b'labels']).astype(np.float64).flatten()
		X.append(X_)
		y.append(y_)
	
	X = np.concatenate(X,axis=-1)
	y = np.concatenate(y,axis=-1)
	print(f"TRAIN X: {X.shape}")
	print(f"TRAIN Y:{y.shape}, here are {len(np.unique(y))} Labels")
	return X, y


def load_test_data():
	"""
	Load Data from the binary file using Pickle 

	Returns: 
		X	: [d,n] 
		Y	: [1,n]  
	"""
	dt = readPickle("test_batch")
	X 		= np.array(dt[b'data']).astype(np.float64).T
	y 		= np.array(dt[b'labels']).astype(np.float64).flatten()
	print(f"TEST X: {X.shape}")
	print(f"TEST Y:{y.shape}, here are {len(np.unique(y))} Labels")
	return X, y

def normal_scaling(X):
	"""
	Pre-processing of the data: normalisation and reshape

	Args:
		X:	Numpy array with shape of Nxd
	Returns: 
		X		: Normalized Input 
		mean_x 	: Mean value of X for sample
		std_x	: STD of X 
	"""
	mean_x = np.repeat(np.mean(X,axis=1).reshape(-1,1),X.shape[-1],1).astype(np.float64)
	std_x  = np.repeat(np.std(X,axis=1).reshape(-1,1),X.shape[-1],1).astype(np.float64)
	# print(f"The Mean={mean_x.shape}; Std = {std_x.shape}")	
	print(f"INFO: Complete Normalisation")

	return (X-mean_x)/std_x, mean_x, std_x

def one_hot_encode(y,K):
	"""
	One-hot encoding for y
	Args:
		y		: [1,n] Un-encoded ground truth 
		K		: (int) Number of labels 

	Returns:
		y_hat 			: [K,n] Encoded ground truth 

	"""

	y_hat = np.zeros(shape=(K, len(y))).astype(np.float64)
	
	for il, yi in enumerate(y):
		y_hat[int(yi),il] = 1
	
	return y_hat

def train_validation_split(X, y, validation_ratio=0.2):
    """
    Split the dataset into training and validation sets.

    Parameters:
    - X: numpy array, shape (n_samples, n_features), input data samples.
    - y: numpy array, shape (n_samples,), input data labels.
    - validation_ratio: float, ratio of validation data to total data.
    - random_seed: int, seed for random number generator.

    Returns:
    - X_train: numpy array, shape (n_train_samples, n_features), training data samples.
    - X_val: numpy array, shape (n_val_samples, n_features), validation data samples.
    - y_train: numpy array, shape (n_train_samples,), training data labels.
    - y_val: numpy array, shape (n_val_samples,), validation data labels.
    """
    
    # Shuffle indices
    indices = np.arange(X.shape[-1])
    np.random.shuffle(indices)
    
    # Calculate number of validation samples
    n_val_samples = int(X.shape[-1] * validation_ratio)
    
    # Split indices into training and validation indices
    val_indices =   indices[:n_val_samples]
    train_indices = indices[n_val_samples:]
    
    # Split data into training and validation sets
    X_train, X_val = X[:,train_indices], X[:,val_indices]
    y_train, y_val = y[:,train_indices], y[:,val_indices]
    
    return X_train, y_train,X_val,y_val
#----------------------------------------------

def dataLoader_OneBatch():
	# Step 1: Load data
	X, Y, X_val ,Y_val = LoadBatch()
	# Define the feature size and label size
	K = len(np.unique(Y)); d = X.shape[0]
	# One-Hot encoded for Y 
	Yenc 	 = one_hot_encode(Y,K)
	Y_val = one_hot_encode(Y_val,K)
	print(f"Global K={K}, d={d}")
	# Load Test Data
	X_test,Y_test = load_test_data()
	Y_test =one_hot_encode(Y_test,K)

	# Step 2: Scaling the data
	X,muX,stdX 		= normal_scaling(X)
	X_val    		= (X_val - muX )/stdX
	X_test    		= (X_test - muX )/stdX

	return X, Yenc, X_val,Y_val,X_test,Y_test

def dataLoader_FullBatch(split_ratio=0.1):
	X,Y = LoadAll()
	K = len(np.unique(Y)); d = X.shape[0]
	Y 				= one_hot_encode(Y,K)
	X,muX,stdX 		= normal_scaling(X)
	
	X,Y,X_val,Y_val =train_validation_split(X,Y,split_ratio)
	X_test,Y_test 	= load_test_data()
	Y_test 			= one_hot_encode(Y_test,K)
	X_test    		= (X_test - muX[:,:X_test.shape[-1]] )/stdX[:,:X_test.shape[-1]]
	print(f"FINISHED NORMALISATION")
	print(f"SUMMARY DATA: TRAIN={X.shape},{Y.shape}")
	print(f"SUMMARY DATA: VAL={X_val.shape},{Y_val.shape}")
	print(f"SUMMARY DATA: TEST={X_test.shape},{Y_test.shape}")
	
	del muX, stdX 
	return X,Y,X_val,Y_val,X_test,Y_test

#----------------------
#	Model
#-----------------------
#----------------------------------------------

class mlp:
	def __init__(self,k_,d_,
			  		h_size=[50],
					lamda=0,
					ifact= True,
					ifBN = True
				):
		self.K  		= k_ 
		self.d  		= d_ 
		self.m  		= h_size 
		self.num_layer  = len(h_size)
		self.lamda 		= lamda
		self.init_WB()
		print(f"INFO: Model initialised: LAMBDA = {self.lamda:3e} K={self.K}, d={self.d}, m={self.m}")
		

		# More dictionary for computation
		self.layerout = {}
		self.W_grad = {}
		self.b_grad = {}
		self.cache  = []
	def forward(self,x):
		"""
		Forward Propagation 
		"""
		
		# if self.num_layer > 1:

		self.layerout[f'h0'] = x
		for il in range(1,self.num_layer+2):
			hkey  = f'h{il}'; hkey_ = f'h{il-1}'
			Wkey  = f"W{il}"; bkey  = f"b{il}"
			# else: # Everytime we save the non-activated output and activate it in the next step
			if il < self.num_layer+1:
				self.layerout[hkey] = ReLU(self.W_dict[Wkey] @ (self.layerout[hkey_]) + self.b_dict[bkey])
			else: 
				p = softmax(self.W_dict[Wkey] @ (self.layerout[hkey_]) + self.b_dict[bkey])
		# For the last layer we use softmax to activate the output 
		return p

	def cost_func(self,X,Y,return_loss = False):
		"""
		Compute the cost function: c = loss + regularisation 

		Args: 
			X	: [d,n] input 
			Y	: [K,n] One-Hot Ground Truth 
		
		Return:
			J	: (float) A scalar of the cost function 
		"""
		import numpy.linalg as LA
		# Part 1: compute the loss:
		## 1 Compute prediction:
		P = self.forward(X)
		## Cross-entropy loss
		# Clip the value to avoid ZERO in log
		P = np.clip(P,1e-16,1-1e-16)
		l_cross =  -np.mean(np.sum(Y*np.log(P),axis=0)).astype(np.float64)
		
		# Part 2: Compute the regularisation 
		reg = 0 
		for il in range(self.num_layer+1):
			reg += np.sum(self.W_dict[f"W{il+1}"]**2)
		reg *= self.lamda
		# Assemble the components
		J = l_cross + reg

		del X, Y, P 
		if return_loss:
			return J, l_cross
		else: 
			return J 

	def compute_acc(self,X,Y):
		"""
		Compute the accuracy of the classification 
		
		Args:

			X	: [d,n] input 
			Y	: [1,n] OR [d,n] Ground Truth 
			
		Returns: 

			acc : (float) a scalar value containing accuracy 
		"""

		lenY = Y.shape[-1]
		# Generate Output with [K,n]
		P = self.forward(X)
		#Compute the maximum prob 
		# [K,n] -> K[1,n]
		P = np.clip(P,1e-16,1-1e-16)
		P = np.argmax(P,axis=0)
		# Compute how many true-positive samples
		if Y.shape[0] != 1: Y = np.argmax(Y,axis=0)

		true_pos = np.sum(P == Y)
		# Percentage on total 
		acc =  true_pos / lenY
		
		del P , X , Y

		return acc

	def computeGradient(self,x,y):
		"""
		Compute the Gradient w.r.t the W&B 

		Args:

			x	: [d,n] input 
			y	: [1,n] Ground Truth 
		"""

		# compute the Prediction 
		p = self.forward(x)
		# Gradient
		g = -(y - p).T 
		# Start Back Prop from the last layer to the first
		for il in reversed(range(2,self.num_layer+2)):
			# print(f"Compute Grad for Layer:{il}")

			self.b_grad[f'b{il}'] = np.sum(g,axis=0,keepdims=True).T.astype(np.float64)
			self.W_grad[f'W{il}']= g.T @ (self.layerout[f'h{il-1}']).T.astype(np.float64)
			#Back prop the gradient
			g = g @ self.W_dict[f'W{il}']
			hbatch = self.layerout[f'h{il-1}'].T 
			hbatch[hbatch<=0.0] = 0
			g = np.multiply(g,hbatch>0)

		self.b_grad[f'b1'] = np.sum(g,axis=0,keepdims=True).T.astype(np.float64)
		self.W_grad[f'W1']= g.T @ x.T.astype(np.float64)
		
		del g 


	def backward(self,x,y,eta_,batch_size):
		"""Back Prop for Update the W&B"""
		self.computeGradient(x,y)
		for il in range(1,self.num_layer+2):
			self.W_dict[f"W{il}"] -= eta_ * ((1/batch_size)*(self.W_grad[f'W{il}']) + 2*self.lamda*self.W_dict[f'W{il}'])
			self.b_dict[f"b{il}"] -= eta_ * ((1/batch_size)*(self.b_grad[f'b{il}']))

		
	
	def train(self,X,Y,X_val,Y_val,lr_sch,n_epochs,n_batch,fix_eta = 1e-3):
		
		print(f'Start Training, Batch Size = {n_batch}, Lr schedule = {lr_sch}')
		lenX = X.shape[-1]
		lenX_val = X_val.shape[-1]
		batch_size = n_batch
		train_batch_range = np.arange(0,lenX//n_batch)

		hist = {}; hist['train_cost'] = []; hist['val_cost'] = []
		hist['train_loss'] = []; hist['val_loss'] = []
		hist["train_acc"] = []; hist["val_acc"] = []
		st = time.time()
		for epoch in (range(n_epochs)): 
			
			epst = time.time()
			# Shuffle the batch indicites 
			indices = np.random.permutation(lenX)
			X = X[:,indices]
			Y = Y[:,indices]
			for b in (train_batch_range):
				
				eta_ = (lr_sch.eta if lr_sch != None else fix_eta)

				X_batch = X[:,b*batch_size:(b+1)*batch_size]
				Y_batch = Y[:,b*batch_size:(b+1)*batch_size]
				
				self.backward(  X_batch,
								Y_batch,
								eta_,
								batch_size)
				if lr_sch !=None:
					lr_sch.update_lr()

				if lr_sch.t % batch_size == 0:
					# Compute the cost func and loss func
					jc,l_train = self.cost_func(X,Y,return_loss=True)
					hist['train_cost'].append(jc)
					hist['train_loss'].append(l_train)
					jc_val,l_val  = self.cost_func(X_val,Y_val,return_loss=True)
					hist['val_cost'].append(jc_val)
					hist['val_loss'].append(l_val)
					# Compute the accuracy 
					train_acc 		= self.compute_acc(X,Y)
					val_acc 		= self.compute_acc(X_val,Y_val)
					hist["train_acc"].append(train_acc)
					hist["val_acc"].append(val_acc)

				
			epet = time.time()
			epct = epet - epst
			
			if lr_sch !=None: 
				print(f"\n Epoch ({epoch+1}/{n_epochs}), At Step =({lr_sch.t}/{n_epochs*lenX//batch_size}), Cost Time = {epct:.2f}s\n"+\
					f" Train Cost ={hist['train_cost'][-1]:.3f}, Val Cost ={hist['val_cost'][-1]:.3f}\n"+\
					f" Train Loss ={hist['train_loss'][-1]:.3f}, Val Loss ={hist['val_loss'][-1]:.3f}\n"+\
					f" Train Acc ={hist['train_acc'][-1]:.3f}, Val Acc ={hist['val_acc'][-1]:.3f}\n"+\
					f" The LR = {lr_sch.eta:.4e}")
			
			else:
				print(f"\n Epoch ({epoch+1}/{n_epochs}),Cost Time = {epct:.2f}s\n"+\
					f" Train Cost ={hist['train_cost'][-1]:.3f}, Val Cost ={hist['val_cost'][-1]:.3f}\n"+\
					f" Train Loss ={hist['train_loss'][-1]:.3f}, Val Loss ={hist['val_loss'][-1]:.3f}\n"+\
					f" Train Acc ={hist['train_acc'][-1]:.3f}, Val Acc ={hist['val_acc'][-1]:.3f}\n"+\
					f" The LR = {fix_eta:.4e}")

		et 	=  time.time()
		self.cost_time = et - st 
		print(f"INFO: Training End, Cost Time = {self.cost_time:.2f}")
		self.hist = hist

		return self.hist
	


	def init_WB(self):
		"""
		Initialising The W&B, we use normal distribution as an initialisation strategy 
		Args:
			K	:	integer of the size of feature size
			d 	:	integer of the size of label size
			m 	:	A list of integer of the size of Hidden layer, here is fixed to 50
		Returns:
			W_dict : dictionary for all the Weight 
			
			b_dict : dictionary for all the bias

		"""
		mu = 0
		self.W_dict= {}
		self.b_dict= {}
		m = self.m
		K = self.K
		d = self.d
		num_hidden = len(m)
		
		for i,h_size in enumerate(m):
			keyW = f"W{i+1}"
			keyb = f"b{i+1}"
			if i==0:
				self.W_dict[keyW] = np.random.normal(loc=mu,scale=1e-3,size=(h_size,self.d)).astype(np.float64)
				self.b_dict[keyb] = np.zeros(shape=(h_size,1)).astype(np.float64)
			else:	
				self.W_dict[keyW] = np.random.normal(loc=mu,scale=1e-3,size=(h_size,m[i-1])).astype(np.float64)
				self.b_dict[keyb] = np.zeros(shape=(h_size,1)).astype(np.float64)

			print(f"INFO:W&B init: {keyW}={self.W_dict[keyW].shape}, {keyb}={self.b_dict[keyb].shape}")

		#Layer 2 
		keyW = f"W{num_hidden+1}"
		keyb = f"b{num_hidden+1}"
		lend = m[-1]
		self.W_dict[keyW] = np.random.normal(loc=mu,scale=1e-3,size=(self.K,lend)).astype(np.float64)
		self.b_dict[keyb] = np.zeros(shape=(self.K,1)).astype(np.float64)
		
		print(f"INFO: Last W&B init: {keyW}={self.W_dict[keyW].shape}, {keyb}={self.b_dict[keyb].shape}")
	



#----------------------
#	Training 
#-----------------------
#----------------------------------------------

class BachNormlizer:

	def __init__(self):
		self.eps = np.finfo(float).eps
		
		# Trainable param
		self.gamma = 1.0
		self.beta = 0.0

		# Mean and Std
		self.mu = 0.0
		self.var = 0.0
		
		# Moving average
		self.alpha = None

	def forward(self,x):
		"""
		Forward prop for BN
		"""
		self.mu = np.mean(x,0)
		self.var = np.std(x,0)

		if self.alpha==None: self.alpha = self.mu
		
		x = (x-self.mu)/(self.var+self.eps)

		return 

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



#----------------------
#	Forward Prop  Utils
#-----------------------
#----------------------------------------------
def ReLU(x):
	"""
	Activation function 
	"""
	# If x>0 x = x; If x<0 x = 0 
	x[x<0] = 0 
	return x


def EvaluateClassifier(x,W_dict,b_dict,num_layer):
	"""
	Forward Prop of the model 
	Args:
		X: [d,n] inputs 
		W: [K,d] Weight 
		b: [K,1] bias
	Returns:
		P: [K,n] The outputs as one-hot classification
	"""
	layerout = {}
	layerout[f'h0'] = x
	for il in range(num_layer+1):
		hkey = f'h{il+1}'
		Wkey = f"W{il+1}"
		bkey = f"b{il+1}"
		if il == 0:
			layerout[hkey] = W_dict[Wkey] @ x + b_dict[bkey]
		else:
			hkey_ = f'h{il}'
			layerout[hkey] = W_dict[Wkey] @ ReLU(layerout[hkey_]) + b_dict[bkey]
		

	return softmax(layerout[hkey]), layerout
	




def ComputeGradsNum(X, Y, model, h=1e-5):
	""" 
	Converted from matlab code 
	
	"""
	from copy import deepcopy
	grad_W = {}
	grad_b = {}
	c1 = model.cost_func(X,Y)

	
	for il in reversed(range(1,model.num_layer + 2)):

		grad_W[f'W{il}'] = np.zeros_like(model.W_dict[f'W{il}']).astype(np.float64)
		grad_b[f'b{il}'] = np.zeros_like(model.b_dict[f'b{il}']).astype(np.float64)
		
		xW, yW = grad_W[f"W{il}"].shape
		xb     = len(grad_b[f"b{il}"])

		for i in range(xW):
			for j in range(yW):
				W_dict = deepcopy(model.W_dict)
				W_try = np.array(W_dict[f"W{il}"]).astype(np.float64)
				W_try[i,j] += h
				W_dict[f'W{il}'] = W_try
				c2 = ComputeCost(X,Y,W_dict,model.b_dict,model.num_layer)
				grad_W[f"W{il}"][i,j] = (c2-c1) / (h)

		
		for i in range(xb):
			b_dict = deepcopy(model.b_dict)
			b_try = np.array(b_dict[f'b{il}']).astype(np.float64)
			b_try[i] += h
			b_dict[f'b{il}'] = b_try
			c2 = ComputeCost(X,Y,model.W_dict,b_dict,model.num_layer)
			grad_b[f"b{il}"][i] = (c2-c1) / (h)

		keyW = f"W{il}"
		keyb = f"b{il}"
		print(f"INFO: Compute {il} Layer, " + \
			 f"gradient Shape = {grad_W[keyW].shape}, {grad_b[keyb].shape}"
			 )
	
	return grad_W, grad_b

def ComputeGradsNumSlow(X, Y, model, h=1e-5):
	""" 
	Converted from matlab code 
	
	"""
	from copy import deepcopy
	grad_W = {}
	grad_b = {}
	c1 = model.cost_func(X,Y)

	
	for il in reversed(range(1,model.num_layer + 2)):

		grad_W[f'W{il}'] = np.zeros_like(model.W_dict[f'W{il}']).astype(np.float64)
		grad_b[f'b{il}'] = np.zeros_like(model.b_dict[f'b{il}']).astype(np.float64)
		
		xW, yW = grad_W[f"W{il}"].shape
		xb     = len(grad_b[f"b{il}"])

		for i in range(xW):
			for j in range(yW):
				W_dict = deepcopy(model.W_dict)
				W_try = np.array(W_dict[f"W{il}"]).astype(np.float64)
				W_try[i,j] -= h
				W_dict[f'W{il}'] = W_try
				c1 = ComputeCost(X,Y,W_dict,model.b_dict,model.num_layer)
				
				W_dict = deepcopy(model.W_dict)
				W_try = np.array(W_dict[f"W{il}"]).astype(np.float64)
				W_try[i,j] += h
				W_dict[f'W{il}'] = W_try
				c2 = ComputeCost(X,Y,W_dict,model.b_dict,model.num_layer)
				
				grad_W[f"W{il}"][i,j] = (c2-c1) / (2*h)

		
		for i in range(xb):

			b_dict = deepcopy(model.b_dict)
			b_try = np.array(b_dict[f'b{il}']).astype(np.float64)
			b_try[i,0] -= h
			b_dict[f'b{il}'] = b_try
			c1 = ComputeCost(X,Y,model.W_dict,b_dict,model.num_layer)
			
			b_dict = deepcopy(model.b_dict)
			b_try = np.array(b_dict[f'b{il}']).astype(np.float64)
			b_try[i,0] += h
			b_dict[f'b{il}'] = b_try
			c2 = ComputeCost(X,Y,model.W_dict,b_dict,model.num_layer)
			
			grad_b[f"b{il}"][i,0] = (c2-c1) / (2*h)

		print(f"INFO: Compute {il} Layer")
	
	return grad_W, grad_b
#----------------------
#	Back Prop  
#-----------------------
#----------------------------------------------




def Prop_Error(ga,gn,h):
	"""
	Compute the propagation Error with uncertainty
	"""

	eps_m = np.ones_like(ga).astype(np.float64) * h
	n,m = ga.shape
	summ  = np.abs(ga).astype(np.float64)+np.abs(gn).astype(np.float64)

	return np.abs(ga-gn)/np.maximum(eps_m,summ)
	# return np.abs(ga-gn)/np.maximum(np.abs(ga)+1e-24, np.abs(gn)+1e-24)



def ComputeCost(X,Y,W_dict,b_dict,num_layer):
	"""
	Compute the cost function: c = loss + regularisation 

	Args: 
		X	: [d,n] input 
		Y	: [K,n] One-Hot Ground Truth 
		W 	: [K,d] Weight 
		b	: [d,1] bias 
		lamb: (float) a regularisation term for the weight
	
	Return:
		J	: (float) A scalar of the cost function 
	"""
	
	# Part 1: compute the loss:
	## 1 Compute prediction:
	P, _ = EvaluateClassifier(X,W_dict,b_dict,num_layer)
	## Cross-entropy loss
	# Clip the value to avoid ZERO in log
	P 		= np.clip(P,1e-16,1-1e-16)
	l_cross =  -np.mean(np.sum(Y*np.log(P),axis=0))
	# Part 2: Compute the regularisation 

	return l_cross

#----------------------------------------------



#----------------------
#	Post-Processing 
#-----------------------
#----------------------------------------------
def plot_loss(interval, loss,fig=None,axs=None,color=None,ls=None):
	if fig==None:
		fig, axs = plt.subplots(1,1,figsize=(6,4))
	
	if color == None: color = "r"
	if ls == None: ls = '-'
	axs.plot(interval, loss,ls,lw=2.5,c=color)
	axs.set_xlabel('Epochs')
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

#----------------------
#	Main Programm
#-----------------------
#----------------------------------------------
def ExamCode():
	"""Test the functions for the assignments """

	print("*"*30)
	print("\t Exericise 1-2")
	print("*"*30)
	print("#"*30)
	labels = ['airplane','automobile','bird',
			'cat','deer','dog','frog',
			"horse",'ship','truck']
	
	print(f"Testing Functions:")

	# Step 1: Load data
	X, Yenc, X_val,Y_val,X_test,Y_test = dataLoader_OneBatch()
	# Define the feature size and label size
	K = 10; d = 3072
	# One-Hot encoded for Y 
	print(f"Global K={K}, d={d}")

	

	#Step 4: Initialisation of the network
	#---------------------------------------------
	#Use the class for model implementation 
	model = mlp(K,d,h_size=[50],lamda=0.0)
	
	# Step 4: Test for forward prop
	batch_size  = 1
	X_batch  	= X[:,:batch_size]
	Y_batch  	= Yenc[:,:batch_size]
	
	# P 		= EvaluateClassifier(X_batch,W,b)
	P 		= model.forward(X_batch)
	print(f"INFO: Test Pred={P.shape}")
	#---------------------------------------------



	# Step 5: Cost Function
	J,l_cross = model.cost_func(X_batch,Y_batch,return_loss=True)
	print(f"INFO: The loss = {J}")


	# Step 6: Examine the acc func:
	acc = model.compute_acc(X,Yenc)
	print(f"INFO:Accuracy Score={acc*100}%") 

	model.computeGradient(X_batch,Y_batch)
	del model 

	# Step 7 Compute the Gradient and compare to analytical solution 
	compute_grad = True
	if_implict = False; if_central = True
	if compute_grad: 
		batch_size  = 1
		trunc 		= 10 # Truncate the size of the imput data
		X_trunc  	= X[:trunc,:1]
		Y_trunc  	= Yenc[:,:1]
		model 		= mlp(k_=K,	d_=trunc, h_size=[50,50],lamda=0.0)	
		h 			= 1e-5 # Given in assignment
		model.computeGradient(X_trunc,Y_trunc)
		grad_error = {}
		if if_implict:
			print("\n----Implict Method----")
			grad_W1_n, grad_b1_n = ComputeGradsNum(X_trunc,
													Y_trunc,
													model,
													h=h)
			
			for il in reversed(range(1,model.num_layer+2)):
				ew = Prop_Error(model.W_grad[f'W{il}'],grad_W1_n[f'W{il}'],h)
				eb = Prop_Error(model.b_grad[f'b{il}'],grad_b1_n[f'b{il}'],h)
				print(f"\nComparison: Prop Error for W{il}:{ew.mean():.3e}")
				print(f"Comparison: Prop Error for B{il}:{eb.mean():.3e}")
				grad_error[f"forward_b{il}"] = eb.mean().reshape(-1,)
				grad_error[f"forward_w{il}"] = ew.mean().reshape(-1,)

				# aw = np.abs(model.W_grad[f'W{il}']-grad_W1_n[f'W{il}'])
				# ab = np.abs(model.b_grad[f'b{il}']-grad_b1_n[f'b{il}'])
				# print(f"Comparison: ABS Error for W{il}:{aw.mean():.3e}")
				# print(f"Comparison: ABS Error for B{il}:{ab.mean():.3e}")

		if if_central:
			print("\n ----Central Method----")
			grad_W1_n, grad_b1_n = ComputeGradsNumSlow( X_trunc,
														Y_trunc,
														model,
														h=h)

			for il in reversed(range(1,model.num_layer+2)):
				ew = Prop_Error(model.W_grad[f'W{il}'],grad_W1_n[f'W{il}'],h)
				eb = Prop_Error(model.b_grad[f'b{il}'],grad_b1_n[f'b{il}'],h)
				print(f"\nComparison: Prop Error for W{il}:{ew.mean():.3e}")
				print(f"Comparison: Prop Error for B{il}:{eb.mean():.3e}")
				grad_error[f"central_b{il}"] = eb.mean().reshape(-1,)
				grad_error[f"central_w{il}"] = ew.mean().reshape(-1,)
				
				# aw = np.abs(model.W_grad[f'W{il}']-grad_W1_n[f'W{il}'])
				# ab = np.abs(model.b_grad[f'b{il}']-grad_b1_n[f'b{il}'])
				# print(f"\nComparison: ABS Error for W{il}:{aw.mean():.3e}")
				# print(f"Comparison: ABS Error for B{il}:{ab.mean():.3e}")

		df = pd.DataFrame(grad_error)
		df.to_csv("Gradient_compute.csv",float_format="%.3e")
	
def train_E11():
	"""
	Exericise 1.1: Use 3-layer MLP for training, NO BN at this stage 
	"""
	print(f"*"*30)
	print(f"\t Exercise 1.1")
	print(f"*"*30)
	lamda = 0.005

	X, Yenc, X_val,Yenc_val,X_test,Y_test = dataLoader_FullBatch(split_ratio=0.1)
	K, N_sample = Yenc.shape; d = X.shape[0]
	
	n_batch = 100
	n_cycle = 2
	n_s		= int(5*45000/n_batch)
	n_epoch = int(n_cycle * n_s * 2 / n_batch)
	
	print(f"General INFO: n_s={n_s}, n_cycle = {n_cycle}, n_epoch={n_epoch}")

	lr_dict = {"n_s":n_s,"eta_min":1e-5,"eta_max":1e-1} 
	train_dict = {'n_batch':n_batch,'n_epochs':n_epoch}
	case_name = name_case(**lr_dict,**train_dict,lamda=lamda)

	model  = mlp(k_=10,d_=3072,h_size=[50,50],lamda=lamda)
	lr_sch = lr_scheduler(**lr_dict)

	print("\n"+"#"*30)
	print(f"Start:\t{case_name}")
	print(f"#"*30)

	hist = model.train( X,Yenc,X_val,Yenc_val,
							lr_sch,**train_dict)
	
	acc = model.compute_acc(X_test,Y_test)
	
	print(f"TEST Acc ={acc*100:.2f}%")
		
	save_as_mat(model,hist,acc,"weights/" + case_name)
	print(f"W&B Saved!")
		
	fig, axs = plot_hist(hist,0,1)
	for ax in axs:
		ax.set_xlabel('Update Step')
	fig.savefig(f'Figs/Loss_{case_name}.jpg',**fig_dict)

##########################################
## Run the programme DOWN Here:
##########################################
if __name__ == "__main__":

	if (args.m == 1):
		ExamCode()
	elif (args.m == 2):
		train_E11()
	# elif (args.m == 3):
		# train_E3()
		# post_E4()
	# elif (args.m == 4):
		
	else:
		raise ValueError