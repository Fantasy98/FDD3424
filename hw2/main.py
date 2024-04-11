"""
Assignment 2

Training for a binary Classifier for 2 Layer 

@yuningw
Apr 6th, 2024 
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
parser.add_argument('-m',default=4,type=int,help='Choose which exercise to do 1,2,3,4,5')
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
global K, d, label

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
	sio.savemat(name + '.mat',
			{
				"W1":model.W1,
				"W2":model.W2,
				"b1":model.b1,
				"b2":model.b2,
				'train_loss':np.array(hist['train_loss']),
				'train_cost':np.array(hist['train_cost']),
				'train_acc':np.array(hist['train_acc']),
				'val_loss':np.array(hist['val_loss']),
				'val_cost':np.array(hist['val_loss']),
				'val_acc':np.array(hist['val_acc']),
				"test_acc":acc
				})
	
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
	X 		= np.array(dt[b'data']).astype(np.float32).T
	y 		= np.array(dt[b'labels']).astype(np.float32).flatten()
	print(f"TRAIN X: {X.shape}")
	print(f"TRAIN Y:{y.shape}, here are {len(np.unique(y))} Labels")
	
	dt = readPickle("data_batch_2")
	X_val 		= np.array(dt[b'data']).astype(np.float32).T
	y_val 		= np.array(dt[b'labels']).astype(np.float32).flatten()
	print(f"Val X: {X.shape}")
	print(f"Val Y:{y.shape}, here are {len(np.unique(y))} Labels")

	del dt 
	return X, y, X_val,y_val

def LoadAll():

	X = []; y = []
	for n in range(5):
		dt = readPickle(f"data_batch_{n+1}")
		X_ = np.array(dt[b'data']).astype(np.float32).T
		y_ = np.array(dt[b'labels']).astype(np.float32).flatten()
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
	X 		= np.array(dt[b'data']).astype(np.float32).T
	y 		= np.array(dt[b'labels']).astype(np.float32).flatten()
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
	mean_x = np.repeat(np.mean(X,axis=1).reshape(-1,1),X.shape[-1],1).astype(np.float32)
	std_x  = np.repeat(np.std(X,axis=1).reshape(-1,1),X.shape[-1],1).astype(np.float32)
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

	y_hat = np.zeros(shape=(K, len(y))).astype(np.float32)
	
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
	def __init__(self,K,d,m=50,
					lamda=0,
				):
		self.K  = K 
		self.d  = d 
		self.m  = m 
		self.init_WB(K,d,m)
		self.lamda = lamda
		print(f"INFO: Model initialised: LAMBDA = {self.lamda:3e} K={self.K}, d={self.d}, m={self.m}")

	def forward(self,x,return_hidden=False):
		"""
		Forward Propagation 
		"""
		hidden_output= self.W1 @ x +self.b1 # ReLU activation

		scores = self.W2 @ ReLU(hidden_output) +self.b2

		
		if return_hidden:
			return softmax(scores),hidden_output
		else:
			return softmax(scores) 

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
		l_cross =  -np.mean(np.sum(Y*np.log(P),axis=0))
		# Part 2: Compute the regularisation 
		reg = self.lamda * (np.sum(self.W1**2) + np.sum(self.W2**2)  )
		# Assemble the components
		J = l_cross + reg
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
		
		del P, Y

		return acc

	def computeGradient(self,x,y):
		"""
		Compute the Gradient w.r.t the W&B 

		Args:

			x	: [d,n] input 
			y	: [1,n] Ground Truth 
		
		Returns:

			grad_W2 : [K,m]
			grad_b2	: [K,1]
			
			grad_W1 : [m,d]
			grad_b2 : [d,1]

		"""

		# compute the Prediction 
		p,hidden_output = self.forward(x,return_hidden=True)
		
		# Gradient for output layer
		g 	    = -(y - p).T
		gW2 	= g.T @ ReLU(hidden_output).T 
		gb2 	= np.sum(g,axis=0,keepdims=True).T
		
		# Gradient for hidden layer 
		g=g @ self.W2
		g[hidden_output.T <=0] = 0
		gW1  = g.T @ x.T 
		gb1  = np.sum(g, axis=0,keepdims=True).T

		return gW2, gb2, gW1, gb1

	def backward(self,x,y,eta_,batch_size):
		"""Back Prop for Update the W&B"""
		grad_W2, grad_b2, grad_W1, grad_b1 = self.computeGradient(x,y)
		
		self.W1 -= eta_*( (1/batch_size)*(grad_W1) + 2*self.lamda*self.W1 )	
		self.b1 -= eta_*( (1/batch_size)*(grad_b1)						 )

		self.W2 -= eta_*( (1/batch_size)*(grad_W2) + 2*self.lamda*self.W2 )
		self.b2 -= eta_*( (1/batch_size)*(grad_b2)						 )

	
	def train(self,X,Y,X_val,Y_val,lr_sch,n_epochs,n_batch,fix_eta = 1e-3):
		
		print(f'Start Training, Batch Size = {n_batch}')
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
			X_ = X[:,indices]
			Y_ = Y[:,indices]
			for b in (train_batch_range):
				
				eta_ = (lr_sch.eta if lr_sch != None else fix_eta)

				X_batch = X_[:,b*batch_size:(b+1)*batch_size]
				Y_batch = Y_[:,b*batch_size:(b+1)*batch_size]
				
				self.backward(  X_batch,
								Y_batch,
								eta_,
								batch_size)
				
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

				if lr_sch !=None:
					lr_sch.update_lr()

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
	


	def init_WB(self,K:int,d:int,m=50):
		"""
		Initialising The W&B, we use normal distribution as an initialisation strategy 
		Args:
			K	:	integer of the size of feature size
			d 	:	integer of the size of label size
			m 	:	integer of the size of Hidden layer, here is fixed to 50
		Returns:
			W1	:	[m,d] Numpy Array as a matrix of W1 
			b1	:	[m,1] Numpy Array as a vector of b1
			W2	:	[K,m] Numpy Array as a matrix of W2 
			b2	:	[K,1] Numpy Array as a vector of b2
		"""
		mu = 0; sigma1 = 1/np.sqrt(d); sigma2 = 1/np.sqrt(m)
		#Layer 1 
		self.W1 = np.random.normal(loc=mu,scale=sigma1,size=(m,d)).astype(np.float32)
		self.b1 = np.zeros(shape=(m,1)).astype(np.float32)
		#Layer 2 
		self.W2 = np.random.normal(loc=mu,scale=sigma2,size=(K,m)).astype(np.float32)
		self.b2 = np.zeros(shape=(K,1)).astype(np.float32)
		
		print(f"INFO:W&B init: W1={self.W1.shape}, b2={self.b1.shape}")
		print(f"INFO:W&B init: W2={self.W2.shape}, b2={self.b2.shape}")
	



#----------------------
#	Training 
#-----------------------
#----------------------------------------------

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
	x = np.maximum(0,x)
	return x


def EvaluateClassifier(x,W1,b1,W2,b2,return_hidden=False):
	"""
	Forward Prop of the model 
	Args:
		X: [d,n] inputs 
		W: [K,d] Weight 
		b: [K,1] bias
	Returns:
		P: [K,n] The outputs as one-hot classification
	"""
	hidden_output= W1 @ x +b1# ReLU activation

	scores = W2 @ ReLU(hidden_output) + b2

	if return_hidden:
		return softmax(scores), hidden_output 
	else:
		return softmax(scores) 



def computeGrad_Explict(x,y,W1,b1,W2,b2):
		# compute the Prediction 
	p,hidden_output = EvaluateClassifier(x,
								W1,b1,W2,b2,
								return_hidden=True)
	
	# Gradient for output layer
	g 	    = -(y - p).T
	gW2 	= g.T @ ReLU(hidden_output).T 
	gb2 	= np.sum(g,axis=0,keepdims=True).reshape(-1,1)

	# Gradient for hidden layer 
	g	 	=	g @ W2
	g[hidden_output.T<=0]=0
	gW1  	= g.T @ x.T 
	gb1  	= np.sum(g, axis=0,keepdims=True).reshape(-1,1)

	return gW2, gb2, gW1, gb1


def ComputeGradsNum(X, Y, W1, b1, W2, b2, h):
	""" 
	Converted from matlab code 
	
	"""

	grad_W1 = np.zeros_like(W1);
	grad_b1 = np.zeros_like(b1);
	grad_W2 = np.zeros_like(W2);
	grad_b2 = np.zeros_like(b2);

	c1 = ComputeCost(X,Y,W1,b1,W2,b2,0)
			
	for i in range(W2.shape[0]):
		for j in range(W2.shape[1]):
			W_try = np.array(W2)
			W_try[i,j] += h
			c2 = ComputeCost(X,Y,W1,b1,W_try,b2,0)
			grad_W2[i,j] = (c2-c1) /h
	print("INFO: Compute W2 Grad")

	
	for i in range(len(b2)):
		b_try = np.array(b2)
		b_try[i] += h
		c2 = ComputeCost(X,Y,W1,b1,W2,b_try,0)
		grad_b2[i] = (c2-c1) / (h)
	print("INFO: Compute b2 Grad")


	
	for i in range(W1.shape[0]):
		for j in range(W1.shape[1]):
			W_try = np.array(W1)
			W_try[i,j] += h
			c2 = ComputeCost(X,Y,W_try,b1,W2,b2,0)
			grad_W1[i,j] = (c2-c1) / (h)
	print("INFO: Compute W1 Grad")

		
	for i in range(len(b1)):
		b_try = np.array(b1)
		b_try[i] += h
		c2 = ComputeCost(X,Y,W1,b_try,W2,b2,0)
		grad_b1[i] = (c2-c1) / (h)
	print("INFO: Compute b1 Grad")
	

	
	return [grad_W1, grad_b1, grad_W2, grad_b2]

def ComputeGradsNumSlow(X, Y,W1, b1, W2, b2, h):
	""" 
	Converted from matlab code 
	
	"""

	grad_W1 = np.zeros_like(W1);
	grad_b1 = np.zeros_like(b1);
	grad_W2 = np.zeros_like(W2);
	grad_b2 = np.zeros_like(b2);

	for i in range(W2.shape[0]):
		for j in range(W2.shape[1]):
			W_try = np.array(W2)
			W_try[i,j] -= h
			c1 = ComputeCost(X,Y,W1,b1,W_try,b2,0)
			W_try = np.array(W2)
			W_try[i,j] += h
			c2 = ComputeCost(X,Y,W1,b1,W_try,b2,0)
			grad_W2[i,j] = (c2-c1) / (2*h)
	print("INFO: Compute W2 Grad")

	
	for i in range(len(b2)):
		b_try = np.array(b2)
		b_try[i] -= h
		c1 = ComputeCost(X,Y,W1,b1,W2,b_try,0)
		b_try = np.array(b2)
		b_try[i] += h
		c2 = ComputeCost(X,Y,W1,b1,W2,b_try,0)
		grad_b2[i] = (c2-c1) / (2*h)
	print("INFO: Compute b2 Grad")


	
	for i in range(W1.shape[0]):
		for j in range(W1.shape[1]):
			W_try = np.array(W1)
			W_try[i,j] -= h
			c1 = ComputeCost(X,Y,W_try,b1,W2,b2,0)
			W_try = np.array(W1)
			W_try[i,j] += h
			c2 = ComputeCost(X,Y,W_try,b1,W2,b2,0)
			grad_W1[i,j] = (c2-c1) / (2*h)
	print("INFO: Compute W1 Grad")

		
	for i in range(len(b1)):
		b_try = np.array(b1)
		b_try[i] -= h
		c1 = ComputeCost(X,Y,W1,b_try,W2,b2,0)
		b_try = np.array(b1)
		b_try[i] += h
		c2 = ComputeCost(X,Y,W1,b_try,W2,b2,0)
		grad_b1[i] = (c2-c1) / (2*h)
	print("INFO: Compute b1 Grad")
	
	return [grad_W1, grad_b1, grad_W2, grad_b2]
#----------------------
#	Back Prop  
#-----------------------
#----------------------------------------------




def Prop_Error(ga,gn,eps):
	"""
	Compute the propagation Error with uncertainty
	"""

	eps_m = np.ones_like(ga).astype(np.float64) * eps
	n,m = ga.shape
	summ  = np.abs(ga)+np.abs(gn)

	return np.abs(ga-gn)/np.maximum(eps_m,summ)



def ComputeCost(X,Y,W1,b1,W2,b2,lamda,return_loss = False):
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
	P = EvaluateClassifier(X,W1,b1,W2,b2)
	## Cross-entropy loss
	# Clip the value to avoid ZERO in log
	P 		= np.clip(P,1e-16,1-1e-16)
	l_cross =  -np.mean(np.sum(Y*np.log(P),axis=0))
	# Part 2: Compute the regularisation 
	reg = lamda * (np.sum(W1**2) + np.sum(W2**2))
	# Assemble the components
	J = l_cross + reg
	if return_loss:
		return J, l_cross
	else: 
		return J 
	
def ComputeAccuracy(X,Y,P):
	"""
	Compute the accuracy of the classification 
	
	Args:

		X	: [d,n] input 
		Y	: [1,n] Ground Truth 
		P	: [K,n] Prediction 

	Returns: 

		acc : (float) a scalar value containing accuracy 
	"""

	
	#Compute the maximum prob 
	# [K,n] -> K[1,n]
	P = np.argmax(P,axis=0)
	
	# Compute how many true-positive samples
	true_pos = np.sum(P == Y)
	# Percentage on total 
	acc =  true_pos / Y.shape[-1]
	return acc
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

def plot_hist(hist,n_start,n_interval):
	fig , axs  = plt.subplots(1,3,figsize=(24,6))
	
	n_range = np.arange(len(hist['train_cost']))

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
	X, Y,X_val,Y_val = LoadBatch()
	# Define the feature size and label size
	K = len(np.unique(Y)); d = X.shape[0]
	# One-Hot encoded for Y 
	Yenc 	 = one_hot_encode(Y,K)
	Y_val 	 = one_hot_encode(Y_val,K)
	print(f"Global K={K}, d={d}")


	# Step 2: Scaling the data
	X, muX, stdX 		= normal_scaling(X)
	X_val				= ( X - muX )/stdX


	# Step 3*: Test the cyclinal_lr
	#---------------------------------------------
	lr_dict = {"n_s":500,"eta_min":1e-5,"eta_max":1e-1} 
	etas = []
	lr_sch = lr_scheduler(**lr_dict)
	n_epoch = 10
	for l in range(n_epoch):
		for b in range(100):
			lr_sch.update_lr()
			
	fig, axs = plt.subplots(1,1,figsize=(6,4))
	print(f"Maximum eta= {max(lr_sch.hist)}, Minimum eta = {min(lr_sch.hist)}")
	axs.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
	axs.plot(range(len(lr_sch.hist)),lr_sch.hist,'o',markersize=1.5,c=colorplate.black,lw=2)
	axs.set_yticks([1e-5,1e-1])
	axs.set_xlabel('Update Step',font_dict)
	axs.set_ylabel(r'$\eta_t$',font_dict)
	fig.savefig('Figs/LR_schedule_1.jpg',bbox_inches='tight',dpi=300)

	lr_dict = {"n_s":800,"eta_min":1e-5,"eta_max":1e-1} 
	etas = []
	lr_sch = lr_scheduler(**lr_dict)
	n_epoch = 48
	for l in range(n_epoch):
		for b in range(100):
			lr_sch.update_lr()
	
	fig, axs = plt.subplots(1,1,figsize=(6,4))
	print(f"Maximum eta= {max(lr_sch.hist)}, Minimum eta = {min(lr_sch.hist)}")
	axs.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
	
	axs.plot(range(len(lr_sch.hist)),lr_sch.hist,'o',markersize=1.5,c=colorplate.black,lw=2)
	axs.set_yticks([1e-5,1e-1])
	axs.set_xlabel('Update Step',font_dict)
	axs.set_ylabel(r'$\eta_t$',font_dict)
	fig.savefig('Figs/LR_schedule_2_log.jpg',bbox_inches='tight',dpi=300)
	#---------------------------------------------

	#Step 4: Initialisation of the network
	#---------------------------------------------
	#Use the class for model implementation 
	model = mlp(K,d,lamda=0.0)
	
	# Step 4: Test for forward prop
	batch_size  = 1
	X_test  	= X[:,:batch_size]
	Y_test  	= Yenc[:,:batch_size]
	
	# P 		= EvaluateClassifier(X_test,W,b)
	P 		= model.forward(X_test)
	print(f"INFO: Test Pred={P.shape}")
	#---------------------------------------------


	# Step 5: Cost Function
	J,l_cross = model.cost_func(X_test,Y_test,return_loss=True)
	print(f"INFO: The loss = {J}")


	# Step 6: Examine the acc func:
	acc = model.compute_acc(X,Y)
	print(f"INFO:Accuracy Score={acc*100}%") 


	# Step 7 Compute the Gradient and compare to analytical solution 
	compute_grad = True
	if compute_grad: 

		batch_size  = 1
		trunc 		= 20
		X_test  	= X[:trunc,:1]
		Y_test  	= Yenc[:,:1]
		
		h 			= 1e-5 # Given in assignment
		
		grad_W2, grad_b2, grad_W1, grad_b1  = computeGrad_Explict(X_test,Y_test,
																model.W1[:,:trunc],
																model.b1,
																model.W2[:,:],
																model.b2,)

		print(f"Compute Gradient: W2:{grad_W2.shape},W1:{grad_W1.shape},b1:{grad_b1.shape},b2:{grad_b2.shape}")

		grad_error = {}
		grad_W1_n, grad_b1_n,grad_W2_n, grad_b2_n = ComputeGradsNumSlow(X_test,
																		Y_test,
																		model.W1[:,:trunc],
																		model.b1,
																		model.W2[:,:],
																		model.b2,
																		h=h)
		print("Central Method")
		ew = Prop_Error(grad_W1,grad_W1_n,h)
		eb = Prop_Error(grad_b1,grad_b1_n,h)
		print(f"Comparison: Prop Error for W1:{ew.mean():.3e}")
		print(f"Comparison: Prop Error for B1:{eb.mean():.3e}")
		grad_error["central_w1"] = ew.mean().reshape(-1,)
		grad_error["central_b1"] = eb.mean().reshape(-1,)

		ew = Prop_Error(grad_W2,grad_W2_n,h)
		eb = Prop_Error(grad_b2,grad_b2_n,h)
		print(f"Comparison: Prop Error for W2:{ew.mean():.3e}")
		print(f"Comparison: Prop Error for B2:{eb.mean():.3e}")
		grad_error["central_w2"] = ew.mean().reshape(-1,)
		grad_error["central_b2"] = eb.mean().reshape(-1,)

		grad_W1_n, grad_b1_n,grad_W2_n, grad_b2_n = ComputeGradsNum(X_test,
																	Y_test,
																	model.W1[:,:trunc],
																	model.b1,
																	model.W2,
																	model.b2,
																	h=h)
		

		print("Implict Method")
		ew = Prop_Error(grad_W1,grad_W1_n,h)
		eb = Prop_Error(grad_b1,grad_b1_n,h)
		print(f"Comparison: Prop Error for W1:{ew.mean():.3e}")
		print(f"Comparison: Prop Error for B1:{eb.mean():.3e}")
		grad_error["forward_b1"] = eb.mean().reshape(-1,)
		grad_error["forward_w1"] = ew.mean().reshape(-1,)

		ew = Prop_Error(grad_W2,grad_W2_n,h)
		eb = Prop_Error(grad_b2,grad_b2_n,h)
		print(f"Comparison: Prop Error for W2:{ew.mean():.3e}")
		print(f"Comparison: Prop Error for B2:{eb.mean():.3e}")
		grad_error["forward_w2"] = ew.mean().reshape(-1,)
		grad_error["forward_b2"] = eb.mean().reshape(-1,)

		df = pd.DataFrame(grad_error)
		df.to_csv("Gradient_compute.csv",float_format="%.3e")

	# Step 8 Train a very small case
	model  = mlp(K,d)
	hist  = model.train(X[:,:100],Yenc[:,:100],X_val,Y_val, lr_sch=None,
					 n_epochs=200,n_batch=10)

	fig, axs = plot_hist(hist,0,1)
	for ax in axs:
		ax.set_xlabel('Update Steps',font_dict)
	fig.savefig('Figs/validate_mini_batch.jpg',**fig_dict)

def train_E3():
	"""
	Train For Exercise 3 
	"""
	print(f"*"*30)
	print(f"\t Exercise 3")
	print(f"*"*30)
	X, Yenc, X_val,Yenc_val,X_test,Y_test = dataLoader_OneBatch()
	K = len(np.unique(Y_test)); d = X.shape[0]

	lr_dict = {"n_s":500,"eta_min":1e-5,"eta_max":1e-1} 
	train_dict = {'n_batch':100,'n_epochs':10}

	lamda  = 0.01
	filename = name_case(**lr_dict, **train_dict,lamda=lamda)
	print(f"INFO: Start CASE: {filename}")
	lr_sch  = lr_scheduler(**lr_dict)
	model 	= mlp(K,d,m=50,lamda=lamda)

	hist = model.train( X,Yenc,
						X_val,Yenc_val,
						lr_sch,**train_dict)
	
	acc = model.compute_acc(X_test,Y_test)
	print(f"Acc ={acc*100}")
	print("#"*30)
	save_as_mat(model,hist,acc,"weights/" + filename)
	print(f"W&B Saved!")
	fig, axs = plot_hist(hist,10,100)
	for ax in axs:
		ax.set_xlabel('Update Step')
	fig.savefig(f'Figs/Loss_{filename}.jpg',**fig_dict)


def train_E4():
	"""
	Training For Longer Epoch with Larger cycle
	"""
	print(f"*"*30)
	print(f"\t Exercise 4")
	print(f"*"*30)
	
	X, Yenc, X_val,Yenc_val,X_test,Y_test = dataLoader_OneBatch()

	K = Yenc.shape[0]; d = X.shape[0]
	# Step 3 Parameter Setting 
	lr_dict = {"n_s":800,"eta_min":1e-5,"eta_max":1e-1} 
	n_cycle = 3 
	n_batch = 100
	n_epoch = int(n_cycle * lr_dict['n_s'] * 2 / n_batch)

	train_dict = {'n_batch':100,'n_epochs':n_epoch}
	lamda  = 0.01
	filename = name_case(**lr_dict, **train_dict,lamda=lamda)
	print(f"INFO: Start CASE: {filename}")
	lr_sch  = lr_scheduler(**lr_dict)
	model 	= mlp(K,d,m=50,lamda=lamda)
	
	hist = model.train( X,Yenc,X_val,Yenc_val,
						lr_sch,**train_dict)
		

	acc = model.compute_acc(X_test,Y_test)
	print(f"Acc ={acc*100:.2f}%")
	print("#"*30)
	save_as_mat(model,hist,acc,"weights/" + filename)
	print(f"W&B Saved!")
	fig, axs = plot_hist(hist,10,100)
	for ax in axs:
		ax.set_xlabel('Update Step')
	fig.savefig(f'Figs/Loss_{filename}.jpg',**fig_dict)
	
def post_E4():
	X, Yenc, X_val,Yenc_val,X_test,Y_test = dataLoader_OneBatch()
	K = len(np.unique(Y_test)); d = X.shape[0]
	lr_dict = {"n_s":800,"eta_min":1e-5,"eta_max":1e-1} 
	n_cycle = 3 
	n_batch = 100
	n_epoch = int(n_cycle * lr_dict['n_s'] * 2 / n_batch)

	train_dict = {'n_batch':100,'n_epochs':n_epoch}
	lamda  = 0.01
	filename = name_case(**lr_dict, **train_dict,lamda=lamda)
	hist = {}
	database = sio.loadmat('weights/' + filename + '.mat')
	hist['train_cost'] = database['train_cost'].flatten()
	hist['val_cost'] = database['val_cost'].flatten()
	hist['train_loss'] = database['train_loss'].flatten()
	hist['val_loss'] = database['val_loss'].flatten()
	hist['train_acc'] = database['train_acc'].flatten()
	hist['val_acc'] = database['val_acc'].flatten()

	model =  mlp(K,d)
	model.W1=database['W1'];model.W2=database['W2']
	model.b1=database['b1'];model.b2=database['b2']
	acc = model.compute_acc(X_test,Y_test)
	print(f"Acc = {acc*100:.3f}")

	fig, axs = plot_hist(hist,10,n_batch)
	for ax in axs:
		ax.set_xlabel('Update Step')
	fig.savefig(f'Figs/Loss_{filename}.jpg',**fig_dict)


def train_E5():
	"""
	Training For Coarse Search the optimal Lambda 
	"""

	print(f"*"*30)
	print(f"\t Exercise 5")
	print(f"*"*30)
	
	X, Yenc, X_val,Yenc_val,X_test,Y_test = dataLoader_FullBatch()
	K, N_sample = Yenc.shape; d = X.shape[0]
	
	n_batch 	 = 300
	n_cycle 	 = 2
	l_min, l_max = -5,-1 
	n_search     = 8
	n_s 	     = 2 * (N_sample//n_batch)
	n_epoch      = int(n_cycle * n_s * 2 / n_batch)
	
	print(f"General INFO: n_s={n_s}, n_cycle = {n_cycle}, n_epoch={n_epoch}")

	lr_dict = {"n_s":n_s,"eta_min":1e-5,"eta_max":1e-1} 
	train_dict = {'n_batch':n_batch,'n_epochs':n_epoch}
	for l in (range(n_search)):
		l = l_min + (l_max - l_min)*np.random.rand()
		lamda = 10**l
		case_name = name_case(**lr_dict,**train_dict,lamda=lamda)
		
		if not os.path.exists('weights/'+case_name+'.mat'):
		
			print("\n"+"#"*30)
			print(f"({l+1}/{n_search}) Start:\t{case_name}")
			print(f"#"*30)
			
			model = mlp(K,d,m=50,lamda=lamda)
			lr_sch  = lr_scheduler(**lr_dict)
			hist = model.train( X,Yenc,X_val,Yenc_val,
								lr_sch,**train_dict)
			
			acc = model.compute_acc(X_test,Y_test)
			print(f"Acc ={acc*100:.2f}%")
			
			save_as_mat(model,hist,acc,"weights/" + case_name)
			print(f"W&B Saved!")
			
			fig, axs = plot_hist(hist,0,200)
			for ax in axs:
				ax.set_xlabel('Update Step')
			fig.savefig(f'Figs/Loss_{case_name}.jpg',**fig_dict)
		else:
			print('The W&B Exist!')	
		
		print("#"*30)

def post_E5():
	print(f"*"*30)
	print(f"\t Exercise 5")
	print(f"*"*30)
	
	X, Yenc, X_val,Yenc_val,X_test,Y_test = dataLoader_FullBatch()
	K, N_sample = Yenc.shape; d = X.shape[0]
	
	n_batch 		= 150
	n_cycle 		= 2
	l_min, l_max 	=-5,-1 
	n_search  		= 8
	n_s 	  		= 2 * (N_sample//n_batch)
	n_epoch 		= int(n_cycle * n_s * 2 / n_batch)
	
	print(f"General INFO: n_s={n_s}, n_cycle = {n_cycle}, n_epoch={n_epoch}")

	lr_dict = {"n_s":n_s,"eta_min":1e-5,"eta_max":1e-1} 
	train_dict = {'n_batch':n_batch,'n_epochs':n_epoch}

	key_words = f"W&B_{n_batch}BS_{n_epoch}Epoch_{n_s}NS"

	list_path = os.listdir('weights/')
	target_file = []
	for filename in list_path:
		if key_words in filename:
			target_file.append(filename)
	
	lmda     = []
	val_acc  =[]
	test_acc = []
	for filename in target_file:
		loc = filename.find('Lambda')
		lamda = float(filename[loc-9:loc])
		print(f"Load Case:{filename}, Lamda = {lamda}")
		dt 	= sio.loadmat('weights/'+filename)
		model = mlp(K,d,lamda=lamda)
		model.W1 = dt['W1']
		model.W2 = dt['W2']
		model.b1 = dt['b1']
		model.b2 = dt['b2']

		acc_val = model.compute_acc(X_val,Yenc_val)
		acc_tst = model.compute_acc(X_test,Y_test)
		lmda.append(lamda)
		val_acc.append(acc_val)
		test_acc.append(acc_tst)

	lmda = np.array(lmda)
	val_acc = np.array(val_acc)
	test_acc = np.array(test_acc)

	isort = np.argsort(val_acc)
	lmda  = lmda[isort]
	val_acc  = val_acc[isort]
	test_acc  = test_acc[isort]
	df = pd.DataFrame({
						"Lambda"  :lmda.flatten(),
						"val_acc" :val_acc.flatten(),
						"test_acc":test_acc.flatten(),
						
						})

	df.to_csv('Coarse_Search.csv',float_format='%.5e')


def train_E6():
	"""
	Training For Coarse Search the optimal Lambda 
	"""

	print(f"*"*30)
	print(f"\t Exercise 6")
	print(f"*"*30)
	
	X, Yenc, X_val,Yenc_val,X_test,Y_test = dataLoader_FullBatch()
	K, N_sample = Yenc.shape; d = X.shape[0]
	
	n_batch = 300
	n_cycle = 3
	l_min, l_max =-4,-3
	n_search  = 4
	n_s 	  = 2 * (N_sample//n_batch)
	n_epoch = int(n_cycle * n_s * 2 / n_batch)
	
	print(f"General INFO: n_s={n_s}, n_cycle = {n_cycle}, n_epoch={n_epoch}")

	lr_dict = {"n_s":n_s,"eta_min":1e-5,"eta_max":1e-1} 
	train_dict = {'n_batch':n_batch,'n_epochs':n_epoch}
	for l in (range(n_search)):
		l = l_min + (l_max - l_min)*np.random.rand()
		lamda = 10**l
		case_name = name_case(**lr_dict,**train_dict,lamda=lamda)
		
		if not os.path.exists('weights/'+case_name+'.mat'):
		
			print("\n"+"#"*30)
			print(f"Start:\t{case_name}")
			print(f"#"*30)
			
			model = mlp(K,d,m=50,lamda=lamda)
			lr_sch  = lr_scheduler(**lr_dict)
			hist = model.train( X,Yenc,X_val,Yenc_val,
								lr_sch,**train_dict)
			
			acc = model.compute_acc(X_test,Y_test)
			print(f"Acc ={acc*100:.2f}%")
			
			save_as_mat(model,hist,acc,"weights/" + case_name)
			print(f"W&B Saved!")
			
			fig, axs = plot_hist(hist,10,100)
			for ax in axs:
				ax.set_xlabel('Update Step')
			fig.savefig(f'Figs/Loss_{case_name}.jpg',**fig_dict)
		else:
			print('The W&B Exist!')	
		
		print("#"*30)

	return

def train_E7():
	"""
	Training For training the optimal Lambda 
	"""

	print(f"*"*30)
	print(f"\t Exercise 7")
	print(f"*"*30)

	lamda = 3e-4


	X, Yenc, X_val,Yenc_val,X_test,Y_test = dataLoader_FullBatch(split_ratio=0.02)
	K, N_sample = Yenc.shape; d = X.shape[0]
	
	n_batch = 150
	n_cycle = 3

	n_s 	  = 2 * (N_sample//n_batch)
	n_epoch   = int(n_cycle * n_s * 2 / n_batch)
	
	print(f"General INFO: n_s={n_s}, n_cycle = {n_cycle}, n_epoch={n_epoch}")

	lr_dict = {"n_s":n_s,"eta_min":1e-5,"eta_max":1e-1} 
	train_dict = {'n_batch':n_batch,'n_epochs':n_epoch}
	
	case_name = name_case(**lr_dict,**train_dict,lamda=lamda)
		
	if not os.path.exists('weights/'+case_name+'.mat'):
	
		print("\n"+"#"*30)
		print(f"Start:\t{case_name}")
		print(f"#"*30)
		
		model = mlp(K,d,m=50,lamda=lamda)
		lr_sch  = lr_scheduler(**lr_dict)
		hist = model.train( X,Yenc,X_val,Yenc_val,
							lr_sch,**train_dict)
		
		acc = model.compute_acc(X_test,Y_test)
		print(f"Acc ={acc*100:.2f}%")
		
		save_as_mat(model,hist,acc,"weights/" + case_name)
		print(f"W&B Saved!")
		
		fig, axs = plot_hist(hist,10,100)
		for ax in axs:
			ax.set_xlabel('Update Step')
		fig.savefig(f'Figs/Loss_{case_name}.jpg',**fig_dict)
	else:
		print('The W&B Exist!')	
	
	print("#"*30)

	return

def post_E7():
	print(f"*"*30)
	print(f"\t Exercise 7")
	print(f"*"*30)
	X, Yenc, X_val,Yenc_val,X_test,Y_test = dataLoader_FullBatch()
	K, N_sample = Yenc.shape; d = X.shape[0]
	
	n_batch = 150
	n_cycle = 3
	n_s 	  = 2 * (N_sample//n_batch)
	n_epoch   = int(n_cycle * n_s * 2 / n_batch)
	
	print(f"General INFO: n_s={n_s}, n_cycle = {n_cycle}, n_epoch={n_epoch}")

	lr_dict = {"n_s":n_s,"eta_min":1e-5,"eta_max":1e-1} 
	train_dict = {'n_batch':n_batch,'n_epochs':n_epoch}
	
	lamda = 3e-4

	case_name = name_case(**lr_dict,**train_dict,lamda=lamda)

	dt = sio.loadmat('weights/'+ case_name + '.mat')
	model = mlp(K,d,m=50)
	model.W1 = dt['W1']
	model.W2 = dt['W2']
	model.b1 = dt['b1']
	model.b2 = dt['b2']

	acc_val = model.compute_acc(X_val,Yenc_val)
	acc_tst = model.compute_acc(X_test,Y_test)
	print(f"Acc TEST = {acc_tst}")



##########################################
## Run the programme DOWN Here:
##########################################
if __name__ == "__main__":

	if (args.m == 1) or (args.m == 2):
		ExamCode()

	elif (args.m == 3):
		train_E3()
	elif (args.m == 4):
		train_E4()
		post_E4()
	elif (args.m == 5):
		# train_E5()
		post_E5()
	elif (args.m==6):
		train_E6()
	elif (args.m==7):
		train_E7()
		post_E7()
	else:
		raise ValueError