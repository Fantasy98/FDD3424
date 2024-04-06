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
import numpy.linalg as LA
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd 
from tqdm import tqdm 
import pathlib
import argparse

# Parse Arguments 
parser = argparse.ArgumentParser()
parser.add_argument('-m',default='test',type=str,help='Choose the mode for run the code: test, run, train, eval')
parser.add_argument('-epoch',default=40,type=int,help='Number of epoch')
parser.add_argument('-batch',default=100,type=int,help='Batch size')
parser.add_argument('-lr',default=1e-3,type=float,help='learning rate')
parser.add_argument('-lamda',default=0,type=float,help='l2 regularisation')
args = parser.parse_args()

# Mkdir 
pathlib.Path('Figs/').mkdir(exist_ok=True)
pathlib.Path('data/').mkdir(exist_ok=True)
pathlib.Path('weights/').mkdir(exist_ok=True)
font_dict = {'size':20,'weight':'bold'}
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

def ComputeGradsNum(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));

	c = ComputeCost(X, Y, W, b, lamda);
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2-c) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)
			grad_W[i,j] = (c2-c) / h

	return [grad_W, grad_b]

def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1 = ComputeCost(X, Y, W, b_try, lamda)
		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h
			c1 = ComputeCost(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)
			grad_W[i,j] = (c2-c1) / (2*h)
	return [grad_W, grad_b]

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

def save_as_mat(data, name):
	""" Used to transfer a python model to matlab """
	import scipy.io as sio
	sio.savemat(name + '.mat',
			data)
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
		X	: [d,n] 
		Y	: [1,n] 
		X_val	: [d,n] 
		Y_val	: [1,n] 
		Y_val	: [1,n] 
	"""
	
	dt = readPickle("data_batch_1")
	X 		= np.array(dt[b'data']).astype(np.float64).T
	y 		= np.array(dt[b'labels']).astype(np.float64).flatten()
	print(f"TRAIN X: {X.shape}")
	print(f"TRAIN Y:{y.shape}, here are {len(np.unique(y))} Labels")
	dt = readPickle("data_batch_2")
	X_val 		= np.array(dt[b'data']).astype(np.float64).T
	y_val 		= np.array(dt[b'labels']).astype(np.float64).flatten()
	labels 	= dt[b'batch_label']
	print(f"Val X: {X.shape}")
	print(f"Val Y:{y.shape}, here are {len(np.unique(y))} Labels")
	return X, y, X_val,y_val

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
	mean_x = np.repeat(np.mean(X,axis=1).reshape(-1,1),X.shape[-1],1)
	std_x  = np.repeat(np.std(X,axis=1).reshape(-1,1),X.shape[-1],1)
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

	y_hat = np.zeros(shape=(K, len(y)))
	
	for il, yi in enumerate(y):
		y_hat[int(yi),il] = 1
	
	return y_hat
#----------------------------------------------




#----------------------
#	W&B Intitialisation  
#-----------------------
#----------------------------------------------


class mlp:
	def __init__(self,K,d,m=50,
					lamda = 0):
		self.K  = K 
		self.d  = d 
		self.init_WB(K,d,m)
		self.lamda = lamda

	def forward(self,x,return_hidden=False):
		"""
		Forward Propagation 
		"""
		hidden_output= ReLU(self.W1 @ x +self.b1)# ReLU activation

		scores = self.W2 @ hidden_output +self.b2

		if return_hidden:
			return softmax(scores), hidden_output 
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
		
		# Part 1: compute the loss:
		## 1 Compute prediction:
		P = self.forward(X)
		## Cross-entropy loss
		# Clip the value to avoid ZERO in log
		P = np.clip(P,1e-16,1-1e-16)
		l_cross =  -np.mean(np.sum(Y*np.log(P),axis=0))
		# Part 2: Compute the regularisation 
		reg = self.lamda * (np.sum(self.W1**2) + np.sum(self.W2**2))
		# Assemble the components
		J = l_cross + reg
		del P 
		if return_loss:
			return J, l_cross
		else: 
			return J 

	def compute_acc(self,X,Y):
		"""
		Compute the accuracy of the classification 
		
		Args:

			X	: [d,n] input 
			Y	: [1,n] Ground Truth 
			b	: [d,1] bias 
			
		Returns: 

			acc : (float) a scalar value containing accuracy 
		"""
		# Generate Output with [K,n]
		P = self.forward(X)
		#Compute the maximum prob 
		# [K,n] -> K[1,n]
		P = np.argmax(P,axis=0)
		# Compute how many true-positive samples
		true_pos = np.sum(P == Y)
		# Percentage on total 
		acc =  true_pos / Y.shape[-1]
		del P 
		return acc

	def computeGradient(self,X,Y):
		"""
		Compute the Gradient w.r.t the W&B 

		Args:

			X	: [d,n] input 
			Y	: [1,n] Ground Truth 
			b	: [d,1] bias 
		
		Returns:

			dW1 : []
		"""

		# compute the difference 
		P, hidden_output = self.forward(X,return_hidden=True)

		g 	    = -(Y - P).T
		grad_W2 = g.T @ hidden_output.T + 2*self.lamda*self.W2
		grad_b2 = np.sum(g,axis=0).reshape(-1,1)
		
		g_hidden = ReLU(g @ self.W2)
		grad_W1  = (X @ g_hidden).T + 2*self.lamda*self.W1
		grad_b1  = np.sum(g_hidden, axis=0).reshape(-1,1)
		
		return grad_W2, grad_b2, grad_W1, grad_b1


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
		mu = 0; sigma1 = np.sqrt(1/d); sigma2 = np.sqrt(1/m)
		#Layer 1 
		self.W1 = np.random.normal(mu,sigma1,size=(m,d)).astype(np.float64)
		self.b1 = np.zeros(shape=(m,1)).astype(np.float64)
		#Layer 2 
		self.W2 = np.random.normal(mu,sigma2,size=(K,m)).astype(np.float64)
		self.b2 = np.zeros(shape=(K,1)).astype(np.float64)
		
		print(f"INFO:W&B init: W1={self.W1.shape}, b2={self.b1.shape}")
		print(f"INFO:W&B init: W2={self.W2.shape}, b2={self.b2.shape}")
	



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


def EvaluateClassifier(X,W,b):
	"""
	Forward Prop of the model 
	Args:
		X: [d,n] inputs 
		W: [K,d] Weight 
		b: [K,1] bias
	Returns:
		P: [K,n] The outputs as one-hot classification
	"""
	P = W @ X + b

	return softmax(P) 


def ComputeCost(X,Y,W,b,lamda,return_loss = False):
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
	P = EvaluateClassifier(X,W,b)
	## Cross-entropy loss
	# Clip the value to avoid ZERO in log
	P = np.clip(P,1e-16,1-1e-16)
	l_cross =  -np.mean(np.sum(Y*np.log(P),axis=0))
	# Part 2: Compute the regularisation 
	reg = lamda * np.sum(W**2)
	# Assemble the components
	J = l_cross + reg
	
	del P, W
	if return_loss:
		return J, l_cross
	else: 
		return J 
	
def ComputeAccuracy(X,Y,W,b):
	"""
	Compute the accuracy of the classification 
	
	Args:

		X	: [d,n] input 
		Y	: [1,n] Ground Truth 
		W 	: [K,d] Weight 
		b	: [d,1] bias 
		
	Returns: 

		acc : (float) a scalar value containing accuracy 
	"""

	# Generate Output with [K,n]
	P = EvaluateClassifier(X,W,b)

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
#	Back Prop  
#-----------------------
#----------------------------------------------
def ComputeGradients(X,Y,P,W,b,lamda):
	"""
	Compute the gradient w.r.t W & B 

	Args: 
		X 		: [d,n] Input 
		Y 		: [K,n] Encoded Ground truth 
		P		: [K,n] Prediction 
		W 		: [K,d] Weight of model 
		b 		: [K,1] Bias of model 
		lamda	: (float) regression 
	
	Returns:

		grad_W  :  [K,d] Gradient w.r.t Weight of J 

		grad_b  :  [K,1] Gradient w.r.t Bias of J 
	"""

	# compute the difference 
	g 	= -(Y - P).T
	grad_W 		= g.T @ X.T + 2*lamda*W
	grad_b 		= np.sum(g,axis=0).reshape(-1,1)

	# delta_scores = probs - y_batch
	# dW_out = np.dot(hidden_output.T, delta_scores)
    # db_out = np.sum(delta_scores, axis=0, keepdims=True)
    # delta_hidden = np.dot(delta_scores, W_out.T)
    # delta_hidden[hidden_output <= 0] = 0  # ReLU derivative
    # dW = np.dot(X_batch.T, delta_hidden)
    # db = np.sum(delta_hidden, axis=0, keepdims=True)


	return grad_W, grad_b

def BackProp(X,Y,W,b,lamda,eta,n_batch):
	"""
	Compute the gradient w.r.t W & B and Back propagation 
	Args: 
		X 		: [d,n] Input 
		Y 		: [K,n] Encoded Ground truth 
		W 		: [K,d] Weight of model 
		b 		: [K,1] Bias of model 
		lamda	: (float) regression 
		eta		: (float) learning rate
		n_batch : (int) Number of batch 
	
	Returns:

		W :  [K,d] Updated Weight 

		b  :  [K,1] Updated Bias 
	"""
	P   = EvaluateClassifier(X,W,b)
	
	g 	= -(Y - P).T
	
	grad_W 		= g.T @ X.T + 2*lamda*W
	grad_b 		= np.sum(g,axis=0).reshape(-1,1)

	Wstar 		= W - (1/n_batch) * eta * grad_W
	bstar 		= b - (1/n_batch) * eta * grad_b

	return Wstar, bstar 


def Prop_Error(ga,gn,eps):
	"""
	Compute the propagation Error with uncertainty
	"""

	eps_m = np.ones_like(ga) * eps
	summ  = np.abs(ga)+np.abs(gn)
	n,m = ga.shape
	diw   = np.stack(
				[eps_m.reshape(1,n,m),
				summ.reshape(1,n,m)],axis=0)
	return np.abs(ga-gn)/np.max(diw,0)



#----------------------
#	Training 
#-----------------------
#----------------------------------------------
def cyclinal_lr(t,
				eta_min, eta_max,
				n_epoch, n_batch,
				k):
	"""
	cyclinal learning rate during training
	
	Args: 
		t: (int) Current Number of iteration 

		eta_min: Lower bound of learning rate 

		eta_max: Upper bound of learning rate 

		n_epoch    :  number of epoch  
		
		n_batch    :   number of batch size 

		k    	   :   An integer between 2 and 8 
	
	Return: 
		eta		: The current learning rate 
	"""

	# Step 1 Compute the step size
	n_s = k * n_epoch/n_batch 

	# Step 2: compute the milestone  
	cycle = np.floor(1+t/(2*n_s))
	x = np.abs(t / n_s - 2 * cycle + 1)

	# Step 3: Compute the learning rate 
	eta = eta_min + (eta_max - eta_min) * np.maximum(0, (1 - x))
	
	return eta
	

def MiniBatchGD(X,Y,X_val,Y_val,GDparams,W,b):
	"""
	MiniBatch Gradient Descent for training the model 

	Args:
		
		X 		:	[d, n] The input with batch size of n 
		
		Y 		:	[K, n] The ground truth 
		
		X_val 	:	[d, n] The input with batch size of n 
		
		Y_val 	:	[K, n] The ground truth 

		GDparams:   (dict) The dictionary for paraemeter 

		W 		:	[K, d] The weight 
		
		b 		:	[K, 1] The bias 
		
		lamda	:	(float) The regularisation

	Returns: 
		Wstar	:	[K,d] The updated Weight

		bstar	:	[K,1] The updated Bias 
	"""
	print('Start Training')
	lenX = X.shape[-1]
	batch_size = GDparams.n_batch
	batch_range = np.arange(1,lenX//GDparams.n_batch)
	
	hist = {}; hist['train_cost'] = []; hist['val_cost'] = []
	hist['train_loss'] = []; hist['val_loss'] = []

	for epoch in (range(GDparams.n_epochs)): 

		# Shuffle the batch indicites 
		indices = np.random.permutation(lenX)
		X_ = X[:,indices]
		Y_ = Y[:,indices]
		
		for b in (batch_range):
			X_batch = X_[:,b*batch_size:(b+1)*batch_size]
			Y_batch = Y_[:,b*batch_size:(b+1)*batch_size]
			W,b = BackProp(X_batch,Y_batch,W,b,
							GDparams.lamda,
							GDparams.eta,
							batch_size)
		
		jc,l_train = ComputeCost(X,Y,W,b,GDparams.lamda,return_loss=True)
		hist['train_cost'].append(jc)
		hist['train_loss'].append(l_train)
	
		jc_val,l_val  = ComputeCost(X_val,Y_val,W,b,GDparams.lamda,return_loss=True)
		hist['val_cost'].append(jc_val)
		hist['val_loss'].append(l_val)

		print(f"\nAt Epoch =({epoch+1}/{GDparams.n_epochs}),\n"+\
			f" Train Cost ={hist['train_cost'][-1]}, Val Cost ={hist['val_cost'][-1]}\n"+\
			f" Train Loss ={hist['train_loss'][-1]}, Val Loss ={hist['val_loss'][-1]}"
			)

	return W, b, hist 



#----------------------
#	Post-Processing 
#-----------------------
#----------------------------------------------
def plot_loss(loss,fig=None,axs=None,color=None,ls=None):
	if fig==None:
		fig, axs = plt.subplots(1,1,figsize=(6,4))
	
	if color == None: color = "r"
	if ls == None: ls = '-'
	axs.plot(loss,ls,lw=2.5,c=color)
	axs.set_xlabel('Epochs')
	axs.set_ylabel('Loss')
	return fig, axs 




#----------------------
#	Main Programm
#-----------------------
#----------------------------------------------
def test_code():
	"""Test the functions for the assignments """

	print("#"*30)
	labels = ['airplane','automobile','bird',
			'cat','deer','dog','frog',
			"horse",'ship','truck']
	
	print(f"Testing Functions:")

	# Step 1: Load data
	X, Y, X_val ,Y_val = LoadBatch()
	# Define the feature size and label size
	K = len(np.unique(Y)); d = X.shape[0]
	# One-Hot encoded for Y 
	Yenc 	 = one_hot_encode(Y,K)
	Yenc_val = one_hot_encode(Y_val,K)
	print(f"Global K={K}, d={d}")

	# Step 2: Scaling the data
	X,muX,stdX 		= normal_scaling(X)
	X_val			= (X_val - muX / stdX)


	# Step 3*: Test the cyclinal_lr 
	n_epoch = 20; n_batch = 100; eta_min = 1e-3; eta_max = 1e-2; k=2
	etas = []
	for l in range(n_epoch):
		eta = cyclinal_lr(l,eta_min, eta_max,n_epoch,n_batch,k)
		etas.append(eta)
	fig, axs = plt.subplots(1,1,figsize=(6,4))
	axs.plot(range(n_epoch),etas,'-o',c=colorplate.black,lw=2)
	axs.set_xlabel('Epoch',font_dict)
	axs.set_ylabel(r'$\eta$',font_dict)
	fig.savefig('Figs/LR_schedule.jpg',bbox_inches='tight',dpi=300)


	# Step 3: Initialisation of the network
	# Use the class for model implementation 
	model = mlp(K,d)
	
	# Step 4: Test for forward prop
	batch_size = 2
	X_test  = X[:,:batch_size]
	Y_test  = Yenc[:,:batch_size]
	# P 		= EvaluateClassifier(X_test,W,b)
	P 		= model.forward(X_test)
	print(f"INFO: Test Pred={P.shape}")


	# Step 5: Cost Function
	J,l_cross = model.cost_func(X_test,Y_test,return_loss=True)
	print(f"INFO: The loss = {J}")

	# Step 6: Examine the acc func:
	acc = model.compute_acc(X,Y)
	print(f"INFO:Accuracy Score={acc*100}%") 

	# Step 7 Compute the Gradient and compare to analytical solution 
	batch_size = 1
	X_test  = X[:,:batch_size]
	Y_test  = Yenc[:,:batch_size]
	P,hid_out 		= model.forward(X_test,True)
	print(f"INFO: Test Pred={P.shape},hidden out = {hid_out.shape}")
	
	grad_W2, grad_b2, grad_W1, grad_b1 = model.computeGradient(X,Y)
	print(f"Compute Gradient: W2:{grad_W2.shape},W1:{grad_W1.shape},b1:{grad_b1.shape},b2:{grad_b2.shape}")
	

	
	lamda = 0
	h 	  = 1e-6
	grad_W , grad_b = ComputeGradients(X_test,
									Y_test,
									P,
									W,b,
									lamda)
	print(f"INFO: Shape of gW = {grad_W.shape}, gb = {grad_b.shape}")
	grad_W_n, grad_b_n = ComputeGradsNumSlow(X_test,
										Y_test,
										P,
										W,b,
										lamda=lamda,
										h=h)
	print(f"INFO: Shape of gW = {grad_W_n.shape}, gb = {grad_b_n.shape}")

	ew = Prop_Error(grad_W,grad_W_n,h)
	eb = Prop_Error(grad_b,grad_b_n,h)
	print(f"Comparison: Prop Error for weight:{ew.mean()}")
	print(f"Comparison: Prop Error for Bias:{eb.mean()}")
	fig, axs = montage(grad_W_n,labels)
	fig.savefig('Figs/Gradient_central.jpg',bbox_inches='tight',dpi=200)
	

	grad_W_n, grad_b_n = ComputeGradsNum(X_test,
										Y_test,
										P,
										W,b,
										lamda=lamda,
										h=h)
	print(f"INFO: Shape of gW = {grad_W_n.shape}, gb = {grad_b_n.shape}")
	
	ew = Prop_Error(grad_W,grad_W_n,h)
	eb = Prop_Error(grad_b,grad_b_n,h)
	print(f"Comparison: Prop Error for weight:{ew.mean()}")
	print(f"Comparison: Prop Error for Bias:{eb.mean()}")
	

	fig, axs = montage(grad_W,labels)
	fig.savefig('Figs/Gradient_test.jpg',bbox_inches='tight',dpi=200)
	fig, axs = montage(grad_W_n,labels)
	fig.savefig('Figs/Gradient_finite.jpg',bbox_inches='tight',dpi=200)
	print("#"*30)


def train():
	
	"""Training for W&B"""

	print("#"*30)
	print(f"Training:")
	filename = f"WB_{GDparams.n_batch}bs_{GDparams.n_epochs}Epoch_{GDparams.eta:.2e}lr_{GDparams.lamda:.3e}lamb"
	print(f"Case:\n{filename}")
	# Step 1: Load data
	X, Y, X_val ,Y_val= LoadBatch()
	# Define the feature size and label size
	K = len(np.unique(Y)); d = X.shape[0]
	# One-Hot encoded for Y 
	Yenc 	 = one_hot_encode(Y,K)
	Yenc_val = one_hot_encode(Y_val,K)
	print(f"Global K={K}, d={d}")

	# Step 2: Scaling the data
	X,muX,stdX 		= normal_scaling(X)
	X_val			= (X_val - muX)/ stdX
	
	# Step 3: Initialisation of the network
	W,b   = init_WB(K,d)
	
	# Step 4: Mini-Batch gradient descent
	W,b, hist = MiniBatchGD(X,Yenc,X_val,Yenc_val,GDparams,W,b)
	
	# Step 5: Save the data
	save_as_mat({"W":W,
				"b":b,
				'train_loss':np.array(hist['train_loss']),
				'train_cost':np.array(hist['train_cost']),
				'val_loss':np.array(hist['val_loss']),
				'val_cost':np.array(hist['val_cost']),
				},

				"weights/" + filename)
	print(f"W&B Saved!")
	
	# Step 6: Visualisation of loss/cost function
	fig, axs = plot_loss(hist['train_cost'],color = colorplate.red,ls = '-')
	fig, axs = plot_loss(hist['train_loss'],fig,axs,color = colorplate.red,ls = '-.')
	fig, axs = plot_loss(hist['val_cost'],fig,axs,color =colorplate.blue, ls = '-')
	fig, axs = plot_loss(hist['val_loss'],fig,axs,color =colorplate.blue, ls = '-.')
	axs.legend(['Train Cost',"Train Loss", "Val Cost", "Val Loss"])
	fig.savefig(f'Figs/Loss_{filename}.jpg',dpi=300,bbox_inches='tight')
	print("#"*30)

	return


def postProcessing():
	"""
	Post-process of all the W&B
	"""
	gdparams = {
				'n_batch':[100,100,100,100],
				'n_epoch':[40,40,40,40],
				'eta'    :[1e-1,1e-3,1e-3,1e-3],
				'lamda'  :[0, 0, 0.1 ,1]
				}
	
	X,Y = load_test_data()
	K = len(np.unique(Y)); d = X.shape[0]
	Xt, _, _ ,_  = LoadBatch() #use training data to scale the test data
	_,muX,stdX     = normal_scaling(Xt)
	X 			   = (X - muX)/stdX

	labels = ['airplane','automobile','bird',
			'cat','deer','dog','frog',
			"horse",'ship','truck']
	
	del Xt
	res_acc		= {}
	names 		= []
	cost_dict 	= {}
	loss_dict 	= {}
	for il in range(4):
		
		n_batch  = gdparams['n_batch'][il]
		n_epochs = gdparams['n_epoch'][il]
		eta      = gdparams['eta'][il]
		lamda    = gdparams['lamda'][il]

		filename = f"WB_{n_batch}bs_"+\
					f"{n_epochs}Epoch_"+\
					f"{eta:.2e}lr_{lamda:.3e}lamb"
		
		d		 =	sio.loadmat("weights/" + filename+'.mat')
		print(f"INFO Loaded: {filename}")

		# Read W&B
		W, b = d["W"],d["b"]

		# Assess Acc on test data
		acc = ComputeAccuracy(X,Y,W,b)
		print(f"ACC = {acc*100:.2f}%")

		names.append(filename)
		res_acc[filename] = np.array(acc).reshape(-1,)

		# Visualise the learned Weight
		fig, axs = montage(W,labels)
		fig.savefig(f"Figs/Weight_Vis_{filename}.jpg",dpi=300,bbox_inches='tight')

		cost_dict["train_" + filename] 	= d['train_cost'].flatten()
		cost_dict["val_" + filename] 	= d['val_cost'].flatten()
		
		loss_dict["train_" + filename] 	= d['train_loss'].flatten()
		loss_dict["val_" + filename] 	= d['val_loss'].flatten()
	
	df = pd.DataFrame(res_acc)
	df.to_csv('Acc.csv')
	
	# Visualisation
	###############################
	
	## 1: All the cost function 
	colors = [colorplate.cyan,colorplate.blue,colorplate.yellow,colorplate.red]
	fig,axs = plt.subplots(1,1,figsize = (8,6))
	legend_label = []
	for il, filename in enumerate(names):
		n_batch  = gdparams['n_batch'][il]
		n_epochs = gdparams['n_epoch'][il]
		eta      = gdparams['eta'][il]
		lamda    = gdparams['lamda'][il]
		label_   = f"n_batch={n_batch}; n_epochs={n_epochs}; " + r"$\eta$" +f"={eta}; "+r"$\lambda$" +f"={lamda}"
		
		axs.plot(cost_dict["train_" + filename], "-",lw = 2,c=colors[il])
		legend_label.append(label_)
		
	axs.legend(legend_label,loc='upper right')
	for il, filename in enumerate(names):
		axs.plot(cost_dict["val_" + filename],   "--",lw = 2,c=colors[il])
		
	axs.set_xlabel('Epoch',font_dict)
	axs.set_ylabel('Cost',font_dict)
	# axs.set_ylim(1.5,6)
	fig.savefig("Figs/Cost_compare.jpg",dpi=500,bbox_inches='tight')

	## 2: All the loss function 
	fig1,axs1 = plt.subplots(1,1,figsize = (8,6))
	for il, filename in enumerate(names):
		n_batch  = gdparams['n_batch'][il]
		n_epochs = gdparams['n_epoch'][il]
		eta      = gdparams['eta'][il]
		lamda    = gdparams['lamda'][il]	
		axs1.plot(loss_dict["train_" + filename], "-",lw = 2,c=colors[il])
		
	axs1.legend(legend_label,loc='upper right')
	for il, filename in enumerate(names):
		axs1.plot(loss_dict["val_" + filename],   "--",lw = 2,c=colors[il])
		
	axs1.set_xlabel('Epoch',font_dict)
	axs1.set_ylabel('Loss',font_dict)
	# axs1.set_ylim(1,3)
	fig1.savefig("Figs/Loss_compare.jpg",dpi=500,bbox_inches='tight')


	## 3: When learning rate = 1e-3
	legend_label = []
	fig1,axs1 = plt.subplots(1,1,figsize = (6,3))
	for il, filename in enumerate(names[1:]):
		n_batch  = gdparams['n_batch'][il+1]
		n_epochs = gdparams['n_epoch'][il+1]
		eta      = gdparams['eta'][il+1]
		lamda    = gdparams['lamda'][il+1]	
		axs1.plot(cost_dict["train_" + filename], "-",lw = 2,c=colors[il+1])
		legend_label.append(r"$\lambda$" +f"={lamda}")

	axs1.legend(legend_label,loc=(0.2,1),ncol=3)
	for il, filename in enumerate(names[1:]):
		axs1.plot(cost_dict["val_" + filename], "--",lw = 2,c=colors[il+1])
	axs1.set_xlabel('Epoch',font_dict)
	axs1.set_ylabel('Cost',font_dict)
	fig1.savefig("Figs/Cost_compare_1e-3.jpg",dpi=500,bbox_inches='tight')


	## 4: When lamda = 0 
	legend_label = []
	fig1,axs1 = plt.subplots(1,1,figsize = (6,3))
	for il, filename in enumerate(names[:2]):
		n_batch  = gdparams['n_batch'][il]
		n_epochs = gdparams['n_epoch'][il]
		eta      = gdparams['eta'][il]
		lamda    = gdparams['lamda'][il]	
		axs1.plot(loss_dict["train_" + filename], "-",lw = 2,c=colors[il])
		legend_label.append(r"$\eta$" +f"={eta}")

	axs1.legend(legend_label,loc=(0.3,1),ncol=2)
	for il, filename in enumerate(names[:2]):
		axs1.plot(loss_dict["val_" + filename], "--",lw = 2,c=colors[il])
	
	axs1.set_xlabel('Epoch',font_dict)
	axs1.set_ylabel('Loss',font_dict)
	fig1.savefig("Figs/Loss_compare_1e-3.jpg",dpi=500,bbox_inches='tight')

#-----------------------------------------------


##########################################
## Run the programme DOWN Here:
##########################################
if __name__ == "__main__":

	if args.m == 'test':
		test_code()
	elif args.m == 'train':
		train()
	elif args.m == 'eval':
		postProcessing()
	elif args.m == 'run':
		train()
		postProcessing()
	else:
		raise ValueError