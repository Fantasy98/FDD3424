"""
Training for a binary Classifier 
@yuningw
"""
##########################################
## Environment and general setup 
##########################################
import numpy as np
import pickle
import numpy.linalg as LA
import matplotlib.pyplot as plt
from tqdm import tqdm 
font_dict = {'size':20,'weight':'bold'}
# Setup the random seed
np.random.seed(400)
# Set the global variables
global K, d


##########################################
## Function from the assignments
##########################################
def LoadBatch(filename):
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

def montage(W):
	""" Display the image for each label in W """
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(2,5)
	for i in range(2):
		for j in range(5):
			im  = W[i*5+j,:].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title("y="+str(5*i+j))
			ax[i][j].axis('off')
	return fig, ax 

def save_as_mat(data, name="model"):
	""" Used to transfer a python model to matlab """
	import scipy.io as sio
	sio.savemat(name + '.mat',
			{name:data})



##########################################
## Functions from scratch 
## Yuningw 
##########################################
def load_data():
	"""
	Load Data from the binary file using Pickle 

	Returns: 
		X	: Array with shape of 
	"""
	import os 

	fileList = os.listdir('data/')
	fileList = [ f for f in fileList if f[:4]=='data']
	# print(f"Existing Data: {fileList}")

	dt = LoadBatch(fileList[0])
	# print(f"Type of data: {type(dt)}")
	# print(f"The key of diction:\n{dt.keys()}")

	X 		= np.array(dt[b'data']).astype(np.float32).T
	y 		= np.array(dt[b'labels']).astype(np.float32).flatten()
	labels 	= dt[b'batch_label']
	print(f"X: \t {X.shape}")
	print(f"Y: \t {y.shape}, here are {len(np.unique(y))} Labels")
	return X, y, labels

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
	mean_x = np.mean(X,axis=0)
	std_x = np.mean(X,axis=0)
	# print(f"The Mean={mean_x.shape}; Std = {std_x.shape}")	
	print(f"INFO: Complete Normalisation")
	return (X-mean_x)/std_x, mean_x, std_x

def init_WB(K:int,d:int,):
	"""
	Initialising The W&B, we use normal distribution as an initialisation strategy 
	Args:
		K	:	integer of the size of feature size
		d 	:	integer of the size of label size

	Returns:
		W	:	[K,d] Numpy Array as a matrix of Weight 
		b	:	[d,1] Numpy Array as a vector of bias
	"""

	mu = 0; sigma = 0.01
	W = np.random.normal(mu,sigma,size=(K,d))
	b = np.random.normal(mu,sigma,size=(K,1))
	print(f"INFO:W&B init: W={W.shape}, b={b.shape}")
	return W,b

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

def ComputeCost(X,Y,W,b,lamda):
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

class GDparams:
	eta 	= 1e-3 
	n_batch = 100
	n_epochs= 100
	lamda 	= 0


def MiniBatchGD(X,Y,GDparams,W,b,lamda):
	"""
	MiniBatch Gradient Descent for training the model 

	Args:
		
		X 		:	[d, n] The input with batch size of n 
		
		Y 		:	[K, n] The ground truth 
		
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
		
		jc = ComputeCost(X,Y,W,b,GDparams.lamda)
		acc = ComputeAccuracy(X,Y,W,b)

		print(f"At Epoch ={epoch+1}, Cost Func ={jc}, acc = {acc*100}")
	return W, b 


def test():
	print("#"*30)
	print(f"Testing Functions:")
	# Step 1: Load data
	X, Y, labels = load_data()
	# Define the feature size and label size
	K = len(np.unique(Y)); d = X.shape[0]
	# One-Hot encoded for Y 
	Yenc = one_hot_encode(Y,K)
	print(f"Global K={K}, d={d}")
	# Step 2: Scaling the data
	X,_,_ = normal_scaling(X)
	# Step 3: Initialisation of the network
	W,b   = init_WB(K,d)
	
	# Step 4: Test for forward prop
	batch_size = 2
	X_test  = X[:,:batch_size]
	Y_test  = Yenc[:,:batch_size]
	P 		= EvaluateClassifier(X_test,W,b)
	print(f"INFO: Test Pred={P.shape}")
	
	# Step 5: Cost Function
	J = ComputeCost(X_test,Y_test,W,b,lamda=0)
	print(f"INFO: The loss = {J}")

	# Step 6: Examine the acc func:
	acc = ComputeAccuracy(X,Y,W,b)

	print(f"INFO:Accuracy Score={acc*100}%") 

	# Step 7 Compute the Gradient and compare to analytical solution 
	# Step 4: Test for forward prop
	
	batch_size = 1
	X_test  = X[:,:batch_size]
	Y_test  = Yenc[:,:batch_size]
	P 		= EvaluateClassifier(X_test,W,b)
	print(f"INFO: Test Pred={P.shape}")
	lamda = 0
	h 	  = 1e-6
	grad_W , grad_b = ComputeGradients(X_test,
									Y_test,
									P,
									W,b,
									lamda)
	print(f"INFO: Shape of gW = {grad_W.shape}, gb = {grad_b.shape}")
	
	

	# grad_W_a, grad_b_a = ComputeGradsNumSlow(X_test,
	# 									Y_test,
	# 									P,
	# 									W,b,
	# 									lamda=lamda,
	# 									h=h)
	# print(f"INFO: Shape of gW = {grad_W_a.shape}, gb = {grad_b_a.shape}")
	# ew = Prop_Error(grad_W,grad_W_a,h)
	# eb = Prop_Error(grad_b,grad_b_a,h)
	# print(f"Comparison: Prop Error for weight:{ew.mean()}")
	# print(f"Comparison: Prop Error for Bias:{eb.mean()}")

	# fig, axs = plt.subplots(2,1,figsize=(4,4),sharex=True,sharey=True)
	# clb = axs[0].contourf(grad_W,
	# 				vmax=grad_W_a.max(),
	# 				vmin=grad_W_a.min(),
	# 				cmap = 'RdBu')
	
	# clb2=axs[1].contourf(grad_W_a,
	# 				vmax=grad_W_a.max(),
	# 				vmin=grad_W_a.min(),
	# 				cmap = 'RdBu')
	# for ax in axs:
	# 	ax.set_xticks([])
	# 	ax.set_yticks([])
	# 	# ax.set_aspect(0.5)
	# axs[0].set_title("Analytical: " + r'$\frac{\partial L}{\partial W}$',font_dict)
	# axs[1].set_title("Numerical: " + r'$\frac{\partial L}{\partial W}$',font_dict)
	# fig.subplots_adjust(hspace=0.5)
	# fig.colorbar(clb,ax=axs[0])
	# fig.colorbar(clb2,ax=axs[1])
	# fig.savefig('resW.jpg',dpi=300,bbox_inches='tight')

	MiniBatchGD(X,Yenc,GDparams,W,b,lamda)
	print("#"*30)

##########################################
## Run the programme DOWN Here:
##########################################
if __name__ == "__main__":

	test()

