"""
Training for a binary Classifier 
@yuningw
"""
##########################################
## Environment and general setup 
##########################################
import numpy as np
import matplotlib.pyplot as plt 
# Setup the random seed

np.random.seed(1024)
##########################################
## Function from the assignments
##########################################
def LoadBatch(filename):
	""" Copied from the dataset website """
	import pickle5 as pickle 
    
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
	print(f"Existing Data: {fileList}")

	dt = LoadBatch(fileList[0])
	print(f"Type of data: {type(dt)}")
	print(f"The key of diction:\n{dt.keys()}")

	X = np.array(dt[b'data']).astype(np.float32)
	y = np.array(dt[b'labels']).astype(np.float32).reshape(-1,1)

	labels 	 = dt[b'batch_label']
	print(f"The data shape = \t {X.shape}")
	print(f"The One-Hot label shape = \t {y.shape}")
	print(f"The label shape = \t {len(labels)}")
	return X, y, labels

def norm_scaling(X):
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
	print(f"The Mean={mean_x.shape}; Std = {std_x.shape}")	
	return (X-mean_x)/std_x, mean_x, std_x

def init_WB(n:int,d:int):
	"""
	Initialising The W&B, we use normal distribution as an initialisation strategy 

	Args:
		n	:	integer of the size of sample 
		d 	:	integer of the size of feature space 
	
	Returns:

		W	:	Numpy Array as a matrix of Weight 
		b	:	Numpy Array as a vector of bias
	"""
	mu = 0; sigma = 0.01
	W = np.random.normal(mu,sigma,size=(n,d))
	b = np.random.normal(mu,sigma,size=(d,1))
	
	return W,b
	




def ComputeCost():
	return



def main():
	
	X, y, labels = load_data()
	X,_,_ = norm_scaling(X)
	return


if __name__ == "__main__":
	
	main()

