# Python 3.XX
# 2 layer neural network
# build from scratch without libraries
# neural network for clarification
# building an XOR predictor 
# 

# using for matrix multiplikation
import numpy as np

# import to track the learing time
import time 

# global variables
#

# number of hidden layers, to improve test error
hidden_num = 10

# input values
input_num = 10 

# output values
output_num = 10

# number of sample data
sample_number = 500

# setting the hyperparameters
#
# variables which determine how the network is trained
# determines structure of the network
#

# how fast the network works
learning_rate = 0.01

# lower the loss funtion for tunement
momentum = 0.9

# generate the same random numbers for testing
# non deterministic seeding
np.random.seed(0)

# implementing sigmoidfunction
# used for activation of neurals, determines the value of the sum of input
# changing values into probabilities, to determine which path is optimal
# values change, if the sum, of one node has changed
# 

# activation function for first layer
def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

# activation function for the second layer
# using tangens hyperbolicus
#
def prime_tanh(x):
	return 1 - np.tanh(x)** 2 

# init training function
# x = input data
# t = the transpose matrix, help to do matrix mult
# V = the first Layer of the network
# W = the second Layer of the network
# bv = bias layer one, for accurate prediction
# bw = bias layer two, for accurate prediction
#

def train(x, t, V, W, bv, bw):

	# feed forward -- matrix multiply + biases, activate activation func tanh
	A = np.dot(x, V) + bv
	Z = np.tanh(A)

	B = np.dot(Z, W) + bw
	Y = sigmoid(B)

	# feeed backward -- sending computed values back to first layer
	# t is used to transform the matrix, because we are going backwards
	# Ev = compute loss and compare to actual loss function, for minimization 
	#
	Ew = Y - t
	Ev = prime_tanh(A) * np.dot(W, Ew)

	# predict the loss
	# deltas used for comparison as actual loss
	#
	dW = np.outer(Z, Ew)
	dV = np.outer(x, Ev)

	# cross entropy 
	# a measurement to determine the quality of a model, in conjunction with the prob
	loss = -np.mean(t * np.log(Y) + (1 - t) * np.log(1 - Y))
	
	return loss, (dV, dW, Ev, Ew)

# prediction function
# A, B = final computation of our variables
# 
def predict(x, V, W, bv, bw):

	A = np.dot(x, V) + bv
	B = np.dot(np.tanh(A), W) + bw

	return (sigmoid(B) > 0.5).astype(int)

# layer creation
# import 10 values and 10 out
#

V = np.random.normal(scale = 0.1, size = (input_num, hidden_num))
W = np.random.normal(scale = 0.1, size = (hidden_num, output_num))

# creation of biases used
bv = np.zeros(hidden_num)
bw = np.zeros(output_num)

# generate parameters into Array
params = [V, W, bv, bw]

# generatiion of random data
# generating 500 samples
#
X = np.random.binomial(1, 0.5, (sample_number, input_num))

# generate trnaspose
T = X ^ 1

# training the network
# using epoch as our starting timestamp
# training throuh for-loop
#
for epoch in range(100):
	err = []
	update = [0] * len(params)

	t0 = time.clock()
	
	# for each data point, we update the weigth of the network, probability
	for i in range(X.shape[0]):
		
		# feed our loss and grad with the train function
		# using the values of params as a list of arguments
		loss,grad = train(X[i], T[i], *params)

		# update the loss and calculate the loss
		for j in range(len(params)):
			params[j] -= update[j]

		for j in range(len(params)):
			update[j] = learning_rate * grad[j] + momentum * update[j]

		err.append(loss)

	print ('Epoch: %d, Loss: %.8f, Time: %.4fs' % (epoch, np.mean(err), time.clock() - t0 ))

	# prediciton try out

	x = np.random.binomial(1, 0.5, input_num)
	
	print ('XOR predicition: ') 
	print (x)
	print (predict(x, *params))