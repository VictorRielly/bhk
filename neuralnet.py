# This python script will be used to implement a neural network.
# The neural network will be nested in a class so it may integrated
# more easily with other code later on.
import sklearn.metrics
from sklearn import svm
import numpy as np
import csv
import pandas as pd
import copy
import time
import math
np.random.seed() 
import gc
class Neuralnet:
	# The network will be set up to allow any number of
	# layers and any number of nodes per layer. The only
	# parameter passed into the constructor is an array
	# holding the number of nodes per layer starting with
	# the input layer and continuing to the output layer	
	def __init__(*arg):
		# by default, we will make a 785 by 25 by 10
		# neural network.
		self = arg[0]
		self.eta = 0.01 
		self.alpha = 0 
		self.netl = [785,26,10]
		self.numl = 3
		self.netw = [np.zeros((25,785)),np.zeros((10,26))]
		self.netwd = [np.zeros((25,785)),np.zeros((10,26))]
		self.netx = [np.ones(785),np.ones(26),np.ones(10)]
		self.netd = [np.ones(785),np.ones(26),np.ones(10)]
		# sets default values for self.train and self.test
		self.train = './mnist_train.csv'
		self.test = './mnist_test.csv'
		# creates a matrix target vector matrix
		self.target = np.ones((10,10))
		self.target = self.target*.1
		temp = np.eye(10)
		temp = temp*.8
		self.target = self.target + temp 
		numi = 0
		for i in arg:
			if numi == 1:
				self.numl = np.size(i)
				self.netl = i
				self.netw = [0,]*(self.numl - 1)
				self.netwd = [0,]*(self.numl - 1)
				self.netx = [1,]*(self.numl)
				self.netd = [1,]*(self.numl)
				for k in range(0,self.numl):
					self.netx[k]=np.ones(self.netl[k]) 
					self.netd[k]=np.ones(self.netl[k]) 
				for j in range(0,self.numl-2):
					self.netw[j] = np.zeros((self.netl[j+1]-1,self.netl[j]))
					self.netwd[j] = np.zeros((self.netl[j+1]-1,self.netl[j]))
				self.netw[self.numl-2] = np.zeros((self.netl[self.numl-1],self.netl[self.numl-2]))
				self.netwd[self.numl-2] = np.zeros((self.netl[self.numl-1],self.netl[self.numl-2]))
				# creates a matrix target vector matrix
				self.target = np.ones((i[self.numl-1],i[self.numl-1]))
				self.target = self.target*.1
				temp = np.eye(i[self.numl-1])*.8
				self.target = self.target + temp 
				
			if numi == 2:
				self.train = i
			if numi == 3:
				self.test = i
			if numi == 4:
				self.eta = i
			if numi == 5:
				self.alpha = i
			numi = numi +1
		# Initializes the weights to be random
		self.rand_weights()
		# reads training and test data into self.train and
		# self.test respectively
		self.traind = (pd.read_csv(self.train,sep=',',header=None)).values
		self.testd = (pd.read_csv(self.test,sep=',',header=None)).values
		self.traindata = np.ones((np.shape(self.traind)[0],np.shape(self.traind)[1]+1))
		self.testdata = np.ones((np.shape(self.testd)[0],np.shape(self.testd)[1]+1))
		self.testdata[:,:-1] = self.testd
		self.traindata[:,:-1] = self.traind
		self.testdata[:,1:-1] = self.testdata[:,1:-1]/255.0
		self.traindata[:,1:-1] = self.traindata[:,1:-1]/255.0
	
	# Randomizes the weights
	def rand_weights(self):
		for i in range(0,self.numl-1):
			self.netw[i] = (np.random.rand(np.shape(self.netw[i])[0],np.shape(self.netw[i])[1])-.5)*.1
	
	# apply sigmoid function to the num'th vector of self.netx  
	def sigmoid(self,num):
		if num < self.numl-1:
			self.netx[num][:-1] = 1.0/(1+np.exp(-self.netx[num][:-1]))
		else :
			self.netx[num] = 1.0/(1+np.exp(-self.netx[num]))
			

	# apply the sgn function to the num'th vector of self.netx 
	def sgn(self,num):
		if num < self.numl-1:
			for i in range(0,self.netl[num]-1):
				if self.netx[num][i] > 0:
					self.netx[num][i] = 1
				else :
					self.netx[num][i] = 0
		else :
			for i in range(0,self.netl[num]):
				if self.netx[num][i] > 0:
					self.netx[num][i] = 1
				else :
					self.netx[num][i] = 0
	
	
	# propagates the first vector of self.netx through all of the vectors in self.netx
	def propagate_sigmoid(self):
		for i in range(0,self.numl-2):
			self.netx[i+1][:-1] = np.dot(self.netw[i],self.netx[i])
			self.sigmoid(i+1)
		self.netx[self.numl-1] = np.dot(self.netw[self.numl-2],self.netx[self.numl-2])
		self.sigmoid(self.numl-1)	
 

	# propagates the first vector of self.netx through all of the vectors in self.netx
	def propagate_sgn(self):
		for i in range(0,self.numl-2):
			self.netx[i+1][:-1] = np.dot(self.netw[i],self.netx[i])
			self.sgn(i+1)
		self.netx[self.numl-2] = np.dot(self.netw[self.numl-2],self.netx[self.numl-2])

	# shuffles the training data
	def shuffle(self):
		np.random.shuffle(self.traindata)

	# This function calculates the accuracy of the network on
	# the train data
	def train_accuracy_sigmoid(self):
		total = 0
		for i in self.traindata:
			self.netx[0] = i[1:]
			self.propagate_sigmoid()
			if (np.argmax(self.netx[self.numl-1]) == i[0]):
				total = total + 1
		self.train_accuracy = (0.0+total)/(np.shape(self.traindata)[0])
	
	# This function calculates the accuracy of the network on
	# the test data
	def test_accuracy_sigmoid(self):
		total = 0
		for i in self.testdata:
			self.netx[0] = i[1:]
			self.propagate_sigmoid()
			if (np.argmax(self.netx[self.numl-1]) == i[0]):
				total = total + 1
		self.test_accuracy = (0.0+total)/(np.shape(self.testdata)[0])

	# This function calculates the accuracy of the network on
	# the train data using the covarience weight matrix for the last layer
	def train_accuracy_sigmoid_conv(self):
		total = 0
		for i in self.traindata:
			self.netx[0] = i[1:]
			self.propagate_sigmoid()
			self.netx[self.numl-1] = np.dot(self.covw,self.netx[self.numl-2])
			if (np.argmax(self.netx[self.numl-1]) == i[0]):
				total = total + 1
		self.train_accuracy = (0.0+total)/(np.shape(self.traindata)[0])
	
	# This function calculates the accuracy of the network on
	# the test data using the covarience weight matrix for the last layer
	def test_accuracy_sigmoid_conv(self):
		total = 0
		for i in self.testdata:
			self.netx[0] = i[1:]
			self.propagate_sigmoid()
			self.netx[self.numl-1] = np.dot(self.covw,self.netx[self.numl-2])
			if (np.argmax(self.netx[self.numl-1]) == i[0]):
				total = total + 1
		self.test_accuracy = (0.0+total)/(np.shape(self.testdata)[0])
	
	# This function calculates the accuracy of the network on
	# the train data this uses the sgn function in place of the sigmoid
	def train_accuracy_sgn(self):
		total = 0
		for i in self.traindata:
			self.netx[0] = i[1:]
			self.propagate_sgn()
			if (np.argmax(self.netx[self.numl-1]) == i[0]):
				total = total + 1
		self.train_accuracy = (0.0+total)/(np.shape(self.traindata)[0])

		
	# This function calculates the accuracy of the network on
	# the test data using the sgn function
	def test_accuracy_sgn(self):
		total = 0
		for i in self.testdata:
			self.netx[0] = i[1:]
			self.propagate_sgn()
			if (np.argmax(self.netx[self.numl-1]) == i[0]):
				total = total + 1
		self.test_accuracy = (0.0+total)/(np.shape(self.testdata)[0])


        #This function prepares target vectors for neural network stock
        # market applications. Prepares training vector
        def stochastic_stock_prep(self):
           self.traintarget = np.zeros((np.shape(self.traindata)[0],123));
           for i in range(0,np.shape(self.traindata)[0]-1):
              rowsum = 0;
              for j in range(0,123):
                 if self.traindata[i+1][-123+j:] > 0:
                    self.traintarget[i][j] = self.traindata[i+1][-123+j:];
                    rowsum = rowsum + self.traindata[i+1][-123+j:];
                 self.traintarget[k][:] = self.traintarget[k][:]/rowsum;
                    

        #This function prepares target vectors for neural network stock
        # market applications prepares the test vector
        def stochastic_stock_prep_test(self):
           self.testtarget = np.zeros((np.shape(self.testdata)[0],123));
           k = 0;
           l = 0;
           for i in self.testdata:
              rowsum = 0;
              for j in i:
                 if j > 0:
                    self.testtarget[k][l] = j;
                    rowsum = rowsum + j;
                 l = l + 1;
              k = k + 1;
              l = 0;

	# This function trains using stochastic gradient descent
	# for 1 epoch with target vectors [.1 .1 .9 .1 ...]
	def stochastic_stocks(self):
                self.stochastic_stock_prep();
                currentc = 0; 
		for i in self.traindata:
			self.netx[0] = i[1:]
			self.propagate_sigmoid()
			# calculates the delta for the last layer
			self.netd[-1] = self.netx[-1]*(1-self.netx[-1])*(self.traintarget[currentc]-self.netx[-1])
			# calculates the delta for all other layers
			self.netd[-2] = self.netx[-2]*(1-self.netx[-2])*(np.dot(self.netw[-1].T,self.netd[-1])) 
			for j in range(0,self.numl-2):
				self.netd[-j-3] = self.netx[-j-3]*(1-self.netx[-j-3])*(np.dot(self.netw[-j-2].T,self.netd[-j-2][:-1]))
			# Now we update all of the weights based on the deltas, etas, and alphas
			for j in range(0,self.numl-2):
				# computes the momentum including delta w
				self.netwd[j] = self.eta*np.outer(self.netd[j+1][:-1],self.netx[j]) + self.alpha*self.netwd[j]
				# updates the weights
				self.netw[j] = self.netw[j] + self.netwd[j]	
			# computes the momentum including delta w
			self.netwd[-1] = self.eta*np.outer(self.netd[-1],self.netx[-2]) + self.alpha*self.netwd[-1]
			# updates the weights
			self.netw[-1] = self.netw[-1] + self.netwd[-1]
                        currentc = currentc + 1;
	


        # This function trains using stochastic gradient descent
	# for 1 epoch with target vectors [.1 .1 .9 .1 ...]
	def stochastic_gradient_sigmoid(self):
		for i in self.traindata:
			self.netx[0] = i[1:]
			self.propagate_sigmoid()
			# calculates the delta for the last layer
			self.netd[-1] = self.netx[-1]*(1-self.netx[-1])*(self.target[int(i[0])]-self.netx[-1])
			# calculates the delta for all other layers
			self.netd[-2] = self.netx[-2]*(1-self.netx[-2])*(np.dot(self.netw[-1].T,self.netd[-1])) 
			for j in range(0,self.numl-2):
				self.netd[-j-3] = self.netx[-j-3]*(1-self.netx[-j-3])*(np.dot(self.netw[-j-2].T,self.netd[-j-2][:-1]))
			# Now we update all of the weights based on the deltas, etas, and alphas
			for j in range(0,self.numl-2):
				# computes the momentum including delta w
				self.netwd[j] = self.eta*np.outer(self.netd[j+1][:-1],self.netx[j]) + self.alpha*self.netwd[j]
				# updates the weights
				self.netw[j] = self.netw[j] + self.netwd[j]	
			# computes the momentum including delta w
			self.netwd[-1] = self.eta*np.outer(self.netd[-1],self.netx[-2]) + self.alpha*self.netwd[-1]
			# updates the weights
			self.netw[-1] = self.netw[-1] + self.netwd[-1]

	# This function trains just one layer at a time
	def stochastic_gradient_sigmoid_2(self,layer):
		for i in self.traindata:
			self.netx[0] = i[1:]
			self.propagate_sigmoid()
			# calculates the delta for the last layer
			self.netd[-1] = (self.target[int(i[0])]-self.netx[-1])*self.netx[-1]*(1-self.netx[-1])
			# calculates the delta for all other layers
			self.netd[-2] = self.netx[-2]*(1-self.netx[-2])*(np.dot(self.netw[-1].T,self.netd[-1])) 
			for j in range(0,self.numl-2):
				self.netd[-j-3] = self.netx[-j-3]*(1-self.netx[-j-3])*(np.dot(self.netw[-j-2].T,self.netd[-j-2][:-1]))
			# Now we update all of the weights based on the deltas, etas, and alphas
			for j in range(0,self.numl-2):
				# computes the momentum including delta w
				self.netwd[j] = self.eta*np.outer(self.netd[j+1][:-1],self.netx[j]) + self.alpha*self.netwd[j]
				# updates the weights
				if j == layer:
					self.netw[j] = self.netw[j] + self.netwd[j]	
			# computes the momentum including delta w
			self.netwd[-1] = self.eta*np.outer(self.netd[-1],self.netx[-2]) + self.alpha*self.netwd[-1]
			if layer == self.numl-2:
				# updates the weights
				self.netw[-1] = self.netw[-1] + self.netwd[-1]


	# This function trains using stochastic gradient descent 
	# for 1 epoch with target vectors [0 0 1 0 0 0 0 ...]
	def stochastic_gradient_sigmoid_1(self):
		temp = np.eye(self.netl[self.numl-1])
		for i in self.traindata:
			self.netx[0] = i[1:]
			self.propagate_sigmoid()
			# calculates the delta for the last layer
			self.netd[-1] = self.netx[-1]*(1-self.netx[-1])
			self.netd[-1] = self.netd[-1]*(temp[int(i[0])]-self.netx[-1])
			# calculates the delta for all other layers
			self.netd[-2] = self.netx[-2]*(1-self.netx[-2])
			self.netd[-2] = self.netd[-2]*(np.dot(self.netw[-1].T,self.netd[-1])) 
			for j in range(0,self.numl-2):
				self.netd[-j-3] = self.netx[-j-3]*(1-self.netx[-j-3])
				self.netd[-j-3] = self.netd[-j-3]*(np.dot(self.netw[-j-2].T,self.netd[-j-2][:-1]))
			# Now we update all of the weights based on the deltas, etas, and alphas
			for j in range(0,self.numl-2):
				# computes the momentum including delta w
				self.netwd[j] = self.eta*np.outer(self.netd[j+1][:-1],self.netx[j]) + self.alpha*self.netwd[j]
				# updates the weights
				self.netw[j] = self.netw[j] + self.netwd[j]	
			# computes the momentum including delta w
			self.netwd[-1] = self.eta*np.outer(self.netd[-1],self.netx[-2]) + self.alpha*self.netwd[-1]
			# updates the weights
			self.netw[-1] = self.netw[-1] + self.netwd[-1]

	

	# This function calculates and stores the confusion matrix with sigmoid function
	def conf_train_sigmoid(self):
		self.conf_train = np.zeros((self.netl[self.numl-1],self.netl[self.numl-1]))
		for i in self.traindata:
			self.netx[0] = i[1:]
			self.propagate_sigmoid()
			temp = np.argmax(self.netx[self.numl-1])
			self.conf_train[int(i[0])][temp] += 1
		
			
	# This function calculates and stores the confusion matrix with sigmoid function
	def conf_test(self):
		self.conf_test = np.zeros((self.netl[self.numl-1],self.netl[self.numl-1]))
		for i in self.testdata:
			self.netx[0] = i[1:]
			self.propagate_sigmoid()
			temp = np.argmax(self.netx[self.numl-1])
			self.conf_test[int(i[0])][temp] += 1

	# This function uses inverse covariance matrix to calculate the weights for the
	# last layer of the neural network, using training data
	def calc_last_layer3(self):
		self.covariancep = np.zeros((self.netl[self.numl-2],self.netl[self.numl-2],self.netl[self.numl-1]))
		self.covariancen = np.zeros((self.netl[self.numl-2],self.netl[self.numl-2],self.netl[self.numl-1]))
		self.meanp = np.zeros((self.netl[self.numl-2],self.netl[self.numl-1]))
		self.meann = np.zeros((self.netl[self.numl-2],self.netl[self.numl-1]))
		self.mean = np.zeros((self.netl[self.numl-2],self.netl[self.numl-1]))
		self.c = np.zeros((self.netl[self.numl-2],self.netl[self.numl-2],self.netl[self.numl-1]))
		self.cinv = np.zeros((self.netl[self.numl-2],self.netl[self.numl-2],self.netl[self.numl-1]))
		self.ptot = np.zeros(self.netl[self.numl-1])
		self.ntot = np.zeros(self.netl[self.numl-1])
		for i in self.traindata:
			self.netx[0] = i[1:]
			self.propagate_sigmoid()
			for k in range(0,self.netl[self.numl-1]):
				if i[0] == k:
					self.ptot[k] += 1
					self.covariancep[:,:,k] += np.outer(self.netx[self.numl-2],self.netx[self.numl-2])
					self.meanp[:,k] += self.netx[self.numl-2]
				else :
					self.ntot[k] += 1
					self.covariancen[:,:,k] += np.outer(self.netx[self.numl-2],self.netx[self.numl-2])
					self.meann[:,k] += self.netx[self.numl-2]
		for k in range(0,self.netl[self.numl-1]):
			self.covariancep[:,:,k] = self.covariancep[:,:,k]/(self.ptot[k]) - np.outer(self.meanp[:,k],self.meanp[:,k])/((self.ptot[k])*(self.ptot[k]))
			self.covariancen[:,:,k] = self.covariancen[:,:,k]/(self.ntot[k]) - np.outer(self.meann[:,k],self.meann[:,k])/((self.ntot[k])*(self.ntot[k]))
			self.mean[:,k] = self.meanp[:,k]/(self.ptot[k]) - self.meann[:,k]/(self.ntot[k])

		self.c = self.covariancep+self.covariancen
		# we begin by splitting the training data into 10 classes one
		# for each digit
		self.numnums = np.zeros(self.netl[-1])
		for i in self.traindata:
			self.numnums[int(i[0])] += 1
		# Creates an array for each of the classes
		self.classes = [0,]*self.netl[-1]
		for i in range(0,self.netl[-1]):
			self.classes[i] = np.zeros((int(self.numnums[i]),np.shape(self.traindata)[1]))
		temp = np.zeros(self.netl[-1])
		for i in self.traindata:
			self.classes[int(i[0])][int(temp[int(i[0])])] = i
			temp[int(i[0])] += 1
		# We will start with the zero weight vector
		self.covw = np.zeros((self.netl[self.numl-1],self.netl[self.numl-2]))
		for epoch in range(0,epochs):
			self.covw += self.covw + (self.c*self.covw - self.mean)
		#for k in range(0,self.netl[self.numl-1]):
		#	self.cinv[:,:,k] = np.linalg.inv(self.c[:,:,k])
		#
		#self.covw = np.zeros((self.netl[self.numl-1],self.netl[self.numl-2]))
		#for k in range(0,self.netl[self.numl-1]):
		#	self.covw[k,:] = np.dot(self.cinv[:,:,k],self.mean[:,k])

	# This function computes the weight matrix for the last layer using
	# gradient descent so as to maximize the area under the roc_auc curve
	# using W^{t+1} = W^{t} - gamma_t(C^{hat}W^{t}-mu^{hat})
	# where C^{hat}= 1/(n-1)(sum(Z*Z^T)-1/n^2sum(Z)sum(Z)^T)
	def calc_last_layer_gradient(self,batchs,epochs):
		# we begin by splitting the training data into 10 classes one
		# for each digit
		self.numnums = np.zeros(self.netl[-1])
		for i in self.traindata:
			self.numnums[int(i[0])] += 1
		# Creates an array for each of the classes
		self.classes = [0,]*self.netl[-1]
		for i in range(0,self.netl[-1]):
			self.classes[i] = np.zeros((int(self.numnums[i]),np.shape(self.traindata)[1]))
		temp = np.zeros(self.netl[-1])
		for i in self.traindata:
			self.classes[int(i[0])][int(temp[int(i[0])])] = i
			temp[int(i[0])] += 1
		# We will start with the zero weight vector
		self.covw = np.zeros((self.netl[self.numl-1],self.netl[self.numl-2]))
		z = np.zeros((self.netl[self.numl-1],self.netl[self.numl-2]))
		c = np.zeros(self.netl[self.numl-1])
		cov = np.zeros((self.netl[self.numl-1],self.netl[self.numl-2]))
		mean = np.zeros((self.netl[self.numl-1],self.netl[self.numl-2]))
		# We attempt to scale all the output weight vectors so the mean
		# of the positive class is sent to -1 and the mean of the negative
		# input vector is sent to +1. This way, the outputs should be more
		# comparable
		self.meanpt = np.zeros((self.netl[self.numl-1],self.netl[self.numl-2]))
		self.meannt = np.zeros((self.netl[self.numl-1],self.netl[self.numl-2]))
		self.b = np.ones(self.netl[self.numl-1])
		for epoch in range(0,epochs):
			for i in range(0,int(np.floor(self.numnums[0]/batchs))):
				for j in range(0,batchs):
					for k in range(0,self.netl[-1]):
						randomc = np.random.randint(self.netl[-1]-1)
						randomc = (k + randomc)%(self.netl[-1])
						self.netx[0] = self.classes[k][i,1:]
						self.propagate_sigmoid()
						xp = copy.copy(self.netx[self.numl-2])
						self.meanpt[k] += xp 
						self.netx[0] = self.classes[randomc][i,1:]
						self.propagate_sigmoid()
						xn = self.netx[self.numl-2]
						self.meannt[k] += xn
						z[k,:] = xp - xn
						c[k] = np.dot(z[k,:],self.covw[k,:])
					cov += z*c[:,None]
					mean += z
				for k in range(0,self.netl[-1]):
					mean[k,:] = (batchs/(batchs-1.0)*1.0/(batchs*batchs)*np.dot(mean[k,:],self.covw[k,:]) + self.b[k]*1.0/batchs)*mean[k,:]
				cov = 1.0/(batchs-1)*cov
				self.covw = self.covw - 1.0/(i+1+epoch*np.floor(self.numnums[0]/batchs))*(cov - mean)								
				# zero the mean and cov
				cov = 0*cov
				mean = 0*mean
				# scales the view vectors by their magnitudes
				for k in range(0,self.netl[-1]):
					blittle = (1./abs(np.linalg.norm(self.covw[k,:])))
					self.covw[k,:] = self.covw[k,:]*blittle
					self.b[k] = self.b[k]*blittle
			for i in range(0,self.netl[-1]):
				np.random.shuffle(self.classes[i])

	# This function computes the weight matrix for the last layer using
	# gradient descent so as to maximize the area under the roc_auc curve
	# using W^{t+1} = W^{t} - gamma_t(C^{hat}W^{t}-mu^{hat})
	# where C^{hat}= 1/(n-1)(sum(Z*Z^T)-1/n^2sum(Z)sum(Z)^T)
	def calc_last_layer_gradient2(self,batchs,epochs):
		# we begin by splitting the training data into 10 classes one
		# for each digit
		self.numnums = np.zeros(self.netl[-1])
		for i in self.traindata:
			self.numnums[int(i[0])] += 1
		# Creates an array for each of the classes
		self.classes = [0,]*self.netl[-1]
		for i in range(0,self.netl[-1]):
			self.classes[i] = np.zeros((int(self.numnums[i]),np.shape(self.traindata)[1]))
		temp = np.zeros(self.netl[-1])
		for i in self.traindata:
			self.classes[int(i[0])][int(temp[int(i[0])])] = i
			temp[int(i[0])] += 1
		# We will start with the zero weight vector
		self.covw = np.zeros((self.netl[self.numl-1],self.netl[self.numl-2]))
		zp = np.zeros((self.netl[self.numl-1],self.netl[self.numl-2]))
		zn = np.zeros((self.netl[self.numl-1],self.netl[self.numl-2]))	
		cp = np.zeros(self.netl[self.numl-1])
		cn = np.zeros(self.netl[self.numl-1])
		cov = np.zeros((self.netl[self.numl-1],self.netl[self.numl-2]))
		mean = np.zeros((self.netl[self.numl-1],self.netl[self.numl-2]))
		# sums of positives and negatives
		self.meanpt = np.zeros((self.netl[self.numl-1],self.netl[self.numl-2]))
		self.meannt = np.zeros((self.netl[self.numl-1],self.netl[self.numl-2]))
		# unit conversion constant
		self.b = np.ones(self.netl[self.numl-1])
		for epoch in range(0,epochs):
			for i in range(0,int(np.floor(self.numnums[0]/batchs))):
				for j in range(0,batchs):
					for k in range(0,self.netl[-1]):
						randomc = np.random.randint(self.netl[-1]-1)
						randomc = (k + randomc)%(self.netl[-1])
						self.netx[0] = self.classes[k][i,1:]
						self.propagate_sigmoid()
						xp = copy.copy(self.netx[self.numl-2])
						self.meanpt[k] += xp 
						self.netx[0] = self.classes[randomc][i,1:]
						self.propagate_sigmoid()
						xn = self.netx[self.numl-2]
						self.meannt[k] += xn
						zp[k,:] = xp
						zn[k,:] = xn
						cp[k] = np.dot(zp[k,:],self.covw[k,:])
						cn[k] = np.dot(zn[k,:],self.covw[k,:])
					cov += zp*cp[:,None] + zn*cn[:,None]
					mean += zp - zn
				for k in range(0,self.netl[-1]):
					#mean[k,:] = (batchs/(batchs-1.0)*1.0/(batchs*batchs)*np.dot(mean[k,:],self.covw[k,:]) + self.b[k]*1.0/batchs)*mean[k,:]
					mean[k,:] = self.b[k]*1.0/batchs*mean[k,:]
					cov[k,:] = cov[k,:]-1.0/batchs*(self.meanpt[k,:]*np.dot(self.meanpt[k,:],self.covw[k,:])+self.meannt[k,:]*np.dot(self.meannt[k,:],self.covw[k,:]))
				cov = 1.0/(batchs-1)*cov
				self.covw = self.covw - 1.0/(i+1+epoch*np.floor(self.numnums[0]/batchs))*(cov - mean)								
				# zero the mean and cov
				cov = 0*cov
				mean = 0*mean
				self.meanpt = 0*self.meanpt
				self.meannt = 0*self.meannt
				# scales the view vectors by their magnitudes
				for k in range(0,self.netl[-1]):
					blittle = (1./abs(np.linalg.norm(self.covw[k,:])))
					self.covw[k,:] = self.covw[k,:]*blittle
					self.b[k] = self.b[k]*blittle
			for i in range(0,self.netl[-1]):
				np.random.shuffle(self.classes[i])

	# This function uses svm to calculate the weight vectors. This needs to be improved to
        # work for second to last layer so it can live at the end of a neuralnet
	def calc_last_layer_svm(self):
		temp = 0
		self.train_target = np.zeros((self.netl[-1],np.shape(self.traindata)[0]))
		for i in range(0,self.netl[-1]):
			self.train_target[i][0:int(self.numnums[i])] = 1
		#self.trainsvmdata = np.zeros((self.netl[-1],np.shape(self.traindata)[0],np.shape(self.traindata)[1]))
		self.trainsvmdata = np.zeros((np.shape(self.traindata)[0],np.shape(self.traindata)[1]-1))
		self.svmw = np.zeros((self.netl[-1],self.netl[-2]))
		self.svm = svm.SVC(kernel='linear')
		for i in range(0,self.netl[-1]):
			self.trainsvmdata[0:int(self.numnums[i])] = self.classes[i][:,0:-1]
			temp = int(self.numnums[i])
			for j in range(0,self.netl[-1]):
				if j != i:
					self.trainsvmdata[temp:temp+int(self.numnums[j])] = self.classes[j][:,0:-1]
					temp += int(self.numnums[j])
			
			self.svm.fit(self.trainsvmdata,self.train_target[i])
			self.svmw[i] = copy.copy(self.svm.coef_)
			print i
			
	def sigmoid2(self,mat):
		mat[:,:-1] = 1.0/(1+np.exp(-mat[:,:-1]))
		return mat

	def sigmoid3(self,mat):
		mat = 1.0/(1+np.exp(-mat))
		return mat

	# apply sigmoid function to the num'th vector of self.netx  
	def sigmoid(self,num):
		if num < self.numl-1:
			self.netx[num][:-1] = 1.0/(1+np.exp(-self.netx[num][:-1]))
		else :
			self.netx[num] = 1.0/(1+np.exp(-self.netx[num]))	

	# This function will use gradient descent but increase the presence of
	# negatively classified instances in the training set.
	def stochastic_gradient_sigmoid_e(self,epochs):
		self.temp = self.traindata[:]
		self.shuffle()
		self.stochastic_gradient_sigmoid()
		for j in range(0,epochs):
			self.traindata = self.temp[:]
			self.train_accuracy_nice()
			print self.train_accuracy
			numerrors = np.sum(self.errors)
			self.newdata = np.zeros((int(np.sum(self.errors)),np.shape(self.traindata)[1]))
			countolaf = 0
			for i in range(0,np.shape(self.traindata)[0]):
				if self.errors[i] == 1:
					self.newdata[countolaf] = self.traindata[i]
			np.vstack((self.traindata,self.newdata))
			self.shuffle()
			self.traindata = self.traindata[:int(-numerrors)]
			print np.shape(self.traindata)
		self.traindata = self.temp[:]

	def train_accuracy_nice(self):
		self.tempx1 = np.dot(self.traindata[:,1:],self.netw[0].T)
		self.errors = np.zeros(np.shape(self.traindata)[0])
		for i in range(1,self.numl-1):
			print np.shape(self.tempx1)
			self.tempx1 = np.concatenate((self.tempx1,np.ones((np.shape(self.traindata)[0],1))),axis=1)
			print np.shape(self.tempx1)
			self.tempx1 = self.sigmoid2(self.tempx1)
			print np.shape(self.tempx1)
			self.tempx1 = np.dot(self.tempx1,self.netw[i].T)
		self.sigmoid3(self.tempx1)
		count = 0
		for i in range(0,np.shape(self.tempx1)[0]):
			if np.argmax(self.tempx1[i,:]) != self.traindata[i,0]:
				self.errors[i] = 1
			else :
				count += 1
		self.train_accuracy = (count+0.0)/(np.shape(self.traindata)[0]) 
	

	def test_accuracy_nice(self):
		self.tempx1 = np.dot(self.testdata[:,1:],self.netw[0].T)
		self.errors = np.zeros(np.shape(self.testdata)[0])
		for i in range(1,self.numl-1):
			print np.shape(self.tempx1)
			self.tempx1 = np.concatenate((self.tempx1,np.ones((np.shape(self.testdata)[0],1))),axis=1)
			print np.shape(self.tempx1)
			self.tempx1 = self.sigmoid2(self.tempx1)
			print np.shape(self.tempx1)
			self.tempx1 = np.dot(self.tempx1,self.netw[i].T)
		self.sigmoid3(self.tempx1)
		count = 0
		for i in range(0,np.shape(self.tempx1)[0]):
			if np.argmax(self.tempx1[i,:]) != self.testdata[i,0]:
				self.errors[i] = 1
			else :
				count += 1
		self.test_accuracy = (count+0.0)/(np.shape(self.testdata)[0]) 

	# This function uses inverse covariance matrix to calculate the weights for the
	# last layer of the neural network, using training data
	def calc_last_layer(self):
		self.covariancep = np.zeros((self.netl[self.numl-2],self.netl[self.numl-2],self.netl[self.numl-1]))
		self.covariancen = np.zeros((self.netl[self.numl-2],self.netl[self.numl-2],self.netl[self.numl-1]))
		self.meanp = np.zeros((self.netl[self.numl-2],self.netl[self.numl-1]))
		self.meann = np.zeros((self.netl[self.numl-2],self.netl[self.numl-1]))
		self.mean = np.zeros((self.netl[self.numl-2],self.netl[self.numl-1]))
		self.c = np.zeros((self.netl[self.numl-2],self.netl[self.numl-2],self.netl[self.numl-1]))
		self.cinv = np.zeros((self.netl[self.numl-2],self.netl[self.numl-2],self.netl[self.numl-1]))
		self.ptot = np.zeros(self.netl[self.numl-1])
		self.ntot = np.zeros(self.netl[self.numl-1])
		for i in self.traindata:
			self.netx[0] = i[1:]
			self.propagate_sigmoid()
			for k in range(0,self.netl[self.numl-1]):
				if i[0] == k:
					self.ptot[k] += 1
					self.covariancep[:,:,k] += np.outer(self.netx[self.numl-2],self.netx[self.numl-2])
					self.meanp[:,k] += self.netx[self.numl-2]
				else :
					self.ntot[k] += 1
					self.covariancen[:,:,k] += np.outer(self.netx[self.numl-2],self.netx[self.numl-2])
					self.meann[:,k] += self.netx[self.numl-2]
		for k in range(0,self.netl[self.numl-1]):
			self.covariancep[:,:,k] = self.covariancep[:,:,k]/(self.ptot[k]) - np.outer(self.meanp[:,k],self.meanp[:,k])/((self.ptot[k])*(self.ptot[k]))
			self.covariancen[:,:,k] = self.covariancen[:,:,k]/(self.ntot[k]) - np.outer(self.meann[:,k],self.meann[:,k])/((self.ntot[k])*(self.ntot[k]))
			self.mean[:,k] = self.meanp[:,k]/(self.ptot[k]) - self.meann[:,k]/(self.ntot[k])

		self.c = self.covariancep+self.covariancen
		for k in range(0,self.netl[self.numl-1]):
			self.cinv[:,:,k] = np.linalg.inv(self.c[:,:,k])

		self.covw = np.zeros((self.netl[self.numl-1],self.netl[self.numl-2]))
		for k in range(0,self.netl[self.numl-1]):
			self.covw[k,:] = np.dot(self.cinv[:,:,k],self.mean[:,k])

	# This function computes the roc_auc_cov over the training set for the output node specified by the
	# parameter
	def roc_auc_train_cov(self,vals):
		x = np.zeros((np.shape(self.traindata)[0],np.size(vals)))
		y = np.zeros((np.shape(self.traindata)[0],np.size(vals)))
		j = 0 
		for i in self.traindata:
			self.netx[0] = i[1:]
			self.propagate_sigmoid()
			self.netx[self.numl - 1] = np.dot(self.covw,self.netx[self.numl - 2])
			for k in range(0,np.size(vals)):
				if i[0] == vals[k]:
					x[j][k] = 1
				y[j][k] = self.netx[self.numl-1][vals[k]]
			j = j + 1

		self.train_auc = np.zeros(np.size(vals))	
		for k in range(0,np.size(vals)):			
			self.train_auc[k] = sklearn.metrics.roc_auc_score(x[:,k],y[:,k])
	
	# This function computes the roc_auc over the training set for the output node specified by the
	# parameter
	def roc_auc_test_cov(self,vals):
		x = np.zeros((np.shape(self.testdata)[0],np.size(vals)))
		y = np.zeros((np.shape(self.testdata)[0],np.size(vals)))
		j = 0 
		for i in self.testdata:
			self.netx[0] = i[1:]
			self.propagate_sigmoid()
			self.netx[self.numl - 1] = np.dot(self.covw,self.netx[self.numl - 2])
			for k in range(0,np.size(vals)):
				if i[0] == vals[k]:
					x[j][k] = 1
				y[j][k] = self.netx[self.numl-1][vals[k]]
			j = j + 1

		self.test_auc = np.zeros(np.size(vals))	
		for k in range(0,np.size(vals)):			
			self.test_auc[k] = sklearn.metrics.roc_auc_score(x[:,k],y[:,k])
		

	# This function computes the roc_auc over the training set for the output node specified by the
	# parameter
	def roc_auc_train(self,vals):
		x = np.zeros((np.shape(self.traindata)[0],np.size(vals)))
		y = np.zeros((np.shape(self.traindata)[0],np.size(vals)))
		j = 0 
		for i in self.traindata:
			self.netx[0] = i[1:]
			self.propagate_sigmoid()
			for k in range(0,np.size(vals)):
				if i[0] == vals[k]:
					x[j][k] = 1
				y[j][k] = self.netx[self.numl-1][vals[k]]
			j = j + 1

		self.train_auc = np.zeros(np.size(vals))	
		for k in range(0,np.size(vals)):			
			self.train_auc[k] = sklearn.metrics.roc_auc_score(x[:,k],y[:,k])
	
	# This function computes the roc_auc over the training set for the output node specified by the
	# parameter
	def roc_auc_test(self,vals):
		x = np.zeros((np.shape(self.testdata)[0],np.size(vals)))
		y = np.zeros((np.shape(self.testdata)[0],np.size(vals)))
		j = 0 
		for i in self.testdata:
			self.netx[0] = i[1:]
			self.propagate_sigmoid()
			for k in range(0,np.size(vals)):
				if i[0] == vals[k]:
					x[j][k] = 1
				y[j][k] = self.netx[self.numl-1][vals[k]]
			j = j + 1

		self.test_auc = np.zeros(np.size(vals))	
		for k in range(0,np.size(vals)):			
			self.test_auc[k] = sklearn.metrics.roc_auc_score(x[:,k],y[:,k])


	# This function computes the roc_auc_cov over the training set for the output node specified by the
	# parameter
	def roc_auc_train_svm(self,vals):
		x = np.zeros((np.shape(self.traindata)[0],np.size(vals)))
		y = np.zeros((np.shape(self.traindata)[0],np.size(vals)))
		j = 0 
		for i in self.traindata:
			self.netx[0] = i[1:]
			self.propagate_sigmoid()
			self.netx[self.numl - 1] = np.dot(self.svmw,self.netx[self.numl - 2])
			for k in range(0,np.size(vals)):
				if i[0] == vals[k]:
					x[j][k] = 1
				y[j][k] = self.netx[self.numl-1][vals[k]]
			j = j + 1

		self.train_auc = np.zeros(np.size(vals))	
		for k in range(0,np.size(vals)):			
			self.train_auc[k] = sklearn.metrics.roc_auc_score(x[:,k],y[:,k])
	
	# This function computes the roc_auc over the training set for the output node specified by the
	# parameter
	def roc_auc_test_svm(self,vals):
		x = np.zeros((np.shape(self.testdata)[0],np.size(vals)))
		y = np.zeros((np.shape(self.testdata)[0],np.size(vals)))
		j = 0 
		for i in self.testdata:
			self.netx[0] = i[1:]
			self.propagate_sigmoid()
			self.netx[self.numl - 1] = np.dot(self.svmw,self.netx[self.numl - 2])
			for k in range(0,np.size(vals)):
				if i[0] == vals[k]:
					x[j][k] = 1
				y[j][k] = self.netx[self.numl-1][vals[k]]
			j = j + 1

		self.test_auc = np.zeros(np.size(vals))	
		for k in range(0,np.size(vals)):			
			self.test_auc[k] = sklearn.metrics.roc_auc_score(x[:,k],y[:,k])
        ###############################################################################################
        # The stuff in the neuralnet class that follows this comment will be used to implement the
	# kernalized version of the binary classifier that optimizes the area under the roc curve.
        # the hypothesis h(x) given the sample x, and training vectors {x_i} is computed with: 
        # h(x) = sum_{i=1}^{n}alpha_{i}K(x_i,x). Notice, to compute the hypothesis, we need to do the
        # computational equivalent of n dot products of d dimensional vectors where n is the number of
        # training points and d is the dimension of the training points. We will assume the user wants
        # the input vectors to these to be the first layer of the neuralnetwork, if the user
        # wants the input to be a hidden layer of the neuralnetwork, then the user must copy the hidden
        # layer to the first layer. To do all computations we need, at a bare minimum, the K_{i} 
        # vectors and the K_{+} and K_{-} vectors. We may do the computations without using a lot of
        # memory, at the expense of computational compexity, or we may save on computational complexity
        # by n^2 storage (for the mnist data, for instance, this amounts to about 10Gb). In either case
        # we need to compute K_{-} and K_{+}.
        ###############################################################################################
        # Computes the polynomial kernel of the vector stored in self.vec1 and self.vec2 and stores the 
        # answer into self.a1
        def linear_kernel_vec(self,n,c):
           self.a1 = (np.dot(self.vec1,self.vec1)+c)**n
        # Computes the linear kernel of the vector stored in self.vec1 with every vector stored in
        # self.mt1 and stores the resulting vector into self.veca1
        def linear_kernel_mt1(self,n,c):
           self.veca1 = (np.dot(self.mt1,self.vec1)+c)**n
        # Computes the linear kernel of every vector stored in self.mt2 with every vector stored in
        # self.mt1 and stores the resulting matrix into self.mta1
        def linear_kernel_mt2(self,n,c):
           self.mta1 = (np.dot(self.mt1,self.mt2)+c)**n
        # Computes the polynomial kernel of the vector stored in self.vec1 and self.vec2 and stores the 
        # answer into self.a1
        def tanh_kernel_vec(self,n,c):
           self.a1 = np.tanh((1.0/n)*np.dot(self.vec1,self.vec1)+c)
        # Computes the linear kernel of the vector stored in self.vec1 with every vector stored in
        # self.mt1 and stores the resulting vector into self.veca1
        def tanh_kernel_mt1(self,n,c):
           self.veca1 = np.tanh((1.0/n)*np.dot(self.mt1,self.vec1)+c)
        # Computes the linear kernel of every vector stored in self.mt2 with every vector stored in
        # self.mt1 and stores the resulting matrix into self.mta1
        def tanh_kernel_mt2(self,n,c):
           self.mta1 = np.tanh((1.0/n)*np.dot(self.mt1,self.mt2)+c)
        # Computes kplus and kminus 5/6 of the training data will be used to create the alphas and
        # 1/6th of the data will be used to find lambda. 
        def compute_K_plus_minus(self,n,c):
           # we want it to concatenate
           self.n = (5*np.shape(self.traindata)[0])/6;
           self.traintrain = self.traindata[:self.n,1:]
           self.mt1 = self.traintrain
           self.traintest = self.traindata[self.n:,1:]
           self.kplus = np.zeros(self.n)
           self.kminus = np.zeros(self.n)
           # we can optimize the following comptation
           # by using a matrix matrix multiplication to compute
           # all the kernals, however, that would require storing
           # an n by n matrix where n is the number of training
           # points. So instead we will have a loop of matrix vector
           # multiplications
           self.posnum = 0
           self.negnum = 0
           for i in range(0,self.n):
              # place the vector in the appropriate register
              self.vec1 = self.traintrain[i]
              # replace this operation with the desired kernal function
              self.linear_kernel_mt1(n,c)
              # if in the negative class
              if self.traindata[i][0] == 0:
                 self.negnum += 1
                 self.kminus += self.veca1
              else :
                 self.kplus += self.veca1
                 self.posnum += 1
           self.kminus *= 1.0/self.negnum
           self.kplus *= 1.0/self.posnum     
        
        # Computes kplus and kminus 5/6 of the training data will be used to create the alphas and
        # 1/6th of the data will be used to find lambda. 
        def compute_all_tanh(self,n,c):
           # we want it to concatenate
           self.n = (5*np.shape(self.traindata)[0])/6;
           self.traintrain = self.traindata[:self.n,1:]
           self.mt1 = self.traintrain
           self.traintest = self.traindata[self.n:,1:]
           self.mt2 = self.traintrain.T
           self.tanh_kernel_mt2(n,c)
           print np.shape(self.mta1)
           self.k = self.mta1
           # we can optimize the following comptation
           # by using a matrix matrix multiplication to compute
           # all the kernals, however, that would require storing
           # an n by n matrix where n is the number of training
           # points. So instead we will have a loop of matrix vector
           # multiplications
           self.posnum = int(np.sum(self.traindata[:self.n,0])) 
           self.negnum = self.n - self.posnum
           print self.posnum 
           self.traintemp = np.zeros(np.shape(self.k))
           self.posind = 0
           self.negind = self.posnum
           for i in range(0,self.n):
              if self.traindata[i][0] == 1:
                 self.traintemp[self.posind]=self.k[i]
                 self.posind += 1
              else :
                 self.traintemp[self.negind] = self.k[i]
                 self.negind += 1
           print self.posind
           print self.negind
           self.kplus = (1.0/self.posnum)*(self.traintemp[:self.posnum][:]).sum(axis = 0)
           self.kminus = (1.0/self.negnum)*(self.traintemp[self.posnum:][:]).sum(axis = 0)
           print np.shape(self.kplus)
           self.gp = np.dot((self.traintemp[:self.posnum,:]-self.kplus).T,(self.traintemp[:self.posnum,:]-self.kplus))
           self.gp = (1.0/self.posnum)*self.gp
           self.gm= np.dot((self.traintemp[self.posnum:,:]-self.kminus).T,(self.traintemp[self.posnum:,:]-self.kminus))
           self.gm = (1.0/self.negnum)*self.gm
           self.g = self.gp + self.gm    

        # Computes kplus and kminus 5/6 of the training data will be used to create the alphas and
        # 1/6th of the data will be used to find lambda. 
        def compute_all(self,n,c):
           # we want it to concatenate
           self.n = (5*np.shape(self.traindata)[0])/6;
           self.traintrain = self.traindata[:self.n,1:]
           self.mt1 = self.traintrain
           self.traintest = self.traindata[self.n:,1:]
           self.mt2 = self.traintrain.T
           self.linear_kernel_mt2(n,c)
           print np.shape(self.mta1)
           self.k = self.mta1
           # we can optimize the following comptation
           # by using a matrix matrix multiplication to compute
           # all the kernals, however, that would require storing
           # an n by n matrix where n is the number of training
           # points. So instead we will have a loop of matrix vector
           # multiplications
           self.posnum = int(np.sum(self.traindata[:self.n,0])) 
           self.negnum = self.n - self.posnum
           print self.posnum 
           self.traintemp = np.zeros(np.shape(self.k))
           self.posind = 0
           self.negind = self.posnum
           for i in range(0,self.n):
              if self.traindata[i][0] == 1:
                 self.traintemp[self.posind]=self.k[i]
                 self.posind += 1
              else :
                 self.traintemp[self.negind] = self.k[i]
                 self.negind += 1
           self.k = 0
           gc.collect()
           print self.posind
           print self.negind
           self.kplus = (1.0/self.posnum)*(self.traintemp[:self.posnum][:]).sum(axis = 0)
           self.kminus = (1.0/self.negnum)*(self.traintemp[self.posnum:][:]).sum(axis = 0)
           print np.shape(self.kplus)
           self.g = (1.0/self.posnum)*np.dot((self.traintemp[:self.posnum,:]-self.kplus).T,(self.traintemp[:self.posnum,:]-self.kplus))
           self.g += (1.0/self.negnum)*np.dot((self.traintemp[self.posnum:,:]-self.kminus).T,(self.traintemp[self.posnum:,:]-self.kminus))

        def evaluate_alpha(self,n,c):    
           self.mt1 = self.traintrain
           self.mt2 = self.traintest.T
           self.linear_kernel_mt2(n,c)
           self.h = np.dot(self.alpha,self.mta1)	
	   return sklearn.metrics.roc_auc_score(self.traindata[self.n:,0],self.h)
        
        def evaluate_alpha_tanh(self,n,c):    
           self.mt1 = self.traintrain
           self.mt2 = self.traintest.T
           self.tanh_kernel_mt2(n,c)
           self.h = np.dot(self.alpha,self.mta1)	
	   return sklearn.metrics.roc_auc_score(self.traindata[self.n:,0],self.h)
       
        def compute_alpha2(self,n,c):
           self.roc_auc = -1
           self.alpha = np.zeros(self.n)
           self.falpha = np.zeros(self.n)
           self.new_roc_auc = self.evaluate_alpha(n,c)
           while self.new_roc_auc >= self.roc_auc:
              self.falpha = self.alpha[:]
              self.roc_auc = self.new_roc_auc
              self.alpha = self.alpha - self.eta*(np.dot(self.g,self.alpha) - (self.kplus - self.kminus))
              self.new_roc_auc = self.evaluate_alpha(n,c)
              print self.new_roc_auc
       
         
        def compute_alpha3(self,n,c,l):
           self.roc_auc = -1
           self.alpha = np.dot(np.linalg.inv(.5*self.g+self.traintemp*l),(self.kplus - self.kminus)) 
           self.new_roc_auc = self.evaluate_alpha(n,c)
           print self.new_roc_auc

        def compute_alpha(self,n,c):
           self.roc_auc = -1
           self.alpha = np.zeros(self.n)
           self.new_roc_auc = self.evaluate_alpha(n,c)
           while self.new_roc_auc >= self.roc_auc:
              self.roc_auc = self.new_roc_auc
              self.Gpalpha = np.zeros(self.n)
              self.Gmalpha = np.zeros(self.n)
              for i in range(0,self.n):
                 self.vec1 = self.traintrain[i]
                 self.linear_kernel_mt1(n,c)
                 if self.traindata[i][0] == 0:
                    self.zp = self.veca1 - self.kplus
                    self.Gpalpha += self.zp*np.dot(self.zp,self.alpha)
                 elif self.traindata[i][0] == 1:
                    self.zm = self.veca1 - self.kminus
                    self.Gmalpha += self.zm*np.dot(self.zm,self.alpha) 
              self.Gpalpha *= (1.0)/(self.posnum*self.posnum)
              self.Gmalpha *= (1.0)/(self.negnum*self.negnum)
              self.alpha -= self.eta*(self.Gpalpha + self.Gmalpha - self.kplus + self.kminus)
              self.new_roc_auc = self.evaluate_alpha(n,c)
              print self.new_roc_auc 
                  
     




# This part of the code will be to try to test the idea of using neural networks
# to come up with numbers of hamming distance close to an inputs smallest factor	
def doPrimeStuff():
	from Crypto.Util import number
	target = open('primes.csv','w')
	target2 = open('primetest.csv','w')
	for i in range(0,6000):
		num1 = number.getPrime(500)
		num2 = number.getPrime(500)
		if num1 > num2:
			num3 = num1
			num1 = num2
			num2 = num3
		num3 = num2*num1
		target.write(str(num1)+','+str(num3)+','+str(num2)+"\n")
	for i in range(0,1000):
		num1 = number.getPrime(500)
		num2 = number.getPrime(500)
		if num1 > num2:
			num3 = num1
			num1 = num2
			num2 = num3
		num3 = num2*num1
		target2.write(str(num1)+','+str(num3)+','+str(num2)+"\n")
	target.close()
	target2.close()
	s = Neuralnet([1000,500,500],'primes.csv','primetest.csv')
	target = open('primes.csv','r')
	index = 0
	s.traindata = []
	s.testdata = []
	for i in target:
		s.traindata.append(i.split(","))
		index += 1
	target.close()
	target = open('primetest.csv')
	index = 0
	for i in target:
		s.testdata.append(i.split(","))
	target.close()
	count = 0
	secount = 0
	for i in s.traindata:
		for j in i:
			s.traindata[count][secount] = []
			s.traindata[count][secount] = np.array([int(x) for x in bin(long(j))[2:]])
			secount += 1
		count += 1
		secount = 0
	count = 0
	secount = 0
	for i in s.testdata:
		for j in i:
			s.testdata[count][secount] = []
			s.testdata[count][secount] = np.array([int(x) for x in bin(long(j))[2:]])
			secount += 1
		count += 1
		secount = 0
	s.target = np.zeros((7000,500))
	for i in range(0,6000):
		s.target[i,500-np.size(s.traindata[i][0]):] = s.traindata[i][0]
	for i in range(0,1000):
		s.target[i+6,500-np.size(s.testdata[i][0]):] = s.testdata[i][0]
	s.target2 = np.zeros((7000,500))
	for i in range(0,6000):
		s.target2[i,500-np.size(s.traindata[i][2]):] = s.traindata[i][2]
	for i in range(0,1000):
		s.target2[i+6,500-np.size(s.testdata[i][2]):] = s.testdata[i][2]
	s.traindata2 = np.zeros((6000,1000))
	for i in range(0,6000):
		s.traindata2[i,1000-np.size(s.traindata[i][1]):] = s.traindata[i][1]
	s.testdata2 = np.zeros((1000,1000))
	for i in range(0,1000):
		s.testdata2[i,1000-np.size(s.testdata[i][1]):] = s.testdata[i][1]
	s.traindata = s.traindata2
	s.testdata = s.testdata2
	return s

# Removes NANs and stores percent differences
def prepnan():
   data = (pd.read_csv('MV1E0081c.csv',',',header=None)).values;
   i = np.shape(data)[0]-1;
   for j in range(1,i):
      for k in range(0,np.shape(data)[1]):
         if math.isnan(float(data[i-1][k])): 
            data[i-1][k] = data[i][k];
         if ((float(data[i-1][k]) <= .0001) and (float(data[i-1][k]) >= -.0001)):
            print "here";
            data[i-1][k] = data[i][k];
      i = i - 1;
   print np.argmin(data[1:][:].astype(float));
   for j in range(1,np.shape(data)[0]-1):
      data[j][:] = 100*255*np.divide((data[j+1][:].astype(float) - data[j][:].astype(float)),data[j][:].astype(float));
   np.savetxt('prepreped.csv',data,delimiter=',',fmt='%s');

# more preprocessing
def prep():
   data = (pd.read_csv('prepreped.csv',',',header=None)).values;
   output = np.zeros((np.shape(data)[0]-10,np.shape(data)[1]*10+1));
   i = 1;
   for row in output:
      for j in range(0,10):
         output[i-1][1+j*(np.shape(data)[1]):1+(j+1)*(np.shape(data)[1])]=data[:][i+j];
      i = i + 1;
   for i in range(0,np.shape(output)[0]-1):
      output[i][0] = np.nanargmax(output[i+1][-122:]);
   np.savetxt('training.csv',output[0:3000][:],delimiter=',',fmt='%f');
   np.savetxt('testing.csv',output[3001:][:],delimiter=',',fmt='%f');

# more preprocessing
def prep2():
   data = (pd.read_csv('prepreped.csv',',',header=None)).values;
   output = np.zeros((np.shape(data)[0]-10,np.shape(data)[1]*10+1));
   i = 1;
   for row in output:
      for j in range(0,10):
         output[i-1][1+j*(np.shape(data)[1]):1+(j+1)*(np.shape(data)[1])]=data[:][i+j];
      i = i + 1;
   for i in range(0,np.shape(output)[0]-1):
      output[i][0] = i;
   np.savetxt('training.csv',output[0:3000][:],delimiter=',',fmt='%f');
   np.savetxt('testing.csv',output[3001:][:],delimiter=',',fmt='%f');

# assumes there are no skipped months
def preprocesstr():
        monthAr1 = [30]*12;
        monthAr1[0] = 31;
        monthAr1[1] = 28;
        monthAr1[2] = 31;
        monthAr1[4] = 31;
        monthAr1[6] = 31;
        monthAr1[7] = 31;
        monthAr1[9] = 31;
        monthAr1[11] = 31;
        monthAr2 = monthAr1[:];
        monthAr2[1] = 29;
        print(monthAr1);
        print(monthAr2);
        data = pd.read_csv('MV1E0081.csv',',',header=None);
        increment = 0;
        last = data[0][1];
        output = np.zeros(np.size(data[0][1:]));
        for row in data[0][1:]:
           if (increment == 0):
              output[increment] = 0;
           else:
              if (int(row[9:11]) == int(last[9:11])) and (int(row[0:4]) == int(last[0:4])):
                 tempdate = data[0][increment+1];
                 output[increment] = output[increment-1] + int(row[16:18]) - int(last[16:18]);
                 last = tempdate; 
              elif (int(row[0:4])%4 != 0):
                 tempdate = data[0][increment+1];
                 output[increment] = output[increment-1] + monthAr1[int(last[9:11])-1] - int(last[16:18]) + int(row[16:18]);
                 last = tempdate; 
              elif (int(row[0:4])%4 == 0) :
                 tempdate = data[0][increment+1];
                 output[increment] = output[increment-1] + monthAr2[int(last[9:11])-1] - int(last[16:18]) + int(row[16:18]);
                 last = tempdate;
           increment = increment + 1;
        output = output.astype(int); 
        np.savetxt('output.csv',output,delimiter=',',fmt='%d');
                

def stockStuff():
    import copy
    stocks = Neuralnet([1231,200,123],'training.csv','testing.csv');
    stocks.testdatacopy = copy.copy(stocks.testdata);
    stocks.traindatacopy = copy.copy(stocks.traindata);
    for i in range(0,1):
       stocks.shuffle();
       stocks.stochastic_gradient_sigmoid();
    stocks.total = 1;
    i = 0;
    for j in stocks.testdatacopy:
       stocks.netx[0] = j[1:];
       stocks.propagate_sigmoid();
       ## calculates the delta for the last layer
       #stocks.netd[-1] = stocks.netx[-1]*(1-stocks.netx[-1])*(stocks.target[int(j[0])]-stocks.netx[-1])
       ## calculates the delta for all other layers
       #stocks.netd[-2] = stocks.netx[-2]*(1-stocks.netx[-2])*(np.dot(stocks.netw[-1].T,stocks.netd[-1])) 
       #for j in range(0,stocks.numl-2):
       #   stocks.netd[-j-3] = stocks.netx[-j-3]*(1-stocks.netx[-j-3])*(np.dot(stocks.netw[-j-2].T,stocks.netd[-j-2][:-1]))
       #   # Now we update all of the weights based on the deltas, etas, and alphas
       #for j in range(0,stocks.numl-2):
       #	  # computes the momentum including delta w
       #	  stocks.netwd[j] = stocks.eta*np.outer(stocks.netd[j+1][:-1],stocks.netx[j]) + stocks.alpha*stocks.netwd[j]
       #	  # updates the weights
       #	  stocks.netw[j] = stocks.netw[j] + stocks.netwd[j]	
       ## computes the momentum including delta w
       #stocks.netwd[-1] = stocks.eta*np.outer(stocks.netd[-1],stocks.netx[-2]) + stocks.alpha*stocks.netwd[-1]
       ## updates the weights
       #stocks.netw[-1] = stocks.netw[-1] + stocks.netwd[-1]
       temp = stocks.netx[2].argsort()[-4:];
       print temp;
       const = 0;
       for k in temp:
          const = const + stocks.netx[2][k];
       ans = 0;
       for k in temp:
          ans = ans + (stocks.netx[2][int(k)]*stocks.testdatacopy[i+10][int(k)+2]/const)/100;
       if (ans < .5) and (ans > -.5):
          stocks.total = stocks.total*(1+ans);
       print stocks.testdatacopy[i][0];
       print ans;
       i = i + 1;
       if (i > np.shape(stocks.testdatacopy)[0] -20):
          return stocks;
    return stocks;



def stockStuff_2p():
    import copy
    stocks = Neuralnet([1231,200,123],'training.csv','testing.csv');
    stocks.testdatacopy = copy.copy(stocks.testdata);
    stocks.traindatacopy = copy.copy(stocks.traindata);
    # creates the target matrix and destroys the last testing datapoint
    stocks.target = np.ones((np.shape(stocks.traindata)[0]+np.shape(stocks.testdata)[0]-1,123))*.1;
    for i in range(0,np.shape(stocks.target)[0]):
       if (i < np.shape(stocks.traindata)[0] - 2):
          for j in range(0,123):
             if (stocks.traindata[i+1][-123+j] > 1):
                stocks.target[i][j] = .9;
       if (i >= np.shape(stocks.traindata)[0] - 2):
          for j in range(0,123):
             if (stocks.testdata[i - np.shape(stocks.traindata)[0]][-123+j] > 1):
                stocks.target[i][j] = .9;
 
    for i in range(0,10):
       stocks.shuffle();
       stocks.stochastic_gradient_sigmoid();
    stocks.total = 1;
    i = 0;
    for j in stocks.testdatacopy:
       stocks.netx[0] = j[1:];
       stocks.propagate_sigmoid();
       temp = stocks.netx[2].argsort()[-4:];
       print temp;
       const = 0;
       for k in temp:
          const = const + stocks.netx[2][k];
       ans = 0;
       for k in temp:
          ans = ans + (stocks.netx[2][int(k)]*stocks.testdatacopy[i+10][int(k)+2]/const)/100;
       if (ans < .5) and (ans > -.5):
          stocks.total = stocks.total*(1+ans);
       print ans;
       print stocks.total;
       i = i + 1;
       if (i > np.shape(stocks.testdatacopy)[0] -20):
          return stocks;
    return stocks;


def stockStuff_3p():
    import copy
    stocks = Neuralnet([1231,200,123],'training.csv','testing.csv');
    stocks.testdatacopy = copy.copy(stocks.testdata);
    stocks.traindatacopy = copy.copy(stocks.traindata);
    # creates the target matrix and destroys the last testing datapoint
    stocks.target = np.zeros((np.shape(stocks.traindata)[0]+np.shape(stocks.testdata)[0]-1,123));
    for i in range(0,np.shape(stocks.target)[0]):
       if (i < np.shape(stocks.traindata)[0] - 1):
          for j in range(0,123):
             if (stocks.traindata[i+1][-123+j] > 2):
                stocks.target[i][j] = 1;
       if (i >= np.shape(stocks.traindata)[0] - 1):
          for j in range(0,123):
             if (stocks.testdata[i - np.shape(stocks.traindata)[0]][-123+j] > 2):
                stocks.target[i][j] = 1;
 
    for i in range(0,20):
       stocks.shuffle();
       stocks.stochastic_gradient_sigmoid();
    stocks.total = 1;
    i = 0;
    for j in stocks.testdatacopy:
       if (i % 20 == 0) and (i != 0):
          stocks.traindata = copy.copy(stocks.traindatacopy);
          stocks.traindata[0:i] = copy.copy(stocks.testdata[0:i]);
          stocks.rand_weights();
          for zk in range(0,20):
             stocks.shuffle(); 
             stocks.stochastic_gradient_sigmoid();
       stocks.netx[0] = j[1:];
       stocks.propagate_sigmoid();
       temp = stocks.netx[2].argsort()[-4:];
       print temp;
       const = 0;
       for k in temp:
          const = const + stocks.netx[2][k];
       ans = 0;
       for k in temp:
          ans = ans + (stocks.netx[2][int(k)]*stocks.testdatacopy[i+10][int(k)+2]/const)/100;
       if (ans < .5) and (ans > -.5):
          stocks.total = stocks.total*(1+ans);
       print ans;
       print stocks.total;
       i = i + 1;
       if (i > np.shape(stocks.testdatacopy)[0] -20):
          return stocks;
    return stocks;


def stockStuff_5():
    import copy
    stocks = Neuralnet([1231,200,123],'training.csv','testing.csv');
    stocks.testdatacopy = copy.copy(stocks.testdata);
    stocks.traindatacopy = copy.copy(stocks.traindata);
    stocks2 = Neuralnet([1231,200,123],'training.csv','testing.csv');
    stocks2.testdatacopy = copy.copy(stocks2.testdata);
    stocks2.traindatacopy = copy.copy(stocks2.traindata);
    # creates the target matrix and destroys the last testing datapoint
    stocks.target = np.ones((np.shape(stocks.traindata)[0]+np.shape(stocks.testdata)[0]-1,123))*.1;
    for i in range(0,np.shape(stocks.target)[0]):
       if (i < np.shape(stocks.traindata)[0] - 2):
          for j in range(0,123):
             if (stocks.traindata[i+1][-123+j] > 2):
                stocks.target[i][j] = 1;
       if (i >= np.shape(stocks.traindata)[0] - 2):
          for j in range(0,123):
             if (stocks.testdata[i - np.shape(stocks.traindata)[0]][-123+j] > 2):
                stocks.target[i][j] = 1;
    stocks2.target = np.ones((np.shape(stocks2.traindata)[0]+np.shape(stocks2.testdata)[0]-1,123))*.1;
    for i in range(0,np.shape(stocks2.target)[0]):
       if (i < np.shape(stocks2.traindata)[0] - 2):
          for j in range(0,123):
             if (stocks2.traindata[i+1][-123+j] < -1):
                stocks2.target[i][j] = 1;
       if (i >= np.shape(stocks2.traindata)[0] - 2):
          for j in range(0,123):
             if (stocks2.testdata[i - np.shape(stocks2.traindata)[0]][-123+j] < -1):
                stocks2.target[i][j] = 1;
 
    for i in range(0,10):
       stocks.shuffle();
       stocks.stochastic_gradient_sigmoid();
    for i in range(0,10):
       stocks2.shuffle();
       stocks2.stochastic_gradient_sigmoid();
    stocks.total = 1;
    i = 0;
    for j in stocks.testdatacopy:
       stocks.netx[0] = j[1:];
       stocks2.netx[0] = j[1:];
       stocks.propagate_sigmoid();
       stocks2.propagate_sigmoid();
       temp = (stocks.netx[2] - stocks2.netx[2]).argsort()[-4:];
       print temp;
       const = 0;
       for k in temp:
          const = const + (stocks.netx[2][k] - stocks2.netx[2][k]);
       ans = 0;
       for k in temp:
          ans = ans + ((stocks.netx[2][int(k)] - stocks2.netx[2][k])*stocks.testdatacopy[i+10][int(k)+2]/const)/100;
       if (ans < .5) and (ans > -.5):
          stocks.total = stocks.total*(1+ans);
       print ans;
       print stocks.total;
       i = i + 1;
       if (i > np.shape(stocks.testdatacopy)[0] -20):
          return stocks;
    return stocks;


def stockStuff_6():
    import copy
    stocks = Neuralnet([1231,200,123],'training.csv','testing.csv');
    stocks.testdatacopy = copy.copy(stocks.testdata);
    stocks.traindatacopy = copy.copy(stocks.traindata);
    # creates the target matrix and destroys the last testing datapoint
    stocks.target = np.ones((np.shape(stocks.traindata)[0]+np.shape(stocks.testdata)[0]-1,123))*.1;
    for i in range(0,np.shape(stocks.target)[0]):
       if (i < np.shape(stocks.traindata)[0] - 2):
          for j in range(0,123):
             if ((stocks.traindata[i+1][-123+j] > 2) and (stocks.traindata[i+1][-123+j] < 20)):
                stocks.target[i][j] = 1;
       if (i >= np.shape(stocks.traindata)[0] - 2):
          for j in range(0,123):
             if ((stocks.testdata[i - np.shape(stocks.traindata)[0]][-123+j] > 2) and stocks.testdata[i-np.shape(stocks.traindata)[0]][-123+j] < 20):
                stocks.target[i][j] = 1;
 
    for i in range(0,10):
       stocks.shuffle();
       stocks.stochastic_gradient_sigmoid();
    
    stocks.total = 1;
    i = 0;
    temptotal = stocks.total;
    tempweights = copy.copy(stocks.netw);
    for k in range(0,10):
       stocks.netx[0] = stocks.testdatacopy[k][1:];
       stocks.propagate_sigmoid();
       temp = stocks.netx[2].argsort()[-4:];
       print temp;
       const = 0;
       for k in temp:
          const = const + stocks.netx[2][k];
       ans = 0;
       for k in temp:
          ans = ans + (stocks.netx[2][int(k)]*stocks.testdatacopy[i+10][int(k)+2]/const)/100;
       if (ans < .5) and (ans > -.5):
          stocks.total = stocks.total*(1+ans);
       print ans;
       print stocks.total;
       i = i + 1;
    print "test1";
    while (stocks.total < 1.10):
       if stocks.total < temptotal:
          stocks.netw = copy.copy(tempweights);
          stocks.total = temptotal;
       else :
          tempweights = copy.copy(stocks.netw);
          temptotal = stocks.total;
       for i in range(0,1):
          stocks.shuffle();
          stocks.stochastic_gradient_sigmoid();
       stocks.total = 1;
       i = 0;
       for k in range(0,10):
          stocks.netx[0] = stocks.testdatacopy[k][1:];
          stocks.propagate_sigmoid();
          temp = stocks.netx[2].argsort()[-4:];
          print temp;
          const = 0;
          for k in temp:
             const = const + stocks.netx[2][k];
          ans = 0;
          for k in temp:
             ans = ans + (stocks.netx[2][int(k)]*stocks.testdatacopy[i+10][int(k)+2]/const)/100;
          if (ans < .5) and (ans > -.5):
             stocks.total = stocks.total*(1+ans);
          print ans;
          print stocks.total;
          print "test";
          i = i + 1;
   
    print "out of test"; 
    stocks.total = 1;
    i = 0;
    for j in stocks.testdatacopy:
       if (i % 20 == 19):
          print "month";
       stocks.netx[0] = j[1:];
       stocks.propagate_sigmoid();
       temp = stocks.netx[2].argsort()[-4:];
       print temp;
       const = 0;
       for k in temp:
             const = const + stocks.netx[2][k];
       ans = 0;
       for k in temp:
             ans = ans + (stocks.netx[2][int(k)]*stocks.testdatacopy[i+10][int(k)+2]/const)/100;
       if (ans < .5) and (ans > -.5):
          stocks.total = stocks.total*(1+ans);
       print ans;
       print stocks.total;
       i = i + 1;
       if (i > np.shape(stocks.testdatacopy)[0] -20):
          return stocks;
    return stocks;

def stockStuff_7():
    import copy
    stocks = Neuralnet([1231,200,123],'training.csv','testing.csv');
    stocks.testdatacopy = copy.copy(stocks.testdata);
    stocks.traindatacopy = copy.copy(stocks.traindata);
    # creates the target matrix and destroys the last testing datapoint
    stocks.target = np.ones((np.shape(stocks.traindata)[0]+np.shape(stocks.testdata)[0]-1,123))*.1;
    for i in range(0,np.shape(stocks.target)[0]):
       if (i < np.shape(stocks.traindata)[0] - 2):
          for j in range(0,123):
             if ((stocks.traindata[i+1][-123+j] > 2) and (stocks.traindata[i+1][-123+j] < 20)):
                stocks.target[i][j] = 1;
       if (i >= np.shape(stocks.traindata)[0] - 2):
          for j in range(0,123):
             if ((stocks.testdata[i - np.shape(stocks.traindata)[0]][-123+j] > 2) and stocks.testdata[i-np.shape(stocks.traindata)[0]][-123+j] < 20):
                stocks.target[i][j] = 1;
    closest = np.zeros((20,2));
    prediction = np.zeros(123);
    i = 0;
    stocks.total = 1;
    for j in stocks.testdata:
       for k in stocks.traindata:
         temp = np.inner(j[1:],k[1:])/(np.linalg.norm(j[1:])*np.linalg.norm(k[1:]));
         if temp > closest[-1][0]:
            for l in range(0,20):
               if temp > closest[i][0]:
                  for t in range(-2,i-20):
                     closest[t+1] = closest[t]; 
                  closest[l][0] = temp;
                  closest[l][1] = k[0];
                  break;
       for q in closest:
          prediction += stocks.target[int(q[1])];
       temp = prediction.argsort()[-4:];
       print temp;
       const = 0;
       for q in temp:
             const = const + stocks.netx[2][q];
       ans = 0;
       for q in temp:
             ans = ans + (prediction[int(q)]*stocks.testdatacopy[i+10][int(q)+2]/const)/100;
       if (ans < .5) and (ans > -.5):
          stocks.total = stocks.total*(1+ans);
       print ans;
       print stocks.total;
       i = i + 1;
       if (i > np.shape(stocks.testdatacopy)[0] -20):
          return stocks;

def tempstuff():
   import matplotlib.pyplot as plt
   data = (pd.read_csv('prepreped.csv',',',header=None)).values;
   ourdata = data[1:,1].astype('float');
   length = np.size(ourdata);
   convolution = np.zeros((length,length));
   for i in range(0,length):
      for j in range(0,length):
         convolution[i][j] = ourdata[i]*ourdata[j];
   stdard = np.std(convolution);
   print stdard;
   for i in range(0,length):
      for j in range(0,length):
         if np.abs(convolution[i][j]) > stdard:
            if convolution[i][j] > 0:
               convolution[i][j] = stdard;
            else:
               convolution[i][j] = -stdard;
   plt.imshow(convolution, cmap='hot',interpolation='nearest')
   plt.show()
   return convolution;

# Creates a csv file called mnist_train_#.csv where # is num
# that is exactly the same as mnist_train.csv except the expected
# value of each row is one if the row is num, and zero otherwise. 
def create_mnist_binary_train(num):
   with open('../mnist_train.csv') as csvfile:
      trainreader = csv.reader(csvfile, delimiter=',', quotechar='|')
      csvwrite = open('../mnist_train_'+str(num)+'.csv','w');
      csvwriter = csv.writer(csvwrite,delimiter=',',quotechar="|",quoting=csv.QUOTE_MINIMAL)
      for row in trainreader:
         if int(row[0]) == num:
            row[0] = '1'
         else :
            row[0] = '0'
         csvwriter.writerow(row) 
      csvwrite.close(); 
         
# Creates a csv file called mnist_train_#.csv where # is num
# that is exactly the same as mnist_train.csv except the expected
# value of each row is one if the row is num, and zero otherwise. 
def create_mnist_binary_minitrain(num):
   with open('../mnist_train.csv') as csvfile:
      trainreader = csv.reader(csvfile, delimiter=',', quotechar='|')
      csvwrite = open('../mnist_minitrain_'+str(num)+'.csv','w');
      csvwriter = csv.writer(csvwrite,delimiter=',',quotechar="|",quoting=csv.QUOTE_MINIMAL)
      num = 6000; 
      for row in trainreader:
         if int(row[0]) == num:
            row[0] = '1'
         else :
            row[0] = '0'
         csvwriter.writerow(row) 
         num -= 1
         if num == 0:
            break
      csvwrite.close(); 

# Creates a csv file called mnist_test_#.csv where # is num
# that is exactly the same as mnist_test.csv except the expected
# value of each row is one if the row is num, and zero otherwise. 
def create_mnist_binary_test(num):
   with open('../mnist_test.csv') as csvfile:
      trainreader = csv.reader(csvfile, delimiter=',', quotechar='|')
      csvwrite = open('../mnist_test_'+str(num)+'.csv','w');
      csvwriter = csv.writer(csvwrite,delimiter=',',quotechar="|",quoting=csv.QUOTE_MINIMAL)
      for row in trainreader:
         if int(row[0]) == num:
            row[0] = '1'
         else :
            row[0] = '0'
         csvwriter.writerow(row)
      csvwrite.close(); 

# small binary classifier test
def small_test():
   output = open("mnist_09_1_11_0_10_test.csv","w")
   for i in range(0,10):
      mn = Neuralnet([785,1],"mnist_train_"+str(i)+".csv","mnist_test_"+str(i)+".csv")
      mn.traindata = mn.traindata[:12000][:]
      for j in range(1,11):
         for k in range(0,10):
            mn.compute_all(j,k)
            s = str(i)+","+str(j)+","+str(k)+","+str(mn.compute_alpha3(j,k,0))
            mn.mt2 = (mn.testdata[:,1:]).T
            mn.linear_kernel_mt2(j,k)
            mn.h = np.dot(mn.alpha,mn.mta1)
            s += str(sklearn.metrics.roc_auc_score(mn.testdata[:,0],mn.h))
            print s
            output.write(s)
   output.close()
               
