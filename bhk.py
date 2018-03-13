from sklearn import svm
import sklearn
import numpy as np
import csv
import pandas as pd
import datetime
import copy
import time
import math
import scipy
np.random.seed()
import gc
class BHK:
        def __init__(*arg):
           self = arg[0]
           self.train = './mnist_train_0.csv'
           self.test = './mnist_test_0.csv'
           numi = 0
           for i in arg:
              if numi == 1:
                 self.train = i
              if numi == 2:
                 self.test = i
              if numi == 3:
                 self.eta = i
              if numi == 4:
                 # not used?
                 self.l = i
              if numi == 5:
                 # will be used to make the arbitrary
                 # polynomial kernel
                 self.kernel = np.zeros(i)
              numi += 1
           self.testd = (pd.read_csv(self.test,sep=',',header=None)).values
           self.traind = (pd.read_csv(self.train,sep=',',header=None)).values
           self.traindata = np.ones((np.shape(self.traind)[0],np.shape(self.traind)[1]))
           self.testdata = np.ones((np.shape(self.testd)[0],np.shape(self.testd)[1]))
           self.testdata[:,:] = self.testd
           self.traindata[:,:] = self.traind
           #self.testdata[:,1:] = self.testdata[:,1:]/255.0
           #self.traindata[:,1:] = self.traindata[:,1:]/255.0
           self.mags = np.zeros(np.shape(self.traindata)[0]) 
           index = 0 
           for i in self.traindata:
              self.mags[index] = np.dot(i,i)
           self.ave = np.sum(self.mags)/np.size(self.mags)
           self.traindata[:,1:] = self.traindata[:,1:]/self.ave
           self.testdata[:,1:] = self.testdata[:,1:]/self.ave
          
     
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
        # answer into self.a1 (x+c)^n
        def linear_kernel_vec(self,n,c):
           self.a1 = ((np.dot(self.vec1,self.vec1)+c)**n)
        # Computes the  polynomial kernel of the vector stored in self.vec1 with every vector stored in
        # self.mt1 and stores the resulting vector into self.veca1
        def linear_kernel_mt1(self,n,c):
           self.veca1 = ((np.dot(self.mt1,self.vec1)+c)**n)
        # Computes the polynomial kernel of every vector stored in self.mt2 with every vector stored in
        # self.mt1 and stores the resulting matrix into self.mta1
        def linear_kernel_mt2(self,n,c):
           self.mta1 = ((np.dot(self.mt1,self.mt2)+c)**n)
        # Computes the angle kernel of the vector stored in self.vec1 and self.vec2 and stores the 
        # answer into self.a1. This is the normalized polynomial kernel
        def angle_kernel_vec(self,n,c):
           self.a1 = ((np.dot(self.vec1,self.vec1)+c)**n)/(np.sqrt(((np.dot(self.vec1,self.vec1)+c)**n)*((np.dot(self.vec2,self.vec2)+c)**n)))
        # Computes the normalized polynomial kernel of the vector stored in self.vec1 with every vector stored in
        # self.mt1 and stores the resulting vector into self.veca1
        def angle_kernel_mt1(self,n,c):
           self.mags = np.zeros(np.shape(self.mt1)[1])
           for i in range(0,np.shape(self.mt1)[1]):
              self.mags[i] = np.sqrt(((np.dot(self.mt1[i,:],self.mt1[i,:])+c)**n)) 
           self.veca1 = ((np.dot(self.mt1,self.vec1)+c)**n)/(np.sqrt((np.dot(self.vec1,self.vec1) + c)**n))
           self.veca1 = self.veca1/self.mags
        # Computes the normalized polynomial kernel of every vector stored in self.mt2 with every vector stored in
        # self.mt1 and stores the resulting matrix into self.mta1
        def angle_kernel_mt2(self,n,c):
           self.mta1 = ((np.dot(self.mt1,self.mt2)+c)**n)
           for i in range(0,np.shape(self.mt1)[1]):
              self.mta1[i,:] = self.mta1[i,:] / np.sqrt((np.dot(self.mt1[i,:],self.mt1[i,:]) + c)**n)
              self.mta1[:,i] = self.mta1[:,i] / np.sqrt((np.dot(self.mt2[:,i],self.mt2[:,i]) + c)**n)

        # Computes kplus and kminus 5/6 of the training data will be used to create the alphas and
        # 1/6th of the data will be used to find lambda. Also computes the covariance matrix
        def compute_all(self,n,c,m):
           # we want it to concatenate
           self.n = (5*np.shape(self.traindata)[0])/6;
           self.traintrain = self.traindata[:self.n,:]
           self.traintest = self.traindata[self.n:,:]
           self.traintrain = self.traintrain[np.lexsort(np.fliplr(self.traintrain).T)]
           self.traintest = self.traintest[np.lexsort(np.fliplr(self.traintest).T)]
           self.traindata[:self.n,:] = self.traintrain[:,:]
           self.traindata[self.n:,:] = self.traintest[:,:]
           self.traintrain = self.traintrain[:,1:]
           self.traintest = self.traintest[:,1:]
           self.testtest = self.testdata[:,1:]
           self.mt1 = self.traintrain
           self.mt2 = self.traintrain.T
           self.linear_kernel_mt2(n,c)
           print np.shape(self.mta1)
           self.traintemp = self.mta1
           self.k = self.traintemp
           # we can optimize the following comptation
           # by using a matrix matrix multiplication to compute
           # all the kernals, however, that would require storing
           # an n by n matrix where n is the number of training
           # points. So instead we will have a loop of matrix vector
           # multiplications
           self.posnum = int(np.sum(self.traindata[:self.n,0])) 
           self.negnum = self.n - self.posnum
           print self.posnum 
           gc.collect()
           self.kplus = (1.0/self.posnum)*(self.traintemp[self.negnum:,:-2-m]).sum(axis = 0)
           self.kminus = (1.0/self.negnum)*(self.traintemp[:self.negnum,:-2-m]).sum(axis = 0)
           print np.shape(self.kplus)
           self.g = (1.0/self.posnum)*np.dot((self.traintemp[self.negnum:,:-2-m]-self.kplus).T,(self.traintemp[self.negnum:,:-2-m]-self.kplus))
           self.g += (1.0/self.negnum)*np.dot((self.traintemp[:self.negnum,:-2-m]-self.kminus).T,(self.traintemp[:self.negnum,:-2-m]-self.kminus))
           print np.shape(self.g)
           print np.shape(self.traintemp[:self.negnum,:])
           print np.shape(self.traintemp[self.negnum:,:])
        # Computes kplus and kminus 5/6 of the training data will be used to create the alphas and
        # 1/6th of the data will be used to find lambda. Also computes the covariance matrix 
        def compute_all_angle(self,n,c,m):
           # we want it to concatenate
           self.n = (5*np.shape(self.traindata)[0])/6;
           self.traintrain = self.traindata[:self.n,:]
           self.traintest = self.traindata[self.n:,:]
           self.traintrain = self.traintrain[np.lexsort(np.fliplr(self.traintrain).T)]
           self.traintest = self.traintest[np.lexsort(np.fliplr(self.traintest).T)]
           self.traindata[:self.n,:] = self.traintrain[:,:]
           self.traindata[self.n:,:] = self.traintest[:,:]
           self.traintrain = self.traintrain[:,1:]
           self.traintest = self.traintest[:,1:]
           self.testtest = self.testdata[:,1:]
           self.mt1 = self.traintrain
           self.mt2 = self.traintrain.T
           self.angle_kernel_mt2(n,c)
           print np.shape(self.mta1)
           self.traintemp = self.mta1
           self.k = self.traintemp
           # we can optimize the following comptation
           # by using a matrix matrix multiplication to compute
           # all the kernals, however, that would require storing
           # an n by n matrix where n is the number of training
           # points. So instead we will have a loop of matrix vector
           # multiplications
           self.posnum = int(np.sum(self.traindata[:self.n,0])) 
           self.negnum = self.n - self.posnum
           print self.posnum 
           gc.collect()
           self.kplus = (1.0/self.posnum)*(self.traintemp[self.negnum:,:-2-m]).sum(axis = 0)
           self.kminus = (1.0/self.negnum)*(self.traintemp[:self.negnum,:-2-m]).sum(axis = 0)
           print np.shape(self.kplus)
           self.g = (1.0/self.posnum)*np.dot((self.traintemp[self.negnum:,:-2-m]-self.kplus).T,(self.traintemp[self.negnum:,:-2-m]-self.kplus))
           self.g += (1.0/self.negnum)*np.dot((self.traintemp[:self.negnum,:-2-m]-self.kminus).T,(self.traintemp[:self.negnum,:-2-m]-self.kminus))
           print np.shape(self.g)
           print np.shape(self.traintemp[:self.negnum,:])
           print np.shape(self.traintemp[self.negnum:,:])
        
        
        # Computes kplus and kminus all training, as well as the covariance matrix
        def compute_all_final(self,n,c,m):
           # we want it to concatenate
           self.n = (np.shape(self.traindata)[0]);
           self.traintrain = self.traindata[:self.n,:]
           self.traintrain = self.traintrain[np.lexsort(np.fliplr(self.traintrain).T)]
           self.traindata[:self.n,:] = self.traintrain[:,:]
           self.traintrain = self.traintrain[:,1:]
           self.testtest = self.testdata[:,1:]
           self.mt1 = self.traintrain
           self.mt2 = self.traintrain.T
           self.linear_kernel_mt2(n,c)
           self.traintemp = self.mta1
           self.k = self.traintemp
           # we can optimize the following comptation
           # by using a matrix matrix multiplication to compute
           # all the kernals, however, that would require storing
           # an n by n matrix where n is the number of training
           # points. So instead we will have a loop of matrix vector
           # multiplications
           self.posnum = int(np.sum(self.traindata[:self.n,0])) 
           self.negnum = self.n - self.posnum
           print self.posnum 
           gc.collect()
           self.kplus = (1.0/self.posnum)*(self.traintemp[self.negnum:,:-2-m]).sum(axis = 0)
           self.kminus = (1.0/self.negnum)*(self.traintemp[:self.negnum,:-2-m]).sum(axis = 0)
           print np.shape(self.kplus)
           self.g = (1.0/self.posnum)*np.dot((self.traintemp[self.negnum:,:-2-m]-self.kplus).T,(self.traintemp[self.negnum:,:-2-m]-self.kplus))
           self.g += (1.0/self.negnum)*np.dot((self.traintemp[:self.negnum,:-2-m]-self.kminus).T,(self.traintemp[:self.negnum,:-2-m]-self.kminus))
           print np.shape(self.g)
           print np.shape(self.traintemp[:self.negnum,:])

        # Returns the roc auc score of current alpha vector on training set        
        def verify_alpha(self,n,c,m):    
           self.mt1 = self.traintrain[:-2-m,:]
           self.mt2 = (self.traintrain[:,:]).T
           self.linear_kernel_mt2(n,c)
           self.h = np.dot(self.alpha,self.mta1)	
	   return sklearn.metrics.roc_auc_score(self.traindata[:self.n,0],self.h)
        
        # Returns the roc auc score of current alpha vector on training set        
        def evaluate_alpha(self,n,c,m):    
           self.mt1 = self.traintrain[:-2-m,:]
           self.mt2 = (self.traintest[:,:]).T
           self.linear_kernel_mt2(n,c)
           self.h = np.dot(self.alpha,self.mta1)
	   return sklearn.metrics.roc_auc_score(self.traindata[self.n:,0],self.h)

        # This function will scale and center the weight vector so the mean of
        # the positive class gets sent to 1 and the mean of the negative class
        # gets sent to -1. The shift constant will be stored in self.shift, and
        # the weight vector self.alpha will be scaled by a constant to make the
        # difference in the means equal to 2. This function assumes self.h stores
        # the h(x) for all x in the train set and self.traindata[self.n:,0] contains
        # all the class labels for the training set.
        def center(self):
           tempPosMean = 0
           tempNegMean = 0
           countp = 0
           countn = 0
           for i in range(0,np.size(self.h)):
              if self.traindata[i,0] == 0:
                 tempNegMean += self.h[i]
                 countn += 1
              else :
                 tempPosMean += self.h[i]
                 countp += 1
           tempPosMean *= 1.0/countp 
           tempNegMean *= 1.0/countn 
           print tempPosMean
           print tempNegMean
           diff = tempPosMean - tempNegMean
           self.alpha *= (2.0/diff)
           tempPosMean *= (2.0/diff) 
           tempNegMean *= (2.0/diff)
           tempDiff = -1 - tempNegMean
           tempPosMean += tempDiff
           tempNegMean += tempDiff
           print tempPosMean
           print tempNegMean 
           self.alpha += tempDiff

        # Returns the roc auc score of current alpha vector on test set        
        def test_alpha(self,n,c,m):    
           self.mt1 = self.traintrain[:-2-m,:]
           self.mt2 = (self.testtest[:,:]).T
           self.angle_kernel_mt2(n,c)
           self.h = np.dot(self.alpha,self.mta1)	
	   return sklearn.metrics.roc_auc_score(self.testdata[:,0],self.h)
        
        # This is a kernel function I tried that sends all magnitude of vectors to 1
        def verify_alpha_angle(self,n,c,m):    
           self.mt1 = self.traintrain[:-2-m,:]
           self.mt2 = (self.traintrain[:,:]).T
           self.angle_kernel_mt2(n,c)
           self.h = np.dot(self.alpha,self.mta1)	
	   return sklearn.metrics.roc_auc_score(self.traindata[:self.n,0],self.h)
        
        # This is a kernel function I tried that sends all magnitude of vectors to 1
        def evaluate_alpha_angle(self,n,c,m):    
           self.mt1 = self.traintrain[:-2-m,:]
           self.mt2 = (self.traintest[:,:]).T
           self.angle_kernel_mt2(n,c)
           self.h = np.dot(self.alpha,self.mta1)
	   return sklearn.metrics.roc_auc_score(self.traindata[self.n:,0],self.h)

        # This is a kernel function I tried that sends all magnitude of vectors to 1
        def test_alpha_angle(self,n,c,m):    
           self.mt1 = self.traintrain[:-2-m,:]
           self.mt2 = (self.testtest[:,:]).T
           self.angle_kernel_mt2(n,c)
           self.h = np.dot(self.alpha,self.mta1)	
	   return sklearn.metrics.roc_auc_score(self.testdata[:,0],self.h)
        
        
        def compute_alpha3(self,n,c,l):
           #self.alpha = scipy.sparse.linalg.cg(.5*self.g+self.traintemp*l,(self.kplus - self.kminus))
           #print self.alpha[1]
           #self.alpha = self.alpha[0] 
           self.chol = scipy.linalg.cho_factor(.5*self.g+self.traintemp*l)
           self.alpha = scipy.linalg.cho_solve(self.chol,self.kplus-self.kminus) 
           self.new_roc_auc = self.evaluate_alpha(n,c)
           print self.new_roc_auc

        def compute_alpha4(self,n,c,l):
           self.alpha = scipy.sparse.linalg.cg(.5*self.g+self.traintemp*l,(self.kplus - self.kminus))
           print self.alpha[1]
           self.alpha = self.alpha[0] 
           self.new_roc_auc = self.evaluate_alpha(n,c)
           print self.new_roc_auc
        
        def compute_alpha6(self,n,c,l):
           self.alpha = np.linalg.lstsq(.5*self.g+self.traintemp*l,(self.kplus - self.kminus))
           self.alpha = self.alpha[0] 
           self.new_roc_auc = self.evaluate_alpha(n,c)
           print self.new_roc_auc
       
        # This is the default function I have been using to compute the
        # weight vector of the final classifier.
        # n is the order of the kernel, c is the constant added to the
        # kernel (x^n+c), l is the term added to reduce overfitting, m tells
        # us how many datapoints will be excluded from the kernel but included
        # in the covariance matrix computations. 
        def compute_alpha7(self,n,c,l,m):
           self.alpha = np.dot(np.linalg.inv(.5*self.g+self.traintemp[:-2-m,:-2-m]*l),(self.kplus - self.kminus))
           self.alpha = self.alpha 
           #self.new_roc_auc = self.evaluate_alpha(n,c)
           #print self.new_roc_auc
        
        # Finds the accuracy over the training data 
        def accuracy_vector_train(self):
           self.temphy = np.zeros([np.shape(self.traindata[self.n:,0])[0],2])
           self.temphy[:,0] = self.h 
           self.temphy[:,1] = self.traindata[self.n:,0]
           self.temphy = self.temphy[np.lexsort(np.fliplr(self.temphy).T)]
           # The accuracy matrix will have the column vectors V (cutoff value), Fp (false
           # positive), Fn (False negative), Tp (True Positive), Tn (True negative),
           # A (accuracy)
           self.accuracy = np.zeros([np.shape(self.temphy)[0],6])
           self.totalp = np.sum(self.h)
           self.totalstuff = np.size(self.h)
           self.totaln = self.totalstuff - self.totalp
           self.currentn = 0
           self.currentp = self.totalp 
           for i in range(0,self.totalstuff):
              if i == 0:
                 self.accuracy[i][0] = self.temphy[i,0] - .5*(self.temphy[i+1,0] - self.temphy[i,0]) 
                 # initial false positive rate = totaln/total percent 
                 self.accuracy[i][1] = self.totaln/(0.0 + self.totalstuff) 
                 # initial False negative rate = 0 percent
                 self.accuracy[i][2] = 0  
                 # Initial True positive rate
                 self.accuracy[i][3] = self.totalp/(0.0+self.totalstuff)
                 # Initial True negative rate = 1
                 self.accuracy[i][4] = 1
                 # initial accuracy = totalp / total
                 self.accuracy[i][5] = self.totalp/(0.0+self.totalstuff)
              else :
                 if self.h[i] == 0:
                    self.currentn += 1
                 if self.h[i] == 1:
                    self.currentp -= 1
                 if i == self.totalstuff - 1:
                    self.accuracy[i][0] = self.temphy[i,0] + .5*(self.temphy[i,0] - self.temphy[i-1,0]) 
                    self.accuracy[i][2] = (self.totalstuff - self.totaln)/(self.totalstuff+ 0.0) 
                 else :
                    self.accuracy[i][0] = self.temphy[i-1,0] + .5*(self.temphy[i,0] - self.temphy[i-1,0])
                 # Total false positive rate
                 self.accuracy[i][1] = (self.currentn - i)/(i+0.0)
                 # True positive rate
                 self.accuracy[i][3] = (self.currentp)/(i + 0.0)
                 # True negative rate
                 self.accuracy[i][5] = (self.currentp + self.currentn)/(self.totalstuff+0.0) 
        
        # Write function description here. Tests the accuracy over the test data
        def accuracy_vector_test(self):
           self.temphy = np.zeros([np.shape(self.testdata[:,0])[0],2])
           self.temphy[:,0] = self.h 
           self.temphy[:,1] = self.testdata[:,0]
           self.temphy = self.temphy[np.lexsort(np.fliplr(self.temphy).T)]
           # The accuracy matrix will have the column vectors V (cutoff value), Fp (false
           # positive), Fn (False negative), Tp (True Positive), Tn (True negative),
           # A (accuracy)
           self.accuracy = np.zeros([np.shape(self.temphy)[0],6])
           self.totalp = np.sum(self.h)
           self.totalstuff = np.size(self.h)
           self.totaln = self.totalstuff - self.totalp
           self.currentn = 0
           self.currentp = self.totalp 
           for i in range(0,self.totalstuff):
              if i == 0:
                 self.accuracy[i][0] = self.temphy[i,0] - .5*(self.temphy[i+1,0] - self.temphy[i,0]) 
                 # initial false positive rate = totaln/total percent 
                 self.accuracy[i][1] = self.totaln/(0.0 + self.totalstuff) 
                 # initial False negative rate = 0 percent
                 self.accuracy[i][2] = 0  
                 # Initial True positive rate
                 self.accuracy[i][3] = self.totalp/(0.0+self.totalstuff)
                 # Initial True negative rate = 1
                 self.accuracy[i][4] = 1
                 # initial accuracy = totalp / total
                 self.accuracy[i][5] = self.totalp/(0.0+self.totalstuff)
              else :
                 if self.h[i] == 0:
                    self.currentn += 1
                 if self.h[i] == 1:
                    self.currentp -= 1
                 if i == self.totalstuff - 1:
                    self.accuracy[i][0] = self.temphy[i,0] + .5*(self.temphy[i,0] - self.temphy[i-1,0]) 
                    self.accuracy[i][2] = (self.totalstuff - self.totaln)/(self.totalstuff+ 0.0) 
                 else :
                    self.accuracy[i][0] = self.temphy[i-1,0] + .5*(self.temphy[i,0] - self.temphy[i-1,0])
                 # Total false positive rate
                 self.accuracy[i][1] = (self.currentn - i)/(i+0.0)
                 # True positive rate
                 self.accuracy[i][3] = (self.currentp)/(i + 0.0)
                 # True negative rate
                 self.accuracy[i][5] = (self.currentp + self.currentn)/(self.totalstuff+0.0)

        # What is this function for?       
        def yhat(self,myV,n,c):
           self.mt1 = self.traintrain[:,:]
           self.vec1 = myV[:]
           self.linear_kernel_mt1(n,c)
           self.yhatans = np.dot(self.alpha,self.veca1)
           return self.yhatans

# This function tests to ensure that
# all the functionality is working
def driver():
   myB = BHK("../mnist_train_0.csv","../mnist_test_0.csv")
   print np.shape(myB.traindata)
   myB.traindata = myB.traindata[20000:25000,:]
   myB.compute_all(2,1,1000)
   myB.compute_alpha7(2,1,0,1000)
   print myB.verify_alpha(2,1,1000)
   print myB.test_alpha(2,1,1000) 

# This driver will implement a forest of kernel
# lda's. It will require a batch size, covariance
# matix size, and number of trees as parameters.
# covariance size is b-m and batch size is b
def driver2(b,m,nt):
   # Tests center
   myB = BHK("../mnist_train_0.csv", "../mnist_test_0.csv")
   myB.traindata = myB.traindata[20000:25000,:]
   print np.shape(myB.traindata)
   myB.compute_all(2,1,200)
   myB.compute_alpha7(2,1,0,200)
   print myB.test_alpha(2,1,200)
   print myB.verify_alpha(2,1,200)
   myB.center()
    

 
# This class processes the google stock data
class ProcGoo:
        # This constructor needs to be commented
        def __init__(*arg):
           self = arg[0]
           numi = 0
           numtrends = np.size(arg)-1
           self.numtrends = numtrends 
           self.data = [0]*numtrends 
           for i in arg:
              if numi > 0:
                 self.data[numi-1] = (pd.read_csv(i,sep=',',header=None)).values
              numi += 1
           # converts dates from 99-Dec-99
           # to 99-99-99 format
           for i in range(0,np.shape(self.data)[0]):
              for j in range(0,np.shape(self.data[i])[0]):
                 # Takes care of the cases where 1-Jan
                 # sends them to 01-Jan
                 if self.data[i][j][0][1] == "-":
                    self.data[i][j][0] = "0"+self.data[i][j][0]
                    self.data[i][j][0] = self.data[i][j][0]
                 if self.data[i][j][0][3:6] == "Jan":
                    self.data[i][j][0] = self.data[i][j][0][:3]+"01"+self.data[i][j][0][6:]
                 elif self.data[i][j][0][3:6] == "Feb":
                    self.data[i][j][0] = self.data[i][j][0][:3]+"02"+self.data[i][j][0][6:]
                 elif self.data[i][j][0][3:6] == "Mar":
                    self.data[i][j][0] = self.data[i][j][0][:3]+"03"+self.data[i][j][0][6:]
                 elif self.data[i][j][0][3:6] == "Apr":
                    self.data[i][j][0] = self.data[i][j][0][:3]+"04"+self.data[i][j][0][6:]
                 elif self.data[i][j][0][3:6] == "May":
                    self.data[i][j][0] = self.data[i][j][0][:3]+"05"+self.data[i][j][0][6:]
                 elif self.data[i][j][0][3:6] == "Jun":
                    self.data[i][j][0] = self.data[i][j][0][:3]+"06"+self.data[i][j][0][6:]
                 elif self.data[i][j][0][3:6] == "Jul":
                    self.data[i][j][0] = self.data[i][j][0][:3]+"07"+self.data[i][j][0][6:]
                 elif self.data[i][j][0][3:6] == "Aug":
                    self.data[i][j][0] = self.data[i][j][0][:3]+"08"+self.data[i][j][0][6:]
                 elif self.data[i][j][0][3:6] == "Sep":
                    self.data[i][j][0] = self.data[i][j][0][:3]+"09"+self.data[i][j][0][6:]
                 elif self.data[i][j][0][3:6] == "Oct":
                    self.data[i][j][0] = self.data[i][j][0][:3]+"10"+self.data[i][j][0][6:]
                 elif self.data[i][j][0][3:6] == "Nov":
                    self.data[i][j][0] = self.data[i][j][0][:3]+"11"+self.data[i][j][0][6:]
                 elif self.data[i][j][0][3:6] == "Dec":
                    self.data[i][j][0] = self.data[i][j][0][:3]+"12"+self.data[i][j][0][6:]
                 # Switches from dd-mm-yy to mm-dd-yy
                 if j > 0:
                    tempday = self.data[i][j][0][3:6]+self.data[i][j][0][0:2]+self.data[i][j][0][5:]
                    self.data[i][j][0] = tempday 
           print self.data[0][1]
           # Finds the earliest data
           temp = "99\99\9999"
           for i in range(0,numtrends):
              bot = np.shape(self.data[i])[0]
              if int(self.data[i][bot-1][0][-2:]) < 40:
                 temptop = self.data[i][bot - 1][0][:-2]+"20"+self.data[i][bot-1][0][-2:]
              else:
                 temptop = self.data[i][bot-1][0][:-2]+"19"+self.data[i][bot - 1][0][-2:]
              if int(temptop[-4:]) < int(temp[-4:]):
                 temp = temptop 
              elif int(temptop[-4:]) == int(temp[-4:]):
                 if int(self.data[i][bot - 1][0][-8:-6]) < int(temp[-10:-8]):
                    temp = temptop 
                 elif int(self.data[i][bot - 1][0][-8:-6]) == int(temp[-10:-8]):
                    if int(self.data[i][bot-1][0][-5:-3]) < int(temp[-7:-5]):
                       temp = temptop 
           # Compute the number of days from temp to now
           now = str(datetime.datetime.now())[:10]
           print temptop
           temptime = datetime.date(int(temp[-4:]),int(temp[:2]),int(temp[-7:-5]))
           numdays = (datetime.date(int(now[:4]),int(now[5:7]),int(now[8:10]))-temptime).days
           # based on time starting with temp - bin 0
           #time = np.zeros([numdays,numtrends*4])
           time = np.zeros([numdays,numtrends])
           for i in range(0,numtrends):
              first = 1 
              for j in self.data[i]:
                 if first == 1:
                    first = 0
                 else:
                    if int(j[0][-2:]) < 40:
                       temptop = j[0][:-2]+"20"+j[0][-2:]
                    else :
                       temptop = j[0][:-2]+"19"+j[0][-2:]
                    datedif = (datetime.date(int(temptop[-4:]),int(temptop[:2]),int(temptop[3:5])) - temptime).days
                    try : 
                    #time[datedif][i*4:(i+1)*4] = j[1:-1].astype(np.float)
                       time[datedif][i] = float(j[1])
                    except :
                       # Nan encountered
                       continue 
           self.fdata = time
           count = 0
           print np.shape(self.fdata)
           # Figures out the number of non zero rows
           for i in self.fdata:
              if np.any(i):
                 count += 1
           self.newdata = np.zeros([count,np.shape(self.fdata)[1]+1])
           index = 0
           for i in range(0,np.shape(self.fdata)[0]):
              if np.any(self.fdata[i]):
                 self.newdata[index][:-1] = self.fdata[i]  
                 self.newdata[index][-1] = i+1
                 index += 1 
           self.final = self.newdata[:]
           print np.shape(self.final)
           # Here we will interpolate any zero values
           bad = 0
           for i in range(0,np.shape(self.final)[1]):
              if np.all(self.final[:][i] == 0):
                 bad =1
           if bad == 0: 
              while np.any(np.absolute(self.final[-1]) == 0):
                 index = np.where(self.final[-1] == 0)[0][0]
                 decrementer = -1 
                 while (decrementer < 0-np.shape(self.final)[0])or(self.final[decrementer-1][index] == 0):
                    decrementer -= 1
                 while decrementer <= -1:
                    try :
                       self.final[decrementer][index] = self.final[decrementer - 1][index]
                    except :
                       self.final[decrementer][index] = 0
                    decrementer += 1
              for i in range(0,np.shape(self.final)[0]):
                 while np.any(self.final[i] == 0):
                    index = np.where(self.final[i] == 0)[0][0]
                    increment = 0 
                    while (self.final[i+increment+1][index] == 0):
                       increment += 1
                    while increment >= 0:
                       self.final[i+increment][index] = self.final[i+increment+1][index]
                       increment -= 1 
          
           print np.shape(self.final) 
           self.final = (self.final[1:] + 0.0 - self.final[:-1])/self.final[:-1]
           print np.shape(self.final)
           self.fintrain = np.zeros([np.shape(self.final)[0]-9,np.shape(self.final)[1]*10]) 
           mean = 0
           for i in range(0,np.shape(self.final)[0]-9):
              for j in range(0,10):
                 self.fintrain[i][j*np.shape(self.final)[1]:(j+1)*np.shape(self.final)[1]]=self.final[i+j]
           # here we will attempt to normalize the data. We want each vector to have a magnitude of around 1
           # on average
           self.mags = np.zeros(np.shape(self.fintrain)[0])
           for i in range(0,np.shape(self.fintrain)[0]):
              self.mags[i] = np.linalg.norm(self.fintrain[i])
           self.avenorm = np.sum(self.mags)/np.size(self.mags)
           self.fintrain = self.fintrain/self.avenorm

        # This function needs to be commented
        def save(self,cutoff,which):
           self.trainingpost = np.zeros([np.shape(self.fintrain)[0],np.shape(self.fintrain)[1]+1])
           # ensures the which number is between 0 and numtrends 
           which  = which % self.numtrends
           self.trainingpost[:,1:] = self.fintrain[:,:]
           for i in range(0,np.shape(self.trainingpost)[0]-2):
              #if (self.fintrain[i+2,-self.numtrends*4-1+which*4])*100*self.avenorm > cutoff:
              if (self.fintrain[i+2,-self.numtrends-1+which])*100*self.avenorm > cutoff:
                 self.trainingpost[i][0] = 1
           np.savetxt(str(cutoff)+","+str(which)+".csv",self.trainingpost,delimiter=',')
          
        # This function needs to be commented
        def new_vec(self):
           self.new_vec = self.fintrain[-1,:]
           return self.new_vec 
        

def run_current():
   myB = ProcGoo("aapl20072017.csv","adbe20072017.csv","amov20072017.csv","amx20072017.csv","amzn20072017.csv","atvi20072017.csv","avx20072017.csv","bac20072017.csv","chl20072017.csv","dis20072017.csv","googl20072017.csv","gsat20072017.csv","ibm20072017.csv","intc20072017.csv","irbt20072017.csv","ko20072017.csv","msft20072017.csv","nke20072017.csv","qcom20072017.csv","sbux20072017.csv","spy20072017.csv","tef20072017.csv","teo20072017.csv","tgt20072017.csv","veon20072017.csv","viv20072017.csv","vod20072017.csv")
   for i in range(-6,10):
      for j in range(0,27):
         myB.save(i*.5,j)
         print "Stock "+str(j)
         print "Cutoff " +str(i*.5)
         print "count " + str(np.sum(myB.trainingpost[:,0]))

def run_current_new(low,high):
   myB = ProcGoo("aapl20072017.csv","adbe20072017.csv","amov20072017.csv","amx20072017.csv","amzn20072017.csv","atvi20072017.csv","avx20072017.csv","bac20072017.csv","chl20072017.csv","dis20072017.csv","googl20072017.csv","gsat20072017.csv","ibm20072017.csv","intc20072017.csv","irbt20072017.csv","ko20072017.csv","msft20072017.csv","nke20072017.csv","qcom20072017.csv","sbux20072017.csv","spy20072017.csv","tef20072017.csv","teo20072017.csv","tgt20072017.csv","veon20072017.csv","viv20072017.csv","vod20072017.csv")
   for i in range(int(low*2),int(high*2)):
      for j in range(0,27):
         myB.save(i*.5,j)
         print "Stock "+str(j)
         print "Cutoff " +str(i*.5)
         print "count " + str(np.sum(myB.trainingpost[:,0]))
def new_vec():
   myB = ProcGoo("aapl20072017.csv","adbe20072017.csv","amov20072017.csv","amx20072017.csv","amzn20072017.csv","atvi20072017.csv","avx20072017.csv","bac20072017.csv","chl20072017.csv","dis20072017.csv","googl20072017.csv","gsat20072017.csv","ibm20072017.csv","intc20072017.csv","irbt20072017.csv","ko20072017.csv","msft20072017.csv","nke20072017.csv","qcom20072017.csv","sbux20072017.csv","spy20072017.csv","tef20072017.csv","teo20072017.csv","tgt20072017.csv","veon20072017.csv","viv20072017.csv","vod20072017.csv")
   return myB.new_vec()

# Need to comment this
# This is used on the stock data
def compute(i):
   myB = BHK("1.0,"+str(i)+".csv","1.0,"+str(i)+".csv")
   tempv = myB.traindata[-2:,:]
   myB.traindata = myB.traindata[:-2,:]
   myB.compute_all_final(2,1,0)
   myB.compute_alpha7(2,1,0,0)
   myB.mt1 = myB.traintrain[:-2,:]
   myB.mt2 = (tempv[:,1:]).T
   myB.linear_kernel_mt2(2,1)
   print np.dot(myB.alpha,myB.mta1)
   myB.verify_alpha(2,1,0)
   pos = 0
   neg = 0
   posfl = 0
   negfl = 0
   stdpos = 0
   stdneg = 0
   for i in range(0,myB.n):
      if myB.traindata[i,0] == 1:
         pos += 1
         posfl += myB.h[i]
      else :
         neg += 1
         negfl += myB.h[i]
   posfl = posfl/pos
   negfl = negfl/neg
   for i in range(0,myB.n):
      if myB.traindata[i,0] == 1:
         stdpos += (myB.h[i] - posfl)**2
      else :
         stdneg += (myB.h[i] - negfl)**2
   stdpos = (stdpos/pos)**(.5)
   stdneg = (stdneg/neg)**(.5)
   print negfl
   print posfl
   print stdneg 
   print stdpos
