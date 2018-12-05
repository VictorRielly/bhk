# Todo: We need to ensure we are removing instances from
# both classes when removing m datapoints from the covariance
# computation

try:
    from sklearn import svm
    import sklearn
    skl = 1
except :
    skl = 0
import numpy as np
import csv
try:
    import pandas as pd
    pandas = 1
except:
    pandas = 0

import datetime
import copy
import time
import math
import scipy
# The following function will be used
# to implement xT*x efficiently for 
# the rbf and general polynomial kernels
from numpy.core.umath_tests import inner1d
np.random.seed()
import gc

# This function will be used in place of the roc_auc function
# in sklearn
def roc_auc(y_true,y_prob):
    if np.size(y_true) != np.size(y_prob):
        return
    total = np.size(y_true)
    nump = np.sum(y_true)
    numn = total - nump
    data = np.zeros([total,6])
    data[:,0] = y_prob
    data[:,1] = y_true
    data = data[np.lexsort((data[:,1],data[:,0]))]
    numTF = numn
    numTP = nump
    FRP = numTF/numn
    TPR = numTP/nump
    for i in range(total):
        if data[i,1] == 1:
            numTP -= 1
        if data[i,1] == 0:
            numTF -= 1
        data[i,2] = numTF/numn
        data[i,3] = numTP/nump
        if i == 0:
            data[i,4] = (1-data[i,2])*(1+data[i,3])/2.0
        else :
            data[i,4] = data[i-1,4] + (data[i-1,2]-data[i,2])*(data[i-1,3]
                   + data[i,3])/2.0
        data[i,5] = (numTP + numn - numTF + .0)/total
        print data[-1,4]
        print np.max(data[:,5])
        return data

class BHK:
    def __init__(*arg):
        self = arg[0]
        self.train = '../mnist_train_0.csv'
        self.test = '../mnist_test_0.csv'
        numi = 0
        # The following are default values for kernel type,
        # the number of datapoints to leave out to make the 
        # covariance matrix more robust, and the
        # regularization constants. Kernel type can be p, 
        # gp or e. e for rbf, p for polynomial, gp for general
        # polynomial kernel.
        self.n = 2
        self.c = 1
        self.e = .05
        self.kernelType = 'p'
        self.l = 2**5
        self.m = .2
        for i in arg:
            if numi == 1:
                if i == "":
                     self.train = "";
                     self.test = "";
                  else :
                     self.train = i
            if numi == 2:
                self.test = i
            if numi == 3:
                self.n = i
            if numi == 4:
                self.c =i 
            if numi == 5:
                self.e = i
            if numi == 6:
                self.l = i
            if numi == 7:
                self.kernelType = i
            if numi == 8:
                self.m = i
            if numi == 9:
                # Will be used to make the arbitrary
                # polynomial kernel eventually
                self.kernel = np.zeros(i)
        # The percentage of data points to remove is 1/5 by
        # default
        self.m = .2
        if self.train != "":
            if pandas == 1:
                self.testd
                = (pd.read_csv(self.test,sep=',',header=None)).values
                self.traind
                = (pd.read_csv(self.train,sep=',',header=None)).values
            else :
                self.testd = np.loadtxt(self.test,delimiter=',')
                self.traind = np.loadtxt(self.train,delimiter=',')
            self.traindata
            = np.ones((np.shape(self.traind)[0],np.shape(self.traind)[1]))
            self.testdata
            = np.ones((np.shape(self.testd)[0],np.shape(self.testd)[1]))
            self.testdata[:,:] = self.testd
            self.traindata[:,:] = self.traind
            # self.testdata[:,1:] = self.testdata[:,1:]/255.0
            # self.traindata[:,1:] = self.traindata[:,1:]/255.0
            # Here we scale the training and test data by
            # a factor such that the average norm of the 
            # training data vectors is 1
            self.mags = np.zeros(np.shape(self.traindata)[0])
            index = 0
            for i in self.traindata:
                self.mags[index] = np.dot(i,i)
                index += 1
            self.ave = np.sum(self.mags)/np.size(self.mags)
            self.traindata[:,1:] = self.traindata[:,1:]/self.ave
            self.testdata[:,1:] = self.testdata[:,1:]/self.ave

    # Implements the polynomial kernel, which includes the
    # linear kernel
    def polynomial_kernel(self):
        self.mta1 = ((np.dot(self.mt1,self.mt2)+self.c)**self.n)
    # Implements the rbf kernel
    def kernele(self.e):
        if np.size(np.shape(self.mt2)) == 2 and np.size(np.shape(self.mt1)) == 2:
            xs = np.zeros([np.shape(self.mt1)[0],np.shape(self.mt2)[0]])
            ys = np.zeros([np.shape(self.mt1)[0],np.shape(self.mt2)[0]])
            vecxs = inner1d(self.mt1,self.mt1) 
            xs = (xs.T + vecxs).T
            vecys = inner1d(self.mt2,self.mt2)
            ys = ys + vecys
        if np.size(np.shape(self.mt2)) == 1:
            xs = np.zeros(np.shape(self.mt1)[0])
            ys = np.zeros(np.shape(self.mt1)[0])
            xs = inner1d(self.mt1,self.mt1)
            ys += np.dot(self.mt2,self.mt2)
        if np.size(np.shape(self.mt1)) == 1:
            xs = np.zeros(np.shape(self.mt2)[0])
            ys = np.zeros(np.shape(self.mt2)[0])
            xs += np.dot(self.mt1,self.mt1)
            ys = inner1d(self.mt2,self.mt2)
        xy = np.dot(self.mt1,(self.mt2).T)
        self.mta1 = np.exp(-(xs-2*xy+ys)/(e**2))

    # Computes kplus and kminus for 5/6 of the training data
    # Also computes the covariance matrix.
    def compute_all(self):
        self.numt = (5*np.shape(self.traindata)[0])/6;
        self.traintrain = self.traindata[:self.numt,:]
        self.traintest = self.traindata[self.numt:,:]
        self.traintrain
        = self.traintrain[np.lexsort(np.fliplr(self.traintrain).T)]
        self.traintest
        = self.traintest[np.lexsort(np.fliplr(self.traintrain).T)]
        self.traindata[:self.numt,:] = self.traintrain[:,:]
        self.traindata[self.numt:,:] = self.traintest[:,:]
        self.traintrain = self.traintrain[:,1:]
        self.traintest = self.traintest[:,1:]
        self.mt1 = self.traintrain
        self.mt2 = self.traintrain.T
        if self.kernelType == "p":
            self.kf = self.polynomial_kernel
        if self.kernelType == "e":
            self.kf = self.kernele
        self.k()
        print np.shape(self.mta1)
        self.traintemp = self.mta1
        self.k = self.traintemp
        self.posnum = int(np.sum(self.traindata[:self.numt,0]))
        self.negnum = self.numt - self.posnum
        gc.collect()
        self.kplus
        = (1.0/self.posnum)*(self.traintemp[self.negnum:,:-2-int(self.m*self.numt)]).sum(axis=0)
        self.kminus=(1.0/self.posnum)*(self.traintemp[:self.negnum,:-2-int(self.m*self.numt)]).sum(axis=0)
        print np.shape(self.kplus)
        self.g
        = (1.0/self.posnum)*np.dot((self.traintemp[self.negnum:,:-2-m]-self.kplus).T,(self.traintemp[self.negnum:,:-2-int(self.m*self.numt)]-self.kplus))
        self.g 
        +=
        (1.0/self.negnum)*np.dot((self.traintemp[:self.negnum,:-2-m]-self.kminus).T,(self.traintemp[:self.negnum,:-2-int(self.m*self.numt)]-self.kminus))
        print np.shape(self.g)
        print np.shape(self.traintemp[:self.negnum,:])
        print np.shape(self.traintemp[self.negnum:,:])

    def compute_all_final(*arg):
        self = arg[0]
        isRandForest = 0
        numi = 0
        for i in arg:
            if numi == 1:
                isRandForest == i
            numi += 1
        # we want it to concatenate
        self.numt = (np.shape(self.traindata)[0]);
        self.traintrain = self.traindata[:self.numt,:]
        self.traintrain = self.traintrain[np.lexsort(np.fliplr(self.traintrain).T)]
        self.traindata[:self.numt,:] = self.traintrain[:,:]
        if isRandForest != 0:
           self.traintrain = self.traintrain[:,1:-1]
        else :
           self.traintrain = self.traintrain[:,1:]
        self.testtest = self.testdata[:,1:]
        self.mt1 = self.traintrain
        self.mt2 = self.traintrain.T
        if self.kernelType == "p":
            self.kf = self.polynomial_kernel
        if self.kernelType == "e" :
            self.kf = self.kernele
        self.kf()
        self.traintemp = self.mta1
        self.k = self.traintemp
        # we can optimize the following comptation
        # by using a matrix matrix multiplication to compute
        # all the kernals, however, that would require storing
        # an n by n matrix where n is the number of training
        # points. So instead we will have a loop of matrix vector
        # multiplications
        self.posnum = int(np.sum(self.traindata[:self.numt,0])) 
        self.negnum = self.numt - self.posnum
        print self.posnum 
        gc.collect()
        print self.negnum
        self.kplus = (1.0/self.posnum)*(self.traintemp[self.negnum:,:-2-m]).sum(axis = 0)
        self.kminus = (1.0/self.negnum)*(self.traintemp[:self.negnum,:-2-m]).sum(axis = 0)
        print np.shape(self.kplus)
        self.g = (1.0/self.posnum)*np.dot((self.traintemp[self.negnum:,:-2-m]-self.kplus).T,(self.traintemp[self.negnum:,:-2-m]-self.kplus))
        self.g += (1.0/self.negnum)*np.dot((self.traintemp[:self.negnum,:-2-m]-self.kminus).T,(self.traintemp[:self.negnum,:-2-m]-self.kminus))
        print np.shape(self.g)


    # Returns the roc auc score of current alpha vector on training set        
    def verify_alpha(self):    
       self.mt1 = self.traintrain[:-2-int(self.m*self.numt),:]
       self.mt2 = (self.traintrain[:,:]).T
       if self.kernelType == "p":
           self.kf = self.polynomial_kernel
       if self.kernelType == "e":
           self.kf = self.kernele
       self.kf()
       self.h = np.dot(self.alpha,self.mta1)
       return roc_auc(self.traindata[:self.numt,0],self.h)	
    
    # Returns the roc auc score of current alpha vector on training set        
    def evaluate_alpha(self):    
       self.mt1 = self.traintrain[:-2-int(self.m*self.numt),:]
       self.mt2 = (self.traintest[:,:]).T
       if self.kernelType == "p":
           self.kf = self.polynomial_kernel
       if self.kernelType == "e":
           self.kf = self.kernele
       self.kf()
       self.h = np.dot(self.alpha,self.mta1)
       return roc_auc(self.traindata[:self.numt,0],self.h)	
        
        
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
       self.shift = tempDiff
    
    # Returns the roc auc score of current alpha vector on test set        
    def test_alpha(self,n,c,m):    
       self.mt1 = self.traintrain[:-2-int(self.m*self.numt),:]
       self.mt2 = (self.testtest[:,:]).T
       if self.kernelType == "p":
           self.kf = self.polynomial_kernel
       if self.kernelType == "e":
           self.kf = self.kernele
       self.kf()
       self.h = np.dot(self.alpha,self.mta1)	
       return roc_auc(self.testdata[:,0],self.h)	
    
    # This is the default function I have been using to compute the
    # weight vector of the final classifier.
    # n is the order of the kernel, c is the constant added to the
    # kernel (x^n+c), l is the term added to reduce overfitting, m tells
    # us how many datapoints will be excluded from the kernel but included
    # in the covariance matrix computations. 
    def compute_alpha7(self,n,c,l,m):
       self.alpha = np.dot(np.linalg.inv(self.g+self.traintemp[:-2-m,:-2-m]*l),(self.kplus - self.kminus))

# This class will implement the random forest
# of BHK classifiers
class BHKRandomForest:
   # It has the same constructor as bhk 
   def __init__(*arg):
      self = arg[0]
      self.train = '../mnist_train_0.csv'
      self.test = '../mnist_test_0.csv'
      numi = 0
      # This will store the epoch number
      self.T = 0
      # It will have a batch size constant
      self.b = 5000
      # This will be the order of the polynomial kernel
      self.n = 2
      # this will be the shift for the polynomial kernel
      self.c = 1
      # this is the sigma for the rbf kernel
      self.e = .05
      self.kernelType = "p"
      self.m = .2
      # along with the batch, it will have the indexes
      # of the instances in the batch
      self.bArray = np.zeros(5000)
      for i in arg:
         if numi == 1:
            self.train = i
         if numi == 2:
            self.test = i
         if numi == 3:
            self.b = i
            self.bArray = np.zeros(i)
         if numi == 4:
            self.n = i
         if numi == 5:
            self.c = i
         if numi == 6:
            self.e = i
         if numi == 7:
            self.l = i
         if numi == 8:
            self.kernelType = i
         if numi == 9:
            self.m = i
         if numi == 10:
            # will be used to make the arbitrary
            # polynomial kernel
            self.kernel = np.zeros(i)
         numi += 1
      if self.train != "":
         if pandas == 1:
            self.testd = (pd.read_csv(self.test,sep=',',header=None)).values
            self.traind = (pd.read_csv(self.train,sep=',',header=None)).values
         else :
            self.testd = np.loadtxt(self.test,delimiter=',')
            self.traind = np.loadtxt(self.train,delimiter=',')
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
            index += 1
         self.ave = np.sum(self.mags)/np.size(self.mags)
         self.traindata[:,1:] = self.traindata[:,1:]/self.ave
         self.testdata[:,1:] = self.testdata[:,1:]/self.ave
         # It will also have an n dimensional alpha vector
         # where n is the number of training points
         self.alpha = np.zeros(np.shape(self.traindata)[0])
         self.h = np.zeros(np.shape(self.testdata)[0]) 
         # and it will have a shift constant
         self.shift = 0
         # Here we define an array that will be used in tandem
         # with the train array. It is not necessary to define
         # this array seperately from the traindata array but
         # it is quickest to implement this way
         self.temptrain = np.zeros([np.shape(self.traindata)[0],np.shape(self.traindata)[1]+1])
         self.temptrain[:,:-1] = copy.copy(self.traindata)
         self.temptrain[:,-1] = range(0,np.shape(self.traindata)[0])
   
   
   # This function grabs a batch, trains on that batch and
   # returns the batch indices as well as the alpha and
   # shift. n is the order of the kernel, c is the constant
   # term in the polynomial kernel, l is the regularization
   # constant, and m determines the size of the covariance matrix
   def trainbatch(self):
      self.myB = BHK("No Data")
      self.myB.n = self.n
      self.myB.c = self.c
      self.myB.e = self.e
      self.myB.kernelType = self.kernelType
      self.myB.m = self.m
      self.myB.l = self.l
      # shuffles temparray 
      np.random.shuffle(self.temptrain)
      # grabs the first b data instances from temptrain
      self.myB.traindata = copy.copy(self.temptrain[:self.b,:])
      self.myB.testdata = self.testdata
      self.myB.compute_all_final(1)
      self.myB.compute_alpha7()
      validatedata = self.myB.test_alpha()
      testdata = self.myB.verify_alpha()
      self.myB.center()
   
   # This function grabs a batch, trains on that batch and
   # returns the batch indices as well as the alpha and
   # shift. n is the order of the kernel, c is the constant
   # term in the polynomial kernel, l is the regularization
   # constant, and m determines the size of the covariance matrix
   def trainbatch(self):
      self.myB = BHK("No Data")
      self.myB.n = self.n
      self.myB.c = self.c
      self.myB.e = self.e
      self.myB.kernelType = self.kernelType
      self.myB.m = self.m
      self.myB.l = self.l
      # shuffles temparray 
      np.random.shuffle(self.temptrain)
      # grabs the first b data instances from temptrain
      self.myB.traindata = copy.copy(self.temptrain[:self.b,:])
      self.myB.testdata = self.testdata
      self.myB.compute_all_final(1)
      self.myB.compute_alpha7()
      self.myB.center()
   
   # This function will be used to aggregate myB.alpha
   # with myRF.alpha as well as the shifts
   def aggregate(self):
      self.T += 1
      if self.T != 1:
         self.alpha *= self.T/(self.T+1.0)
         self.shift *= self.T/(self.T+1.0)
         self.myB.alpha *= 1/(self.T+1.0)
         self.myB.shift *= 1/(self.T+1.0)
      self.shift += self.myB.shift
      for i in range(0,np.size(self.myB.alpha)):
         self.alpha[int(self.myB.traindata[i,-1])] += self.myB.alpha[i]
   
   # Implements the polynomial kernel, which includes the
   # linear kernel
   def polynomial_kernel(self):
       self.mta1 = ((np.dot(self.mt1,self.mt2)+self.c)**self.n)
   
   # Implements the rbf kernel
   def kernele(self.e):
       if np.size(np.shape(self.mt2)) == 2 and np.size(np.shape(self.mt1)) == 2:
           xs = np.zeros([np.shape(self.mt1)[0],np.shape(self.mt2)[0]])
           ys = np.zeros([np.shape(self.mt1)[0],np.shape(self.mt2)[0]])
           vecxs = inner1d(self.mt1,self.mt1) 
           xs = (xs.T + vecxs).T
           vecys = inner1d(self.mt2,self.mt2)
           ys = ys + vecys
       if np.size(np.shape(self.mt2)) == 1:
           xs = np.zeros(np.shape(self.mt1)[0])
           ys = np.zeros(np.shape(self.mt1)[0])
           xs = inner1d(self.mt1,self.mt1)
           ys += np.dot(self.mt2,self.mt2)
       if np.size(np.shape(self.mt1)) == 1:
           xs = np.zeros(np.shape(self.mt2)[0])
           ys = np.zeros(np.shape(self.mt2)[0])
           xs += np.dot(self.mt1,self.mt1)
           ys = inner1d(self.mt2,self.mt2)
       xy = np.dot(self.mt1,(self.mt2).T)
       self.mta1 = np.exp(-(xs-2*xy+ys)/(e**2))
   
   # Returns the roc auc score of current alpha vector on test set
   # breaks up a large matrix multiplication into byte size chunks        
   def test_alpha(self):    
      self.mt1 = self.traindata[:,1:]
      self.mt2 = (self.testdata[:,1:]).T
      if self.kernelType == "p":
          self.kf = self.polynomial_kernel
      if self.kernelType == "e":
          self.kf = self.kernele
      self.kf()
      self.h = np.dot(self.alpha,self.mta1)
      return roc_auc(self.testdata[:,0],self.h)	
   
   # runs the random forest for the desired number of trees
   def runForest(*arg):
      self = arg[0]
      numTrees = 10
      if len(arg) > 0:
         self.n = arg[1]
      if len(arg) > 1:
         self.c = arg[2]
      if len(arg) > 3:
         self.l = arg[4]
      if len(arg) > 2:
         self.e = arg[3]
      if len(arg) > 4:
         self.b = arg[5]
      if len(arg) > 5:
         self.m = arg[6]
      if len(arg) > 6:
         self.kernelType = arg[7]
      if len(arg) > 7:
         numTrees = arg[8]
      for i in range(0,numTrees):
         self.trainbatch()
         self.aggregate()
         self.testRes = self.test_alpha()
   
   # runs the random forest for the desired number of trees
   def runForest2(*arg):
      self = arg[0]
      numTrees = 10
      if len(arg) > 0:
         self.n = arg[1]
      if len(arg) > 1:
         self.c = arg[2]
      if len(arg) > 3:
         self.l = arg[4]
      if len(arg) > 2:
         self.e = arg[3]
      if len(arg) > 4:
         self.b = arg[5]
      if len(arg) > 5:
         self.m = arg[6]
      if len(arg) > 6:
         self.kernelType = arg[7]
      if len(arg) > 7:
         numTrees = arg[8]
      for i in range(0,numTrees):
         self.trainbatch()
         self.aggregate()
      self.testRes = self.test_alpha()
         
# Here we construct a ten class classifier. The 10 class 
# classifier will consist of 10 binary classifiers. Classifier
# 0 will be 0 versus all, classifier 1 will be 1 versus all
# and so on and so forth. Each classifier is a function
# onto the real numbers and sends will send the mean
# of all data to 0 the mean of the negative classes to -1 and
# the mean of the positive class to 1. The final classification
# will be taken as the class corresponding to the i versus all
# classifier that evaluates to the highest value. n is the order
# of the polynomial kernel, c is the constant in the polynomial
# kernel. So if n = 2, and c = 1, our kernel function is (xy+1)^2
# l is the regularization constant when computing the covariance 
# matrix C = C_+ + C_- + l*K where K is the grahm matrix. b is
# the batch size. We use ensemble learning with a kernel
# so we make a collection of small classifiers, of batch size
# b. m determines how mush of the data per batch is set asside 
# to make the covariance matrix robust. We suggest about 1/5 of
# the batch size be set asside to make the covariance matrix more
# robust. t is the number of trees in per classifier in our
# ensemble learner.
def total_classifier(n,c,l,e,b,m,ktype,t):
   # The following matrix will hold
   # the 10 alpha vectors
   # We start by getting an alpha vector
   # for each of the binary classification
   # problems since we have a ram constraint,
   # and this classifier is very ram intensive,
   # we will get these one at a time.
   # We will use batch sizes of 10000
   # First, we create the alpha vector for
   # the 0 versus all classification problem. 
   myR = BHKRandomForest("../mnist_train_0.csv","../mnist_test_0.csv",b)
   # The following matrix will hold
   # the 10 alpha vectors
   alphas = np.zeros([10,np.size(myR.alpha)])
   hypos = np.zeros([10,np.size(myR.h)]) 
   # our batch size is 10000, each epoch is 6 classifiers
   # we will run for 5 epochs, which will aggregate over 
   # 120 classifiers. We will have the (xy+1)^2 kernel
   # function, no regularization constant, and we will set asside
   # 1/5 of 10000 or about 20000 datapoints to ensure a robust
   # covariance matrix.
   myR.runForest2(n,c,l,e,b,m,ktype,t)
   alphas[0,:] = copy.copy(myR.alpha)
   hypos[0,:] = copy.copy(myR.h + myR.shift)
   print np.sum(hypos[0,:])
   myR = BHKRandomForest("../mnist_train_1.csv","../mnist_test_1.csv",b)
   myR.runForest2(n,c,l,e,b,m,ktype,t)
   alphas[1,:] = copy.copy(myR.alpha)
   hypos[1,:] = copy.copy(myR.h + myR.shift)
   print np.sum(hypos[1,:])
   myR = BHKRandomForest("../mnist_train_2.csv","../mnist_test_2.csv",b)
   myR.runForest2(n,c,l,e,b,m,ktype,t)
   alphas[2,:] = copy.copy(myR.alpha)
   hypos[2,:] = copy.copy(myR.h + myR.shift)
   print np.sum(hypos[2,:])
   myR = BHKRandomForest("../mnist_train_3.csv","../mnist_test_3.csv",b)
   myR.runForest2(n,c,l,e,b,m,ktype,t)
   alphas[3,:] = copy.copy(myR.alpha)
   hypos[3,:] = copy.copy(myR.h + myR.shift)
   print np.sum(hypos[3,:])
   myR = BHKRandomForest("../mnist_train_4.csv","../mnist_test_4.csv",b)
   myR.runForest2(n,c,l,e,b,m,ktype,t)
   alphas[2,:] = copy.copy(myR.alpha)
   alphas[4,:] = copy.copy(myR.alpha)
   hypos[4,:] = copy.copy(myR.h + myR.shift)
   print np.sum(hypos[4,:])
   myR = BHKRandomForest("../mnist_train_5.csv","../mnist_test_5.csv",b)
   myR.runForest2(n,c,l,e,b,m,ktype,t)
   alphas[2,:] = copy.copy(myR.alpha)
   alphas[5,:] = copy.copy(myR.alpha)
   hypos[5,:] = copy.copy(myR.h + myR.shift)
   print np.sum(hypos[5,:])
   myR = BHKRandomForest("../mnist_train_6.csv","../mnist_test_6.csv",b)
   myR.runForest2(n,c,l,e,b,m,ktype,t)
   alphas[6,:] = copy.copy(myR.alpha)
   hypos[6,:] = copy.copy(myR.h + myR.shift)
   print np.sum(hypos[6,:])
   myR = BHKRandomForest("../mnist_train_7.csv","../mnist_test_7.csv",b)
   myR.runForest2(n,c,l,e,b,m,ktype,t)
   alphas[7,:] = copy.copy(myR.alpha)
   hypos[7,:] = copy.copy(myR.h + myR.shift)
   print np.sum(hypos[7,:])
   myR = BHKRandomForest("../mnist_train_8.csv","../mnist_test_8.csv",b)
   myR.runForest2(n,c,l,e,b,m,ktype,t)
   alphas[8,:] = copy.copy(myR.alpha)
   hypos[8,:] = copy.copy(myR.h + myR.shift)
   print np.sum(hypos[8,:])
   myR = BHKRandomForest("../mnist_train_9.csv","../mnist_test_9.csv",b)
   myR.runForest2(n,c,l,e,b,m,ktype,t)
   alphas[9,:] = copy.copy(myR.alpha)
   hypos[9,:] = copy.copy(myR.h + myR.shift)
   print np.sum(hypos[9,:])
   # finally, we go through the rows of hypos and choose the largest as the 
   # class label
   final = np.zeros(np.size(myR.h)) 
   for i in range(0,np.size(final)):
      final[i] = np.argmax(hypos[:,i])
   # Now, we see how we did.
#    testd = (pd.read_csv("../mnist_test.csv",sep=',',header=None)).values
   self.testd = np.loadtxt("../mnist_test.csv",delimiter=',')
   testdata = np.ones((np.shape(testd)[0],np.shape(testd)[1]))
   testdata[:,:] = testd
   correct = 0
   total = np.size(myR.h)
   for i in range(0, total):
      if final[i] == testdata[i,0]:
         correct += 1
   print correct
   print (correct + .0)/total
   return [alphas, hypos, final]
 
