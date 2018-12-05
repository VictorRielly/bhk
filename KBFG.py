############################################################
# This code was written by Victor Rielly to implement the
# BHK binary classifier. To use this code, 
#
# 1: Create a BHK object
#
# myBHK = BHK("trainfile","testfile",pownum,const,m)
# where "trainfile" is the name of your file to be used for
# training data, "testfile" is the name of your file to be
# used for testing, and the kernel function is 
# (x*y + const)**pownum. Notice, x is scaled to have 
# average norm 1. m is a number between 0 and 1 that 
# controls the regularization of the covariance matrix. Any
# number of the parameters may be entered. All that are not
# entered will be filled in with default values
# Training and test files are assumed to be csv files of
# format:
# class1,x_{1,1},x_{1,2},...,x_{1,d}
# .
# .
# .
# classn,x_{n,1},x_{n,2},...,x_{n,d}
# with no text headers. For a computer with 12 Gigs of ram
# the training set should have no more than 20000 instances,
# otherwise a memory error will be thrown.
#
# 2: call compute_all(p)
#
# m regularizes the covariance matrix while p sets some
# data asside for training hyperparameters. m is a float 
# that ranges from 0 to 1 indicating the percent of the 
# data to be used to regularize the covariance matrix.
# (usually .2) while p is a float from 0 to 1 indicating
# what percent of the data to set asside.
#
# 3: call compute_alpha(l)
#
# l is the regularization constant.
# The alpha vector is then stored in the alpha member of
# your BHK instance. This vector corresponds to the training
# data in the traintrain member of your class with positive
# entries from 0 to a and negative entries from c to b.
# where a is the total number of positive instances minus
# the instances set aside to make the covariance matrix 
# more robust, and c is the index that corresponds to the
# first negative instance, and b is the total number of
# datapoints minus the negative instances set asside to make
# the covariance matrix more robust. Consult the verify 
# function for help.
#
# 4: call verify()
#
# This runs the classifier on the subset of the training
# set used for training, the subset of the training set 
# used for validation, as well as the test set.
############################################################
from sklearn import svm
import sklearn
import numpy as np
import csv
# To be used to read csv files
# more efficiently
import pandas as pd
import copy
import time
import math
import scipy
# Seed the random number generator
np.random.seed()
# garbage collector. We will not need this
import gc
class BHK:
   #########################################################
   # Constructor for a BHK binary classifier
   # object. It takes in an array of arguments
   # the first (zeroeth argument is self, the second,
   # if it exists, is the string pointing to the input
   # csv file for the training set, the third is a
   # string indicating the test csv file, etc.
   # The imput training and test files are assumed to have
   # the test field in the first entry of every row.
   #########################################################
   def __init__(*arg):
      self = arg[0]
      self.train = '../mnist_train_0.csv'
      self.test = '../mnist_test_0.csv'
      numi = 0
      # our default polynomial classifier will be
      # of order 2
      self.pow = 2
      # and have a constant 1
      self.c = 1
      # by default, m will be .2
      self.m = .2
      # Here we process and store the arguments
      for i in arg:
         if numi == 1:
            if i == "":
               self.train = "";
               self.test = "";
            else :
               self.train = i;
         if numi == 2:
            self.test = i
         if numi == 3:
            self.pow = i
         if numi == 4:
            self.c = i
         if numi == 5:
            self.m = i
         numi += 1
      # If we don't want to process any files of data but
      # just want a BHK object, we will pass an empty 
      # string to the arguments, in which case train and
      # test will be set to "". 
      if self.train != "":
         self.testd=(pd.read_csv(self.test,sep=',',header=None)).values
         self.traind=(pd.read_csv(self.train,sep=',',header=None)).values
         self.traindata=np.ones((np.shape(self.traind)[0],np.shape(self.traind)[1]))
         self.testdata=np.ones((np.shape(self.testd)[0],np.shape(self.testd)[1]))
         self.testdata[:,:] = self.testd
         self.traindata[:,:] = self.traind
         # The following vector will hold the magnitudes
         # of the data in the training set. The only
         # preprocessing we do is we scale by the average
         # magnitude of our training vectors so that the
         # average magnitude of the preprocessed vectors is
         # 1.
         self.mags=np.zeros(np.shape(self.traindata)[0])
         index=0
         for i in self.traindata[1:,:]:
            self.mags[index]=np.sqrt(np.dot(i,i))
            index += 1
         self.ave=np.sum(self.mags)/np.size(self.mags)
         # the first index of each of the rows of training
         # data and test data are the classes. For mnist, 
         # the class is an integer from 0 to 9.
         self.traindata[:,1:]=self.traindata[:,1:]/self.ave
         self.testdata[:,1:]=self.testdata[:,1:]/self.ave 
   #########################################################
   # The following set of functions determine our kernels,
   # the linear, as well as polynomial kernels will be 
   # implemented with the pol_kernel function. For
   # computational complexity considerations, it would 
   # benefit us to separate linear and polynomial kernels
   # but for code simplicity we implement both in one
   # function
   #########################################################
   # Computes the polynomial kernel of the matrix stored in
   # self.mt1 and self.mt2 and stores the answer in
   # self.mta1. The kernel function is (x*y+c)^n
   def pol_kernel(self):
      self.mta1 = ((np.dot(self.mt1,self.mt2)+self.c)**self.pow)
   # The following block of code computes the covariance 
   # matrix of our classifier. We may only want to use a 
   # certain portion of our training set for the covariance
   # matrix. The rest of the training set we may want to use
   # for finding our hyper parameters. M tells us how many 
   # points to set asside to make the covariance matrix 
   # robust while p is used to set asside a portion of the
   # training set for hyper parameters. This function assumes
   # the class labels are 1 for positive class and 0 for 
   # negative.
   def compute_all(self,p):
      # shuffle our training set rows
      np.random.shuffle(self.traindata);
      # In self.n we will hold the number of training points
      # used in computing the covariance matrix
      self.n = int(p*np.shape(self.traindata)[0]);
      print self.n;
      # these will be stored in self.traintrain
      self.traintrain = self.traindata[:self.n,:]
      self.traintest = self.traindata[self.n:,:]
      # This sorts the training set so the rows of class 1
      # come before the rows of class 0
      self.traintrain = self.traintrain[np.lexsort(np.fliplr(self.traintrain).T)]
      # This sorts the training test set similarly.
      self.traintest = self.traintest[np.lexsort(np.fliplr(self.traintest).T)]
      self.traindata[:self.n,:] = self.traintrain[:,:]
      self.traindata[self.n:,:] = self.traintest[:,:]
      print self.traindata[:10,0]
      self.traintrain = self.traindata[:self.n,1:];
      self.traintest = self.traindata[self.n:,1:];
      # The true test set (from the test data) will be stored
      # in self.testtest
      self.testtest = self.testdata[:,1:];
      # Store the matrices for computing gram matrix
      self.mt1 = self.traintrain;
      self.mt2 = self.traintrain.T;
      # compute the gram matrix and store it in self.mta1
      self.pol_kernel();
      self.k = self.mta1;
      # Here is where we use the assumption that the classes
      # are labeled 1 for all positive instances and 0 for
      # all negative instances. We shouldn't need to cast
      # to an int but its better to be safe than sorry.
      self.posnum = int(np.sum(self.traindata[:self.n,0]));
      print self.posnum;
      self.negnum = self.n - self.posnum;
      # In these lines we pull out the positive and negative
      # rows we will use in our covariance matrix. The rest
      # will be used to improve our estimates of the
      # covariance matrices but not used as rows.
      # First, we compute our positive and negative class
      # gram matrix row means.
      self.M = int(self.m*np.shape(self.k)[0]);
      self.kplus = np.zeros(self.n - 2 - self.M);
      # this will make our code prettier. negdif is the 
      # proportion on the m rows that will be removed from
      # the negative instances. The rest will be removed
      # from the positive instances.
      negdif = int(self.negnum*self.M/(.0+self.n))
      self.negdif = negdif;
      # train temp is a submatrix of k that removes the last
      # 1+negdif negative class columns of k and the last
      # 1+m-negdif positive class columns of k.
      self.traintemp = np.zeros([np.shape(self.k)[0],np.shape(self.k)[0]-2-self.M]);
      print "Defined traintemp";
      print np.shape(self.traintemp);
      print negdif;
      print self.negnum;
      self.traintemp[:,:self.negnum-1-negdif]=self.k[:,:self.negnum-1-negdif];
      print "set part of traintemp";
      self.traintemp[:,self.negnum-1-negdif:]=self.k[:,self.negnum:-1-self.M+negdif];
      # compute the means of our positive and negative 
      # rows of traintemp
      self.kplus=(1.0/self.posnum)*(self.traintemp[self.negnum:,:]).sum(axis=0);
      self.kminus=(1.0/self.negnum)*(self.traintemp[:self.negnum,]).sum(axis=0);
      # computes our positive C_{+} matrix
      self.g=(1.0/self.posnum)*np.dot((self.traintemp[self.negnum:,:]-self.kplus).T,(self.traintemp[self.negnum:,:]-self.kplus));
      # adds to the positive covariance matrix the negative
      # covariance matrix. To make this code FDA, just
      # change the first part of these two lines 
      # (1.0/self.posnum), and (1.0/self.negnum) to the
      # appropriate constant.
      self.g+=(1.0/self.negnum)*np.dot((self.traintemp[:self.negnum,:]-self.kminus).T,(self.traintemp[:self.negnum,:]-self.kminus));
   # Computes alpha with (C+l)^{-1}(kplus-kminus)
   def compute_alpha(self,l):
      squareTemp=np.zeros([np.shape(self.k)[0]-2-self.M,np.shape(self.k)[0]-2-self.M]);
      a = self.negnum-1-self.negdif;
      b = -1-self.M+self.negdif;
      squareTemp[:a,:]=self.traintemp[:a,:];
      squareTemp[a:,:]=self.traintemp[self.negnum:b,:];
      self.alpha=np.dot(np.linalg.inv(.5*self.g+squareTemp*l),(self.kplus-self.kminus));      
   # Prints the roc auc scores for test set, the training
   # set and the set from the training set that is set
   # asside for training hyperparameters
   def verify(self):
      c = self.negnum
      a = c-1-self.negdif;
      b = -1-self.M+self.negdif;
      self.mt1 = np.concatenate((self.traintrain[:a,:],self.traintrain[c:b,:]),axis=0) 
      self.mt2 = (self.traintrain[:,:]).T
      self.pol_kernel()
      self.h = np.dot(self.alpha,self.mta1)
      print "The roc score on the training set is:";
      print "(It should be close to 1)";
      print sklearn.metrics.roc_auc_score(self.traindata[:self.n,0],self.h);
      self.mt2 = (self.traintest[:,:]).T
      self.pol_kernel()
      self.h = np.dot(self.alpha,self.mta1)
      print "The roc score on the test subset of the"
      print "training set is:";
      print sklearn.metrics.roc_auc_score(self.traindata[self.n:,0],self.h);
      self.mt2 = (self.testtest[:,:]).T
      self.pol_kernel()
      self.h = np.dot(self.alpha,self.mta1)
      print "The roc_auc on the test set is:";
      print sklearn.metrics.roc_auc_score(self.testdata[:,0],self.h);
      

