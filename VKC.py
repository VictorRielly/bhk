import sklearn as sk
import numpy as np
from scipy import linalg as la
import csv
import pandas as pd
import copy
import time
import math
import scipy
from sklearn.svm import LinearSVC
# seed the random number generator
np.random.seed()
class AUC:
   #########################################################
   # Constructor for a generic AUC based binary classifier.
   # This class makes a transformation of variables to turn
   # a kernel AUC classification problem into a linear 
   # accuracy classification problem which can be solved 
   # using off the shelf linear classifiers such as 
   # logistic regression, 1-svm, 2-svm, or ridge regression.
   # the sklearn package is used to implement these off the
   # shelf classifiers. For the time being, we will only 
   # evaluate polynomial kernels, with our default
   # polynomial kernel of (x^Ty + 1)^2
   #########################################################
   def __init__(*arg):
      self = arg[0]
      self.train = '../mnist_train_0.csv'
      self.test = '../mnist_test_0.csv'
      numi = 0
      self.pow = 2
      self.c = 1
      for i in arg:
         if numi == 1:
            self.train = i
         if numi == 2:
            self.test = i
         if numi == 3:
            self.pow = i
         if numi == 4:
            self.c = i
         numi += 1
      if self.train != "":
         self.testd = (pd.read_csv(self.test,sep=',',header=None)).values
         self.traind = (pd.read_csv(self.train,sep=',',header=None)).values
         self.traindata = np.ones((np.shape(self.traind)[0],np.shape(self.traind)[1]))
         self.testdata = np.ones((np.shape(self.testd)[0],np.shape(self.testd)[1]))
         self.testdata[:,:] = self.testd
         self.traindata[:,:] = self.traind
         self.testd = None;
         self.traind = None;
         # We will scale our feature vectors so the average
         # l^2 norm of the vectors is 1 this will improve
         # the stability of the kernel matrix computations
         self.mags = np.zeros(np.shape(self.traindata)[0])
         index = 0
         for i in self.traindata:
            self.mags[index] = np.sqrt(np.dot(i,i))
            index += 1
         self.ave = np.sum(self.mags)/np.size(self.mags)
         self.traindata[:,1:]=self.traindata[:,1:]/self.ave
         self.testdata[:,1:] = self.testdata[:,1:]/self.ave
   def pol_kernel(self):
      self.mta1 = ((np.dot(self.mt1,self.mt2)+self.c)**self.pow)
   def comp_k(self):
      self.n = np.shape(self.traindata)[0]
      self.traindata = self.traindata[np.lexsort(np.fliplr(self.traindata).T)]
      self.mt1 = self.traindata[:,1:]
      self.mt2 = (self.traindata[:,1:]).T
      self.pol_kernel();
      self.k = self.mta1;

   def comp_svd(self):
      self.U, self.S, self.U = la.svd(self.k);
      self.sqS = np.sqrt(self.S)
      self.nsqS = 1.0/np.sqrt(self.S)

   def comp_kern_alph(self,regK):
      self.z = np.dot((self.U).T,(self.sqS*(self.U).T).T);
      self.svm = LinearSVC(fit_intercept=False,C=regK)
      self.svm.fit(self.z,self.traindata[:,0])
      self.beta = self.svm.coef_[0];
      self.alpha = np.dot((self.U).T,self.nsqS*np.dot(self.U,self.beta));
      self.h = self.svm.predict(self.z);
      # lets test the test set!
      # first we create gram matrix
      self.mt1 = self.testdata[:,1:];
      self.mt2 = (self.traindata[:,1:]).T;
      self.pol_kernel();
      self.k2 = self.mta1;
#      self.z = np.dot(self.U,np.dot(np.diag(self.nsqS),np.dot(self.V,(self.k).T)));
#      self.h = self.svm.predict((self.z).T);
      self.svm.coef_[0] = self.alpha;

   def comp_auc_alpha(self,regK,split):
      self.n = int(np.shape(self.traindata)[0]);
      self.nump = int(np.sum(self.traindata[:,0]));
      self.z = np.zeros((int(self.nump*(self.n-self.nump)),self.n));
      self.aurocy = np.ones(np.shape(self.z)[0]);
      par = 0;
      for i in range(0,self.nump):
         for j in range(0,self.n - self.nump):
            # send 1/split of the points to the other side
            self.z[i*(self.n - self.nump)+j] = (-1)**(par)*(self.k[self.n-self.nump+i,:] - self.k[j,:]);
            if par == 1:
               self.aurocy[i*(self.n-self.nump)+j] = -1;
            if j % split == 0: 
               par = 1;
            else :
               par = 0
      print np.shape(self.z)
      print np.shape(self.aurocy)
      print "continuing";
      self.z = np.dot((self.U).T,(self.nsqS*np.dot(self.U,(self.z).T)));
      self.svm = LinearSVC(fit_intercept=False,C=regK)
      self.svm.fit((self.z).T,self.aurocy)
      self.beta = self.svm.coef_[0];
      self.alpha = np.dot((self.U).T,self.nsqS*np.dot(self.U,self.beta));
      # self.h = self.svm.predict(self.z);
      # lets test the test set!
      # first we create gram matrix
      self.mt1 = self.testdata[:,1:];
      self.mt2 = (self.traindata[:,1:]).T;
      self.pol_kernel();
      self.k2 = self.mta1;
#      self.z = np.dot(self.U,np.dot(np.diag(self.nsqS),np.dot(self.V,(self.k).T)));
#      self.h = self.svm.predict((self.z).T);
      self.svm.coef_[0] = self.alpha;
   def eval_on_test(self):
      return sk.metrics.roc_auc_score(self.testdata[:,0],np.dot(self.k2,self.svm.coef_[0]));
        
