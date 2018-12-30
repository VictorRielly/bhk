# This is the script used to compare
# svm to roc optimization if an error 
# exitsts it would probably live somewhere
# in here.
import bhk
import numpy as np
import scipy

class SVMTest:
   def __init__(*arg):
       self = arg[0]
       self.kernelType = 'k'
       numi = 0
       self.n = 2
       self.c = 1
       self.train = '../mnist_train_0.csv'
       self.test = '../mnist_test_0.csv'
       for i in arg:
           if numi == 1:
               self.train = i
           if numi == 2:
               self.test = i
           numi += 1
       self.myB = bhk.BHK(self.train,self.test)
       self.temp = bhk.copy.copy(self.myB.traindata)
       self.temp2 = bhk.copy.copy(self.myB.testdata)
       # This is the pretest, it finds the ideal regularization
       # constant for the problem, it assumes the regularization
       # constant is greater than 2^(-100). Then it doubles the
       # regularization constant until the bestRoc has not improved
       # for 20 steps.
       self.bestRoc = 0
       self.bestRocL = 0
       self.numDec = 0
       self.myL = 2**(0)
       self.myB.traindata = bhk.copy.copy(self.temp[:3000,:])
       while self.numDec < 2:
           self.myB.compute_all(self.n,self.c,600)
           self.myB.compute_alpha7(self.n,self.c,self.myL,600)
           self.ea =  self.myB.evaluate_alpha(self.n,self.c,600)
           self.ta =  self.myB.test_alpha(self.n,self.c,600)
           self.va =  self.myB.verify_alpha(self.n,self.c,600)
           print "evaluation: " + str(self.ea)
           print "test: " + str(self.ta)
           print "evaluation: " + str(self.va)
           if not (self.ea < self.bestRoc):
               self.bestRoc = self.ea
               self.bestRocL = self.myL
               self.numDec = 0
           else :
               self.numDec += 1
           self.myL *= 2

       self.bestRoc = 0
       self.bestRocL = 0
       self.numDec = 0
       self.gausL = 2**(-10)
       notChanged = 0
       while (self.numDec < 2) and (notChanged < 10):
          svmMachine=bhk.sklearn.svm.SVC(C=self.gausL,cache_size=8000,kernel='poly',degree=2,coef0=1)
          svmMachine.fit(self.myB.traindata[:,1:],self.myB.traindata[:,0])
          self.temptean = svmMachine.predict(self.myB.testtest)
          self.ta=bhk.sklearn.metrics.roc_auc_score(self.myB.testdata[:,0],self.temptean)
          if self.ta == self.bestRoc:
              notChanged += 1
          if not (self.ta < self.bestRoc):
              self.bestRoc = self.ta
              self.bestRocL = self.gausL
              self.numDec = 0
          else :
              self.numDec += 1
          print self.gausL
          print self.ta
          self.gausL *= 2

       # This array will store the roc auc's on the 
       # test set of all 60 classifications.
       self.answers = bhk.np.zeros(20)
       self.svmAnswers = bhk.np.zeros(20)
       print self.gausL
       print self.bestRocL
       for i in range(0,20):
          self.myB.traindata = bhk.copy.copy(self.temp[i*3000:(i+1)*3000,:])
          self.myB.testdata = bhk.copy.copy(self.temp2[i*500:(i+1)*500,:])
          self.myB.compute_all(self.n,self.c,600)
          self.myB.compute_alpha7(self.n,self.c,self.bestRocL,600)
          self.answers[i] = self.myB.test_alpha(self.n,self.c,600)
          svmMachine=bhk.sklearn.svm.SVC(C=self.gausL,cache_size=8000,kernel='poly',degree=2,coef0=1)
          svmMachine.fit(self.myB.traindata[:,1:],self.myB.traindata[:,0])
          self.tempans = svmMachine.predict(self.myB.testtest)
          print self.tempans
          self.svmAnswers[i]=bhk.sklearn.metrics.roc_auc_score(self.myB.testdata[:,0],self.tempans)

runtest = SVMTest('../mnist_train_0.csv','../mnist_test_0.csv')
x = runtest.answers - runtest.svmAnswers
results = scipy.stats.wilcoxon(x,y=None,zero_method='wilcox',correction=False)
print results.pvalue
runtest = SVMTest('../mnist_train_1.csv','../mnist_test_1.csv')
x = runtest.answers - runtest.svmAnswers
results = scipy.stats.wilcoxon(x,y=None,zero_method='wilcox',correction=False)
print results.pvalue
runtest = SVMTest('../mnist_train_2.csv','../mnist_test_2.csv')
x = runtest.answers - runtest.svmAnswers
results = scipy.stats.wilcoxon(x,y=None,zero_method='wilcox',correction=False)
print results.pvalue
runtest = SVMTest('../mnist_train_3.csv','../mnist_test_3.csv')
x = runtest.answers - runtest.svmAnswers
results = scipy.stats.wilcoxon(x,y=None,zero_method='wilcox',correction=False)
print results.pvalue
runtest = SVMTest('../mnist_train_4.csv','../mnist_test_4.csv')
x = runtest.answers - runtest.svmAnswers
results = scipy.stats.wilcoxon(x,y=None,zero_method='wilcox',correction=False)
print results.pvalue
runtest = SVMTest('../mnist_train_5.csv','../mnist_test_5.csv')
x = runtest.answers - runtest.svmAnswers
results = scipy.stats.wilcoxon(x,y=None,zero_method='wilcox',correction=False)
print results.pvalue
runtest = SVMTest('../mnist_train_6.csv','../mnist_test_6.csv')
x = runtest.answers - runtest.svmAnswers
results = scipy.stats.wilcoxon(x,y=None,zero_method='wilcox',correction=False)
print results.pvalue
runtest = SVMTest('../mnist_train_7.csv','../mnist_test_7.csv')
x = runtest.answers - runtest.svmAnswers
results = scipy.stats.wilcoxon(x,y=None,zero_method='wilcox',correction=False)
print results.pvalue
runtest = SVMTest('../mnist_train_8.csv','../mnist_test_8.csv')
x = runtest.answers - runtest.svmAnswers
results = scipy.stats.wilcoxon(x,y=None,zero_method='wilcox',correction=False)
print results.pvalue
runtest = SVMTest('../mnist_train_9.csv','../mnist_test_9.csv')
x = runtest.answers - runtest.svmAnswers
results = scipy.stats.wilcoxon(x,y=None,zero_method='wilcox',correction=False)
print results.pvalue
