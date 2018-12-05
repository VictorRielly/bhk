from sklearn import svm
import sklearn
import numpy as np
import csv
import pandas as pd
import datetime
import copy
import math
import scipy
import gc # GC is a manual garbage collector

np.random.seed()
class BHK:
    def __init__(*arg):
        self = arg[0]
        self.train = './mnist_train_0.csv'
        self.test = './mnist_test_0.csv'
        numi = 0
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
                self.eta = i
            numi += 1
        if self.train != "":
            self.testd =(pd.read_csv(self.te:x

