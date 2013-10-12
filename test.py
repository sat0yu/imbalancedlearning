#coding: utf-8;
import numpy as np
import os
import sys
from sklearn import svm
from dataset import *
import multiprocessing

import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
from nonvectorial import *

if __name__ == '__main__':
    dataset = np.loadtxt("data/SMSSpamCollection.rplcd", delimiter='\t', dtype={'names':('0','1'), 'formats':('f8','S512')})
    label = dataset['0']
    X = dataset['1']

    sp = SpectrumKernel(2)
    print sp.gram(X[:100])
