#coding: utf-8;
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from matplotlib import lines
from sklearn import svm

import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
from kernel import *
from mlutil import *

class KernelDensityEstimater():
    def __init__(self, dim, beta):
        self.dim = dim
        self.variance = 1. / (2 * beta)
        self.kernel = GaussKernel(beta)

    def estimate(self, sample):
        self.sample = sample
        n = np.sqrt( 2. * np.pi * self.variance )**self.dim
        h = sample.shape[0]
        self.nConst = n * h
        print "normalize constant: ", self.nConst

    def prob(self, x):
        buf = [ self.kernel.val(x, xi) for xi in self.sample ]
        return sum(buf) / self.nConst

if __name__ == '__main__':
    N = 15
    X = np.zeros((N,2))

    mean = [0, 0]
    cov = [[4,0],[0,4]]
    X[:,0], X[:,1] = np.random.multivariate_normal(mean, cov, N).T

    kde = KernelDensityEstimater(2, 0.50)
    kde.estimate(X)

    coodinates = [-8,8,8,-8]
    density = 10
    x0 = coodinates[0]
    y0 = coodinates[1]
    x1 = coodinates[2]
    y1 = coodinates[3]
    w = int(abs(x1 - x0)*density)
    h = int(abs(y1 - y0)*density)
    I = np.linspace(x0, x1, num = w)
    J = np.linspace(y1, y0, num = h)

    buf = 0.
    i,j = np.meshgrid(I, J)
    for p in range(h):
        for q in range(w):
            buf += kde.prob( np.array( [i[p,q], j[p,q]] ) )
            print buf
    print buf / density**2

