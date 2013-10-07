#coding: utf-8;
import numpy as np
import os
import sys
from matplotlib import pyplot as plt

import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
from kernel import *
from mlutil import *


if __name__ == '__main__':
    # initialize random seed
    np.random.seed(0)

    # parameter settings
    mean = [10.]
    cov = [[3.]]
    numData = 50
    numBins = 25
    beta = 10.
    degree = 2
    alpha = 10.

    dist = NormalDistribution(mean, cov)
    X = dist.create(numData)
    #print X

    plt.subplot(221)
    plt.hist(X, bins=numBins)
    plt.title("Histgram")


    gk = GaussKernel(beta)
    gram = gk.gram(X)
    membership = np.sum(gram, axis=0)
    y = np.c_[X, membership]
    y = y[y[:,0].argsort()]
    plt.subplot(222)
    plt.stem(y[:,0], y[:,1])
    plt.plot(X, [0.1]*len(X), 'xr')
    plt.title("Gauss Kernel (beta=10.0)")

    pk = PolyKernel(degree)
    lpk = NormalizedKernel(pk)
    gram = lpk.gram(X)
    membership = np.sum(gram, axis=0)
    y = np.c_[X, membership]
    y = y[y[:,0].argsort()]
    plt.subplot(223)
    plt.stem(y[:,0], y[:,1])
    plt.plot(X, [0.0005]*len(X), 'xr')
    plt.title("Normalized 2-Poly Kernel (degree=2, coef=0)")


    lk = LaplaceKernel(alpha)
    gram = lk.gram(X)
    membership = np.sum(gram, axis=0)
    y = np.c_[X, membership]
    y = y[y[:,0].argsort()]
    plt.subplot(224)
    plt.stem(y[:,0], y[:,1])
    plt.plot(X, [0.1]*len(X), 'xr')
    plt.title("Laplace Kernel (alpha=10.)")

    plt.show()
