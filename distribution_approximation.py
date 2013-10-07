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
    mean = [1.]
    cov = [[5.]]
    numData = 50
    numBins = 25
    beta = 10.
    degree = 3
    coef = 1.
    alpha = 10.

    dist = NormalDistribution(mean, cov)
    X = dist.create(numData)
    #print X

    plt.subplot(321)
    plt.hist(X, bins=numBins)
    plt.title("Histgram")

    gk = GaussKernel(beta)
    gram = gk.gram(X)
    membership = np.sum(gram, axis=0)
    y = np.c_[X, membership]
    y = y[y[:,0].argsort()]
    plt.subplot(322)
    plt.stem(y[:,0], y[:,1])
    plt.plot(X, [0.1]*len(X), 'xr')
    plt.title("Gauss Kernel (beta=%f)" % beta)

    lk = LaplaceKernel(alpha)
    gram = lk.gram(X)
    membership = np.sum(gram, axis=0)
    y = np.c_[X, membership]
    y = y[y[:,0].argsort()]
    plt.subplot(323)
    plt.stem(y[:,0], y[:,1])
    plt.plot(X, [0.1]*len(X), 'xr')
    plt.title("Laplace Kernel (alpha=%f)" % alpha)

    flk = FloatLinearKernel()
    nlk = NormalizedKernel(flk)
    gram = nlk.gram(X)
    membership = np.sum(gram, axis=0)
    y = np.c_[X, membership]
    y = y[y[:,0].argsort()]
    plt.subplot(324)
    plt.stem(y[:,0], y[:,1])
    plt.plot(X, np.zeros(len(X)), 'xr')
    plt.title("Linear Kernel")

    pk = PolyKernel(degree)
    lpk = NormalizedKernel(pk)
    gram = lpk.gram(X)
    membership = np.sum(gram, axis=0)
    y = np.c_[X, membership]
    y = y[y[:,0].argsort()]
    plt.subplot(325)
    plt.stem(y[:,0], y[:,1])
    plt.plot(X, [0.1]*len(X), 'xr')
    plt.title("Normalized %d-Poly Kernel (degree=%d, coef=0.0)" % (degree, degree))

    pk = PolyKernel(degree, coef)
    lpk = NormalizedKernel(pk)
    gram = lpk.gram(X)
    membership = np.sum(gram, axis=0)
    y = np.c_[X, membership]
    y = y[y[:,0].argsort()]
    plt.subplot(326)
    plt.stem(y[:,0], y[:,1])
    plt.plot(X, [0.1]*len(X), 'xr')
    plt.title("Normalized %d-Poly Kernel (degree=%d, coef=%f)" % (degree, degree, coef))

    plt.show()
