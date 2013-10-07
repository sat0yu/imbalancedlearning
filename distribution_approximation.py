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
    mean1 = [-10., 10.]
    cov1 = [[1.,0.],[0., 1.]]

    mean2 = [-5., 4.]
    cov2 = [[0.5,0.],[0., 0.5]]

    mean3 = [-5., 15.]
    cov3 = [[0.5,0.],[0., 0.5]]

    mean4 = [-15., 15.]
    cov4 = [[0.5,0.],[0., 0.5]]

    mean5 = [-15., 4.]
    cov5 = [[0.5,0.],[0., 0.5]]

    rect = [-17,-2,1,18]
    numData1 = 200
    numData2 = 50
    numData3 = 50
    numData4 = 50
    numData5 = 50
    beta = 10.
    degree = 3
    coef = 1.
    alpha = 10.

    dist1 = NormalDistribution(mean1, cov1)
    dist2 = NormalDistribution(mean2, cov2)
    dist3 = NormalDistribution(mean3, cov3)
    dist4 = NormalDistribution(mean4, cov4)
    dist5 = NormalDistribution(mean5, cov5)

    X = np.r_[
            dist1.create(numData1),
            dist2.create(numData2),
            dist3.create(numData3),
            dist4.create(numData4),
            dist5.create(numData5),
            ]
    #print X

    plt.subplot(321)
    plt.axis(rect)
    plt.scatter(X[:,0], X[:,1])
    plt.title("Scatter")

    gk = GaussKernel(beta)
    gram = gk.gram(X)
    membership = np.sum(gram, axis=0)
    plt.subplot(322)
    plt.axis(rect)
    plt.scatter(X[:,0], X[:,1], c=membership)
    plt.title("Gauss Kernel (beta=%f)" % beta)

    lk = LaplaceKernel(alpha)
    gram = lk.gram(X)
    membership = np.sum(gram, axis=0)
    plt.subplot(323)
    plt.axis(rect)
    plt.scatter(X[:,0], X[:,1], c=membership)
    plt.title("Laplace Kernel (alpha=%f)" % alpha)

    flk = FloatLinearKernel()
    nlk = NormalizedKernel(flk)
    gram = nlk.gram(X)
    membership = np.sum(gram, axis=0)
    y = np.c_[X, membership]
    y = y[y[:,0].argsort()]
    plt.subplot(324)
    plt.axis(rect)
    plt.scatter(X[:,0], X[:,1], c=membership)
    plt.title("Linear Kernel")

    pk = PolyKernel(degree)
    lpk = NormalizedKernel(pk)
    gram = lpk.gram(X)
    membership = np.sum(gram, axis=0)
    y = np.c_[X, membership]
    y = y[y[:,0].argsort()]
    plt.subplot(325)
    plt.axis(rect)
    plt.scatter(X[:,0], X[:,1], c=membership)
    plt.title("Normalized %d-Poly Kernel (degree=%d, coef=0.0)" % (degree, degree))

    pk = PolyKernel(degree, coef)
    lpk = NormalizedKernel(pk)
    gram = lpk.gram(X)
    membership = np.sum(gram, axis=0)
    y = np.c_[X, membership]
    y = y[y[:,0].argsort()]
    plt.subplot(326)
    plt.axis(rect)
    plt.scatter(X[:,0], X[:,1], c=membership)
    plt.title("Normalized %d-Poly Kernel (degree=%d, coef=%f)" % (degree, degree, coef))

    plt.show()
