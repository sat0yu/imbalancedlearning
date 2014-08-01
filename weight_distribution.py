#coding: utf-8;
import numpy as np
import os
import sys
from matplotlib import pyplot as plt

import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
from kernel import *
from mlutil import *

def convert2monogradation(X, RGB='R', alpha=1.):
    cidx = None
    if RGB == 'R':
        cidx = (1,2)
    if RGB == 'G':
        cidx = (0,2)
    if RGB == 'B':
        cidx = (0,1)
    else:
        raise ValueError("an invalid value is given for RGB")

    N = X.shape[0]
    RGBA = np.ones((N, 4))
    RGBA[:, 3] = alpha
    grad = 1 - regularize(X)
    RGBA[:,cidx[0]] = grad
    RGBA[:,cidx[1]] = grad

    return RGBA

def regularize(X):
    return np.array([ i / float(max(X)) for i in (X[:] - min(X)) ])

if __name__ == '__main__':
    # initialize random seed
    np.random.seed(0)

    # parameter settings
    mean1 = [0., 0.]
    cov1 = [[1.,0.],[0., 1.]]

    mean2 = [-2.5, -2.5]
    cov2 = [[0.2,0.],[0., 0.2]]

    mean3 = [2.5, -2.5]
    cov3 = [[0.2,0.],[0., 0.2]]

    mean4 = [-2.5, 2.5]
    cov4 = [[0.2,0.],[0., 0.2]]

    mean5 = [2.5, 2.5]
    cov5 = [[0.2,0.],[0., 0.2]]

    rect = [-4,4,-4,4]
    numData1 = 160
    numData2 = 80
    numData3 = 80
    numData4 = 80
    numData5 = 80
    beta = 1.
    delta = 10.**-6
    alpha = 0.7
    linewidths = 0.4
    scale = 100

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

    gk = GaussKernel(beta)
    gram = gk.gram(X)
    membership = np.sum(gram, axis=0) / float(gram.shape[0])
    plt.subplot(212)
    plt.axis(rect)
    plt.scatter(X[:,0], X[:,1], s=scale*regularize(membership), c=convert2monogradation(membership, 'B', alpha=alpha), linewidths=linewidths)
    #plt.colorbar()

    def linear_decay_function(X, delta):
        denominator = max(X) + delta
        return 1. - (X / denominator)

    def dist_from_center(X):
        # calc. center
        center = np.average(X, axis=0)
        # calc. distance between from center for each sample
        return np.sum(np.abs(X - center)**2, axis=-1)**(1/2.)

    plt.subplot(211)
    plt.axis(rect)
    fx = linear_decay_function(dist_from_center(X), delta)
    plt.scatter(X[:,0], X[:,1], s=scale*regularize(fx), c=convert2monogradation(fx, 'B', alpha=alpha), linewidths=linewidths)
    #plt.colorbar()

    plt.show()
