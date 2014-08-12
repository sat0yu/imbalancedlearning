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
    elif RGB == 'G':
        cidx = (0,2)
    elif RGB == 'B':
        cidx = (0,1)
    else:
        raise ValueError("an invalid value is given as RGB: %s" % RGB)

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
    rect = [-4,4,-4,4]
    beta = 1.
    delta = 10.**-6
    alpha = 0.70
    linewidths = 0.4
    scale = 100

    #create NEG_X
    nPosData1 = 180
    nPosData2 = 320

    mean1 = [0., 0.]
    cov1 = [[1.,0.],[0., 1.]]
    mean2 = [-2.5, -2.5]
    mean3 = [2.5, -2.5]
    mean4 = [-2.5, 2.5]
    mean5 = [2.5, 2.5]
    cov2 = [[0.2,0.],[0., 0.2]]

    dist1 = NormalDistribution(mean1, cov1)
    dist2 = NormalDistribution(mean2, cov2)
    dist3 = NormalDistribution(mean3, cov2)
    dist4 = NormalDistribution(mean4, cov2)
    dist5 = NormalDistribution(mean5, cov2)

    NEG_X = np.r_[
            dist1.create(nPosData1),
            dist2.create(nPosData2/4),
            dist3.create(nPosData2/4),
            dist4.create(nPosData2/4),
            dist5.create(nPosData2/4),
        ]

    #create POS_X
    nNegData = 52

    mean1 = [-1.5, -1.5]
    mean2 = [1.5, -1.5]
    mean3 = [-1.5, 1.5]
    mean4 = [1.5, 1.5]
    cov = [[0.1,0.],[0., 0.1]]

    dist1 = NormalDistribution(mean1, cov)
    dist2 = NormalDistribution(mean2, cov)
    dist3 = NormalDistribution(mean3, cov)
    dist4 = NormalDistribution(mean4, cov)

    POS_X = np.r_[
            dist1.create(nNegData/4),
            dist2.create(nNegData/4),
            dist3.create(nNegData/4),
            dist4.create(nNegData/4),
        ]

    #plot examples
    plt.subplot(211)
    plt.axis(rect)
    plt.scatter(NEG_X[:,0], NEG_X[:,1], c='b', linewidths=linewidths)
    plt.scatter(POS_X[:,0], POS_X[:,1], c='r', linewidths=linewidths)

    #plot weighted examples
    gk = GaussKernel(beta)

    NEG_gram = gk.gram(NEG_X)
    NEG_membership = np.sum(NEG_gram, axis=0) / float(NEG_gram.shape[0])
    POS_gram = gk.gram(POS_X)
    POS_membership = np.sum(POS_gram, axis=0) / float(POS_gram.shape[0])

    plt.subplot(212)
    plt.axis(rect)
    plt.scatter(
            NEG_X[:,0], NEG_X[:,1],
            s=scale*regularize(NEG_membership),
            c=convert2monogradation(NEG_membership, RGB='B', alpha=alpha),
            linewidths=linewidths
        )
    ratio = (NEG_gram.shape[0] / POS_gram.shape[0])
    plt.scatter(
            POS_X[:,0], POS_X[:,1],
            s=ratio*scale*regularize(POS_membership),
            c=convert2monogradation(ratio*POS_membership, RGB='R', alpha=alpha),
            linewidths=linewidths
        )

    plt.savefig('fig_example_dist.eps')
