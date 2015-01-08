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

def circleDist(mu, var, num):
    R = np.random.normal(mu, var, num)
    PHI = np.random.uniform(0, 2*np.pi, num)
    D = np.zeros((num,2))
    for i,(r,p) in enumerate(zip(R,PHI)):
        D[i][0] = r * np.sin(p)
        D[i][1] = r * np.cos(p)
    return np.array(D)

def regularize(X):
    maxX, minX = max(X), min(X)
    return np.array([ i / float(maxX - minX) for i in (X[:] - minX) ])

if __name__ == '__main__':
    # initialize random seed
    np.random.seed(0)

    # parameter settings
    rect = [-4,4,-4,4]
    beta = 1.
    alpha = 0.70
    linewidths = 0.4
    scale = 10

    #create NEG_X
    nPosData1 = 850
    nPosData2 = 150
    r_mean1 = 2.5
    r_var1 = 0.5
    r_mean2 = 0.75
    r_var2 = 0.2

    NEG_X = np.r_[
            circleDist(r_mean1, r_var1, nPosData1),
            circleDist(r_mean2, r_var2, nPosData2),
        ]

    #create POS_X
    nNegData = 100
    r_mean1 = 1.5
    r_var1 = 0.2

    POS_X = np.r_[
            circleDist(r_mean1, r_var1, nNegData),
        ]

    #adjust the space between subplots
    plt.subplots_adjust(hspace=0.3)

    #plot examples
    plt.subplot(221)
    plt.axis(rect)
    plt.scatter(NEG_X[:,0], NEG_X[:,1], c='b', linewidths=linewidths)
    plt.scatter(POS_X[:,0], POS_X[:,1], c='r', linewidths=linewidths)

    #plot weighted examples by GaussKernel
    #--------------------------------------------------
    gk = GaussKernel(beta)

    NEG_gram = gk.gram(NEG_X)
    NEG_membership = np.sum(NEG_gram, axis=0) / float(NEG_gram.shape[0])
    POS_gram = gk.gram(POS_X)
    POS_membership = np.sum(POS_gram, axis=0) / float(POS_gram.shape[0])

    plt.subplot(222)
    plt.title("Gaussian Kernel (beta=%.2f)" % beta)
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

    #plot weighted examples by PolyKernel
    #--------------------------------------------------
    pk = PolyKernel(2, 0)
    npk = NormalizedKernel(pk)

    NEG_gram = npk.gram(NEG_X)
    NEG_membership = np.sum(NEG_gram, axis=0) / float(NEG_gram.shape[0])
    POS_gram = npk.gram(POS_X)
    POS_membership = np.sum(POS_gram, axis=0) / float(POS_gram.shape[0])

    plt.subplot(223)
    plt.title("poly kernel (d=2, c=0.00)")
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

    #plot weighted examples by High-dimensional PolyKernel
    #--------------------------------------------------
    pk = PolyKernel(32, 0)
    npk = NormalizedKernel(pk)

    NEG_gram = npk.gram(NEG_X)
    NEG_membership = np.sum(NEG_gram, axis=0) / float(NEG_gram.shape[0])
    POS_gram = npk.gram(POS_X)
    POS_membership = np.sum(POS_gram, axis=0) / float(POS_gram.shape[0])

    plt.subplot(224)
    plt.title("poly kernel (d=32, c=0.00)")
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
