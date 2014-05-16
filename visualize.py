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

def effectCausedByClassImbalance():
    N = 1000
    rate = 100.
    X = np.zeros((N,2))
    label = np.zeros(N)
    m = int( N * ( 1 / (rate+1) ) )

    mean = [-10, -10]
    cov = [[50,0],[0,50]]
    label[:m] = 1.
    X[:m,0], X[:m,1] = np.random.multivariate_normal(mean, cov, m).T

    mean = [25, 25]
    cov = [[75,0],[0,75]]
    label[m:] = -1.
    X[m:,0], X[m:,1] = np.random.multivariate_normal(mean, cov, N-m).T

    print "positive: ", m
    print "negative: ", N-m

    kernel = GaussKernel(0.0010)
    #kernel = PolyKernel(7)
    #kernel = FloatLinearKernel()
    gram = kernel.gram(X)
    clf = svm.SVC(kernel='precomputed')
    clf.fit(gram, label)

    yi = label[clf.support_[0]]
    xi = X[clf.support_[0], :]
    f = DecisionFunction(kernel, clf, X, label)
    plt = draw_contour(f.eval, [-50,50,50,-50], plot=plt, density=0.5)

    plt.plot(X[:m,0],X[:m,1], "bo")
    plt.plot(X[m:,0],X[m:,1], "ro")
    plt.plot(X[clf.support_,0],X[clf.support_,1], "go")
    plt.show()

def fuzzyMembership():
    N = 50
    X = np.array([ (i,j) for j in range(-N/2,N/2) for i in range(-N/2,N/2) ], dtype=np.float)
    Y = np.ones(len(X), dtype=np.int)

    lower, upper = -0.4*N, 0.4*N
    h_w = 1 # the half of width
    for i,x in enumerate(X):
        if (lower - h_w <= x[0] < upper + h_w) and (lower - h_w <= x[1] < upper + h_w):
            if (-h_w <= x[1] < h_w) or (x[1] < lower + h_w) or (upper - h_w <= x[1]):
                Y[i] = -1
            elif ((x[0] < lower + h_w) and (x[1] < 0)) \
                    or ((upper - h_w <= x[0]) and (x[1] >= 0)):
                Y[i] = -1

    dataset = np.c_[X, Y]
    dataset = np.r_[dataset[Y[:]==-1,:], dataset[Y[:]==1,:]]
    X, Y = dataset[:,:-1], dataset[:,-1]

    from matplotlib import pyplot as plt
    plt.plot(X[Y[:]==1,0],X[Y[:]==1,1], "bo")
    plt.plot(X[Y[:]==-1,0],X[Y[:]==-1,1], "ro")
    plt.show()

    W = proposed(X, Y)
    W_map = np.zeros((N,N))
    for i in np.c_[X[:,0]+N/2, X[:,1]+N/2+1, Y*W]:
        W_map[int(i[0])][-int(i[1])] = i[2]
    hinton(W_map)
    plt.show()

    from matplotlib import pyplot as plt
    plt.plot(X[Y[:]==1,0],X[Y[:]==1,1], "bo")
    plt.plot(X[Y[:]==-1,0],X[Y[:]==-1,1], "ro")
    plt.show()

def proposed(X, Y):
    kernel = GaussKernel(0.001)
    gram = kernel.gram(X)
    numN = len(Y[:]==-1)
    gramN, gramP = gram[:numN,:numN], gram[numN:,numN:]
    weightN, weightP = np.sum(gramN, axis=1), np.sum(gramP, axis=1)
    return np.r_[weightN/numN,  weightP/(len(Y)-numN)]

def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x,y),w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

if __name__ == '__main__':
    fuzzyMembership()
