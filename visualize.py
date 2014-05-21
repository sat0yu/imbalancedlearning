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
from cil import *

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

    plt.plot(X[Y[:]==1,0],X[Y[:]==1,1], "bo")
    plt.plot(X[Y[:]==-1,0],X[Y[:]==-1,1], "ro")
    plt.show()

    #X, Y, W = proposed(X, Y)
    #X, Y, W = fsvmcil(X, Y, 'center', 'linear')
    X, Y, W = fsvmcil(X, Y, 'estimate', 'linear')
    #X, Y, W = fsvmcil(X, Y, 'hyperplane', 'linear')
    #X, Y, W = fsvmcil(X, Y, 'center', 'exp')
    #X, Y, W = fsvmcil(X, Y, 'estimate', 'exp')
    #X, Y, W = fsvmcil(X, Y, 'hyperplane', 'exp')

    print "W is in [%s, %s] and has a mean %s, a variance %s" % (min(W), max(W), np.mean(W), np.var(W))
    W_map = np.zeros((N,N))
    for i in np.c_[X[:,0]+N/2, X[:,1]+N/2, Y*W]:
        W_map[int(i[1])][-int(i[0])] = i[2]
    fig, ax = plt.subplots()
    ax.imshow(W_map, cmap=plt.cm.seismic_r, interpolation='nearest')
    plt.show()

def fsvmcil(X, Y, distance_function='center', decay_function='linear'):
    beta = 0.001
    delta = 1.
    gamma = 0.1
    clf = FSVMCIL(beta, distance_function=distance_function, decay_function=decay_function, delta=delta, gamma=gamma)

    if distance_function is 'center':
        X, Y, d = clf.dist_from_center(X, Y)
    elif distance_function is 'estimate':
        X, Y, d = clf.dist_from_estimated_hyperplane(X, Y)
    elif distance_function is 'hyperplane':
        X, Y, d = clf.dist_from_hyperplane(X, Y)
    else:
        raise ValueError()

    if decay_function is 'linear':
        W = clf.linear_decay_function(d)
    elif decay_function is 'exp':
        W = clf.exp_decay_function(d)
    else:
        raise ValueError()

    return (X, Y, W)

def proposed(X, Y):
    kernel = GaussKernel(0.001)
    gram = kernel.gram(X)
    numN = len(Y[:]==-1)
    gramN, gramP = gram[:numN,:numN], gram[numN:,numN:]
    weightN, weightP = np.sum(gramN, axis=1), np.sum(gramP, axis=1)
    return (X, Y, np.r_[weightN/numN,  weightP/(len(Y)-numN)])

if __name__ == '__main__':
    fuzzyMembership()
