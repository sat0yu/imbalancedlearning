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

    magic = 20000
    kde = KernelDensityEstimater(2, 0.005)
    kde.estimate(X[:m,:])
    posWeight = magic*np.array([ kde.prob(xi) for xi in X[:m] ])
    print posWeight
    kde.estimate(X[m:,:])
    negWeight = magic*np.array([ kde.prob(xi) for xi in X[m:] ])
    print negWeight
    weights = np.r_[posWeight, negWeight]

    kernel = GaussKernel(0.005)
    gram = kernel.gram(X)
    clf = svm.SVC(kernel='precomputed')

    fig, axes = plt.subplots(1, 2)

    clf.fit(gram, label)
    yi = label[clf.support_[0]]
    xi = X[clf.support_[0], :]
    f = DecisionFunction(kernel, clf, X, label)
    axes[0] = draw_contour(f.eval, [-50,50,50,-50], plot=axes[0], density=0.5)
    axes[0].plot(X[:m,0],X[:m,1], "bo")
    axes[0].plot(X[m:,0],X[m:,1], "ro")
    axes[0].plot(X[clf.support_,0],X[clf.support_,1], "go")

    clf.fit(gram, label, sample_weight=weights)
    yi = label[clf.support_[0]]
    xi = X[clf.support_[0], :]
    f = DecisionFunction(kernel, clf, X, label)
    axes[1] = draw_contour(f.eval, [-50,50,50,-50], plot=axes[1], density=0.5)
    axes[1].plot(X[:m,0],X[:m,1], "bo")
    axes[1].plot(X[m:,0],X[m:,1], "ro")
    axes[1].plot(X[clf.support_,0],X[clf.support_,1], "go")

    plt.show()
