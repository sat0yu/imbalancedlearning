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

if __name__ == '__main__':
    N = 1000
    rate = 30.
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

    #kernel = GaussKernel(0.0010)
    #kernel = PolyKernel(7)
    kernel = FloatLinearKernel()
    gram = kernel.gram(X)
    clf = svm.SVC(kernel='precomputed')
    clf.fit(gram, label)

    yi = label[clf.support_[0]]
    xi = X[clf.support_[0], :]
    f = create_dicision_function(kernel, clf, X, label)
    plt = draw_contour(f, [-50,50,50,-50], plot=plt, density=0.5)

    plt.plot(X[:m,0],X[:m,1], "bo")
    plt.plot(X[m:,0],X[m:,1], "ro")
    plt.plot(X[clf.support_,0],X[clf.support_,1], "go")
    plt.show()
