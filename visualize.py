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
    N = 500
    rate = 10.
    X = np.zeros((N,2))
    label = np.zeros(N)
    m = int( N * ( 1 / (rate+1) ) )

    mean = [-25, -25]
    cov = [[50,0],[0,50]]
    label[:m] = 1
    X[:m,0], X[:m,1] = np.random.multivariate_normal(mean, cov, m).T

    mean = [25, 25]
    cov = [[75,0],[0,75]]
    label[m:] = -1
    X[m:,0], X[m:,1] = np.random.multivariate_normal(mean, cov, N-m).T

    print "positive: ", m
    print "negative: ", N-m

    kernel = GaussKernel(0.005)
    #kernel = FloatLinearKernel()
    gram = kernel.gram(X)
    clf = svm.SVC(kernel='precomputed')
    clf.fit(gram, label)

    cclf = svm.SVC(kernel="rbf", gamma=0.005)
    #cclf = svm.SVC(kernel='linear')
    cclf.fit(X,label)

    print 'support vector indices: ', clf.support_
    print 'coefficients of support vectors(whose sign are reversed): ', clf.dual_coef_
    print 'constants in dicision funcitons: ', clf.intercept_

    yi = label[clf.support_[0]]
    xi = X[clf.support_[0], :]
    sv = X[clf.support_, :]
    coef = clf.dual_coef_[0]
    b = yi + np.sum([ coef[j] * kernel.val(xi, xj) for j,xj in enumerate(sv) ])
    f = create_dicision_function(kernel, -coef, label, sv)
    print 'calculted bias b: ', b
    print 'df calculted bias b: ', f(np.array([0.,0.]))

    plt = draw_contour(f, [-50,50,50,-50], plot=plt, linewidths=3, colors='r')
    plt = draw_contour(cclf.decision_function, [-50,50,50,-50], plot=plt, colors='b')

    plt.plot(X[:m,0],X[:m,1], "bo")
    plt.plot(X[m:,0],X[m:,1], "ro")
    plt.plot(X[clf.support_,0],X[clf.support_,1], "go")
    plt.show()
