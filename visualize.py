#coding: utf-8;
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from matplotlib import lines
from sklearn import svm

if __name__ == '__main__':
    N = 1000
    rate = 10.
    X = np.zeros((N,3))
    m = int( N * ( 1 / (rate+1) ) )

    mean = [-25, -25]
    cov = [[50,0],[0,50]]
    X[:m,0] = 1
    X[:m,1], X[:m,2] = np.random.multivariate_normal(mean, cov, m).T

    mean = [10, 10]
    cov = [[75,0],[0,75]]
    X[m:,0] = -1
    X[m:,1], X[m:,2] = np.random.multivariate_normal(mean, cov, N-m).T

    print "positive: ", len(X[X[:,0]>0,:])
    print "negative: ", len(X[X[:,0]<0,:])

    #clf = svm.SVC(kernel="poly", degree=2)
    #clf = svm.SVC(kernel="rbf", gamma=0.01)
    clf = svm.SVC(kernel="linear")
    clf.fit(X[:,1:], X[:,0])

    I = np.arange(-50,50)
    J = np.arange(-50,50)
    i,j = np.meshgrid(I,J)
    K = np.zeros((100,100))
    for s in range(100):
        for t in range(100):
            K[s,t] = clf.decision_function([i[s,t],j[s,t]])
    CS = plt.contour(I, J, K)
    plt.clabel(CS)
        
    plt.plot(X[X[:,0]>0,1],X[X[:,0]>0,2], "bo")
    plt.plot(X[X[:,0]<0,1],X[X[:,0]<0,2], "ro")
    plt.show()
