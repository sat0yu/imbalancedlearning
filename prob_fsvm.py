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

def createSamplesFromNormalDistribution(numTrain, numTest, dim=2, classRatio=1.):
    #[ToDo] should modisy to assign means ans covariances.

    X = np.zeros((numTrain,dim))
    Y = np.zeros((numTest,dim))
    label = np.zeros(numTrain)
    answer = np.zeros(numTest)
    mTrain = int( numTrain * ( 1 / (classRatio+1) ) )
    mTest = int( numTest * ( 1 / (classRatio+1) ) )

    mean = [-10, -10]
    cov = [[50,0],[0,100]]
    label[:mTrain] = 1
    answer[:mTest] = 1
    X[:mTrain,0], X[:mTrain,1] = np.random.multivariate_normal(mean, cov, mTrain).T
    Y[:mTest,0], Y[:mTest,1] = np.random.multivariate_normal(mean, cov, mTest).T

    mean = [10, 10]
    cov = [[100,0],[0,50]]
    label[mTrain:] = -1
    answer[mTest:] = -1
    X[mTrain:,0], X[mTrain:,1] = np.random.multivariate_normal(mean, cov, numTrain-mTrain).T
    Y[mTest:,0], Y[mTest:,1] = np.random.multivariate_normal(mean, cov, numTest-mTest).T

    print "given positive samples (train): ", mTrain
    print "given negative samples (train): ", numTrain-mTrain
    print "given positive samples (test): ", mTest
    print "given negative samples (test): ", numTest-mTest

    return (X,label,Y,answer)

def evaluation(predict, answer, posLabel=1, negLabel=-1):
    idxPos = answer[:]==posLabel
    idxNeg = answer[:]==negLabel
    numPos = len(answer[idxPos])
    numNeg = len(answer[idxNeg])

    acc = sum(predict == answer) / float(len(answer))
    accPos = sum(predict[idxPos] == answer[idxPos]) / float(numPos)
    accNeg = sum(predict[idxNeg] == answer[idxNeg]) / float(numNeg)
    g = np.sqrt(accPos * accNeg)

    return (acc,accPos,accNeg,g)

def procedure(numTest, numTrain, classRatio):
    #[ToDo] should modisy to assign dim, magic, and beta param.

    dim = 2
    mTrain = int( numTrain * ( 1 / (classRatio+1) ) )
    mTest = int( numTest * ( 1 / (classRatio+1) ) )
    X,label,Y,answer = createSamplesFromNormalDistribution(numTrain, numTest, dim, classRatio)

    magic = 20000
    beta = 0.005

    kde = KernelDensityEstimater(dim, beta)
    kde.estimate(X[:mTrain,:])
    posWeight = magic*np.array([ kde.prob(xi) for xi in X[:mTrain] ])
    print posWeight

    kde.estimate(X[mTrain:,:])
    negWeight = magic*np.array([ kde.prob(xi) for xi in X[mTrain:] ])
    print negWeight
    weights = np.r_[posWeight, negWeight]

    kernel = GaussKernel(beta)
    gram = kernel.gram(X)
    mat = kernel.matrix(Y, X)

    clf = svm.SVC(kernel='precomputed')
    clf.fit(gram, label)
    predict = clf.predict(mat)
    acc,accPos,accNeg,g = evaluation(predict, answer)
    print "[svm] ", "acc: ", acc
    print "[svm] ", "acc on pos: ", accPos
    print "[svm] ", "acc on neg: ", accNeg
    print "[svm] ", "g-mean: ", g

    clf = svm.SVC(kernel='precomputed')
    clf.fit(gram, label, sample_weight=weights)
    predict = clf.predict(mat)
    acc,accPos,accNeg,g = evaluation(predict, answer)
    print "[prob_fsvm] ", "acc: ", acc
    print "[prob_fsvm] ", "acc on pos: ", accPos
    print "[prob_fsvm] ", "acc on neg: ", accNeg
    print "[prob_fsvm] ", "g-mean: ", g

if __name__ == '__main__':
    for cr in [1., 2., 5., 10., 20., 50., 100.]:
        #procedure(5000, 1000, cr)
        procedure(1250, 250, cr)
