#coding: utf-8;
import numpy as np
import os
import sys
from sklearn import svm

import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
from kernel import *
from mlutil import *

class DifferentErrorCost():
    def __init__(self, kernel=None, beta=1., magic=1.):
        self.kernel = kernel if kernel is not None else GaussKernel(beta)
        self.magic = magic

    def fit(self, posData, negData, label=[1,-1]):
        # equip weight of each class
        numPos, numNeg = float(len(posData)), float(len(negData))
        if numPos < numNeg:
            cPos, cNeg = 1., numPos / numNeg
        else:
            cPos, cNeg = numNeg / numPos, 1.

        # equip sample with these label
        posData = np.c_[ [label[0]]*posData.shape[0], posData]
        negData = np.c_[ [label[1]]*negData.shape[0], negData]

        # concatenate sample matrices
        self.sample = np.r_[posData, negData]

        # ready and fit SVM to given sample
        self.clf = svm.SVC(kernel='precomputed', class_weight={label[0]:cPos, label[1]:cNeg})
        gram = self.kernel.gram(self.sample[:,1:])
        self.clf.fit(gram, self.sample[:,0])

    def predict(self, target):
        mat = self.kernel.matrix(target, self.sample[:,1:])
        return self.clf.predict(mat)

class ProbFuzzySVM():
    def __init__(self, beta, magic):
        self.beta = beta
        self.magic = magic

    def fit(self, posData, negData, label=[1,-1]):
        # equip weights based on KDE
        kde = KernelDensityEstimater(self.beta)

        kde.fit(posData)
        posProb = kde.estimate(posData)
        kde.fit(negData)
        negProb = kde.estimate(negData)

        # create weights array, putting positive's one before negative' one
        self.weights = self.magic * np.r_[posProb, negProb]

        # equip sample with these label
        posData = np.c_[ [label[0]]*posData.shape[0], posData]
        negData = np.c_[ [label[1]]*negData.shape[0], negData]

        # concatenate sample matrices, putting positive's one before negative' one
        self.sample = np.r_[posData, negData]

        # ready and fit SVM to given sample
        self.clf = svm.SVC(kernel='precomputed')
        self.kernel = GaussKernel(self.beta)
        gram = self.kernel.gram(self.sample[:,1:])
        self.clf.fit(gram, self.sample[:,0], sample_weight=self.weights)

    def predict(self, target):
        mat = self.kernel.matrix(target, self.sample[:,1:])
        return self.clf.predict(mat)

def procedure(numTest, numTrain, classRatio):
    mean, cov = [-10, -10], [[50,0],[0,100]]
    posDist = NormalDistribution(mean, cov)

    mean, cov = [10, 10], [[100,0],[0,50]]
    negDist = NormalDistribution(mean, cov)

    id = ImbalancedData(posDist, negDist, classRatio)
    trainset = id.getSample(numTrain)
    testset = id.getSample(numTest)
    label, X = trainset[:,0], trainset[:,1:]
    answer, Y = testset[:,0], testset[:,1:]

    print "given positive samples (train): ", len(label[label[:]==1])
    print "given negative samples (train): ", len(label[label[:]==-1])
    print "given positive samples (test): ", len(answer[answer[:]==1])
    print "given negative samples (test): ", len(answer[answer[:]==-1])

    # params definition
    magic = 20000
    beta = 0.005

    # default SVM
    clf = svm.SVC(kernel='rbf', gamma=beta)
    clf.fit(X, label)
    predict = clf.predict(Y)
    acc,accPos,accNeg,g = evaluation(predict, answer)
    print "[svm] ", "acc: ", acc
    print "[svm] ", "acc on pos: ", accPos
    print "[svm] ", "acc on neg: ", accNeg
    print "[svm] ", "g-mean: ", g

    # DEC
    dec = DifferentErrorCost(beta=beta, magic=magic)
    dec.fit(X[label[:]==1,:], X[label[:]==-1,:])
    predict = dec.predict(Y)
    acc,accPos,accNeg,g = evaluation(predict, answer)
    print "[DEC] ", "acc: ", acc
    print "[DEC] ", "acc on pos: ", accPos
    print "[DEC] ", "acc on neg: ", accNeg
    print "[DEC] ", "g-mean: ", g

    # ProvFuzzySVM
    pfsvm = ProbFuzzySVM(beta, magic)
    pfsvm.fit(X[label[:]==1,:], X[label[:]==-1,:])
    predict = pfsvm.predict(Y)
    acc,accPos,accNeg,g = evaluation(predict, answer)
    print "[prob_fsvm] ", "acc: ", acc
    print "[prob_fsvm] ", "acc on pos: ", accPos
    print "[prob_fsvm] ", "acc on neg: ", accNeg
    print "[prob_fsvm] ", "g-mean: ", g

if __name__ == '__main__':
    for cr in [1., 2., 5., 10., 20., 50., 100.]:
        #procedure(5000, 1000, cr)
        procedure(1250, 250, cr)
